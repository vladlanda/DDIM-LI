"""
model_ar.py — Autoregressive single-step EDM denoiser for METSAT.

Architecture is identical to model.py (same UNet / EDMPrecond backbone)
but the MultiStepDenoiser is replaced by ARDenoiser which:
  - Predicts ONE next-step residual (no lead-time embedding needed)
  - At inference rolls out autoregressively for T_ar steps
  - Context is always the last T_in observed / predicted frames

Key difference vs direct multi-step model
------------------------------------------
  Direct model : predicts residual for ANY lead time in one shot
                 (lead-time embedding selects which step)
  AR model     : always predicts "next step" residual
                 (no lead-time needed — context carries temporal info)

This also means in_channels = C * (T_in + 1) instead of C * (T_in + 2)
because we no longer need a separate channel-mask plane fed through the
lead-time conditioning. The channel mask is still used for loss weighting
but not injected into the network input.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Re-use all building blocks from model.py
from model import (
    FourierEmbedding,
    AdaGroupNorm,
    ResBlock,
    SelfAttentionBlock,
    UNet,
    EDMPrecond,
    EDMSchedule,
    edm_sampler,
    spectral_loss,
    channel_weighted_mse,
)


# ===================================================================
# AR Preconditioner  (no lead-time embedding)
# ===================================================================

class ARPrecond(nn.Module):
    """
    EDM preconditioner for the AR model.

    Identical to EDMPrecond except the UNet receives no lead_time
    argument — the temporal position is implicit in the rolling context.

    Input concatenation:
        [c_in * x_noisy  | context_flat | ch_mask_spatial]
        shapes:
            x_noisy     : (B, C, H, W)
            context_flat: (B, T_in*C, H, W)
            ch_mask     : (B, C, H, W)  broadcast from (B, C)
        total in_channels = C + T_in*C + C = C*(T_in + 2)

    This matches model.py's input layout so the same UNet architecture
    can be used.  The only difference is that lead_time is set to a
    constant zero tensor (the UNet embedding maps it to a neutral vector).
    """
    def __init__(self, unet: UNet, sigma_data: float = 0.5):
        super().__init__()
        self.unet       = unet
        self.sigma_data = sigma_data

    def forward(
        self,
        x_noisy:  torch.Tensor,   # (B, C, H, W)
        sigma:    torch.Tensor,   # (B,)
        context:  torch.Tensor,   # (B, T_in*C, H, W)  pre-flattened
        ch_mask:  torch.Tensor,   # (B, C)
    ) -> torch.Tensor:
        sd = self.sigma_data

        c_skip  = sd**2 / (sigma**2 + sd**2)
        c_out   = sigma * sd / (sigma**2 + sd**2).sqrt()
        c_in    = 1.0   / (sigma**2 + sd**2).sqrt()
        c_noise = sigma.log() / 4.0

        c_skip = c_skip[:, None, None, None]
        c_out  = c_out [:, None, None, None]
        c_in   = c_in  [:, None, None, None]

        x_in_scaled  = c_in * x_noisy
        mask_spatial = ch_mask[:, :, None, None].expand_as(x_noisy)
        net_input    = torch.cat([x_in_scaled, context, mask_spatial], dim=1)

        # Pass lead_time=zeros — the UNet still has a lead-time pathway
        # but a constant input means it contributes nothing, effectively
        # disabling the conditioning without changing the architecture.
        lead_time = torch.zeros(x_noisy.shape[0], device=x_noisy.device)

        raw = self.unet(net_input, c_noise, lead_time, ch_mask)
        return c_skip * x_noisy + c_out * raw


# ===================================================================
# AR Denoiser — single-step forward + autoregressive rollout
# ===================================================================

class ARDenoiser(nn.Module):
    """
    Wraps ARPrecond for training and autoregressive inference.

    Training: one forward call predicts the next-step residual.
    Inference: roll out for T_ar steps by feeding each prediction
               back as the most recent context frame.
    """
    def __init__(self, precond: ARPrecond, T_in: int, dt_min: int = 10):
        super().__init__()
        self.precond = precond
        self.T_in    = T_in
        self.dt_min  = dt_min

    def forward(
        self,
        x_noisy:  torch.Tensor,   # (B, C, H, W)  noisy next-step residual
        sigma:    torch.Tensor,   # (B,)
        context:  torch.Tensor,   # (B, T_in, C, H, W)  rolling context
        ch_mask:  torch.Tensor,   # (B, C)
    ) -> torch.Tensor:            # (B, C, H, W)  denoised residual
        B, T_in, C, H, W = context.shape
        ctx_flat = context.view(B, T_in * C, H, W)
        return self.precond(x_noisy, sigma, ctx_flat, ch_mask)

    @torch.no_grad()
    def rollout(
        self,
        context_abs: torch.Tensor,  # (B, T_in, C, H, W) absolute normalised frames
        ch_mask:     torch.Tensor,  # (B, C)
        sampler_fn,                  # callable(denoiser_fn, shape, device) -> tensor
        device:      torch.device,
        T_ar:        int,            # number of AR steps to generate
    ) -> torch.Tensor:              # (B, T_ar, C, H, W) absolute normalised frames
        """
        Autoregressive rollout:
          1. Predict next residual from current T_in context frames.
          2. Reconstruct absolute frame: pred_abs = residual + last_ctx.
          3. Slide context window: drop oldest frame, append predicted frame.
          4. Repeat T_ar times.
        """
        B, T_in, C, H, W = context_abs.shape
        ctx = context_abs.clone()   # rolling window (B, T_in, C, H, W)
        predictions = []

        for _ in range(T_ar):
            last_abs = ctx[:, -1]   # (B, C, H, W)

            def denoiser_fn(x, sigma):
                return self(x, sigma, ctx, ch_mask)

            pred_residual = sampler_fn(denoiser_fn, (B, C, H, W), device)
            pred_abs      = pred_residual + last_abs   # residual → absolute

            predictions.append(pred_abs)
            # Slide window: drop oldest, append prediction
            ctx = torch.cat([ctx[:, 1:], pred_abs.unsqueeze(1)], dim=1)

        return torch.stack(predictions, dim=1)   # (B, T_ar, C, H, W)


# ===================================================================
# AR training loss
# ===================================================================

def ar_training_loss(
    denoiser:       ARDenoiser,
    schedule:       EDMSchedule,
    batch:          dict,
    device:         torch.device,
    cfg_drop_prob:  float = 0.15,
    spectral_weight: float = 0.1,
    li_weight:      float = 3.0,
) -> torch.Tensor:
    """
    EDM training loss for the AR model.

    Each batch item has T_out target frames.  We randomly sample ONE
    target step per item — this is the "next step" the model should
    predict given the context ending at the previous frame.

    Context for step t = [frame_{T_in-t_offset}, ..., frame_{t-1}]
    (the T_in frames immediately preceding the target).

    Because the dataset stores residuals relative to last_ctx (frame
    at t=0), we reconstruct the rolling absolute context on-the-fly.
    """
    context  = batch["context"].to(device)   # (B, T_in, C, H, W) normalised abs
    target   = batch["target"].to(device)    # (B, T_out, C, H, W) normalised residuals
    tgt_mask = batch["tgt_mask"].to(device)  # (B, T_out, C)
    last_ctx = batch["last_ctx"].to(device)  # (B, C, H, W)

    B, T_out, C, H, W = target.shape

    # Randomly sample one target step per batch item
    step_idx = torch.randint(0, T_out, (B,), device=device)

    # Reconstruct absolute target frames from residuals
    # target[b, t] is the residual relative to last_ctx[b]
    # absolute[b, t] = target[b, t] + last_ctx[b]
    target_abs = target + last_ctx[:, None]  # (B, T_out, C, H, W)

    # The context for predicting step t is the T_in frames ending at t-1.
    # context (from dataset) = frames[-T_in:] ending at last_ctx.
    # For step 0: use the original context window (ends at last_ctx).
    # For step t>0: slide window to include t-1 predicted absolute frames.
    # At training time we use GROUND TRUTH frames (teacher forcing).
    #
    # Full frame sequence (absolute): [ctx_0, ..., ctx_{T_in-1}, abs_0, ..., abs_{T_out-1}]
    # Context for step t: frames[t : t+T_in]
    all_frames = torch.cat([context, target_abs], dim=1)  # (B, T_in+T_out, C, H, W)

    ctx_for_step = torch.stack(
        [all_frames[b, step_idx[b]:step_idx[b] + T_in] for b in range(B)],
        dim=0,
    )  # (B, T_in, C, H, W)

    # Ground-truth next residual = target_abs[step] - target_abs[step-1]
    # (for step 0: target_abs[0] - last_ctx)
    prev_abs = torch.where(
        (step_idx == 0).view(B, 1, 1, 1).expand_as(last_ctx),
        last_ctx,
        torch.stack([target_abs[b, max(0, step_idx[b].item() - 1)] for b in range(B)]),
    )
    y       = target_abs[torch.arange(B), step_idx] - prev_abs   # residual
    ch_mask = tgt_mask[torch.arange(B), step_idx]                 # (B, C)

    # EDM noise
    sigma   = schedule.sample_sigma(B, device)
    x_noisy = y + torch.randn_like(y) * sigma[:, None, None, None]

    # CFG dropout
    if cfg_drop_prob > 0:
        drop = torch.rand(B, device=device) < cfg_drop_prob
        ctx_used = ctx_for_step.clone()
        ctx_used[drop] = 0.0
    else:
        ctx_used = ctx_for_step

    pred = denoiser(x_noisy, sigma, ctx_used, ch_mask)

    lw  = schedule.edm_loss_weight(sigma)[:, None, None, None]
    mse = channel_weighted_mse(pred * lw.sqrt(), y * lw.sqrt(), ch_mask, li_weight)
    spec = spectral_loss(pred, y, spectral_weight)
    return mse + spec
