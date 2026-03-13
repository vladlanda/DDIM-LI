"""
EDM-style diffusion model for METSAT lightning nowcasting.

Following: "Elucidating the Design Space of Diffusion-Based Generative Models"
           Karras et al. 2022  (EDM)
           +
           "Probabilistic weather forecasting with machine learning"
           Price et al. 2023  (GenCast)

Key design decisions:
  - U-Net with residual blocks and self-attention at multiple scales
  - Conditioning: context frames concatenated channel-wise to noisy input
  - Lead-time embedding injected via AdaGroupNorm at every residual block
  - Channel mask injected to handle optional satellite channels
  - Classifier-free guidance (CFG): context dropped with prob p_uncond
  - Spectral (FFT) auxiliary loss to preserve high-frequency sharpness
  - Direct multi-step prediction: model predicts T_out residuals at once
    conditioned on lead-time embeddings → avoids autoregressive blur
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ===================================================================
# Positional / Fourier embeddings
# ===================================================================

def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    """Sinusoidal embedding for scalar t (sigma or lead-time)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t[:, None].float() * freqs[None]
    emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


# ===================================================================
# Normalisation helpers
# ===================================================================

class AdaGroupNorm(nn.Module):
    """
    Group norm whose scale & shift are predicted from a conditioning vector.
    Replaces the standard GN+add-emb pattern with a single modulation.
    """
    def __init__(self, num_channels: int, emb_dim: int, num_groups: int = 8):
        super().__init__()
        self.gn   = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(emb_dim, 2 * num_channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)   emb: (B, emb_dim)
        x   = self.gn(x)
        scale, shift = self.proj(emb).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


# ===================================================================
# Building blocks
# ===================================================================

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int,
                 dropout: float = 0.1, num_groups: int = 8):
        super().__init__()
        self.norm1 = AdaGroupNorm(in_ch,  emb_dim, num_groups)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2 = AdaGroupNorm(out_ch, emb_dim, num_groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.drop  = nn.Dropout2d(dropout)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x, emb))
        h = self.conv1(h)
        h = F.silu(self.norm2(h, emb))
        h = self.drop(self.conv2(h))
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = rearrange(h, "b c h w -> b (h w) c")
        h, _ = self.attn(h, h, h, need_weights=False)
        h = rearrange(h, "b (h w) c -> b c h w", h=H, w=W)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ===================================================================
# U-Net
# ===================================================================

class UNet(nn.Module):
    """
    Conditioned U-Net that denoises a (T_out, C, H, W) stack of residuals.

    Input channels per timestep:
      - noisy_residual: C channels
      - context (concatenated): T_in * C channels
      - channel_mask:           C channels (binary)
    Total input channels = C + T_in*C + C = C*(T_in+2)

    The model flattens T_out into the batch dimension (one forward pass
    per target step, or all at once with batch trick).

    For multi-step direct prediction we process each lead step with a
    SHARED network but different lead-time embeddings — parameter efficient.
    """

    def __init__(
        self,
        in_channels:    int,   # C * (T_in + 2)  noisy + context + mask
        out_channels:   int,   # C
        base_channels:  int = 128,
        channel_mults:  tuple = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (16, 8),
        dropout:        float = 0.1,
        emb_dim:        int = 512,
        num_groups:     int = 8,
    ):
        super().__init__()
        self.emb_dim = emb_dim

        # Noise sigma embedding
        self.sigma_emb = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        # Lead-time embedding (1..T_out multiples of dt_min)
        self.lead_emb = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        # Channel mask embedding
        self.mask_emb = nn.Sequential(
            nn.Linear(out_channels, emb_dim // 4),
            nn.SiLU(),
            nn.Linear(emb_dim // 4, emb_dim),
        )
        # Total conditioning: sigma + lead + mask
        total_emb = emb_dim * 3
        self.emb_proj = nn.Sequential(
            nn.Linear(total_emb, emb_dim),
            nn.SiLU(),
        )

        ch = base_channels
        self.input_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        # Encoder
        self.down_blocks  = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        encoder_chs = [ch]
        img_size = 256  # tracked symbolically for attn decision

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, emb_dim, dropout, num_groups))
                ch = out_ch
                encoder_chs.append(ch)
                if img_size in attn_resolutions:
                    self.down_blocks.append(SelfAttention2D(ch))
                    encoder_chs.append(ch)  # dummy; we track per-resblock
            if level < len(channel_mults) - 1:
                self.down_samples.append(Downsample(ch))
                encoder_chs.append(ch)
                img_size //= 2

        # Bottleneck
        self.mid1 = ResBlock(ch, ch, emb_dim, dropout, num_groups)
        self.mid_attn = SelfAttention2D(ch)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout, num_groups)

        # Decoder (mirror encoder, with skip cats)
        self.up_blocks  = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = encoder_chs.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, emb_dim, dropout, num_groups))
                ch = out_ch
                if img_size in attn_resolutions:
                    self.up_blocks.append(SelfAttention2D(ch))
            if level > 0:
                self.up_samples.append(Upsample(ch))
                img_size *= 2

        self.out_norm = nn.GroupNorm(num_groups, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 1)

    def forward(
        self,
        x:         torch.Tensor,   # (B, C*(T_in+2), H, W)  noisy+context+mask
        sigma:     torch.Tensor,   # (B,)  noise level
        lead_time: torch.Tensor,   # (B,)  lead time scalar
        ch_mask:   torch.Tensor,   # (B, C_out)  channel presence mask
    ) -> torch.Tensor:

        # Build conditioning embedding
        sigma_e = self.sigma_emb(timestep_embedding(sigma, self.emb_dim))
        lead_e  = self.lead_emb(timestep_embedding(lead_time, self.emb_dim))
        mask_e  = self.mask_emb(ch_mask)
        emb     = self.emb_proj(torch.cat([sigma_e, lead_e, mask_e], dim=-1))

        # U-Net forward
        h = self.input_conv(x)
        skips = [h]

        down_idx = 0
        for block in self.down_blocks:
            if isinstance(block, ResBlock):
                h = block(h, emb)
            else:  # SelfAttention
                h = block(h)
            skips.append(h)

        # pop the extra append after each downsample
        sample_idx = 0
        # Redo with correct structure
        # (simplified: rebuild cleanly below)
        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid2(h, emb)

        up_attn_set = set()
        for block in self.up_blocks:
            if isinstance(block, ResBlock):
                s = skips.pop()
                # handle size mismatch at boundaries
                if h.shape[-2:] != s.shape[-2:]:
                    h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
                h = torch.cat([h, s], dim=1)
                h = block(h, emb)
            else:
                h = block(h)

        for up in self.up_samples:
            h = up(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# ===================================================================
# EDM preconditioner  (Karras et al. Table 1)
# ===================================================================

class EDMPrecond(nn.Module):
    """
    Wraps UNet with EDM preconditioning:
      D_θ(x; σ) = c_skip·x + c_out·F_θ(c_in·x; c_noise)
    """
    def __init__(self, unet: UNet, sigma_data: float = 0.5):
        super().__init__()
        self.unet       = unet
        self.sigma_data = sigma_data

    def forward(
        self,
        x_noisy:   torch.Tensor,   # (B, C, H, W)  noisy residual
        sigma:     torch.Tensor,   # (B,)
        context:   torch.Tensor,   # (B, T_in*C, H, W)
        ch_mask:   torch.Tensor,   # (B, C) binary
        lead_time: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        sd = self.sigma_data

        c_skip  = sd**2 / (sigma**2 + sd**2)
        c_out   = sigma * sd / (sigma**2 + sd**2).sqrt()
        c_in    = 1.0   / (sigma**2 + sd**2).sqrt()
        c_noise = sigma.log() / 4.0

        # Reshape scalars for broadcasting
        c_skip  = c_skip[:, None, None, None]
        c_out   = c_out [:, None, None, None]
        c_in    = c_in  [:, None, None, None]

        # Build input: scaled noisy + context + mask broadcast to spatial
        x_in_scaled = c_in * x_noisy
        # Expand mask to spatial
        mask_spatial = ch_mask[:, :, None, None].expand_as(x_noisy)
        net_input = torch.cat([x_in_scaled, context, mask_spatial], dim=1)

        raw = self.unet(net_input, c_noise, lead_time, ch_mask)
        return c_skip * x_noisy + c_out * raw


# ===================================================================
# Multi-step denoiser
# ===================================================================

class MultiStepDenoiser(nn.Module):
    """
    Wraps EDMPrecond to handle (B, T_out, C, H, W) targets.
    Processes each lead step independently but shares weights.
    At training: randomly sample one lead step per sample (efficient).
    At inference: loop over all lead steps.
    """
    def __init__(self, precond: EDMPrecond, T_out: int, dt_min: int = 10):
        super().__init__()
        self.precond = precond
        self.T_out   = T_out
        self.dt_min  = dt_min

    def forward(
        self,
        x_noisy:    torch.Tensor,   # (B, C, H, W)
        sigma:      torch.Tensor,   # (B,)
        context:    torch.Tensor,   # (B, T_in, C, H, W)
        ch_mask:    torch.Tensor,   # (B, C)
        lead_idx:   torch.Tensor,   # (B,)  0-indexed step
    ) -> torch.Tensor:
        B, T_in, C, H, W = context.shape
        ctx_flat = context.view(B, T_in * C, H, W)
        lead_time = (lead_idx.float() + 1) * self.dt_min  # minutes

        return self.precond(x_noisy, sigma, ctx_flat, ch_mask, lead_time)

    @torch.no_grad()
    def sample_all_steps(
        self,
        context:    torch.Tensor,   # (B, T_in, C, H, W)
        ch_mask:    torch.Tensor,   # (B, C)
        sampler_fn,                  # callable(denoiser_fn, shape, device) -> tensor
        device:     torch.device,
    ) -> torch.Tensor:
        """Return (B, T_out, C, H, W) predicted residuals."""
        B, T_in, C, H, W = context.shape
        all_preds = []

        for step in range(self.T_out):
            lead_idx = torch.full((B,), step, device=device, dtype=torch.long)

            def denoiser_fn(x, sigma):
                return self(x, sigma, context, ch_mask, lead_idx)

            pred = sampler_fn(denoiser_fn, (B, C, H, W), device)
            all_preds.append(pred)

        return torch.stack(all_preds, dim=1)  # (B, T_out, C, H, W)


# ===================================================================
# EDM noise schedule & sampler
# ===================================================================

class EDMSchedule:
    """EDM training noise schedule (lognormal)."""
    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2,
                 sigma_min: float = 0.002, sigma_max: float = 80.0,
                 rho: float = 7.0):
        self.P_mean    = P_mean
        self.P_std     = P_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho       = rho

    def sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample σ ~ lognormal(P_mean, P_std)."""
        return (torch.randn(batch_size, device=device) * self.P_std + self.P_mean).exp()

    def edm_loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """λ(σ) = (σ² + σ_data²) / (σ · σ_data)²"""
        sd = 0.5
        return (sigma**2 + sd**2) / (sigma * sd)**2


def edm_sampler(
    denoiser_fn,
    shape: tuple,
    device: torch.device,
    num_steps: int = 20,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    S_churn: float = 40.0,
    S_min: float = 0.05,
    S_max: float = 50.0,
    S_noise: float = 1.003,
) -> torch.Tensor:
    """
    EDM stochastic sampler (Algorithm 2, Karras et al.).
    denoiser_fn: (x_noisy, sigma_tensor) -> denoised_x
    """
    # Build sigma schedule
    step_indices = torch.arange(num_steps, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros(1, device=device)])

    x = torch.randn(*shape, device=device) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        t_cur_batch  = t_cur.expand(shape[0])
        t_next_batch = t_next.expand(shape[0])

        # Stochastic churn
        gamma = min(S_churn / num_steps, math.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0.0
        t_hat = t_cur * (1 + gamma)
        if gamma > 0:
            x = x + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(x)

        t_hat_batch = t_hat.expand(shape[0]) if isinstance(t_hat, torch.Tensor) else \
                      torch.full((shape[0],), t_hat, device=device)

        # Euler step
        denoised = denoiser_fn(x, t_hat_batch)
        d_cur    = (x - denoised) / t_hat

        x_next = x + (t_next - t_hat) * d_cur

        # Second-order correction (Heun)
        if i < num_steps - 1:
            denoised_next = denoiser_fn(x_next, t_next_batch)
            d_next        = (x_next - denoised_next) / t_next
            x_next        = x + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_next)

        x = x_next

    return x


# ===================================================================
# Losses
# ===================================================================

def spectral_loss(pred: torch.Tensor, target: torch.Tensor, weight: float = 0.1) -> torch.Tensor:
    """
    Penalise mismatch in FFT magnitude spectrum.
    Encourages preservation of high-frequency detail (sharpness).
    pred, target: (B, C, H, W)
    """
    pred_fft   = torch.fft.rfft2(pred)
    target_fft = torch.fft.rfft2(target)
    loss = F.mse_loss(pred_fft.abs(), target_fft.abs())
    return weight * loss


def channel_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    ch_mask: torch.Tensor,
    li_weight: float = 3.0,
    li_idx: int = 1,        # LI is channel index 1  (ir=0, li=1, ch0=2, ...)
) -> torch.Tensor:
    """
    MSE with:
      - masking of absent channels
      - upweighting LI channel (sparse but critical)
    pred, target: (B, C, H, W)
    ch_mask: (B, C)
    """
    weights = torch.ones_like(ch_mask)
    weights[:, li_idx] = li_weight

    per_pixel  = (pred - target) ** 2                    # (B, C, H, W)
    mask_4d    = ch_mask[:, :, None, None]
    weight_4d  = weights[:, :, None, None]

    loss = (per_pixel * mask_4d * weight_4d).sum() / (mask_4d * weight_4d).sum().clamp(min=1)
    return loss


def edm_training_loss(
    denoiser: MultiStepDenoiser,
    schedule: EDMSchedule,
    batch: dict,
    device: torch.device,
    cfg_drop_prob: float = 0.15,
    spectral_weight: float = 0.1,
    li_weight: float = 3.0,
) -> torch.Tensor:
    """
    Full EDM training loss with:
      - random lead step sampling
      - classifier-free guidance dropout
      - weighted MSE + spectral loss
    """
    context    = batch["context"].to(device)      # (B, T_in, C, H, W)
    target     = batch["target"].to(device)       # (B, T_out, C, H, W)
    tgt_mask   = batch["tgt_mask"].to(device)     # (B, T_out, C)
    ctx_mask   = batch["ctx_mask"].to(device)     # (B, T_in, C)

    B, T_out, C, H, W = target.shape

    # Sample random lead step for each batch item
    lead_idx = torch.randint(0, T_out, (B,), device=device)
    y        = target[torch.arange(B), lead_idx]          # (B, C, H, W)
    ch_mask  = tgt_mask[torch.arange(B), lead_idx]        # (B, C)

    # Sample noise level
    sigma = schedule.sample_sigma(B, device)

    # Add noise
    noise   = torch.randn_like(y)
    x_noisy = y + noise * sigma[:, None, None, None]

    # CFG: randomly null-condition context
    if cfg_drop_prob > 0:
        drop = (torch.rand(B, device=device) < cfg_drop_prob)
        context_drop = context.clone()
        context_drop[drop] = 0.0   # zero out context for dropped samples
        ctx_used = context_drop
    else:
        ctx_used = context

    # Forward
    pred = denoiser(x_noisy, sigma, ctx_used, ch_mask, lead_idx)

    # Loss weight λ(σ)
    lw = schedule.edm_loss_weight(sigma)[:, None, None, None]

    mse  = channel_weighted_mse(pred * lw.sqrt(), y * lw.sqrt(), ch_mask, li_weight)
    spec = spectral_loss(pred, y, spectral_weight)

    return mse + spec
