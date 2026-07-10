"""
Conditional score-based denoising network for METSAT lightning nowcasting.

Architecture: EDM-style UNet (Karras et al. 2022) with composite task-specific loss.
We do not claim strict ELBO optimisation. We train a denoising network with a
composite loss designed for simultaneous binary skill and calibrated uncertainty:

    L = L_denoise + λ_a · L_asymmetric + λ_n · L_neighbourhood + λ_s · L_spectral

References:
  - Karras et al. 2022  (EDM architecture and sampler)
  - Cui et al. 2019     (effective number weighting)
  - Gao et al. 2022     (asymmetric loss for convective nowcasting)
  - Zhang et al. 2023   (neighbourhood spatial consistency loss)
"""

import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ===================================================================
# Positional embeddings
# ===================================================================

def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    """Sinusoidal embedding for scalar t (sigma or lead-time)."""
    half  = dim // 2
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
    """Group norm with scale & shift predicted from a conditioning vector."""
    def __init__(self, num_channels: int, emb_dim: int, num_groups: int = 8):
        super().__init__()
        self.gn   = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(emb_dim, 2 * num_channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.gn(x)
        scale, shift = self.proj(emb).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


# ===================================================================
# UNet building blocks
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
        # Standard nearest-upsample + conv (learned anti-aliasing).
        # F.interpolate alone is correct but the conv refines the result.
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ===================================================================
# U-Net
# ===================================================================

class UNet(nn.Module):
    """
    Conditioned U-Net denoiser.

    Input layout (channels concatenated):
      [noisy_residual | context_frames | channel_mask]
      = C + T_in*C_ctx + C   total channels
    where C_ctx = C + 1 when binary_li_ctx=True (extra binary LI channel per frame).
    """

    def __init__(
        self,
        in_channels:      int,
        out_channels:     int,
        base_channels:    int   = 128,
        channel_mults:    tuple = (1, 2, 3, 4),
        num_res_blocks:   int   = 2,
        attn_resolutions: tuple = (16, 8),
        dropout:          float = 0.1,
        emb_dim:          int   = 512,
        num_groups:       int   = 8,
        img_size:         int   = 64,
    ):
        super().__init__()
        self.emb_dim    = emb_dim
        self.num_groups = num_groups
        attn_res = set(attn_resolutions)

        # ── Conditioning embeddings ─────────────────────────────────
        def mlp(d): return nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.sigma_emb = mlp(emb_dim)
        self.lead_emb  = mlp(emb_dim)
        self.mask_emb  = nn.Sequential(
            nn.Linear(out_channels, emb_dim // 4), nn.SiLU(),
            nn.Linear(emb_dim // 4, emb_dim),
        )
        self.emb_proj = nn.Sequential(nn.Linear(emb_dim * 3, emb_dim), nn.SiLU())

        # ── Input projection ────────────────────────────────────────
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ── Build encoder plan ──────────────────────────────────────
        # res tracks the spatial resolution as we go deeper.
        # Must be initialised to the actual input image size, not a fixed value.
        # With img_size=64, attn_resolutions=[16,8]: attention fires at levels 2,3.
        # With img_size=256, attn_resolutions=[16,8]: attention fires at levels 4,5.
        enc_plan: list[dict] = []
        skip_channels: list[int] = []
        ch  = base_channels
        # Infer starting resolution from in_channels is not possible here;
        # use the known default. For non-square or non-power-of-2 images,
        # pass img_size explicitly via a future refactor.
        res = img_size   # must match actual spatial input size

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                enc_plan.append({"type": "res", "in": ch, "out": out_ch})
                ch = out_ch
                skip_channels.append(ch)
                if res in attn_res:
                    enc_plan.append({"type": "attn", "ch": ch})
                    skip_channels.append(ch)
            if level < len(channel_mults) - 1:
                enc_plan.append({"type": "down", "ch": ch})
                res //= 2

        # ── Build decoder plan ──────────────────────────────────────
        dec_plan: list[dict] = []
        skips_copy = list(skip_channels)

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = skips_copy.pop()
                dec_plan.append({"type": "res", "in": ch + skip_ch, "out": out_ch})
                ch = out_ch
                if res in attn_res:
                    skips_copy.pop()
                    dec_plan.append({"type": "attn", "ch": ch})
            if level > 0:
                dec_plan.append({"type": "up", "ch": ch})
                res *= 2

        # ── Instantiate modules ─────────────────────────────────────
        self.enc_blocks = nn.ModuleList()
        for p in enc_plan:
            if p["type"] == "res":
                self.enc_blocks.append(ResBlock(p["in"], p["out"], emb_dim, dropout, num_groups))
            elif p["type"] == "attn":
                self.enc_blocks.append(SelfAttention2D(p["ch"], num_heads=max(1, p["ch"] // 64)))
            elif p["type"] == "down":
                self.enc_blocks.append(Downsample(p["ch"]))
        self.enc_plan = enc_plan

        bot_ch = base_channels
        for p in reversed(enc_plan):
            if p["type"] == "res":
                bot_ch = p["out"]; break

        self.mid1     = ResBlock(bot_ch, bot_ch, emb_dim, dropout, num_groups)
        self.mid_attn = SelfAttention2D(bot_ch, num_heads=max(1, bot_ch // 64))
        self.mid2     = ResBlock(bot_ch, bot_ch, emb_dim, dropout, num_groups)

        self.dec_blocks = nn.ModuleList()
        for p in dec_plan:
            if p["type"] == "res":
                self.dec_blocks.append(ResBlock(p["in"], p["out"], emb_dim, dropout, num_groups))
            elif p["type"] == "attn":
                self.dec_blocks.append(SelfAttention2D(p["ch"], num_heads=max(1, p["ch"] // 64)))
            elif p["type"] == "up":
                self.dec_blocks.append(Upsample(p["ch"]))
        self.dec_plan = dec_plan

        final_ch = bot_ch
        for p in dec_plan:
            if p["type"] == "res":
                final_ch = p["out"]

        self.out_norm = nn.GroupNorm(num_groups, final_ch)
        self.out_conv = nn.Conv2d(final_ch, out_channels, 1)

    def forward(
        self,
        x:         torch.Tensor,
        sigma:     torch.Tensor,
        lead_time: torch.Tensor,
        ch_mask:   torch.Tensor,
    ) -> torch.Tensor:

        sigma_e = self.sigma_emb(timestep_embedding(sigma,     self.emb_dim))
        lead_e  = self.lead_emb( timestep_embedding(lead_time, self.emb_dim))
        mask_e  = self.mask_emb(ch_mask.float())
        emb     = self.emb_proj(torch.cat([sigma_e, lead_e, mask_e], dim=-1))

        h     = self.input_conv(x)
        skips = []

        for block, plan in zip(self.enc_blocks, self.enc_plan):
            if plan["type"] == "res":
                h = block(h, emb); skips.append(h)
            elif plan["type"] == "attn":
                h = block(h);      skips.append(h)
            elif plan["type"] == "down":
                h = block(h)

        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid2(h, emb)

        for block, plan in zip(self.dec_blocks, self.dec_plan):
            if plan["type"] == "res":
                s = skips.pop()
                if h.shape[-2:] != s.shape[-2:]:
                    h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
                h = torch.cat([h, s], dim=1)
                h = block(h, emb)
            elif plan["type"] == "attn":
                skips.pop()
                h = block(h)
            elif plan["type"] == "up":
                h = block(h)

        return self.out_conv(F.silu(self.out_norm(h)))


# ===================================================================
# EDM preconditioner  (Karras et al. Table 1)
# ===================================================================

class EDMPrecond(nn.Module):
    """
    D_θ(x; σ) = c_skip·x + c_out·F_θ(c_in·x; c_noise)
    """
    def __init__(self, unet: UNet, sigma_data: float = 1.0):
        super().__init__()
        self.unet       = unet
        self.sigma_data = sigma_data

    def forward(
        self,
        x_noisy:   torch.Tensor,
        sigma:     torch.Tensor,
        context:   torch.Tensor,
        ch_mask:   torch.Tensor,
        lead_time: torch.Tensor,
    ) -> torch.Tensor:
        sd = self.sigma_data
        c_skip  = sd**2 / (sigma**2 + sd**2)
        c_out   = sigma * sd / (sigma**2 + sd**2).sqrt()
        c_in    = 1.0   / (sigma**2 + sd**2).sqrt()
        c_noise = sigma.log() / 4.0

        c_skip = c_skip[:, None, None, None]
        c_out  = c_out [:, None, None, None]
        c_in   = c_in  [:, None, None, None]

        mask_spatial = ch_mask[:, :, None, None].expand_as(x_noisy)
        net_input    = torch.cat([c_in * x_noisy, context, mask_spatial], dim=1)

        raw = self.unet(net_input, c_noise, lead_time, ch_mask)
        return c_skip * x_noisy + c_out * raw


# ===================================================================
# Multi-step denoiser
# ===================================================================

class MultiStepDenoiser(nn.Module):
    """
    Wraps EDMPrecond for (B, T_out, C, H, W) targets.
    Each lead step is processed independently with shared weights.
    """
    def __init__(self, precond: EDMPrecond, T_out: int, dt_min: int = 10):
        super().__init__()
        self.precond = precond
        self.T_out   = T_out
        self.dt_min  = dt_min

    def forward(
        self,
        x_noisy:  torch.Tensor,   # (B, C_data, H, W)  — data channels only
        sigma:    torch.Tensor,
        context:  torch.Tensor,   # (B, T_in, C_ctx, H, W)  C_ctx = C_data or C_data+1
        ch_mask:  torch.Tensor,   # (B, C_data)
        lead_idx: torch.Tensor,
    ) -> torch.Tensor:
        B, T_in, C_ctx, H, W = context.shape
        ctx_flat  = context.reshape(B, T_in * C_ctx, H, W)
        lead_time = (lead_idx.float() + 1) * self.dt_min
        return self.precond(x_noisy, sigma, ctx_flat, ch_mask, lead_time)

    @torch.no_grad()
    def sample_all_steps(
        self,
        context:    torch.Tensor,
        ch_mask:    torch.Tensor,
        sampler_fn,
        device:     torch.device,
    ) -> torch.Tensor:
        """Return (B, T_out, C, H, W) predicted residuals."""
        B, T_in, C_ctx, H, W = context.shape
        # C_data from output conv — not C_ctx (context may have extra binary LI channel)
        C_data = self.precond.unet.out_conv.weight.shape[0]
        all_preds = []
        for step in range(self.T_out):
            lead_idx = torch.full((B,), step, device=device, dtype=torch.long)
            def denoiser_fn(x, sigma, _step=step):
                _lead = torch.full((x.shape[0],), _step, device=device, dtype=torch.long)
                return self(x, sigma, context, ch_mask, _lead)
            all_preds.append(sampler_fn(denoiser_fn, (B, C_data, H, W), device))
        return torch.stack(all_preds, dim=1)


# ===================================================================
# EDM noise schedule & sampler
# ===================================================================

class EDMSchedule:
    """EDM training noise schedule (lognormal)."""
    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2,
                 sigma_min: float = 0.002, sigma_max: float = 80.0,
                 sigma_data: float = 1.0):
        self.P_mean     = P_mean
        self.P_std      = P_std
        self.sigma_min  = sigma_min
        self.sigma_max  = sigma_max
        self.sigma_data = sigma_data  # RMS of training data (residuals)

    def sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample σ ~ lognormal(P_mean, P_std)."""
        return (torch.randn(batch_size, device=device) * self.P_std + self.P_mean).exp()

    def edm_loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        λ(σ) = (σ² + σ_data²) / (σ · σ_data)²    (Karras et al. 2022 Table 1)

        σ_data must be the RMS of the training targets (residuals, not raw frames).
        For normalised residuals this is typically 0.2–0.5, NOT 1.0.
        Use check_residual_stats.py to measure and set sigma_data in default.yaml.
        Wrong sigma_data shifts the transition point where c_skip=0.5, biasing
        the preconditioner toward copying the noisy input at low noise levels.
        """
        sd = self.sigma_data
        return (sigma**2 + sd**2) / (sigma * sd)**2


def edm_sampler(
    denoiser_fn,
    shape:     tuple,
    device:    torch.device,
    num_steps: int   = 20,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho:       float = 7.0,
    S_churn:   float = 40.0,
    S_min:     float = 0.05,
    S_max:     float = 50.0,
    S_noise:   float = 1.003,
) -> torch.Tensor:
    """EDM stochastic sampler — Algorithm 2, Karras et al. 2022."""
    step_indices = torch.arange(num_steps, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros(1, device=device)])

    x = torch.randn(*shape, device=device) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        t_cur_b  = t_cur.expand(shape[0])
        t_next_b = t_next.expand(shape[0])

        gamma = min(S_churn / num_steps, math.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0.0
        t_hat = t_cur * (1 + gamma)
        if gamma > 0:
            x = x + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(x)

        t_hat_b = t_hat.expand(shape[0]) if isinstance(t_hat, torch.Tensor) else \
                  torch.full((shape[0],), float(t_hat), device=device)

        denoised = denoiser_fn(x, t_hat_b)
        d_cur    = (x - denoised) / t_hat
        x_next   = x + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised_next = denoiser_fn(x_next, t_next_b)
            d_next        = (x_next - denoised_next) / t_next
            x_next        = x + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_next)

        x = x_next

    return x


# ===================================================================
# Loss functions
# ===================================================================

def effective_li_weight(
    li_density:  torch.Tensor,
    base_weight: float = 30.0,
    beta:        float = 0.9999,
    min_weight:  float = 1.0,
    max_weight:  float = 200.0,
    ref_density: float = 0.05,
) -> torch.Tensor:
    """
    Per-sample dynamic LI channel weight using the Effective Number of Samples
    formula (Cui et al. 2019, CVPR).

    E_n = (1 - β^n) / (1 - β)

    Weight is normalised so that reference density 0.05 → base_weight.
    Zero-density sequences receive weight=1.0 (learn to predict zero residual).
    """
    eps   = 1e-6
    n     = li_density.clamp(min=eps)
    E_n   = (1.0 - beta ** n)    / (1.0 - beta)
    E_ref = (1.0 - beta ** ref_density) / (1.0 - beta)
    w     = base_weight * (E_ref / E_n)
    w     = torch.where(li_density < eps, torch.ones_like(w), w)
    return w.clamp(min_weight, max_weight)


def channel_weighted_mse(
    pred:      torch.Tensor,
    target:    torch.Tensor,
    ch_mask:   torch.Tensor,
    li_weight: Union[float, torch.Tensor] = 30.0,
    li_idx:    int = 1,
) -> torch.Tensor:
    """
    Masked MSE with per-channel (and optionally per-sample) LI upweighting.
    pred, target : (B, C, H, W)
    ch_mask      : (B, C)
    li_weight    : scalar or (B,) tensor
    """
    weights = torch.ones_like(ch_mask)   # (B, C), float
    # li_weight can be scalar or (B,) tensor — assignment works for both
    weights[:, li_idx] = li_weight if isinstance(li_weight, (int, float)) \
                         else li_weight.to(ch_mask.device)

    per_px  = (pred - target) ** 2
    mask_4d = ch_mask[:, :, None, None]
    w_4d    = weights[:, :, None, None]
    return (per_px * mask_4d * w_4d).sum() / (mask_4d * w_4d).sum().clamp(min=1)


def asymmetric_li_loss(
    pred:           torch.Tensor,
    target:         torch.Tensor,
    last_ctx:       torch.Tensor,
    alpha:          float = 0.1,
    li_idx:         int   = 1,
    norm_threshold: float = 0.16,
) -> torch.Tensor:
    """
    Asymmetric pixel loss for the LI channel.

    Operates in ABSOLUTE normalised space (residual + last_ctx) because:
    - pred and target are RESIDUALS (change from last context frame)
    - A residual ≈ 0 means "LI unchanged", NOT "no lightning"
    - Thresholding residuals confuses "LI continues" with "no LI"
    - We want: GT=0 ↔ no lightning in the ABSOLUTE target frame

    For pixels where absolute GT=0 (no lightning), false positives are
    penalised by factor alpha < 1 — making the model more spatially
    selective, directly reducing FAR.

    For pixels where absolute GT>0 (lightning present), standard MSE applies.

    Following Gao et al. 2022 (EarthFormer) applied to convective nowcasting.

    Args:
        pred, target   : (B, C, H, W) NORMALISED RESIDUALS
        last_ctx       : (B, C, H, W) last context frame (normalised absolute)
        alpha          : FP cost relative to FN (0.1 → FP penalised at 10%%).
        li_idx         : index of LI channel
        norm_threshold : threshold in ABSOLUTE normalised space.
                         norm_threshold=0.16 ≈ 5/255 in physical space.
    """
    # Convert residuals to absolute normalised values
    pred_abs_li   = pred[:, li_idx]   + last_ctx[:, li_idx]   # (B, H, W)
    target_abs_li = target[:, li_idx] + last_ctx[:, li_idx]

    # GT=0: no lightning in absolute target frame
    gt_pos = (target_abs_li >= norm_threshold).float()
    gt_neg = 1.0 - gt_pos

    # Loss on residuals (not absolutes) — we want the model to predict
    # the correct residual, penalised asymmetrically based on absolute GT
    sq_err = (pred[:, li_idx] - target[:, li_idx]) ** 2

    loss = (alpha * gt_neg + gt_pos) * sq_err
    return loss.mean()


def neighbourhood_li_loss(
    pred:      torch.Tensor,
    target:    torch.Tensor,
    li_idx:    int         = 1,
    scales:    List[int]   = [5, 11],
) -> torch.Tensor:
    """
    Neighbourhood consistency loss for the LI channel.

    Penalises spatial displacement by comparing spatially smoothed predictions
    to smoothed targets at multiple neighbourhood scales. A model that places
    lightning in the right general region but slightly wrong pixel position
    will still be penalised by standard MSE; this loss rewards spatial proximity.

    L_nbr = Σ_k ||avg_pool(pred_li, k) - avg_pool(target_li, k)||²

    Following Zhang et al. 2023 (NowcastNet) spatial consistency loss.

    Args:
        scales : kernel sizes for avg_pool2d (in pixels).
                 k=5 ≈ 20km, k=11 ≈ 44km at 4km/pixel resolution.
    """
    # Spatial consistency on residuals is correct — a spatially smooth
    # residual produces a spatially smooth absolute prediction.
    pred_li   = pred[:, li_idx:li_idx+1]      # (B, 1, H, W)
    target_li = target[:, li_idx:li_idx+1]

    loss = 0.0
    for k in scales:
        pad        = k // 2
        pred_sm    = F.avg_pool2d(pred_li,   k, stride=1, padding=pad)
        target_sm  = F.avg_pool2d(target_li, k, stride=1, padding=pad)
        loss       = loss + F.mse_loss(pred_sm, target_sm)
    return loss / len(scales)


def spectral_loss(
    pred:     torch.Tensor,
    target:   torch.Tensor,
    ch_mask:  torch.Tensor,
    li_idx:   int = 1,
) -> torch.Tensor:
    """
    FFT magnitude spectrum loss applied to non-LI channels only.

    Preserves high-frequency cloud texture (sharpness) without
    sharpening LI false positives.

    pred, target : (B, C, H, W)
    ch_mask      : (B, C)
    """
    loss  = 0.0
    count = 0
    C = pred.shape[1]
    for ci in range(C):
        if ci == li_idx:
            continue   # skip LI
        mask = ch_mask[:, ci].float()
        if mask.sum() < 1:
            continue
        p_fft = torch.fft.rfft2(pred[:, ci])
        t_fft = torch.fft.rfft2(target[:, ci])
        err   = ((p_fft.abs() - t_fft.abs()) ** 2) * mask[:, None, None]
        loss  = loss + err.mean()
        count += 1
    return loss / max(count, 1)


def training_loss(
    denoiser:           MultiStepDenoiser,
    schedule:           EDMSchedule,
    batch:              dict,
    device:             torch.device,
    cfg_drop_prob:      float = 0.15,
    # Channel weighting
    li_weight:          float = 30.0,
    li_weight_beta:     float = 0.9999,
    li_weight_ref_density: float = 0.05,
    # Asymmetric LI loss (Gao et al. 2022)
    asym_weight:        float = 1.0,
    asym_alpha:         float = 0.1,
    asym_norm_threshold: float = 0.16,  # normalised equivalent of li_event_threshold
    # Neighbourhood spatial loss (Zhang et al. 2023)
    nbr_weight:         float = 0.5,
    nbr_scales:         List[int] = [5, 11],
    # Spectral loss on cloud channels
    spectral_weight:    float = 0.1,
    # Lead-time weighted sampling
    lead_time_weights:  Optional[List[float]] = None,
    # Channel config
    channels:           Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Composite training loss:

        L = L_denoise  +  λ_a · L_asymmetric  +  λ_n · L_neighbourhood  +  λ_s · L_spectral

    L_denoise      : EDM-weighted MSE with dynamic per-sample LI channel weight
    L_asymmetric   : asymmetric FP/FN pixel loss for LI (reduces FAR)
    L_neighbourhood: spatial consistency loss at multiple scales (improves FSS)
    L_spectral     : FFT magnitude loss on IR/cloud channels only (preserves texture)

    lead_time_weights: optional per-step sampling probabilities (length=T_out).
        Default None = uniform sampling across all lead steps.
        Example [1,1,1,1,2,3] oversamples +50min and +60min 2× and 3× respectively.
        Normalised internally to a probability distribution.
        This is a pure data-side change — the loss function itself is unchanged.
        Following curriculum/importance sampling principles to allocate more
        gradient steps to harder long-range predictions.
    """
    context  = batch["context"].to(device)     # (B, T_in, C, H, W)
    target   = batch["target"].to(device)      # (B, T_out, C, H, W) residuals
    tgt_mask = batch["tgt_mask"].to(device)    # (B, T_out, C)

    B, T_out, C, H, W = target.shape
    # Derive li_idx from channels list if provided, else fall back to hardcoded 1
    if channels is not None and "li" in channels:
        li_idx = channels.index("li")
    else:
        li_idx = 1   # default: ir=0, li=1, ch0=2, ch1=3

    # Sample one lead step per batch item.
    # lead_time_weights allows oversampling later (harder) steps to improve
    # long-range skill without changing the loss function itself.
    if lead_time_weights is not None:
        w = torch.tensor(lead_time_weights, dtype=torch.float32, device=device)
        w = w[:T_out] / w[:T_out].sum()   # normalise, truncate to T_out if needed
        lead_idx = torch.multinomial(w.expand(B, -1), num_samples=1).squeeze(1)
    else:
        lead_idx = torch.randint(0, T_out, (B,), device=device)
    y        = target[torch.arange(B), lead_idx]       # (B, C, H, W)
    ch_mask  = tgt_mask[torch.arange(B), lead_idx]     # (B, C)

    # Sample noise level σ ~ lognormal
    sigma   = schedule.sample_sigma(B, device)
    x_noisy = y + torch.randn_like(y) * sigma[:, None, None, None]

    # CFG: randomly null-condition context
    if cfg_drop_prob > 0:
        drop     = torch.rand(B, device=device) < cfg_drop_prob
        ctx_used = context.clone()
        ctx_used[drop] = 0.0
    else:
        ctx_used = context

    # Forward pass
    pred = denoiser(x_noisy, sigma, ctx_used, ch_mask, lead_idx)

    # ── L_denoise: EDM-weighted MSE with dynamic LI channel weight ────
    lw = schedule.edm_loss_weight(sigma)[:, None, None, None]

    if "li_density" in batch:
        dyn_w = effective_li_weight(
            batch["li_density"].to(device),
            base_weight  = li_weight,
            beta         = li_weight_beta,
            ref_density  = li_weight_ref_density,
        )
    else:
        dyn_w = li_weight

    L_denoise = channel_weighted_mse(pred * lw.sqrt(), y * lw.sqrt(), ch_mask, dyn_w)

    # ── L_asymmetric: reduces FAR by penalising FP less than FN ──────
    # Uses absolute normalised values (residual + last_ctx) so that GT=0
    # correctly means "no lightning in absolute target frame", not
    # "no change from context frame" (which residual=0 would mean).
    last_ctx = batch["last_ctx"].to(device)   # (B, C, H, W)
    L_asym = asymmetric_li_loss(pred, y, last_ctx, alpha=asym_alpha, li_idx=li_idx,
                                norm_threshold=asym_norm_threshold)

    # ── L_neighbourhood: spatial consistency at 2 scales ─────────────
    L_nbr = neighbourhood_li_loss(pred, y, li_idx=li_idx, scales=nbr_scales)

    # ── L_spectral: FFT on cloud channels only ────────────────────────
    L_spec = spectral_loss(pred, y, ch_mask, li_idx=li_idx)

    return (L_denoise
            + asym_weight    * L_asym
            + nbr_weight     * L_nbr
            + spectral_weight * L_spec)


# =====================================================================
# Shared input-channel arithmetic — single source of truth.
# Previously this formula was independently duplicated in train.py,
# evaluate.py, and infer.py. All three were consistent, but duplication
# is exactly how such formulas silently drift after a future edit.
# =====================================================================
def compute_in_ch(C: int, T_in: int, ctx_channels, binary_li_ctx: bool) -> int:
    """
    Total UNet input channels: noisy target (C) + flattened context
    (T_in * C_ctx) + channel-presence mask (C).

    C_ctx = number of channels per context frame:
        len(ctx_channels) if a subset is specified, else C (all channels),
        +1 if binary_li_ctx is on AND "li" is among the context channels
        (an extra binary LI-presence mask is appended per frame).
    """
    C_ctx_sel = len(ctx_channels) if ctx_channels else C
    li_in_ctx = (ctx_channels is None) or ("li" in ctx_channels)
    C_ctx     = C_ctx_sel + 1 if (binary_li_ctx and li_in_ctx) else C_ctx_sel
    return C + T_in * C_ctx + C
