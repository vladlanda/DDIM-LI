"""
Evaluation for METSAT lightning nowcasting.

Metrics:
  - CRPS         : Continuous Ranked Probability Score (all channels)
  - Energy Score : multivariate probabilistic score
  - CSI          : Critical Success Index for lightning (binary)
  - POD / FAR    : Probability of Detection / False Alarm Rate (lightning)
  - Spread-Skill : ensemble spread vs RMSE
  - SSIM         : Structural similarity (sharpness proxy)

All metrics are computed per lead time step so you can plot
skill vs. forecast horizon.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from model import MultiStepDenoiser, EDMSchedule, edm_sampler
from dataset import denormalize

logger = logging.getLogger(__name__)


# ===================================================================
# Helper: generate an ensemble of forecasts
# ===================================================================

@torch.no_grad()
def generate_ensemble(
    model:      MultiStepDenoiser,
    context:    torch.Tensor,      # (B, T_in, C, H, W)
    ch_mask:    torch.Tensor,      # (B, C)
    device:     torch.device,
    n_members:  int  = 10,
    num_steps:  int  = 20,
    cfg_scale:  float = 1.5,       # > 1 → sharper via CFG
) -> torch.Tensor:
    """
    Returns ensemble of shape (B, M, T_out, C, H, W).
    Uses classifier-free guidance: final = uncond + cfg_scale*(cond - uncond)
    """
    model.eval()
    B, T_in, C, H, W = context.shape
    T_out = model.T_out

    members = []
    for _ in range(n_members):
        # Predict each step independently
        all_steps = []
        for step in range(T_out):
            lead_idx = torch.full((B,), step, device=device, dtype=torch.long)

            def denoiser_fn(x, sigma, _step=step, _lead_idx=lead_idx):
                # Conditional denoised estimate
                cond   = model(x, sigma, context,              ch_mask, _lead_idx)
                # Unconditional (null context)
                uncond = model(x, sigma, torch.zeros_like(context), ch_mask, _lead_idx)
                # CFG blend
                return uncond + cfg_scale * (cond - uncond)

            pred = edm_sampler(
                denoiser_fn, (B, C, H, W), device,
                num_steps=num_steps,
                sigma_min=model.precond.sigma_data * 0.01,
                sigma_max=80.0,
            )
            all_steps.append(pred)

        members.append(torch.stack(all_steps, dim=1))  # (B, T_out, C, H, W)

    return torch.stack(members, dim=1)  # (B, M, T_out, C, H, W)


# ===================================================================
# CRPS  (energy form, sample-based)
# ===================================================================

def crps_energy(
    ensemble: np.ndarray,   # (M, ...) member dimension first
    obs:      np.ndarray,   # (...) same shape sans M
) -> float:
    """
    CRPS via energy score decomposition:
      CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    Computed pixelwise, returned as scalar mean.
    """
    M = ensemble.shape[0]
    term1 = np.abs(ensemble - obs[None]).mean(axis=0)                      # (...)
    diffs = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            diffs += np.abs(ensemble[i] - ensemble[j])
    term2 = diffs / (M * (M - 1) / 2 + 1e-8)
    return float((term1 - 0.5 * term2).mean())


# ===================================================================
# Lightning-specific metrics (binary events)
# ===================================================================

def lightning_contingency(
    pred_prob: np.ndarray,  # (H, W) probability from ensemble fraction
    obs:       np.ndarray,  # (H, W) binary 0/1
    threshold: float = 0.5,
) -> Dict[str, float]:
    pred_bin = (pred_prob >= threshold).astype(float)
    obs_bin  = (obs > 0).astype(float)

    TP = (pred_bin * obs_bin).sum()
    FP = (pred_bin * (1 - obs_bin)).sum()
    FN = ((1 - pred_bin) * obs_bin).sum()
    TN = ((1 - pred_bin) * (1 - obs_bin)).sum()

    csi = TP / (TP + FP + FN + 1e-8)
    pod = TP / (TP + FN + 1e-8)
    far = FP / (TP + FP + 1e-8)
    bias = (TP + FP) / (TP + FN + 1e-8)
    return {"csi": float(csi), "pod": float(pod), "far": float(far), "bias": float(bias)}


# ===================================================================
# Spread-skill ratio
# ===================================================================

def spread_skill(
    ensemble: np.ndarray,  # (M, ...) values
    obs:      np.ndarray,  # (...)
) -> float:
    spread = ensemble.std(axis=0).mean()
    skill  = np.abs(ensemble.mean(axis=0) - obs).mean()
    return float(spread / (skill + 1e-8))


# ===================================================================
# Main evaluation function
# ===================================================================

def evaluate_epoch(
    model:       MultiStepDenoiser,
    val_loader:  DataLoader,
    schedule:    EDMSchedule,
    device:      torch.device,
    stats:       Dict,
    channels:    List[str],
    n_members:   int   = 10,
    num_samples: int   = 5,       # batches to evaluate on
    cfg_scale:   float = 1.5,
    li_threshold: float = 0.1,    # normalised units for lightning binary
    dt_min:      int   = 10,
) -> Dict[str, float]:
    """
    Run probabilistic evaluation on val_loader.
    Returns dict of aggregated metrics.
    """
    model.eval()
    C = len(channels)
    li_idx = channels.index("li") if "li" in channels else None

    all_crps   = []
    all_csi    = []
    all_ss     = []
    T_out      = model.T_out   # read from model — never depends on loop executing

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= num_samples:
            break

        context  = batch["context"].to(device)
        target   = batch["target"].to(device)
        ch_mask  = batch["tgt_mask"][:, 0].to(device)

        ens = generate_ensemble(
            model, context, ch_mask, device,
            n_members=n_members, cfg_scale=cfg_scale,
        )  # (B, M, T_out, C, H, W)

        ens_np = ens.cpu().numpy()
        tgt_np = target.cpu().numpy()

        B = ens_np.shape[0]

        for b in range(B):
            crps_per_step = []
            csi_per_step  = []
            ss_per_step   = []

            for t in range(T_out):
                ens_t = ens_np[b, :, t]   # (M, C, H, W)
                tgt_t = tgt_np[b, t]      # (C, H, W)

                crps_per_step.append(crps_energy(ens_t, tgt_t))
                ss_per_step.append(spread_skill(ens_t, tgt_t))

                if li_idx is not None:
                    ens_li    = ens_t[:, li_idx]
                    tgt_li    = tgt_t[li_idx]
                    pred_prob = (ens_li > li_threshold).mean(axis=0)
                    obs_bin   = (tgt_li > li_threshold).astype(float)
                    ct        = lightning_contingency(pred_prob, obs_bin)
                    csi_per_step.append(ct["csi"])

            all_crps.append(crps_per_step)
            all_ss.append(ss_per_step)
            if csi_per_step:
                all_csi.append(csi_per_step)

    # Guard: return empty metrics if val_loader had no batches
    if not all_crps:
        logger.warning("evaluate_epoch: no batches evaluated — val loader may be empty.")
        return {}

    all_crps = np.array(all_crps)   # (N, T_out)
    all_ss   = np.array(all_ss)

    metrics = {
        "crps_mean":    float(all_crps.mean()),
        "crps_1h":      float(all_crps[:, :6].mean())  if T_out >= 6  else float(all_crps.mean()),
        "crps_3h":      float(all_crps[:, :18].mean()) if T_out >= 18 else float(all_crps.mean()),
        "crps_6h":      float(all_crps[:, -1].mean()),
        "spread_skill": float(all_ss.mean()),
    }
    if all_csi:
        all_csi = np.array(all_csi)
        metrics.update({
            "csi_mean": float(all_csi.mean()),
            "csi_1h":   float(all_csi[:, :6].mean()) if T_out >= 6 else float(all_csi.mean()),
            "csi_6h":   float(all_csi[:, -1].mean()),
        })

    return metrics


# ===================================================================
# Inference utility: produce a forecast from a single context window
# ===================================================================

@torch.no_grad()
def forecast(
    model:      MultiStepDenoiser,
    context_np: np.ndarray,         # (T_in, C, H, W)  already normalised
    ch_mask_np: np.ndarray,         # (C,) binary
    device:     torch.device,
    n_members:  int  = 20,
    cfg_scale:  float = 1.5,
) -> np.ndarray:
    """
    Run a full probabilistic 6-hour forecast.
    Returns ensemble: (M, T_out, C, H, W) in normalised space.
    User should denormalise per channel using stats.
    """
    ctx  = torch.from_numpy(context_np).unsqueeze(0).to(device)   # (1, T_in, C, H, W)
    mask = torch.from_numpy(ch_mask_np).unsqueeze(0).to(device)   # (1, C)

    ens = generate_ensemble(model, ctx, mask, device,
                            n_members=n_members, cfg_scale=cfg_scale)
    return ens[0].cpu().numpy()   # (M, T_out, C, H, W)


# ===================================================================
# Quick visualisation (matplotlib)
# ===================================================================

def plot_forecast(
    ens_np:   np.ndarray,      # (M, T_out, C, H, W)
    tgt_np:   np.ndarray,      # (T_out, C, H, W)
    channels: List[str],
    steps_to_plot: List[int] = [0, 5, 11, 17, 23, 35],
    save_path: Optional[str] = None,
):
    """
    Plot ensemble mean, spread (std), and ground truth for selected steps.
    One row per step, columns: [GT, Ens Mean, Spread] per channel.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    M, T_out, C, H, W = ens_np.shape
    n_steps = len(steps_to_plot)
    n_ch    = min(C, 4)  # plot at most 4 channels

    fig = plt.figure(figsize=(n_ch * 9, n_steps * 3))
    gs  = gridspec.GridSpec(n_steps, n_ch * 3, figure=fig, hspace=0.05, wspace=0.05)

    for row, step in enumerate(steps_to_plot):
        if step >= T_out:
            continue
        ens_mean = ens_np[:, step].mean(axis=0)  # (C, H, W)
        ens_std  = ens_np[:, step].std(axis=0)
        gt       = tgt_np[step]

        for col, ch in enumerate(channels[:n_ch]):
            lead_min = (step + 1) * 10

            ax_gt   = fig.add_subplot(gs[row, col * 3])
            ax_mean = fig.add_subplot(gs[row, col * 3 + 1])
            ax_std  = fig.add_subplot(gs[row, col * 3 + 2])

            vmin, vmax = gt[col].min(), gt[col].max()
            ax_gt.imshow(gt[col],       cmap="viridis", vmin=vmin, vmax=vmax)
            ax_mean.imshow(ens_mean[col], cmap="viridis", vmin=vmin, vmax=vmax)
            ax_std.imshow(ens_std[col],  cmap="plasma")

            for ax, title in zip(
                [ax_gt, ax_mean, ax_std],
                [f"{ch} GT +{lead_min}m", "Ens Mean", "Spread"],
            ):
                ax.set_title(title, fontsize=7)
                ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info(f"Saved forecast plot: {save_path}")
    else:
        plt.show()
    plt.close()
