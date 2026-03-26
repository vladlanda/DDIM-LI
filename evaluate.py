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

import csv
import json
import logging
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# cuML (RAPIDS) — GPU-accelerated metrics.  Falls back to sklearn silently.
try:
    import cuml.metrics as _cuml_metrics
    from cuml.metrics import precision_recall_curve as _pr_curve
    from cuml.metrics import auc as _auc
    HAS_CUML = True
except ImportError:
    from sklearn.metrics import precision_recall_curve as _pr_curve
    from sklearn.metrics import auc as _auc
    from sklearn.metrics import brier_score_loss as _sk_brier
    HAS_CUML = False

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
    n_members:  int   = 10,
    num_steps:  int   = 20,
    cfg_scale:  float = 1.5,
) -> torch.Tensor:
    """
    Returns ensemble of shape (B, M, T_out, C, H, W).
    Uses classifier-free guidance: final = uncond + cfg_scale*(cond - uncond)
    """
    model.eval()
    B, T_in, C, H, W = context.shape
    T_out = model.T_out

    members = []
    member_bar = tqdm(
        range(n_members),
        desc         = "    Ensemble members",
        unit         = "member",
        dynamic_ncols = True,
        leave        = False,
    )
    for _ in member_bar:
        all_steps = []
        step_bar  = tqdm(
            range(T_out),
            desc         = "      Steps",
            unit         = "step",
            dynamic_ncols = True,
            leave        = False,
        )
        for step in step_bar:
            lead_idx = torch.full((B,), step, device=device, dtype=torch.long)

            def denoiser_fn(x, sigma, _step=step, _lead_idx=lead_idx):
                cond   = model(x, sigma, context,                 ch_mask, _lead_idx)
                uncond = model(x, sigma, torch.zeros_like(context), ch_mask, _lead_idx)
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
    CRPS via energy score decomposition (Gneiting & Raftery 2007):
      CRPS = E[|X - y|] - (1/2) * E[|X - X'|]

    E[|X - X'|] is the mean over ALL M² pairs (diagonal = 0) — i.e.
    divide by M², not by M(M-1)/2.  Dividing by unique pairs only is a
    biased estimator that under-penalises spread.

    The Python loop over unique pairs is kept intentionally: for small M
    (typically 10) numpy broadcast allocates an (M,M,C,H,W) tensor whose
    memory cost and allocation overhead exceeds the loop cost at this scale.
    """
    M     = ensemble.shape[0]
    term1 = np.abs(ensemble - obs[None]).mean(axis=0)   # (C,H,W)
    # Sum unique pairs, then scale to the M² mean: sum_unique * 2 / M²
    diffs = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            diffs += np.abs(ensemble[i] - ensemble[j])
    term2 = diffs * 2.0 / (M * M)                       # ÷ M²  (unbiased)
    return float((term1 - 0.5 * term2).mean())


# ===================================================================
# Cloud metrics  (continuous spatial fields: IR / CH0 / CH1)
# ===================================================================

def cloud_metrics(
    ens_mean:   np.ndarray,
    obs:        np.ndarray,
    ch_indices: List[int],
    ch_names:   List[str],
    stats:      Dict,
) -> Dict[str, Dict[str, float]]:
    """
    Per-channel RMSE and MAE in physical units + SSIM.

    Linear channels (ir, ch0-ch9): error_K = error_norm * stats[ch]["std"]
    cbrt channels (li): kept in normalised units, unit="norm"

    Returns dict keyed by channel name:
        {"ir": {"rmse": K, "mae": K, "ssim": float, "unit": "K"}, ...}
    """
    out = {}
    for ci, ch in zip(ch_indices, ch_names):
        pred = ens_mean[ci].astype(np.float32)
        gt   = obs[ci].astype(np.float32)

        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse_norm = float(np.sqrt(mean_squared_error(gt.ravel(), pred.ravel())))
        mae_norm  = float(mean_absolute_error(gt.ravel(), pred.ravel()))

        is_cbrt = ch in stats and stats[ch].get("transform") == "cbrt"
        if not is_cbrt and ch in stats:
            scale     = stats[ch]["std"]
            rmse_phys = rmse_norm * scale
            mae_phys  = mae_norm  * scale
            unit = "K"
        else:
            rmse_phys = rmse_norm
            mae_phys  = mae_norm
            unit = "norm"

        entry = {"rmse": rmse_phys, "mae": mae_phys, "unit": unit}
        if HAS_SKIMAGE:
            data_range = float(gt.max() - gt.min()) + 1e-8
            entry["ssim"] = float(ssim_fn(gt, pred, data_range=data_range))
        out[ch] = entry
    return out


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

    csi  = TP / (TP + FP + FN + 1e-8)
    pod  = TP / (TP + FN + 1e-8)
    far  = FP / (TP + FP + 1e-8)
    bias = (TP + FP) / (TP + FN + 1e-8)
    return {"csi": float(csi), "pod": float(pod),
            "far": float(far), "bias": float(bias)}


def fss(
    pred_prob: np.ndarray,   # (H, W) ensemble probability [0,1]
    obs_bin:   np.ndarray,   # (H, W) binary observation
    scale:     int = 8,      # neighbourhood half-width in pixels
) -> float:
    """
    Fractions Skill Score at a given spatial scale.
    FSS=1 → perfect, FSS=0 → no skill, FSS<0 → worse than climatology.

    Uses uniform box filtering to compute neighbourhood fractions.
    """
    from scipy.ndimage import uniform_filter
    size = 2 * scale + 1
    pred_frac = uniform_filter(pred_prob.astype(np.float32), size=size)
    obs_frac  = uniform_filter(obs_bin.astype(np.float32),   size=size)

    fss_num   = np.mean((pred_frac - obs_frac) ** 2)
    fss_ref   = np.mean(pred_frac ** 2) + np.mean(obs_frac ** 2)
    return float(1.0 - fss_num / (fss_ref + 1e-8))


def brier_score(
    pred_prob: np.ndarray,   # (H, W) ensemble probability
    obs_bin:   np.ndarray,   # (H, W) binary
) -> float:
    """
    Mean squared error between predicted probability and binary observation.
    Uses cuML if available, sklearn otherwise, plain numpy as final fallback.
    """
    y_true = obs_bin.ravel().astype(np.float32)
    y_prob = pred_prob.ravel().astype(np.float32)
    if HAS_CUML:
        try:
            import cupy as cp
            # cuML brier_score_loss expects 1-D arrays
            return float(_cuml_metrics.brier_score_loss(
                cp.asarray(y_true), cp.asarray(y_prob)
            ))
        except Exception:
            pass
    # sklearn brier_score_loss = mean((p - y)^2), identical formula
    return float(_sk_brier(y_true, y_prob))


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
    model:        MultiStepDenoiser,
    val_loader:   DataLoader,
    schedule:     EDMSchedule,
    device:       torch.device,
    stats:        Dict,
    channels:     List[str],
    n_members:    int   = 10,
    val_samples:  int   = -1,      # batches to evaluate; -1 = full val set
    cfg_scale:    float = 1.5,
    li_threshold: float = 0.1,
    dt_min:       int   = 10,
) -> Dict[str, float]:
    """
    Run probabilistic evaluation on val_loader.

    val_samples: number of batches to evaluate.
                 -1 (default) = use the entire val loader.
                 >0 = randomly sample that many batches without replacement,
                      so the evaluation window varies each time it is called.
    """
    model.eval()
    li_idx = channels.index("li") if "li" in channels else None
    T_out  = model.T_out

    # --- Build the list of batch indices to evaluate ---
    total_batches = len(val_loader)
    if val_samples == -1 or val_samples >= total_batches:
        # Use the full val set sequentially
        chosen = None
        budget = total_batches
    else:
        # Randomly sample val_samples batches without replacement so the
        # evaluation window varies each call
        chosen = set(random.sample(range(total_batches), val_samples))
        budget = val_samples

    all_crps, all_csi, all_ss = [], [], []

    pbar = tqdm(
        enumerate(val_loader),
        total        = budget,
        desc         = "  Validation",
        unit         = "batch",
        dynamic_ncols = True,
        leave        = True,
    )

    evaluated = 0
    for batch_idx, batch in pbar:
        # Skip batches not in our random sample
        if chosen is not None and batch_idx not in chosen:
            continue


        if evaluated >= budget:
            break

        context  = batch["context"].to(device)
        target   = batch["target"].to(device)       # (B, T_out, C, H, W) residuals
        last_ctx = batch["last_ctx"].to(device)     # (B, C, H, W) last context frame
        # Use per-timestep channel mask (not just step 0)
        tgt_mask = batch["tgt_mask"].to(device)     # (B, T_out, C)
        ch_mask  = tgt_mask[:, 0]                   # (B, C) for ensemble generation

        ens = generate_ensemble(
            model, context, ch_mask, device,
            n_members=n_members, cfg_scale=cfg_scale,
        )  # (B, M, T_out, C, H, W)  — model output is residuals

        # Reconstruct absolute frames: residual + last context frame
        # Both ens and target are in normalised residual space.
        # Adding last_ctx (also normalised) gives normalised absolute values,
        # which is the correct space for thresholding and physical metrics.
        last_ctx_np = last_ctx.cpu().numpy()          # (B, C, H, W)
        ens_np  = ens.cpu().numpy() + last_ctx_np[:, None, None]  # (B,M,T_out,C,H,W)
        tgt_np  = target.cpu().numpy() + last_ctx_np[:, None]     # (B,T_out,C,H,W)
        B       = ens_np.shape[0]

        for b in range(B):
            crps_per_step, csi_per_step, ss_per_step = [], [], []

            for t in range(T_out):
                ens_t = ens_np[b, :, t]   # (M, C, H, W) absolute normalised
                tgt_t = tgt_np[b, t]      # (C, H, W)    absolute normalised

                crps_per_step.append(crps_energy(ens_t, tgt_t))
                ss_per_step.append(spread_skill(ens_t, tgt_t))

                if li_idx is not None:
                    # Threshold on absolute normalised LI — meaningful signal
                    pred_prob = (ens_t[:, li_idx] > li_threshold).mean(axis=0)
                    obs_bin   = (tgt_t[li_idx] > li_threshold).astype(float)
                    csi_per_step.append(
                        lightning_contingency(pred_prob, obs_bin)["csi"]
                    )

            all_crps.append(crps_per_step)
            all_ss.append(ss_per_step)
            if csi_per_step:
                all_csi.append(csi_per_step)

        evaluated += 1

        # Live metrics in the tqdm postfix
        if all_crps:
            pbar.set_postfix(
                crps = f"{np.mean(all_crps):.4f}",
                csi  = f"{np.mean(all_csi):.3f}" if all_csi else "n/a",
                refresh = False,
            )

    if not all_crps:
        logger.warning("evaluate_epoch: no batches evaluated — val loader may be empty.")
        return {}

    # --- DDP aggregation: reduce per-sample arrays across all ranks ----------
    # Convert lists to tensors, all_reduce SUM, divide by global sample count.
    import torch.distributed as dist_mod
    ddp = dist_mod.is_available() and dist_mod.is_initialized()

    def _reduce_array(lst):
        """Stack list of per-step arrays → (N, T_out) tensor, all_reduce, return numpy."""
        t = torch.tensor(lst, device=device)          # (N_local, T_out)
        if ddp:
            # Gather counts so we can weight the mean correctly across ranks
            count = torch.tensor([len(lst)], device=device, dtype=torch.float32)
            dist_mod.all_reduce(t.sum(0, keepdim=True), op=dist_mod.ReduceOp.SUM)
            dist_mod.all_reduce(count, op=dist_mod.ReduceOp.SUM)
            return (t.sum(0) / count).cpu().numpy(), count.item()
        return t.cpu().numpy(), float(len(lst))

    crps_arr = np.array(all_crps)   # (N_local, T_out)
    ss_arr   = np.array(all_ss)

    if ddp:
        # Reduce sum of per-sample arrays and total sample count across ranks
        crps_sum = torch.tensor(crps_arr.sum(0), device=device)   # (T_out,)
        ss_sum   = torch.tensor(ss_arr.sum(0),   device=device)
        n_t      = torch.tensor([len(all_crps)],  device=device, dtype=torch.float32)
        dist_mod.all_reduce(crps_sum, op=dist_mod.ReduceOp.SUM)
        dist_mod.all_reduce(ss_sum,   op=dist_mod.ReduceOp.SUM)
        dist_mod.all_reduce(n_t,      op=dist_mod.ReduceOp.SUM)
        crps_arr = (crps_sum / n_t).cpu().numpy()   # (T_out,) global mean per step
        ss_arr   = (ss_sum   / n_t).cpu().numpy()
        n_global = n_t.item()
    else:
        n_global = len(all_crps)

    # Rank 0 returns the final dict; other ranks return {}.
    # Both ranks MUST have called all_reduce above before reaching this point.
    if ddp and dist_mod.get_rank() != 0:
        return {}

    metrics = {
        "crps_mean":    float(crps_arr.mean()),
        "crps_1h":      float(crps_arr[:6].mean())  if T_out >= 6  else float(crps_arr.mean()),
        "crps_3h":      float(crps_arr[:18].mean()) if T_out >= 18 else float(crps_arr.mean()),
        "crps_6h":      float(crps_arr[-1]),
        "spread_skill": float(ss_arr.mean()),
    }
    if all_csi:
        csi_arr = np.array(all_csi)
        if ddp:
            csi_sum = torch.tensor(csi_arr.sum(0), device=device)
            n_c     = torch.tensor([len(all_csi)],  device=device, dtype=torch.float32)
            dist_mod.all_reduce(csi_sum, op=dist_mod.ReduceOp.SUM)
            dist_mod.all_reduce(n_c,     op=dist_mod.ReduceOp.SUM)
            csi_arr = (csi_sum / n_c).cpu().numpy()
        metrics.update({
            "csi_mean": float(csi_arr.mean()),
            "csi_1h":   float(csi_arr[:6].mean()) if T_out >= 6 else float(csi_arr.mean()),
            "csi_6h":   float(csi_arr[-1]),
        })

    return metrics


# ===================================================================
# Fast validation — cheap training-time quality proxy
# ===================================================================

@torch.no_grad()
def fast_val_metrics(
    model:        "MultiStepDenoiser",
    val_loader:   "DataLoader",
    schedule:     "EDMSchedule",
    device:       "torch.device",
    channels:     List[str],
    val_samples:  int   = -1,      # batches to use; -1 = full val set
    li_threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Cheap validation metrics for use every training epoch.

    DDP-aware: all ranks run inference on their own val_loader shard in
    parallel, accumulate weighted sums, then all_reduce across ranks so
    rank 0 can compute the global mean.  Non-rank-0 ranks return {}.

    For each batch:
      1. Sample σ from the EDM log-normal (same distribution as training).
      2. Add noise to a randomly chosen target lead-time frame.
      3. One denoiser forward pass → val_loss / val_mse / val_mae / val_li_mse.

    val_samples=-1 uses the full (per-rank) shard; >0 randomly samples that
    many batches so the per-epoch cost is bounded.
    """
    import torch.distributed as dist_mod
    from model import channel_weighted_mse

    model.eval()
    li_idx = channels.index("li") if "li" in channels else None
    ddp    = dist_mod.is_available() and dist_mod.is_initialized()

    total_batches = len(val_loader)
    if val_samples == -1 or val_samples >= total_batches:
        chosen_fast = None
        budget_fast = total_batches
    else:
        chosen_fast = set(random.sample(range(total_batches), val_samples))
        budget_fast = val_samples

    # Accumulators: sum and count kept as GPU tensors for efficient all_reduce.
    # Layout: [loss_sum, mse_sum, mae_sum, li_mse_sum, n_batches, n_li_batches]
    acc = torch.zeros(6, device=device)

    rank        = dist_mod.get_rank() if ddp else 0
    pbar = tqdm(
        enumerate(val_loader),
        total        = budget_fast,
        desc         = f"  Fast val (rank {rank})",
        unit         = "batch",
        dynamic_ncols = True,
        leave        = False,
    )

    evaluated = 0
    for batch_idx, batch in pbar:
        if chosen_fast is not None and batch_idx not in chosen_fast:
            continue
        if evaluated >= budget_fast:
            break

        context  = batch["context"].to(device)
        target   = batch["target"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)

        B, T_out, C, H, W = target.shape
        lead_idx = torch.randint(0, T_out, (B,), device=device)
        y        = target[torch.arange(B), lead_idx]
        ch_mask  = tgt_mask[torch.arange(B), lead_idx]

        sigma   = schedule.sample_sigma(B, device)
        x_noisy = y + torch.randn_like(y) * sigma[:, None, None, None]
        pred    = model(x_noisy, sigma, context, ch_mask, lead_idx)

        # val_loss: EDM-weighted (comparable to train_loss)
        lw   = schedule.edm_loss_weight(sigma)[:, None, None, None]
        loss = channel_weighted_mse(pred * lw.sqrt(), y * lw.sqrt(),
                                    ch_mask, li_weight=3.0)

        # val_mse / val_mae: unweighted pixel errors
        err     = pred - y
        mask_hw = ch_mask[:, :, None, None].float()
        denom   = mask_hw.sum() * H * W + 1e-8
        mse     = (err ** 2 * mask_hw).sum() / denom
        mae     = (err.abs() * mask_hw).sum() / denom

        acc[0] += loss
        acc[1] += mse
        acc[2] += mae
        acc[4] += 1.0

        if li_idx is not None:
            li_err  = err[:, li_idx]
            li_mask = ch_mask[:, li_idx].float()
            li_mse  = (li_err ** 2).mean(dim=(-2, -1))
            li_mse_m = (li_mse * li_mask).sum() / (li_mask.sum() + 1e-8)
            acc[3] += li_mse_m
            acc[5] += 1.0

        evaluated += 1
        pbar.set_postfix(
            loss = f"{(acc[0] / max(acc[4], 1)).item():.4f}",
            mse  = f"{(acc[1] / max(acc[4], 1)).item():.4f}",
            refresh = False,
        )

    # Aggregate across all ranks
    if ddp:
        dist_mod.all_reduce(acc, op=dist_mod.ReduceOp.SUM)

    # Rank 0 builds and returns the global dict; other ranks return {}.
    # Both ranks MUST reach this point — the all_reduce above is a collective
    # and requires all ranks to call it before any rank can proceed.
    if ddp and dist_mod.get_rank() != 0:
        return {}

    n       = acc[4].item()
    n_li    = acc[5].item()
    metrics = {
        "val_loss": acc[0].item() / max(n, 1),
        "val_mse":  acc[1].item() / max(n, 1),
        "val_mae":  acc[2].item() / max(n, 1),
    }
    if n_li > 0:
        metrics["val_li_mse"] = acc[3].item() / n_li
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

def _make_rgb(frames: np.ndarray, channels: List[str]) -> np.ndarray:
    """
    Compose a false-colour RGB image from ir/ch0/ch1 channels.
    frames: (C, H, W)  denormalised
    Returns (H, W, 3) uint8.
    """
    def _pull(ch):
        if ch in channels:
            arr = frames[channels.index(ch)]
        else:
            arr = np.zeros(frames.shape[-2:], dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        return ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

    r = _pull("ir")
    g = _pull("ch0")
    b = _pull("ch1")
    return np.stack([r, g, b], axis=-1)   # (H, W, 3)


def _make_li(frames: np.ndarray, channels: List[str]) -> np.ndarray:
    """
    Return the LI channel as a (H, W) float array, normalised 0-1.
    """
    if "li" in channels:
        arr = frames[channels.index("li")]
    else:
        arr = np.zeros(frames.shape[-2:], dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def plot_forecast(
    context_np:    np.ndarray,                  # (T_in,  C, H, W) denormalised
    ens_np:        np.ndarray,                  # (M, T_out, C, H, W) denormalised
    channels:      List[str],
    gt_np:         Optional[np.ndarray] = None, # (T_out, C, H, W) or None
    steps_to_plot: Optional[List[int]]  = None, # None = all steps
    save_path:     Optional[str]        = None,
):
    """
    Layout: rows × columns grid.

      Rows (2 or 3):
        Row 0 — Context    : last context frame  [IR+CH0+CH1 | LI]
        Row 1 — Prediction : ens-mean            [IR+CH0+CH1 | LI | LI-spread]
        Row 2 — Ground Truth (only if gt_np):    [IR+CH0+CH1 | LI]

      Columns: one group of 3 sub-columns per forecast step
               (+10min, +20min, … up to T_out×dt_min)

    With T_out=36 this produces a 36-column × 2-3 row image.
    Each cell is kept small (cell_w=1.6in) so the full figure is ~58in wide
    at 120 dpi → a wide but readable PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless — no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    M, T_out, C, H, W = ens_np.shape
    steps   = list(range(T_out)) if steps_to_plot is None else \
              [s for s in steps_to_plot if s < T_out]
    n_steps = len(steps)
    has_gt  = gt_np is not None
    n_rows  = 3 if has_gt else 2

    # 3 sub-columns per step: [rgb | li | spread(pred only)]
    n_subcols  = n_steps * 3
    cell_w     = 1.6     # inches per sub-column
    cell_h     = 2.2     # inches per row
    label_w    = 1.0     # extra left margin for row labels

    fig_w = label_w + n_subcols * cell_w
    fig_h = n_rows  * cell_h

    fig, axes = plt.subplots(
        n_rows, n_subcols,
        figsize     = (fig_w, fig_h),
        squeeze     = False,
        gridspec_kw = {"wspace": 0.02, "hspace": 0.12},
    )
    fig.patch.set_facecolor("white")
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")
            ax.set_facecolor("white")

    # Pre-compute last context frame composites (shared across all columns)
    ctx_last = context_np[-1]              # (C, H, W)
    ctx_rgb  = _make_rgb(ctx_last, channels)
    ctx_li   = _make_li(ctx_last,  channels)

    font_title = max(4, min(7, int(120 / n_steps)))   # shrinks gracefully

    for col_idx, step in enumerate(steps):
        base     = col_idx * 3
        lead_min = (step + 1) * 10

        ens_mean = ens_np[:, step].mean(axis=0)   # (C, H, W)
        ens_std  = ens_np[:, step].std(axis=0)

        # ── Row 0: Context (repeated for every column so time labels align) ──
        axes[0, base    ].imshow(ctx_rgb)
        axes[0, base + 1].imshow(ctx_li, cmap="hot", vmin=0, vmax=1)
        axes[0, base + 2].set_visible(False)
        if col_idx == 0:
            axes[0, base    ].set_title("IR/CH0/CH1", fontsize=font_title, color="black", pad=2)
            axes[0, base + 1].set_title("LI",         fontsize=font_title, color="black", pad=2)

        # ── Row 1: Prediction ──
        pred_rgb = _make_rgb(ens_mean, channels)
        pred_li  = _make_li(ens_mean,  channels)
        if "li" in channels:
            s = ens_std[channels.index("li")]
            spread_li = (s - s.min()) / (s.max() - s.min() + 1e-8)
        else:
            spread_li = np.zeros((H, W), dtype=np.float32)

        axes[1, base    ].imshow(pred_rgb)
        axes[1, base + 1].imshow(pred_li,  cmap="hot",    vmin=0, vmax=1)
        axes[1, base + 2].imshow(spread_li, cmap="plasma", vmin=0, vmax=1)
        axes[1, base    ].set_title(f"+{lead_min}m", fontsize=font_title, color="black", pad=2)

        # ── Row 2: Ground truth ──
        if has_gt:
            gt_frame = gt_np[step]
            axes[2, base    ].imshow(_make_rgb(gt_frame, channels))
            axes[2, base + 1].imshow(_make_li(gt_frame,  channels), cmap="hot", vmin=0, vmax=1)
            axes[2, base + 2].set_visible(False)

    # Row labels on the far left of each row
    row_labels = ["Context", "Prediction", "Ground Truth"] if has_gt \
                 else ["Context", "Prediction"]
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=8, color="black",
                              rotation=90, labelpad=4, va="center")
        axes[r, 0].yaxis.set_label_position("left")
        axes[r, 0].axis("on")
        axes[r, 0].tick_params(left=False, bottom=False,
                               labelleft=False, labelbottom=False)
        for spine in axes[r, 0].spines.values():
            spine.set_visible(False)

    # Sub-column header legend (once, top-right)
    fig.text(0.99, 0.99,
             "Pred cols: [IR+CH0+CH1 | LI | LI-spread]",
             ha="right", va="top", fontsize=7, color="#555555",
             transform=fig.transFigure)

    plt.suptitle(
        f"METSAT Lightning Nowcast — {n_steps} steps × 10min  "
        f"({'with GT' if has_gt else 'no GT'})",
        fontsize=10, color="black", y=1.005,
    )

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight",
                    facecolor="white")
        logger.info(f"  Saved plot: {save_path}  ({n_steps} steps, {n_rows} rows)")
    else:
        plt.show()
    plt.close(fig)


# ===================================================================
# Full test-set evaluation  (called from __main__)
# ===================================================================


# ===================================================================
# Plot regeneration from saved plot_data.npz (--plot_only)
# ===================================================================

def _regenerate_plots(npz_path: str, args) -> None:
    """
    Reload all arrays saved during a previous run_test_evaluation call
    and regenerate every skill-curve / PR / calibration figure without
    re-running model inference.  Called automatically when --plot_only
    is passed.
    """
    import json as _json_inner
    data = np.load(npz_path, allow_pickle=True)

    per_step     = _json_inner.loads(str(data["per_step_json"]))
    lead_times   = data["lead_times"].tolist()
    channels     = data["channels"].tolist()
    cloud_chs    = data["cloud_chs"].tolist()
    fss_scales   = data["fss_scales"].tolist()
    T_out        = int(data["T_out"])
    dt_min       = int(data["dt_min"])

    def _mean(lst): return float(np.mean(lst)) if len(lst) > 0 else float("nan")

    # Reconstruct fss_by_scale_step from saved per-step means
    fss_by_scale_step = {}
    for s in fss_scales:
        key = f"fss_s{s}"
        arr = data[key].tolist() if key in data else [float("nan")] * T_out
        fss_by_scale_step[s] = [[v] for v in arr]

    # Reconstruct PR / calibration arrays
    pr_steps_arr = data["pr_steps"].tolist()
    pr_probs  = {t: [] for t in pr_steps_arr}
    pr_labels = {t: [] for t in pr_steps_arr}
    cal_probs  = {t: [] for t in pr_steps_arr}
    cal_labels = {t: [] for t in pr_steps_arr}
    for t in pr_steps_arr:
        if f"pr_prob_{t}" in data:
            pr_probs[t]   = [data[f"pr_prob_{t}"]]
            pr_labels[t]  = [data[f"pr_label_{t}"]]
            cal_probs[t]  = [data[f"cal_prob_{t}"]]
            cal_labels[t] = [data[f"cal_label_{t}"]]

    li_idx   = channels.index("li") if "li" in channels else None
    pr_steps = set(pr_steps_arr)
    ch_unit  = {ch: "K" for ch in cloud_chs}

    ssim_by_ch_step = {
        ch: [([per_step[t][f"ssim_{ch}"]] if f"ssim_{ch}" in per_step[t] else [])
             for t in range(T_out)]
        for ch in cloud_chs
    }

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"--plot_only: regenerating plots -> {args.output_dir}")

    import matplotlib
    # Use interactive backend when displaying, headless Agg when only saving
    if getattr(args, "plot_only", False):
        try:
            matplotlib.use("TkAgg")
        except Exception:
            try:
                matplotlib.use("Qt5Agg")
            except Exception:
                matplotlib.use("Agg")
                logger.warning("No interactive backend available — saving only")
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    # ── Journal-style white theme ─────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "black",
        "axes.labelcolor":   "black",
        "xtick.color":       "black",
        "ytick.color":       "black",
        "text.color":        "black",
        "grid.color":        "#cccccc",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "#cccccc",
        "font.size":         9,
    })

    def _styled_ax(ax):
        """Apply journal-style white formatting to an axis."""
        ax.set_facecolor("white")
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        for s in ax.spines.values():
            s.set_edgecolor("black")
            s.set_linewidth(0.8)
        ax.grid(True, color="#cccccc", linestyle="--", linewidth=0.5, zorder=0)

    lt = lead_times

    # ── Figure 1: Cloud skill curves — 2×N grid (no 1×N strips) ────
    ch_palette = [
        "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    ch_colors = {ch: ch_palette[i % len(ch_palette)]
                 for i, ch in enumerate(cloud_chs)}

    has_ssim   = any(ssim_by_ch_step[ch][0] for ch in cloud_chs)
    n_subplots = 4 + (1 if has_ssim else 0)

    # Choose a grid that avoids 1×N: prefer 2 rows
    import math
    ncols1 = math.ceil(n_subplots / 2)
    nrows1 = math.ceil(n_subplots / ncols1)
    fig1, axes1_2d = plt.subplots(nrows1, ncols1,
                                   figsize=(4.5 * ncols1, 3.8 * nrows1))
    fig1.patch.set_facecolor("white")
    axes1 = axes1_2d.flatten() if hasattr(axes1_2d, "flatten") else [axes1_2d]
    # Hide any unused axes
    for ax in axes1[n_subplots:]:
        ax.set_visible(False)

    def _ch_unit_label(ch):
        return ch_unit.get(ch, "norm")

    unit_lbl = "/".join(sorted(set(ch_unit.values())))

    ax = axes1[0]
    for ch in cloud_chs:
        vals = [r.get(f"rmse_{ch}", float("nan")) for r in per_step]
        ax.plot(lt, vals, color=ch_colors[ch], linewidth=1.5, label=ch)
    ax.set_title("RMSE per Channel")
    ax.set_xlabel("Lead time (min)")
    ax.set_ylabel(f"RMSE ({unit_lbl})")
    ax.legend(fontsize=8)
    _styled_ax(ax)

    ax = axes1[1]
    for ch in cloud_chs:
        vals = [r.get(f"mae_{ch}", float("nan")) for r in per_step]
        ax.plot(lt, vals, color=ch_colors[ch], linewidth=1.5, label=ch)
    ax.set_title("MAE per Channel")
    ax.set_xlabel("Lead time (min)")
    ax.set_ylabel(f"MAE ({unit_lbl})")
    ax.legend(fontsize=8)
    _styled_ax(ax)

    ax_idx = 2
    if has_ssim:
        ax = axes1[ax_idx]
        for ch in cloud_chs:
            vals = [r.get(f"ssim_{ch}", float("nan")) for r in per_step]
            ax.plot(lt, vals, color=ch_colors[ch], linewidth=1.5, label=ch)
        ax.set_ylim(0, 1)
        ax.axhline(1.0, color="#888888", linestyle=":", linewidth=0.8)
        ax.set_title("SSIM per Channel")
        ax.set_xlabel("Lead time (min)")
        ax.set_ylabel("SSIM")
        ax.legend(fontsize=8)
        _styled_ax(ax)
        ax_idx += 1

    ax = axes1[ax_idx]
    ax.plot(lt, [r["crps"] for r in per_step], color="#1f77b4", linewidth=1.5)
    ax.set_title("CRPS (All Channels)")
    ax.set_xlabel("Lead time (min)")
    ax.set_ylabel("CRPS (normalised)")
    _styled_ax(ax)
    ax_idx += 1

    ax = axes1[ax_idx]
    ax.plot(lt, [r["spread_skill"] for r in per_step], color="#2ca02c", linewidth=1.5)
    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.0, label="Ideal = 1.0")
    ax.legend(fontsize=8)
    ax.set_title("Spread-Skill Ratio")
    ax.set_xlabel("Lead time (min)")
    ax.set_ylabel("Spread / Skill")
    _styled_ax(ax)

    fig1.suptitle("Cloud Prediction Skill", fontsize=12, fontweight="bold", y=1.01)
    fig1.tight_layout()
    cloud_path = os.path.join(args.output_dir, "skill_curves_cloud.png")
    fig1.savefig(cloud_path, dpi=150, bbox_inches="tight", facecolor="white")
    if getattr(args, 'plot_only', False): plt.show()
    plt.close(fig1)
    logger.info(f"Cloud skill     -> {cloud_path}")

    if li_idx is not None:
        import matplotlib.cm as mcm

        lt_cmap   = mcm.get_cmap("turbo")
        lt_colors = [lt_cmap(i / max(T_out - 1, 1)) for i in range(T_out)]

        def _lt_legend(ax, labeled_steps, ncol=4, loc="best", extra_handles=None):
            handles = list(extra_handles or [])
            for t in labeled_steps:
                handles.append(
                    plt.Line2D([0], [0], color=lt_colors[t], linewidth=2,
                               label=f"+{(t+1)*dt_min}m")
                )
            ax.legend(handles=handles, fontsize=7.5, ncol=ncol, loc=loc,
                      framealpha=0.9, handlelength=1.4,
                      columnspacing=0.8, handletextpad=0.4)

        # ── Figure 2: CSI / POD / FAR / Brier — 2×2 grid ────────
        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
        fig2.patch.set_facecolor("white")
        li_cfg = [
            ([r["csi"]   for r in per_step], "#1f77b4", "CSI vs Lead Time",    "CSI"),
            ([r["pod"]   for r in per_step], "#2ca02c", "POD vs Lead Time",    "POD"),
            ([r["far"]   for r in per_step], "#d62728", "FAR vs Lead Time",    "FAR"),
            ([r["brier"] for r in per_step], "#9467bd", "Brier Score vs Lead Time", "Brier Score"),
        ]
        for ax, (vals, color, title, ylabel) in zip(axes2.flatten(), li_cfg):
            ax.plot(lt, vals, color=color, linewidth=1.5)
            if ylabel in ("CSI", "POD"):
                ax.set_ylim(0, 1)
            ax.set_title(title)
            ax.set_xlabel("Lead time (min)")
            ax.set_ylabel(ylabel)
            _styled_ax(ax)
        fig2.suptitle("Lightning Detection Skill", fontsize=12, fontweight="bold")
        fig2.tight_layout()
        li_path = os.path.join(args.output_dir, "skill_curves_lightning.png")
        fig2.savefig(li_path, dpi=150, bbox_inches="tight", facecolor="white")
        if getattr(args, 'plot_only', False): plt.show()
        plt.close(fig2)
        logger.info(f"Lightning skill -> {li_path}")

        # ── Figure 3: FSS vs scale ────────────────────────────────
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        fig3.patch.set_facecolor("white")
        scale_km = [s * args.pixel_size_km for s in fss_scales]
        for t in pr_steps:
            fss_vals = [_mean(fss_by_scale_step[s][t]) for s in fss_scales]
            ax3.plot(scale_km, fss_vals, color=lt_colors[t],
                     linewidth=1.2, marker="o", markersize=3, alpha=0.9)
        skill_line = plt.Line2D([0], [0], color="#d62728", linestyle="--",
                                linewidth=1.5, label="FSS = 0.5 (useful skill)")
        ax3.axhline(0.5, color="#d62728", linestyle="--", linewidth=1.5)
        ax3.set_xlabel("Neighbourhood Scale (km)")
        ax3.set_ylabel("FSS")
        ax3.set_title(f"FSS vs Spatial Scale — all {T_out} lead times",
                      fontweight="bold")
        _styled_ax(ax3)
        _lt_legend(ax3, pr_steps, ncol=4, loc="lower right",
                   extra_handles=[skill_line])
        fig3.tight_layout()
        fss_path = os.path.join(args.output_dir, "fss_vs_scale.png")
        fig3.savefig(fss_path, dpi=150, bbox_inches="tight", facecolor="white")
        if getattr(args, 'plot_only', False): plt.show()
        plt.close(fig3)
        logger.info(f"FSS plot        -> {fss_path}")

        # ── Figure 4 + 5: PR and Calibration — 1×2 side by side ──
        # (two square panels = 2×1 is fine for a matched pair)
        fig45, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 5.5))
        fig45.patch.set_facecolor("white")

        # PR curves — uses cuML (GPU) if available, sklearn otherwise.
        # Both are imported at module level as _pr_curve / _auc.
        auc_by_step = {}
        for t in pr_steps:
            if not pr_probs[t]:
                continue
            all_prob = np.concatenate(pr_probs[t]).astype(np.float32)
            all_lbl  = np.concatenate(pr_labels[t]).astype(np.int32)
            if HAS_CUML:
                try:
                    import cupy as cp
                    prec, rec, _ = _pr_curve(
                        cp.asarray(all_lbl), cp.asarray(all_prob)
                    )
                    prec = cp.asnumpy(prec)
                    rec  = cp.asnumpy(rec)
                except Exception:
                    prec, rec, _ = _pr_curve(all_lbl, all_prob)
            else:
                prec, rec, _ = _pr_curve(all_lbl, all_prob)
            auc_val        = float(_auc(rec, prec))
            auc_by_step[t] = auc_val
            ax4.plot(rec, prec, color=lt_colors[t], linewidth=1.2, alpha=0.9)
        ax4.set_xlabel("Recall (POD)")
        ax4.set_ylabel("Precision (1 − FAR)")
        ax4.set_title("Precision-Recall Curves", fontweight="bold")
        ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
        _styled_ax(ax4)
        handles_pr = [
            plt.Line2D([0], [0], color=lt_colors[t], linewidth=2,
                       label=f"+{(t+1)*dt_min}m  AUC={auc_by_step[t]:.2f}")
            for t in pr_steps if t in auc_by_step
        ]
        ax4.legend(handles=handles_pr, fontsize=7.5, ncol=4,
                   loc="upper right", framealpha=0.9,
                   handlelength=1.4, columnspacing=0.8, handletextpad=0.4)

        # Reliability / calibration diagram
        n_bins      = 10
        bin_edges   = np.linspace(0, 1, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        diag_line   = plt.Line2D([0], [0], color="black", linestyle="--",
                                 linewidth=1.2, label="Perfect calibration")
        ax5.plot([0, 1], [0, 1], color="black", linestyle="--",
                 linewidth=1.2, zorder=5)
        for t in pr_steps:
            if not cal_probs[t]:
                continue
            all_prob = np.concatenate(cal_probs[t])
            all_lbl  = np.concatenate(cal_labels[t])
            bin_idx  = np.digitize(all_prob, bin_edges[1:-1])
            obs_freq = np.full(n_bins, np.nan)
            for b_i in range(n_bins):
                mask = bin_idx == b_i
                if mask.sum() > 10:
                    obs_freq[b_i] = all_lbl[mask].mean()
            valid = ~np.isnan(obs_freq)
            ax5.plot(bin_centers[valid], obs_freq[valid],
                     color=lt_colors[t], linewidth=1.2,
                     marker="o", markersize=3, alpha=0.9)
        ax5.set_xlabel("Mean Predicted Probability")
        ax5.set_ylabel("Observed Frequency")
        ax5.set_title("Reliability Diagram", fontweight="bold")
        ax5.set_xlim(0, 1); ax5.set_ylim(0, 1)
        _styled_ax(ax5)
        _lt_legend(ax5, pr_steps, ncol=4, loc="upper left",
                   extra_handles=[diag_line])

        fig45.suptitle(f"Lightning Probabilistic Skill — all {T_out} lead times",
                       fontsize=12, fontweight="bold")
        fig45.tight_layout()
        pr_path  = os.path.join(args.output_dir, "precision_recall.png")
        cal_path = os.path.join(args.output_dir, "calibration.png")
        fig45.savefig(pr_path,  dpi=150, bbox_inches="tight", facecolor="white")
        if getattr(args, 'plot_only', False): plt.show()
        # Also save calibration panel alone
        fig_cal, ax_cal = plt.subplots(figsize=(6, 5.5))
        fig_cal.patch.set_facecolor("white")
        ax_cal.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2)
        for line in ax5.lines:
            ax_cal.plot(line.get_xdata(), line.get_ydata(),
                        color=line.get_color(), linewidth=line.get_linewidth(),
                        marker=line.get_marker() if line.get_marker() != "None" else "",
                        markersize=line.get_markersize(), alpha=line.get_alpha() or 1.0)
        ax_cal.set_xlabel("Mean Predicted Probability")
        ax_cal.set_ylabel("Observed Frequency")
        ax_cal.set_title("Reliability Diagram", fontweight="bold")
        ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
        _styled_ax(ax_cal)
        _lt_legend(ax_cal, pr_steps, ncol=4, loc="upper left",
                   extra_handles=[diag_line])
        fig_cal.tight_layout()
        fig_cal.savefig(cal_path, dpi=150, bbox_inches="tight", facecolor="white")
        if getattr(args, 'plot_only', False): plt.show()
        plt.close(fig45)
        plt.close(fig_cal)
        logger.info(f"PR curve        -> {pr_path}")
        logger.info(f"Calibration     -> {cal_path}")



def run_test_evaluation(args):
    """
    Load a checkpoint, run ensemble inference on a held-out test folder,
    compute per-lead-time metrics, and save results.

    Outputs written to args.output_dir:
        metrics_summary.json   — scalar summary (crps_mean, csi_mean, ...)
        metrics_per_step.csv   — one row per lead-time step
        skill_curves.png       — CRPS / spread-skill / CSI vs lead time
        plots/                 — one PNG per evaluated sequence (if --plot)
    """
    import os, json, csv
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # ---- tqdm-safe logging ----
    from tqdm import tqdm as _tqdm

    class _TqdmHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                _tqdm.write(self.format(record))
            except Exception:
                self.handleError(record)

    root_log = logging.getLogger()
    root_log.handlers.clear()
    h = _TqdmHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_log.addHandler(h)
    root_log.setLevel(logging.INFO)

    # ---- plot_only: load saved arrays and jump straight to plots ----
    if args.plot_only:
        npz_path = os.path.join(args.output_dir, "plot_data.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"--plot_only requires {npz_path}\n"
                "Run evaluation once first (without --plot_only) to generate it."
            )
        logger.info(f"--plot_only: loading plot data from {npz_path}")
        _regenerate_plots(npz_path, args)
        return {}

    # ---- Device ----
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- Load checkpoint ----
    from model import UNet, EDMPrecond, MultiStepDenoiser, EDMSchedule

    ckpt      = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt["args"]
    channels  = ckpt["channels"]
    stats     = ckpt["stats"]
    T_in      = ckpt_args["T_in"]
    T_out     = ckpt_args["T_out"]
    dt_min    = ckpt_args["dt_min"]
    C         = len(channels)

    unet = UNet(
        in_channels      = C * (T_in + 2),
        out_channels     = C,
        base_channels    = ckpt_args["base_channels"],
        channel_mults    = tuple(ckpt_args["channel_mults"]),
        num_res_blocks   = ckpt_args["num_res_blocks"],
        attn_resolutions = tuple(ckpt_args["attn_resolutions"]),
        dropout          = 0.0,
        emb_dim          = ckpt_args["emb_dim"],
    )
    precond  = EDMPrecond(unet, sigma_data=ckpt_args.get("sigma_data", 0.5))
    model    = MultiStepDenoiser(precond, T_out=T_out, dt_min=dt_min)
    # Prefer EMA weights for evaluation
    state    = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    logger.info(f"Checkpoint loaded: {args.checkpoint}")
    logger.info(f"  channels={channels}  T_in={T_in}  T_out={T_out}  dt={dt_min}min")

    # ---- Test loader (non-overlapping) ----
    from dataset import make_test_loader

    test_loader = make_test_loader(
        test_roots   = args.test_roots,
        channel_list = channels,
        stats        = stats,
        T_in         = T_in,
        T_out        = T_out,
        img_size     = tuple(args.img_size),
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
    )
    logger.info(f"Test sequences (non-overlapping): {len(test_loader.dataset)}")

    # ---- Channel index helpers ----
    li_idx      = channels.index("li") if "li" in channels else None
    cloud_chs   = [ch for ch in channels if ch != "li"]   # names
    cloud_idxs  = [channels.index(ch) for ch in cloud_chs]

    fss_scales = args.fss_scales

    # ---- Accumulators ----
    # CRPS / spread-skill: single scalar per (sample, step)
    crps_by_step = [[] for _ in range(T_out)]
    ss_by_step   = [[] for _ in range(T_out)]

    # Per-channel cloud metrics: dict[ch] -> list-per-step -> list of scalars
    # keys: "rmse", "mae", "ssim"  (ssim only if HAS_SKIMAGE)
    rmse_by_ch_step = {ch: [[] for _ in range(T_out)] for ch in cloud_chs}
    mae_by_ch_step  = {ch: [[] for _ in range(T_out)] for ch in cloud_chs}
    ssim_by_ch_step = {ch: [[] for _ in range(T_out)] for ch in cloud_chs}

    # Lightning
    csi_by_step   = [[] for _ in range(T_out)]
    pod_by_step   = [[] for _ in range(T_out)]
    far_by_step   = [[] for _ in range(T_out)]
    brier_by_step = [[] for _ in range(T_out)]
    fss_by_scale_step = {s: [[] for _ in range(T_out)] for s in fss_scales}

    # All lead-time steps — pixel subsampling keeps memory bounded
    pr_steps  = list(range(T_out))
    pr_probs  = {t: [] for t in pr_steps}
    pr_labels = {t: [] for t in pr_steps}
    cal_probs  = {t: [] for t in pr_steps}
    cal_labels = {t: [] for t in pr_steps}

    os.makedirs(args.output_dir, exist_ok=True)
    if args.plot:
        os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    # ---- Evaluation loop ----
    seq_counter = 0
    batch_bar   = tqdm(test_loader, desc="Evaluating", unit="batch",
                       dynamic_ncols=True, leave=True)

    for batch in batch_bar:
        context  = batch["context"].to(device)
        target   = batch["target"].to(device)       # (B, T_out, C, H, W) residuals
        last_ctx = batch["last_ctx"].to(device)     # (B, C, H, W) last context frame
        ch_mask  = batch["tgt_mask"][:, 0].to(device)

        ens = generate_ensemble(
            model, context, ch_mask, device,
            n_members = args.n_members,
            cfg_scale = args.cfg_scale,
        )  # (B, M, T_out, C, H, W) residuals

        # Reconstruct absolute normalised frames before all metric computation
        last_ctx_np = last_ctx.cpu().numpy()                            # (B, C, H, W)
        ens_np  = ens.cpu().numpy() + last_ctx_np[:, None, None]        # (B,M,T_out,C,H,W)
        tgt_np  = target.cpu().numpy() + last_ctx_np[:, None]           # (B,T_out,C,H,W)
        ctx_np  = batch["context"].numpy()                              # (B,T_in,C,H,W)
        B       = ens_np.shape[0]

        for b in range(B):
            for t in range(T_out):
                ens_t    = ens_np[b, :, t]   # (M, C, H, W) absolute normalised
                tgt_t    = tgt_np[b, t]      # (C, H, W)    absolute normalised
                ens_mean = ens_t.mean(axis=0)

                crps_by_step[t].append(crps_energy(ens_t, tgt_t))
                ss_by_step[t].append(spread_skill(ens_t, tgt_t))

                if cloud_idxs:
                    cm = cloud_metrics(ens_mean, tgt_t, cloud_idxs,
                                       cloud_chs, stats)
                    for ch in cloud_chs:
                        if ch in cm:
                            rmse_by_ch_step[ch][t].append(cm[ch]["rmse"])
                            mae_by_ch_step[ch][t].append(cm[ch]["mae"])
                            if "ssim" in cm[ch]:
                                ssim_by_ch_step[ch][t].append(cm[ch]["ssim"])

                if li_idx is not None:
                    # Threshold on absolute normalised LI
                    pred_prob = (ens_t[:, li_idx] > args.li_threshold).mean(axis=0)
                    obs_bin   = (tgt_t[li_idx] > args.li_threshold).astype(np.float32)

                    ct = lightning_contingency(pred_prob, obs_bin)
                    csi_by_step[t].append(ct["csi"])
                    pod_by_step[t].append(ct["pod"])
                    far_by_step[t].append(ct["far"])
                    brier_by_step[t].append(brier_score(pred_prob, obs_bin))
                    for s in fss_scales:
                        fss_by_scale_step[s][t].append(fss(pred_prob, obs_bin, scale=s))

                    if t in pr_steps:
                        flat_prob = pred_prob.ravel()
                        flat_lbl  = obs_bin.ravel()
                        stride    = max(1, len(flat_prob) // 4096)
                        pr_probs[t].append(flat_prob[::stride])
                        pr_labels[t].append(flat_lbl[::stride])
                        cal_probs[t].append(flat_prob[::stride])
                        cal_labels[t].append(flat_lbl[::stride])

            if args.plot and seq_counter < args.max_plots:
                from dataset import denormalize as _denorm
                ctx_b   = ctx_np[b]
                ctx_den = np.zeros_like(ctx_b)
                ens_den = np.zeros_like(ens_np[b])
                tgt_den = np.zeros_like(tgt_np[b])
                for ci, ch in enumerate(channels):
                    fn = (lambda x, _ch=ch: _denorm(x, stats, _ch)) if ch in stats                          else (lambda x: x)
                    ctx_den[:, ci]    = fn(ctx_b[:, ci])
                    # ens_np and tgt_np are already absolute — just denormalise
                    tgt_den[:, ci]    = fn(tgt_np[b, :, ci])
                    for m in range(ens_np.shape[1]):
                        ens_den[m, :, ci] = fn(ens_np[b, m, :, ci])
                png = os.path.join(args.output_dir, "plots", f"eval_{seq_counter:04d}.png")
                plot_forecast(context_np=ctx_den, ens_np=ens_den,
                              channels=channels, gt_np=tgt_den, save_path=png)

            seq_counter += 1

        live_crps = float(np.mean([np.mean(v) for v in crps_by_step if v]))
        live_csi  = float(np.mean([np.mean(v) for v in csi_by_step  if v]))                     if li_idx is not None else float("nan")
        batch_bar.set_postfix(crps=f"{live_crps:.4f}", csi=f"{live_csi:.3f}", refresh=False)

    # ---- Aggregate per-lead-time ----
    def _mean(lst): return float(np.mean(lst)) if lst else float("nan")

    lead_times = [(t + 1) * dt_min for t in range(T_out)]

    # Resolve unit label per cloud channel
    ch_unit = {}
    for ch in cloud_chs:
        is_cbrt = ch in stats and stats[ch].get("transform") == "cbrt"
        ch_unit[ch] = "norm" if is_cbrt else "K"

    per_step = []
    for t in range(T_out):
        row = {"lead_min": lead_times[t],
               "crps":     _mean(crps_by_step[t]),
               "spread_skill": _mean(ss_by_step[t])}
        for ch in cloud_chs:
            row[f"rmse_{ch}"] = _mean(rmse_by_ch_step[ch][t])
            row[f"mae_{ch}"]  = _mean(mae_by_ch_step[ch][t])
            if ssim_by_ch_step[ch][t]:
                row[f"ssim_{ch}"] = _mean(ssim_by_ch_step[ch][t])
        if li_idx is not None:
            row.update({
                "csi":   _mean(csi_by_step[t]),
                "pod":   _mean(pod_by_step[t]),
                "far":   _mean(far_by_step[t]),
                "brier": _mean(brier_by_step[t]),
                "fss":   _mean(fss_by_scale_step[fss_scales[len(fss_scales)//2]][t]),
            })
        per_step.append(row)

    fss_summary = {
        f"fss_scale{s}": _mean([_mean(fss_by_scale_step[s][t]) for t in range(T_out)])
        for s in fss_scales
    } if li_idx is not None else {}

    # Scalar summary (mean over time for each channel)
    summary = {
        "checkpoint":   args.checkpoint,
        "test_roots":   args.test_roots,
        "n_sequences":  seq_counter,
        "n_members":    args.n_members,
        "crps_mean":    _mean([r["crps"] for r in per_step]),
        "crps_1h":      _mean([r["crps"] for r in per_step[:6]]),
        "crps_3h":      _mean([r["crps"] for r in per_step[:18]]),
        "crps_6h":      per_step[-1]["crps"],
        "spread_skill": _mean([r["spread_skill"] for r in per_step]),
    }
    for ch in cloud_chs:
        summary[f"rmse_{ch}_mean"] = _mean([r[f"rmse_{ch}"] for r in per_step])
        summary[f"mae_{ch}_mean"]  = _mean([r[f"mae_{ch}"]  for r in per_step])
        ssim_vals = [r.get(f"ssim_{ch}", float("nan")) for r in per_step]
        if not all(np.isnan(ssim_vals)):
            summary[f"ssim_{ch}_mean"] = _mean(ssim_vals)
    if li_idx is not None:
        summary.update({
            "csi_mean":   _mean([r["csi"]   for r in per_step]),
            "csi_1h":     _mean([r["csi"]   for r in per_step[:6]]),
            "csi_6h":     per_step[-1]["csi"],
            "pod_mean":   _mean([r["pod"]   for r in per_step]),
            "far_mean":   _mean([r["far"]   for r in per_step]),
            "brier_mean": _mean([r["brier"] for r in per_step]),
            **fss_summary,
        })

    # ---- Save JSON ----
    json_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary    -> {json_path}")

    # ---- Save CSV ----
    csv_path = os.path.join(args.output_dir, "metrics_per_step.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_step[0].keys())
        writer.writeheader()
        writer.writerows(per_step)
    logger.info(f"Per-step   -> {csv_path}")

    # ---- Save plot data (all arrays needed to regenerate plots) ----
    # Saved as .npz so plots can be regenerated instantly with --plot_only
    # without re-running the full evaluation.
    npz_path = os.path.join(args.output_dir, "plot_data.npz")
    npz_payload = {
        "per_step_json": np.array(json.dumps(per_step)),   # serialised as scalar str
        "lead_times":    np.array(lead_times),
        "channels":      np.array(channels),
        "cloud_chs":     np.array(cloud_chs),
        "fss_scales":    np.array(fss_scales),
        "T_out":         np.array(T_out),
        "dt_min":        np.array(dt_min),
        "pixel_size_km": np.array(args.pixel_size_km),
    }
    # FSS per scale per step
    for s in fss_scales:
        npz_payload[f"fss_s{s}"] = np.array(
            [_mean(fss_by_scale_step[s][t]) for t in range(T_out)]
        )
    # PR / calibration: concatenated and subsampled (already bounded)
    pr_steps_arr = list(pr_steps)
    npz_payload["pr_steps"] = np.array(pr_steps_arr)
    for t in pr_steps_arr:
        if pr_probs[t]:
            npz_payload[f"pr_prob_{t}"]   = np.concatenate(pr_probs[t])
            npz_payload[f"pr_label_{t}"]  = np.concatenate(pr_labels[t])
            npz_payload[f"cal_prob_{t}"]  = np.concatenate(cal_probs[t])
            npz_payload[f"cal_label_{t}"] = np.concatenate(cal_labels[t])
    np.savez_compressed(npz_path, **npz_payload)
    logger.info(f"Plot data  -> {npz_path}  (use --plot_only to regenerate plots)")

    # ---- All plots ----
    # ---- All plots ----
    _regenerate_plots(npz_path, args)

    # ---- Print summary ----
    logger.info("\n" + "=" * 52)
    logger.info("  CLOUD METRICS")
    logger.info("=" * 52)
    for k in ["crps_mean","crps_1h","crps_3h","crps_6h",
              "rmse_mean","mae_mean","ssim_mean","spread_skill"]:
        if k in summary:
            logger.info(f"  {k:<22s}: {summary[k]:.4f}")
    if li_idx is not None:
        logger.info("\n" + "=" * 52)
        logger.info("  LIGHTNING METRICS")
        logger.info("=" * 52)
        for k in ["csi_mean","csi_1h","csi_6h","pod_mean",
                  "far_mean","brier_mean"]:
            if k in summary:
                logger.info(f"  {k:<22s}: {summary[k]:.4f}")
        for k in sorted([k for k in summary if k.startswith("fss_scale")]):
            logger.info(f"  {k:<22s}: {summary[k]:.4f}")
    logger.info("=" * 52)

    return summary


# ===================================================================
# Entry point
# ===================================================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Run full test-set evaluation on a trained METSAT nowcasting model."
    )
    p.add_argument("--checkpoint",    required=True,
                   help="Path to best.pt or latest.pt")
    p.add_argument("--test_roots",    required=True, nargs="+",
                   help="One or more dataset root directories for the test set")
    p.add_argument("--output_dir",    default="outputs/evaluation",
                   help="Directory to write metrics, CSV and plots")
    p.add_argument("--n_members",     type=int,   default=10,
                   help="Ensemble members (more = slower but better CRPS)")
    p.add_argument("--cfg_scale",     type=float, default=1.5)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--img_size",      nargs=2, type=int, default=[256, 256])
    p.add_argument("--li_threshold",  type=float, default=0.1,
                   help="Normalised LI threshold for binary metrics")
    p.add_argument("--gpu",           type=int,   default=0)
    p.add_argument("--plot",          action="store_true",
                   help="Save full forecast PNGs for each test sequence")
    p.add_argument("--plot_only",     action="store_true",
                   help="Skip inference — reload plot_data.npz from --output_dir "
                        "and regenerate all figures instantly.")
    p.add_argument("--max_plots",     type=int,   default=20)
    p.add_argument("--fss_scales",    nargs="+", type=int,
                   default=[1, 2, 4, 8, 16, 32],
                   help="Neighbourhood half-widths (pixels) for FSS curve")
    p.add_argument("--pixel_size_km", type=float, default=4.0,
                   help="Pixel size in km — used to label FSS x-axis in km")
    args = p.parse_args()
    run_test_evaluation(args)