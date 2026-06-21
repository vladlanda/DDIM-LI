"""
Persistence baseline for METSAT lightning nowcasting.

Strategy: for each forecast step t, predict that the LI field at time
t equals the LAST OBSERVED LI frame in the context window.
This is the simplest physically-motivated baseline — it assumes
"whatever lightning was present at the last observation will persist."

Usage:
    python persistence_baseline.py --config configs/evaluate.yaml

Output:
    persistence_metrics.csv  — same format as metrics_per_step.csv
                               from evaluate.py, for direct comparison.
"""

import argparse
import csv
import logging
import os
import sys
from typing import Dict, List

import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import precision_recall_curve as _pr_curve
    from sklearn.calibration import calibration_curve as _cal_curve
    from sklearn.metrics import auc as _auc
except ImportError:
    raise ImportError("scikit-learn required: pip install scikit-learn")

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    logger.warning("scikit-image not found — SSIM will not be computed")

# Reuse helpers from evaluate.py
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import (
    _li_to_physical, crps_energy, lightning_skill_curve, fss, spread_skill
)
from dataset import make_test_loader, denormalize


# --------------------------------------------------------------------------
# Persistence forecast
# --------------------------------------------------------------------------

def persistence_forecast(
    last_ctx_norm: np.ndarray,   # (C, H, W) last context frame, normalised
    T_out:         int,
) -> np.ndarray:
    """
    Return a persistence forecast: repeat last_ctx for all T_out steps.

    Returns (T_out, C, H, W) in normalised absolute space.
    The residual for each step is 0 (no change from last context frame).
    In absolute space: pred_abs[t] = last_ctx for all t.
    """
    return np.stack([last_ctx_norm] * T_out, axis=0)   # (T_out, C, H, W)


# --------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------

def run_persistence_evaluation(args):
    # ── Load stats from an existing checkpoint (needed for denormalisation)
    # We don't need the model weights — just the channel stats and args.
    import torch
    ckpt      = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt["args"]
    stats     = ckpt["stats"]
    channels  = ckpt["channels"]

    C       = len(channels)
    T_in    = ckpt_args["T_in"]
    T_out   = ckpt_args["T_out"]
    dt_min  = ckpt_args["dt_min"]
    li_idx  = channels.index("li") if "li" in channels else None

    binary_li_ctx = ckpt_args.get("binary_li_ctx", False)
    ctx_channels  = ckpt_args.get("ctx_channels", None)

    logger.info(f"Channels: {channels}  T_in={T_in}  T_out={T_out}  dt_min={dt_min}")
    logger.info(f"LI index: {li_idx}")

    # ── Test loader
    test_loader = make_test_loader(
        test_roots    = args.test_roots,
        channel_list  = channels,
        stats         = stats,
        T_in          = T_in,
        T_out         = T_out,
        img_size      = tuple(args.img_size),
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
        binary_li_ctx = binary_li_ctx,
        ctx_channels  = ctx_channels,
    )
    logger.info(f"Test sequences: {len(test_loader.dataset)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Accumulators — same structure as evaluate.py
    lead_times   = [(t + 1) * dt_min for t in range(T_out)]
    pr_steps     = list(range(T_out))

    crps_by_step  = [[] for _ in range(T_out)]
    ss_by_step    = [[] for _ in range(T_out)]
    skill_by_step = [[] for _ in range(T_out)]
    fss_by_step   = {thr: [[] for _ in range(T_out)]
                     for thr in args.fss_prob_thresholds}
    ssim_by_step  = [[] for _ in range(T_out)]
    rmse_by_step  = [[] for _ in range(T_out)]
    pr_probs      = {t: [] for t in pr_steps}
    pr_labels     = {t: [] for t in pr_steps}

    # ── Evaluation loop
    batch_bar = tqdm(test_loader, desc="Persistence eval", unit="batch",
                     dynamic_ncols=True)

    for batch in batch_bar:
        context  = batch["context"].numpy()    # (B, T_in, C_ctx, H, W)
        target   = batch["target"].numpy()     # (B, T_out, C, H, W) residuals
        last_ctx = batch["last_ctx"].numpy()   # (B, C, H, W)

        # Reconstruct absolute target
        tgt_abs = target + last_ctx[:, None]   # (B, T_out, C, H, W)

        B = context.shape[0]

        for b in range(B):
            # Persistence forecast: predict last_ctx for all steps
            # Shape: (T_out, C, H, W) in normalised absolute space
            pred_abs = persistence_forecast(last_ctx[b], T_out)  # (T_out, C, H, W)

            for t in range(T_out):
                tgt_t  = tgt_abs[b, t]   # (C, H, W) normalised absolute
                pred_t = pred_abs[t]     # (C, H, W) normalised absolute

                # ── CRPS and spread-skill
                # Persistence is deterministic → M=1 ensemble
                # CRPS with M=1 reduces to MAE
                pred_ens = pred_t[None]   # (1, C, H, W) — single "member"
                crps_by_step[t].append(crps_energy(pred_ens, tgt_t))
                ss_by_step[t].append(spread_skill(pred_ens, tgt_t))

                # ── RMSE on IR channel
                ir_idx = channels.index("ir") if "ir" in channels else 0
                rmse   = float(np.sqrt(np.mean(
                    (pred_t[ir_idx] - tgt_t[ir_idx])**2
                )))
                rmse_by_step[t].append(rmse)

                # ── SSIM on IR channel
                if HAS_SKIMAGE:
                    ir_pred = pred_t[ir_idx]
                    ir_tgt  = tgt_t[ir_idx]
                    vrange  = float(ir_tgt.max() - ir_tgt.min()) + 1e-6
                    ssim_val = float(ssim_fn(ir_pred, ir_tgt,
                                             data_range=vrange))
                    ssim_by_step[t].append(ssim_val)

                # ── LI binary metrics
                if li_idx is not None:
                    obs_phys  = _li_to_physical(tgt_t[li_idx],  stats)
                    pred_phys = _li_to_physical(pred_t[li_idx], stats)

                    obs_bin   = (obs_phys  >= args.li_event_threshold
                                 ).astype(np.float32)
                    # Persistence pred_prob is binary: either 1 (last LI > thresh)
                    # or 0 (last LI == 0). Use continuous physical value as prob.
                    # Clip to [0,1]: persistence LI value IS the "probability."
                    pred_prob = np.clip(pred_phys / max(pred_phys.max(), 1e-6),
                                        0.0, 1.0).astype(np.float32)

                    skill_by_step[t].append(
                        lightning_skill_curve(pred_prob, obs_bin))

                    # FSS at multiple thresholds
                    for thr in args.fss_prob_thresholds:
                        pred_bin = (pred_prob >= thr).astype(float)
                        scales   = [1, 2, 4, 8, 16, 32]
                        for scale in scales:
                            pass  # simplified — just threshold FSS at scale=1
                        fss_val  = fss(pred_bin, obs_bin, scale=1)
                        fss_by_step[thr][t].append(fss_val)

                    # PR curve data
                    stride    = max(1, obs_bin.size // 4096)
                    flat_prob = pred_prob.ravel()[::stride]
                    flat_lbl  = obs_bin.ravel()[::stride]
                    pr_probs[t].append(flat_prob)
                    pr_labels[t].append(flat_lbl)

    # ── Compute PR-AUC + save full PR curves for overlay plotting
    auc_by_step = {}
    pr_curves   = {}   # t -> (precision, recall) arrays for dashed overlay
    for t in pr_steps:
        if not pr_probs[t]:
            continue
        try:
            all_prob = np.concatenate(pr_probs[t]).astype(np.float32)
            all_lbl  = np.concatenate(pr_labels[t]).astype(np.int32)
            prec, rec, _ = _pr_curve(all_lbl, all_prob)
            auc_by_step[t] = float(_auc(rec, prec))
            pr_curves[t]   = (prec.astype(np.float32), rec.astype(np.float32))
        except Exception as e:
            logger.warning(f"PR-AUC failed at step {t}: {e}")

    # Save persistence PR + calibration curves to npz for overlay plots.
    npz_payload = {}
    for t, (prec, rec) in pr_curves.items():
        npz_payload[f"prec_{t}"] = prec
        npz_payload[f"rec_{t}"]  = rec
        npz_payload[f"auc_{t}"]  = np.array(auc_by_step.get(t, float("nan")))
        # Calibration curve from the same prob/label data
        try:
            all_prob = np.concatenate(pr_probs[t]).astype(np.float32)
            all_lbl  = np.concatenate(pr_labels[t]).astype(np.int32)
            frac_pos, mean_pred = _cal_curve(all_lbl, all_prob,
                                             n_bins=10, strategy="uniform")
            npz_payload[f"cal_mean_{t}"] = mean_pred.astype(np.float32)
            npz_payload[f"cal_frac_{t}"] = frac_pos.astype(np.float32)
        except Exception as e:
            logger.warning(f"Persistence calibration failed at step {t}: {e}")
    npz_payload["pr_steps"] = np.array(list(pr_curves.keys()))
    npz_payload["dt_min"]   = np.array(dt_min)
    pr_npz = os.path.join(args.output_dir, "persistence_pr_curves.npz")
    np.savez_compressed(pr_npz, **npz_payload)
    logger.info(f"Persistence PR + calibration curves -> {pr_npz}")

    # ── Build per-step CSV rows
    _mean = lambda lst: float(np.mean(lst)) if lst else float("nan")

    per_step = []
    for t in range(T_out):
        row = {
            "lead_min":    lead_times[t],
            "crps":        _mean(crps_by_step[t]),
            "spread_skill": _mean(ss_by_step[t]),
            "ssim_ir":     _mean(ssim_by_step[t]) if HAS_SKIMAGE else float("nan"),
            "rmse_ir":     _mean(rmse_by_step[t]),
        }

        if skill_by_step[t]:
            # CSI at fixed prob thresholds
            for thr in args.fss_prob_thresholds:
                csi_at = [float(np.interp(thr, sc["thresholds"], sc["csi"]))
                          for sc in skill_by_step[t] if len(sc["thresholds"])]
                pod_at = [float(np.interp(thr, sc["thresholds"], sc["pod"]))
                          for sc in skill_by_step[t] if len(sc["thresholds"])]
                far_at = [float(np.interp(thr, sc["thresholds"], sc["far"]))
                          for sc in skill_by_step[t] if len(sc["thresholds"])]
                row[f"csi_{thr}"] = _mean(csi_at)
                row[f"pod_{thr}"] = _mean(pod_at)
                row[f"far_{thr}"] = _mean(far_at)

            # CSI at optimal threshold
            csi_max_vals, pod_max_vals, far_max_vals = [], [], []
            for sc in skill_by_step[t]:
                if len(sc["csi"]):
                    best = int(np.argmax(sc["csi"]))
                    csi_max_vals.append(sc["csi"][best])
                    pod_max_vals.append(sc["pod"][best])
                    far_max_vals.append(sc["far"][best])
            row["csi_max"]    = _mean(csi_max_vals)
            row["pod_at_max"] = _mean(pod_max_vals)
            row["far_at_max"] = _mean(far_max_vals)

        for thr in args.fss_prob_thresholds:
            row[f"fss_{thr}"] = _mean(fss_by_step[thr][t])

        if t in auc_by_step:
            row["pr_auc"] = auc_by_step[t]

        per_step.append(row)

    # ── Save CSV
    csv_path = os.path.join(args.output_dir, "persistence_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_step[0].keys())
        writer.writeheader()
        writer.writerows(per_step)
    logger.info(f"Saved → {csv_path}")

    # ── Print summary
    print("\n=== Persistence Baseline — Summary ===")
    print(f"{'Lead':>8}  {'PR-AUC':>8}  {'CSI_max':>8}  {'FAR@max':>8}  "
          f"{'SSIM_IR':>8}  {'CRPS':>8}")
    print("-" * 60)
    for row in per_step:
        print(f"  +{int(row['lead_min']):3d}min  "
              f"{row.get('pr_auc', float('nan')):>8.3f}  "
              f"{row.get('csi_max', float('nan')):>8.3f}  "
              f"{row.get('far_at_max', float('nan')):>8.3f}  "
              f"{row.get('ssim_ir', float('nan')):>8.3f}  "
              f"{row['crps']:>8.3f}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Persistence baseline evaluation for METSAT lightning nowcasting."
    )
    p.add_argument("--config",     default=None,
                   help="Path to evaluate.yaml — reads checkpoint, test_roots, etc.")
    p.add_argument("--checkpoint", default=None,
                   help="Path to best.pt or latest.pt (for stats and args only)")
    p.add_argument("--test_roots", nargs="+", default=None)
    p.add_argument("--output_dir", default="outputs/persistence_baseline")
    p.add_argument("--img_size",   nargs=2, type=int, default=[256, 256])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--li_event_threshold", type=float, default=5.0/255.0)
    p.add_argument("--fss_prob_thresholds", nargs="+", type=float,
                   default=[0.1, 0.3, 0.5])
    args = p.parse_args()

    # Load YAML config if provided
    if args.config is not None:
        import pathlib, sys
        if not pathlib.Path(args.config).exists():
            p.error(f"Config not found: {args.config}")
        try:
            import yaml
        except ImportError:
            p.error("PyYAML required: pip install pyyaml")
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        # YAML values as defaults; CLI overrides
        for k, v in cfg.items():
            if k != "config" and hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)

    if args.checkpoint is None:
        p.error("--checkpoint is required")
    if args.test_roots is None:
        p.error("--test_roots is required")

    return args


if __name__ == "__main__":
    args = parse_args()
    run_persistence_evaluation(args)
