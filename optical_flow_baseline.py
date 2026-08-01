"""
Optical-flow (pySTEPS-style) extrapolation baseline for METSAT lightning
nowcasting.

Two variants, matching the convention established in nowcasting literature:

  --flow_source li  (PRIMARY / standard convention)
      The motion field is estimated from the SAME field being forecast (LI),
      exactly as precipitation-nowcasting pySTEPS baselines derive flow from
      radar reflectivity and advect radar reflectivity. This is what a
      reviewer expects by default when "optical flow baseline" is claimed.
      Expected to perform poorly given how sparse LI is -- consistent with
      the literature's own observation that extrapolation baselines struggle
      when the pre-existing signal is weak (Cintineo-style convective
      initiation problem).

  --flow_source ir  (SECONDARY / robustness variant)
      The motion field is estimated from the denser, more reliable IR
      channel and applied to advect LI. Direct precedent: severe-convection
      nowcasting papers derive one motion field from the primary/densest
      field and apply it to advect several other target fields separately.
      Gives the optical-flow method its strongest reasonable chance.

Method (semi-Lagrangian extrapolation, the pySTEPS convention):
  1. Estimate a dense flow field (u, v) between the last two available
     context frames of the flow-source channel via Farneback optical flow.
  2. Hold that flow field CONSTANT across the forecast horizon (Lagrangian
     persistence of the motion field -- the standard extrapolation
     assumption; no growth/decay is modelled).
  3. For lead step k, backward-warp the LAST OBSERVED frame of every
     channel by k*(u,v) (semi-Lagrangian: sample the source at the
     backward-traced position). This produces a full (T_out, C, H, W)
     deterministic forecast, directly comparable to persistence_baseline.py.

Evaluation loop, metrics, and output schema are a deliberate near-mirror of
persistence_baseline.py so the three baselines (persistence, pysteps_li,
pysteps_ir) are directly, apples-to-apples comparable.

Usage:
  python optical_flow_baseline.py --config configs/evaluate.yaml \
      --flow_source li --output_dir pysteps_li
  python optical_flow_baseline.py --config configs/evaluate.yaml \
      --flow_source ir --output_dir pysteps_ir
"""

import argparse
import csv
import logging
import os
import sys
from typing import Dict, List

import numpy as np
import cv2
from tqdm import tqdm

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import (
    _li_to_physical, crps_energy, lightning_skill_curve, fss, spread_skill
)
from dataset import make_test_loader, denormalize


# --------------------------------------------------------------------------
# Flow estimation + semi-Lagrangian advection
# --------------------------------------------------------------------------

def _to_uint8(field: np.ndarray) -> np.ndarray:
    """Min-max scale a float field to uint8 for Farneback (which requires
    8-bit single-channel input). Flow estimation only needs relative
    gradient structure, so the absolute intensity scale doesn't matter."""
    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-8:
        return np.zeros_like(field, dtype=np.uint8)
    return np.clip((field - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def estimate_flow(frame_prev: np.ndarray, frame_curr: np.ndarray) -> np.ndarray:
    """
    Dense Farneback optical flow between two frames.
    Returns (H, W, 2) flow field: flow[...,0]=dx, flow[...,1]=dy, in pixels
    per (frame_curr - frame_prev) time step.
    """
    a, b = _to_uint8(frame_prev), _to_uint8(frame_curr)
    flow = cv2.calcOpticalFlowFarneback(
        a, b, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    return flow.astype(np.float32)   # (H, W, 2)


def advect(field: np.ndarray, flow: np.ndarray, k: int) -> np.ndarray:
    """
    Semi-Lagrangian backward advection: sample `field` at each pixel's
    position traced backward by k*flow. Out-of-domain source -> 0
    (no signal), matching the fill convention used elsewhere in this
    codebase for missing/out-of-bounds data.

    field: (H, W) float32.  flow: (H, W, 2) from estimate_flow.
    """
    H, W = field.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = xx - k * flow[..., 0]
    map_y = yy - k * flow[..., 1]
    return cv2.remap(
        field.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )


def optical_flow_forecast(
    context_norm: np.ndarray,   # (T_in, C, H, W) normalised, most-recent last
    flow_source_idx: int,       # channel index to derive flow from
    T_out: int,
) -> np.ndarray:
    """
    Returns (T_out, C, H, W) deterministic forecast in normalised absolute
    space: every channel advected by the SAME flow field, held constant
    across the forecast horizon (standard pySTEPS extrapolation assumption).
    """
    T_in, C, H, W = context_norm.shape
    frame_prev = context_norm[-2, flow_source_idx]
    frame_curr = context_norm[-1, flow_source_idx]
    flow = estimate_flow(frame_prev, frame_curr)   # (H, W, 2)

    last_ctx = context_norm[-1]   # (C, H, W) — the field being advected
    out = np.zeros((T_out, C, H, W), dtype=np.float32)
    for k in range(1, T_out + 1):
        for c in range(C):
            out[k - 1, c] = advect(last_ctx[c], flow, k)
    return out


# --------------------------------------------------------------------------
# Main evaluation (near-mirror of persistence_baseline.py's structure, for
# direct apples-to-apples comparability across the three baselines)
# --------------------------------------------------------------------------

def run_optical_flow_evaluation(args):
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
    ir_idx  = channels.index("ir") if "ir" in channels else 0

    if args.flow_source not in channels:
        raise ValueError(f"--flow_source '{args.flow_source}' not in "
                         f"model channels {channels}")
    flow_source_idx = channels.index(args.flow_source)

    binary_li_ctx = ckpt_args.get("binary_li_ctx", False)
    ctx_channels  = ckpt_args.get("ctx_channels", None)

    logger.info(f"Channels: {channels}  T_in={T_in}  T_out={T_out}  dt_min={dt_min}")
    logger.info(f"Flow source: '{args.flow_source}' (idx={flow_source_idx})  "
               f"LI index: {li_idx}")

    if T_in < 2:
        raise ValueError(f"T_in={T_in} < 2: need at least 2 context frames "
                         f"to estimate optical flow.")

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
    pr_seqids     = {t: [] for t in pr_steps}
    seq_counter = 0

    batch_bar = tqdm(test_loader, desc=f"Optical-flow[{args.flow_source}] eval",
                     unit="batch", dynamic_ncols=True)

    for batch in batch_bar:
        context  = batch["context"].numpy()    # (B, T_in, C_ctx, H, W)
        target   = batch["target"].numpy()     # (B, T_out, C, H, W) residuals
        last_ctx = batch["last_ctx"].numpy()   # (B, C, H, W)

        # context may include an extra binary-LI-presence channel appended
        # per frame (binary_li_ctx); flow/advection only need the C "real"
        # data channels, which are the first C entries of each context frame.
        context_data = context[:, :, :C]       # (B, T_in, C, H, W)

        tgt_abs = target + last_ctx[:, None]   # (B, T_out, C, H, W)
        B = context.shape[0]

        for b in range(B):
            pred_abs = optical_flow_forecast(
                context_data[b], flow_source_idx, T_out)  # (T_out, C, H, W)

            for t in range(T_out):
                tgt_t  = tgt_abs[b, t]
                pred_t = pred_abs[t]

                pred_ens = pred_t[None]   # (1, C, H, W) — deterministic, M=1
                crps_by_step[t].append(crps_energy(pred_ens, tgt_t))
                ss_by_step[t].append(spread_skill(pred_ens, tgt_t))

                rmse = float(np.sqrt(np.mean(
                    (pred_t[ir_idx] - tgt_t[ir_idx])**2)))
                rmse_by_step[t].append(rmse)

                if HAS_SKIMAGE:
                    ir_pred, ir_tgt = pred_t[ir_idx], tgt_t[ir_idx]
                    vrange = float(ir_tgt.max() - ir_tgt.min()) + 1e-6
                    ssim_by_step[t].append(
                        float(ssim_fn(ir_pred, ir_tgt, data_range=vrange)))

                if li_idx is not None:
                    obs_phys  = _li_to_physical(tgt_t[li_idx],  stats)
                    pred_phys = _li_to_physical(pred_t[li_idx], stats)

                    obs_bin = (obs_phys >= args.li_event_threshold
                              ).astype(np.float32)
                    # Advected LI can go negative/overshoot at edges (linear
                    # interpolation across a sharp cbrt-transformed sparse
                    # field); clip to a valid probability-like range exactly
                    # as persistence_baseline.py does for its deterministic
                    # forecast.
                    pred_prob = np.clip(
                        pred_phys / max(pred_phys.max(), 1e-6),
                        0.0, 1.0).astype(np.float32)

                    skill_by_step[t].append(
                        lightning_skill_curve(pred_prob, obs_bin))

                    for thr in args.fss_prob_thresholds:
                        pred_bin = (pred_prob >= thr).astype(float)
                        fss_val  = fss(pred_bin, obs_bin, scale=1)
                        fss_by_step[thr][t].append(fss_val)

                    stride    = max(1, obs_bin.size // 4096)
                    flat_prob = pred_prob.ravel()[::stride]
                    flat_lbl  = obs_bin.ravel()[::stride]
                    pr_probs[t].append(flat_prob)
                    pr_labels[t].append(flat_lbl)
                    pr_seqids[t].append(np.full(flat_prob.shape, seq_counter,
                                                dtype=np.int32))

            seq_counter += 1   # one test sequence fully processed

    auc_by_step = {}
    pr_curves   = {}
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

    npz_payload = {}
    for t, (prec, rec) in pr_curves.items():
        npz_payload[f"prec_{t}"] = prec
        npz_payload[f"rec_{t}"]  = rec
        npz_payload[f"auc_{t}"]  = np.array(auc_by_step.get(t, float("nan")))
        try:
            all_prob = np.concatenate(pr_probs[t]).astype(np.float32)
            all_lbl  = np.concatenate(pr_labels[t]).astype(np.int32)
            frac_pos, mean_pred = _cal_curve(all_lbl, all_prob,
                                             n_bins=10, strategy="uniform")
            npz_payload[f"cal_mean_{t}"] = mean_pred.astype(np.float32)
            npz_payload[f"cal_frac_{t}"] = frac_pos.astype(np.float32)
        except Exception as e:
            logger.warning(f"Calibration failed at step {t}: {e}")
        if t in pr_probs and pr_probs[t]:
            npz_payload[f"pr_prob_{t}"]  = np.concatenate(pr_probs[t]).astype(np.float32)
            npz_payload[f"pr_label_{t}"] = np.concatenate(pr_labels[t]).astype(np.int32)
            npz_payload[f"pr_seqid_{t}"] = np.concatenate(pr_seqids[t]).astype(np.int32)
    npz_payload["pr_steps"] = np.array(list(pr_curves.keys()))
    npz_payload["dt_min"]   = np.array(dt_min)
    npz_payload["n_sequences"] = np.array(seq_counter)
    tag = f"optical_flow_{args.flow_source}"
    pr_npz = os.path.join(args.output_dir, f"{tag}_pr_curves.npz")
    np.savez_compressed(pr_npz, **npz_payload)
    logger.info(f"PR + calibration curves -> {pr_npz}")

    _mean = lambda lst: float(np.mean(lst)) if lst else float("nan")

    per_step = []
    for t in range(T_out):
        row = {
            "lead_min":     lead_times[t],
            "crps":         _mean(crps_by_step[t]),
            "spread_skill": _mean(ss_by_step[t]),
            "ssim_ir":      _mean(ssim_by_step[t]) if HAS_SKIMAGE else float("nan"),
            "rmse_ir":      _mean(rmse_by_step[t]),
        }

        if skill_by_step[t]:
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

    csv_path = os.path.join(args.output_dir, f"{tag}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_step[0].keys())
        writer.writeheader()
        writer.writerows(per_step)
    logger.info(f"Saved -> {csv_path}")

    print(f"\n=== Optical-Flow Baseline (flow_source={args.flow_source}) — Summary ===")
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
        description="Optical-flow (pySTEPS-style) extrapolation baseline."
    )
    p.add_argument("--config",     default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--test_roots", nargs="+", default=None)
    p.add_argument("--output_dir", default="outputs/optical_flow_baseline")
    p.add_argument("--flow_source", choices=["li", "ir"], default="li",
                   help="Channel to derive the motion field from. 'li' is "
                        "the standard convention (flow-source == forecast "
                        "target). 'ir' is a robustness variant using a "
                        "denser field for motion estimation.")
    p.add_argument("--img_size",   nargs=2, type=int, default=[256, 256])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--li_event_threshold", type=float, default=5.0/255.0)
    p.add_argument("--fss_prob_thresholds", nargs="+", type=float,
                   default=[0.1, 0.3, 0.5])
    args = p.parse_args()

    if args.config is not None:
        import pathlib
        if not pathlib.Path(args.config).exists():
            p.error(f"Config not found: {args.config}")
        try:
            import yaml
        except ImportError:
            p.error("PyYAML required: pip install pyyaml")
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
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
    run_optical_flow_evaluation(args)
