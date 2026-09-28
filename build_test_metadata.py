"""
Build per-pixel / per-sequence metadata for the test set, aligned 1:1 with
the PR samples every evaluation script already saves (pr_prob_t, pr_label_t,
pr_seqid_t in plot_data.npz / *_pr_curves.npz).

Why this works without re-running any model: all evaluation scripts
(evaluate.py, persistence_baseline.py, optical_flow_baseline.py,
evaluate_cnn.py, evaluate_lightgbm.py) iterate the SAME deterministic test
loader (make_test_loader: shuffle=False, augment=False, non-overlapping
stride), label pixels with the SAME rule (physical LI >= li_event_threshold)
and subsample pixels with the SAME stride (flat[::max(1, H*W // 4096)]).
So the k-th saved sample of lead t in any of those files refers to the same
(sequence, pixel). This script recomputes the labels independently and
analysis_regimes.py ASSERTS they match each model file exactly -- if the
alignment assumption were ever wrong, it fails loudly instead of silently
mixing up pixels.

Model-free: needs the test data and normalisation stats only (stats/args are
read from the diffusion checkpoint, exactly as evaluate.py does), no GPU.

Per strided pixel it stores, for each sequence:
  dist_recent_px : Chebyshev distance (pixels) to the nearest LI-active pixel
                   in the last --recent_frames context frames (default 3 =
                   30 min at 10-min cadence). 9999 if none.
  dist_all_px    : same over the full context window (T_in frames = 6 h).
Pixels far from any recent activity that become active at the target lead
are lightning INITIATION; pixels near recent activity are CONTINUATION.
The radius threshold is applied later (analysis_regimes.py --radius_km), so
it can be varied without rerunning this script.

Per sequence it stores: region index/name and the context-end time (UTC),
for diurnal and per-region breakdowns.

With --train_roots (or train_roots in --config) it also prints each
region's train/test date ranges, confirming the chronological split.

Usage:
  python build_test_metadata.py --config configs/evaluate.yaml \
      --checkpoint outputs/nature_256_T36_ir_li_only/best.pt \
      --output outputs/test_metadata.npz
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_cdt
from tqdm import tqdm

from config import load_yaml
from dataset import build_index, make_test_loader
from evaluate import _li_to_physical

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NO_ACTIVITY = 9999


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", required=True,
                   help="Diffusion checkpoint -- used ONLY for stats and data "
                        "args (T_in/T_out/channels), so the loader matches "
                        "evaluate.py exactly. No model is built.")
    p.add_argument("--test_roots", nargs="+", default=None)
    p.add_argument("--train_roots", nargs="+", default=None)
    p.add_argument("--output", default="outputs/test_metadata.npz")
    p.add_argument("--img_size", nargs=2, type=int, default=[256, 256])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--li_event_threshold", type=float, default=5.0 / 255.0)
    p.add_argument("--recent_frames", type=int, default=3,
                   help="Context frames defining 'recent' activity (3 = 30 min).")
    args = p.parse_args()
    if args.config is not None:
        cfg = load_yaml(args.config)
        for k, v in cfg.items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)
    if args.test_roots is None:
        p.error("--test_roots is required (CLI or --config)")
    return args


def nearest_activity_distance(active):
    """Chebyshev distance (pixels) from each pixel to the nearest True pixel."""
    if not active.any():
        return np.full(active.shape, NO_ACTIVITY, dtype=np.int32)
    return distance_transform_cdt(~active, metric="chessboard").astype(np.int32)


def report_split(train_roots, test_roots):
    logger.info("Train/test date ranges per region (chronological split check):")
    for tr, te in zip(train_roots or [], test_roots):
        try:
            tr_t = sorted(build_index(tr).keys())
            te_t = sorted(build_index(te).keys())
            gap = te_t[0] - tr_t[-1]
            logger.info(f"  {Path(te).name}: train {tr_t[0]} -> {tr_t[-1]} | "
                        f"test {te_t[0]} -> {te_t[-1]} | gap {gap}")
            if te_t[0] <= tr_t[-1]:
                logger.warning(f"  !! {Path(te).name}: test period OVERLAPS training")
        except Exception as e:  # report only, never block the run
            logger.warning(f"  could not index {tr} / {te}: {e}")


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ca = ckpt["args"]
    stats = ckpt["stats"]
    channels = ckpt.get("channels", ca.get("channels"))
    ctx_channels = ca.get("ctx_channels", None)
    ctx_chs = ctx_channels if ctx_channels else channels
    T_in, T_out = ca["T_in"], ca["T_out"]
    li_idx = channels.index("li")
    li_ctx_idx = ctx_chs.index("li")
    thr = args.li_event_threshold
    K = min(args.recent_frames, T_in)

    report_split(args.train_roots, args.test_roots)

    loader = make_test_loader(
        test_roots=args.test_roots, channel_list=channels, stats=stats,
        T_in=T_in, T_out=T_out, img_size=tuple(args.img_size),
        batch_size=args.batch_size, num_workers=args.num_workers,
        binary_li_ctx=ca.get("binary_li_ctx", False), ctx_channels=ctx_channels,
    )
    ds = loader.dataset
    region_names = [Path(r).name for r in args.test_roots]

    labels = {t: [] for t in range(T_out)}
    dist_recent, dist_all, seqids = [], [], []
    seq_region, seq_time = [], []
    seq = 0
    for batch in tqdm(loader, desc="metadata", dynamic_ncols=True):
        context = batch["context"].numpy()
        target = batch["target"].numpy()
        last_ctx = batch["last_ctx"].numpy()
        for b in range(context.shape[0]):
            ds_idx = int(np.searchsorted(ds.cumlen[1:], seq, side="right"))
            local = seq - int(ds.cumlen[ds_idx])
            times = ds.datasets[ds_idx].valid_sequences[local]
            seq_region.append(ds_idx)
            seq_time.append(times[T_in - 1].strftime("%Y-%m-%dT%H:%M:%S"))

            li_ctx = _li_to_physical(context[b, :, li_ctx_idx], stats)  # (T_in,H,W)
            active = li_ctx >= thr
            d_rec = nearest_activity_distance(active[-K:].any(axis=0))
            d_all = nearest_activity_distance(active.any(axis=0))

            n_pix = d_rec.size
            stride = max(1, n_pix // 4096)
            dist_recent.append(d_rec.ravel()[::stride])
            dist_all.append(d_all.ravel()[::stride])
            seqids.append(np.full(dist_recent[-1].shape, seq, dtype=np.int32))
            for t in range(T_out):
                obs = _li_to_physical(target[b, t, li_idx] + last_ctx[b, li_idx], stats)
                labels[t].append((obs >= thr).ravel()[::stride].astype(np.int8))
            seq += 1

    payload = {
        "n_sequences": np.array(seq), "T_out": np.array(T_out),
        "recent_frames": np.array(K), "li_event_threshold": np.array(thr),
        "region_names": np.array(region_names),
        "seq_region": np.array(seq_region, dtype=np.int16),
        "seq_time_utc": np.array(seq_time),
        "dist_recent_px": np.concatenate(dist_recent),
        "dist_all_px": np.concatenate(dist_all),
        "seqid": np.concatenate(seqids),
    }
    for t in range(T_out):
        payload[f"label_{t}"] = np.concatenate(labels[t])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, **payload)
    logger.info(f"Saved {seq} sequences, {payload['seqid'].size} pixels/lead -> {args.output}")


if __name__ == "__main__":
    main()
