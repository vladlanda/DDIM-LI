"""
infer_ar.py — Single-sequence autoregressive forecast from a trained AR model.

Usage:
    python infer_ar.py \\
        --checkpoint outputs/run_ar/best.pt \\
        --data_dir   /path/to/data/ \\
        --output_dir outputs/forecast_ar \\
        --T_ar 36 --n_members 20 --plot
"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch

from dataset import METSATDataset, denormalize
from model_ar import ARDenoiser, ARPrecond
from model    import UNet
from evaluate import plot_forecast
from evaluate_ar import generate_ar_ensemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ar_model(checkpoint: str, device: torch.device):
    from evaluate_ar import _load_ar_model
    return _load_ar_model(checkpoint, device)


def run_ar_inference(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, channels, stats, T_in, T_out, dt_min = load_ar_model(args.checkpoint, device)
    T_ar = args.T_ar
    C    = len(channels)

    logger.info(f"AR model loaded: T_in={T_in}  T_ar={T_ar}  channels={channels}")

    # ---- Load context window ----
    ds = METSATDataset(
        root         = args.data_dir,
        channel_list = channels,
        T_in         = T_in,
        T_out        = T_out,
        img_size     = tuple(args.img_size),
        stats        = stats,
    )
    if len(ds) == 0:
        raise RuntimeError(f"No valid sequences found in {args.data_dir}")

    idx     = args.sequence_idx
    sample  = ds[idx]
    logger.info(f"Using sequence index {idx} / {len(ds) - 1}")

    context_abs = sample["context"].unsqueeze(0).to(device)   # (1, T_in, C, H, W)
    ch_mask     = sample["tgt_mask"][0].unsqueeze(0).to(device)  # (1, C)
    last_ctx    = sample["last_ctx"].unsqueeze(0)              # (1, C, H, W)

    # Ground truth absolute (for plotting)
    target_res = sample["target"].unsqueeze(0)                 # (1, T_out, C, H, W)
    gt_abs     = (target_res + last_ctx.unsqueeze(1))[:, :T_ar, 0]  # (T_ar, C, H, W)

    # ---- Generate ensemble ----
    logger.info(f"Generating {args.n_members} AR members × {T_ar} steps ...")
    ens = generate_ar_ensemble(
        model, context_abs, ch_mask, device,
        T_ar=T_ar, n_members=args.n_members,
        cfg_scale=args.cfg_scale,
    )  # (1, M, T_ar, C, H, W)
    ens = ens[0]   # (M, T_ar, C, H, W)

    # ---- Denormalise ----
    def _denorm(arr, ch_idx, ch_name):
        if ch_name in stats:
            return denormalize(arr, stats, ch_name)
        return arr

    ctx_np  = sample["context"].numpy()             # (T_in, C, H, W)
    ens_np  = ens.cpu().numpy()                     # (M, T_ar, C, H, W)
    gt_np   = gt_abs.numpy()                        # (T_ar, C, H, W)

    ctx_den = np.zeros_like(ctx_np)
    ens_den = np.zeros((args.n_members, T_ar, C) + ctx_np.shape[-2:])
    gt_den  = np.zeros_like(gt_np)

    for ci, ch in enumerate(channels):
        fn = (lambda x: denormalize(x, stats, ch)) if ch in stats else (lambda x: x)
        ctx_den[:, ci]    = fn(ctx_np[:, ci])
        gt_den[:, ci]     = fn(gt_np[:, ci])
        for m in range(args.n_members):
            ens_den[m, :, ci] = fn(ens_np[m, :, ci])

    # ---- Save / plot ----
    os.makedirs(args.output_dir, exist_ok=True)

    if args.plot:
        save_path = os.path.join(args.output_dir, f"forecast_ar_seq{idx:04d}.png")
        plot_forecast(
            context_np = ctx_den,
            ens_np     = ens_den,
            channels   = channels,
            gt_np      = gt_den if args.use_gt else None,
            save_path  = save_path,
        )
        logger.info(f"Forecast plot -> {save_path}")

    # Save ensemble numpy arrays
    out_path = os.path.join(args.output_dir, f"ensemble_ar_seq{idx:04d}.npz")
    np.savez_compressed(
        out_path,
        ensemble    = ens_den,
        context     = ctx_den,
        ground_truth = gt_den,
        channels    = np.array(channels),
        dt_min      = dt_min,
    )
    logger.info(f"Ensemble saved -> {out_path}")
    logger.info(f"  Shape: {ens_den.shape}  (members × T_ar × C × H × W)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AR single-sequence inference for METSAT.")
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--data_dir",      required=True,
                   help="Directory containing satellite data for the context window")
    p.add_argument("--output_dir",    default="outputs/forecast_ar")
    p.add_argument("--T_ar",          type=int,   default=36,
                   help="Number of AR rollout steps")
    p.add_argument("--n_members",     type=int,   default=20)
    p.add_argument("--cfg_scale",     type=float, default=1.5)
    p.add_argument("--sequence_idx",  type=int,   default=0,
                   help="Which sequence in the dataset to forecast")
    p.add_argument("--img_size",      nargs=2, type=int, default=[256, 256])
    p.add_argument("--gpu",           type=int,   default=0)
    p.add_argument("--plot",          action="store_true")
    p.add_argument("--use_gt",        action="store_true",
                   help="Include ground-truth future frames in the plot")
    args = p.parse_args()
    run_ar_inference(args)
