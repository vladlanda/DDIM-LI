"""
Inference script: run a 6-hour ensemble forecast from a context window.

Usage:
    python infer.py \
        --checkpoint outputs/run1/best.pt \
        --data_dir   /path/to/data/ \
        --output_dir outputs/forecast \
        --n_members  20 \
        --cfg_scale  1.5 \
        --plot
    python infer.py \
        --checkpoint outputs/h1_dim64/best.pt \
        --data_dir   /home/vladlanda/Workplace/LI-DATASETS/inference/central_africa_4/ \
        --output_dir outputs/forecast/h1_dim64 \
        --n_members  20 \
        --cfg_scale  1.5 \
        --plot

    Pass --use_gt to also load ground-truth future frames (if available)
    and include them in the plots.
"""

import argparse
import logging
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from dataset import METSATDataset, denormalize
from model   import UNet, EDMPrecond, MultiStepDenoiser
from evaluate import forecast, plot_forecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Model loader
# -------------------------------------------------------------------
def load_model(checkpoint: str, device: torch.device):
    ckpt     = torch.load(checkpoint, map_location=device)
    args     = ckpt["args"]
    channels = ckpt["channels"]
    stats    = ckpt["stats"]
    T_in     = args["T_in"]
    T_out    = args["T_out"]
    dt_min   = args["dt_min"]
    C        = len(channels)

    unet    = UNet(
        in_channels      = C + T_in * (C + 1 if ckpt_args.get("binary_li_ctx", False) else C) + C,
        out_channels     = C,
        base_channels    = args["base_channels"],
        channel_mults    = tuple(args["channel_mults"]),
        num_res_blocks   = args["num_res_blocks"],
        attn_resolutions = tuple(args["attn_resolutions"]),
        dropout          = 0.0,
        emb_dim          = args["emb_dim"],
    )
    precond = EDMPrecond(unet, sigma_data=args["sigma_data"])
    model   = MultiStepDenoiser(precond, T_out=T_out, dt_min=dt_min)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, channels, stats, T_in, T_out, dt_min


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, channels, stats, T_in, T_out, dt_min = load_model(args.checkpoint, device)
    logger.info(f"Loaded model: channels={channels}  T_in={T_in}  T_out={T_out}")

    # ----------------------------------------------------------------
    # Dataset:
    #   T_out=1 (dummy) when ground truth is not needed/available.
    #   T_out=T_out when --use_gt is set so future frames are loaded too.
    # ----------------------------------------------------------------
    effective_T_out = T_out if args.use_gt else 1
    ds = METSATDataset(
        root         = args.data_dir,
        channel_list = channels,
        T_in         = T_in,
        T_out        = effective_T_out,
        img_size     = tuple(args.img_size),
        stats        = stats,
        augment      = False,
    )
    logger.info(f"Dataset: {len(ds)} sequences  (use_gt={args.use_gt})")

    os.makedirs(args.output_dir, exist_ok=True)

    for sample_idx in range(min(args.n_forecasts, len(ds))):
        item    = ds[sample_idx]
        ctx_np  = item["context"].numpy()       # (T_in, C, H, W)  normalised
        mask_np = item["ctx_mask"][0].numpy()   # (C,)

        # ------------------------------------------------------------------
        # Filename: last context frame timestamp
        # valid_sequences[sample_idx] is the full list of actual datetimes
        # ------------------------------------------------------------------
        seq         = ds.valid_sequences[sample_idx]   # List[datetime]
        last_ctx_ts = seq[T_in - 1]                    # last context timestep
        stem        = f"forecast_{last_ctx_ts.strftime('%Y%m%dT%H%M%SZ')}"

        logger.info(f"[{sample_idx+1}/{min(args.n_forecasts, len(ds))}] "
                    f"context ends at {last_ctx_ts}  →  {stem}")

        # ---- Generate ensemble ----
        ens_norm = forecast(
            model, ctx_np, mask_np, device,
            n_members = args.n_members,
            cfg_scale = args.cfg_scale,
        )   # (M, T_out, C, H, W)  normalised

        # ---- Denormalise ensemble ----
        ens_denorm = np.zeros_like(ens_norm)
        for ci, ch in enumerate(channels):
            ens_denorm[:, :, ci] = (
                denormalize(ens_norm[:, :, ci], stats, ch)
                if ch in stats else ens_norm[:, :, ci]
            )

        # ---- Denormalise context for plotting ----
        ctx_denorm = np.zeros_like(ctx_np)
        for ci, ch in enumerate(channels):
            ctx_denorm[:, ci] = (
                denormalize(ctx_np[:, ci], stats, ch)
                if ch in stats else ctx_np[:, ci]
            )

        # ---- Ground truth (optional) ----
        # item["target"] is a residual (target_abs - last_ctx), so we
        # reconstruct: abs = residual + last_ctx_norm, then denormalise.
        gt_denorm = None
        if args.use_gt and "target" in item:
            tgt_residual  = item["target"].numpy()        # (T_out, C, H, W) normalised residual
            last_ctx_norm = ctx_np[-1]                    # (C, H, W)
            tgt_norm_abs  = tgt_residual + last_ctx_norm  # (T_out, C, H, W)

            gt_denorm = np.zeros_like(tgt_norm_abs)
            for ci, ch in enumerate(channels):
                gt_denorm[:, ci] = (
                    denormalize(tgt_norm_abs[:, ci], stats, ch)
                    if ch in stats else tgt_norm_abs[:, ci]
                )

        # ---- Save .npz ----
        npz_path  = os.path.join(args.output_dir, f"{stem}.npz")
        save_dict = dict(
            ensemble      = ens_denorm.astype(np.float32),
            context       = ctx_denorm.astype(np.float32),
            channels      = np.array(channels),
            dt_min        = dt_min,
            last_ctx_time = str(last_ctx_ts),
        )
        if gt_denorm is not None:
            save_dict["ground_truth"] = gt_denorm.astype(np.float32)
        np.savez_compressed(npz_path, **save_dict)
        logger.info(f"  Saved: {npz_path}  ensemble={ens_denorm.shape}")

        # ---- Plot ----
        if args.plot:
            png_path = os.path.join(args.output_dir, f"{stem}.png")
            plot_forecast(
                context_np    = ctx_denorm,
                ens_np        = ens_denorm,
                channels      = channels,
                gt_np         = gt_denorm,
                steps_to_plot = args.steps_to_plot,
                save_path     = png_path,
            )

    logger.info("Done.")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True,
                   help="Path to best.pt or latest.pt")
    p.add_argument("--data_dir",      required=True,
                   help="Dataset root (same structure as training data)")
    p.add_argument("--output_dir",    default="outputs/inference")
    p.add_argument("--n_members",     type=int,   default=20)
    p.add_argument("--cfg_scale",     type=float, default=1.5)
    p.add_argument("--n_forecasts",   type=int,   default=10)
    p.add_argument("--img_size",      nargs=2, type=int, default=[256, 256])
    p.add_argument("--plot",          action="store_true",
                   help="Save forecast plots as PNG")
    p.add_argument("--use_gt",        action="store_true",
                   help="Load ground-truth future frames for plotting/eval")
    p.add_argument("--steps_to_plot", nargs="+", type=int,
                   default=None,
                   help="Forecast steps (0-indexed) to plot. Default: all steps.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
