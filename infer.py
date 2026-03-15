"""
Inference script: run a 6-hour ensemble forecast from a context window.

Usage:
    python infer.py \
        --checkpoint outputs/run1/best.pt \
        --context_dir /path/to/context_frames/ \
        --output_dir  outputs/forecast_20250130 \
        --n_members 20 \
        --cfg_scale 1.5
"""

import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch

from dataset import METSATDataset, denormalize
from model import UNet, EDMPrecond, MultiStepDenoiser
from evaluate import forecast, plot_forecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint: str, device: torch.device) -> tuple:
    ckpt     = torch.load(checkpoint, map_location=device)
    args     = ckpt["args"]
    channels = ckpt["channels"]
    stats    = ckpt["stats"]
    C        = len(channels)
    T_in     = args["T_in"]
    T_out    = args["T_out"]
    dt_min   = args["dt_min"]

    in_ch  = C * (T_in + 2)
    unet   = UNet(
        in_channels      = in_ch,
        out_channels     = C,
        base_channels    = args["base_channels"],
        channel_mults    = tuple(args["channel_mults"]),
        num_res_blocks   = args["num_res_blocks"],
        attn_resolutions = tuple(args["attn_resolutions"]),
        dropout          = 0.0,   # no dropout at inference
        emb_dim          = args["emb_dim"],
    )
    precond = EDMPrecond(unet, sigma_data=args["sigma_data"])
    model   = MultiStepDenoiser(precond, T_out=T_out, dt_min=dt_min)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    return model, channels, stats, T_in, T_out, dt_min


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, channels, stats, T_in, T_out, dt_min = load_model(args.checkpoint, device)
    logger.info(f"Loaded model: {len(channels)} channels, T_in={T_in}, T_out={T_out}")

    # Build a minimal dataset just to load and normalise context frames
    ds = METSATDataset(
        root         = args.context_dir,
        channel_list = channels,
        T_in         = T_in,
        T_out        = 1,         # dummy, we just want context loading
        img_size     = tuple(args.img_size),
        stats        = stats,
        augment      = False,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    C = len(channels)

    for sample_idx in range(min(args.n_forecasts, len(ds))):
        item       = ds[sample_idx]
        ctx_np     = item["context"].numpy()     # (T_in, C, H, W)
        mask_np    = item["ctx_mask"][0].numpy() # (C,)  use first step mask

        logger.info(f"Running forecast {sample_idx+1}/{args.n_forecasts}")
        print(item.keys())

        # (M, T_out, C, H, W)
        ens_norm = forecast(
            model, ctx_np, mask_np, device,
            n_members = args.n_members,
            cfg_scale = args.cfg_scale,
        )

        # Denormalise
        ens_denorm = np.zeros_like(ens_norm)
        for ci, ch in enumerate(channels):
            if ch in stats:
                ens_denorm[:, :, ci] = denormalize(ens_norm[:, :, ci], stats, ch)
            else:
                ens_denorm[:, :, ci] = ens_norm[:, :, ci]

        # Save
        out_path = os.path.join(args.output_dir, f"forecast_{sample_idx:04d}.npz")
        np.savez_compressed(
            out_path,
            ensemble  = ens_denorm.astype(np.float32),
            channels  = np.array(channels),
            dt_min    = dt_min,
        )
        logger.info(f"  Saved: {out_path}  shape={ens_denorm.shape}")

        # Plot
        if args.plot:
            # We don't have ground truth at inference time, use ensemble mean as ref
            ens_mean = ens_denorm.mean(axis=0)
            plot_forecast(
                ens_denorm, ens_mean, channels,
                save_path=os.path.join(args.output_dir, f"forecast_{sample_idx:04d}.png"),
            )

    logger.info("Done.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--context_dir",  required=True)
    p.add_argument("--output_dir",   default="outputs/inference")
    p.add_argument("--n_members",    type=int,   default=20)
    p.add_argument("--cfg_scale",    type=float, default=1.5)
    p.add_argument("--n_forecasts",  type=int,   default=6)
    p.add_argument("--img_size",     nargs=2, type=int, default=[256, 256])
    p.add_argument("--plot",         action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
