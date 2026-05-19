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
    # Use CPU for diagnostic to avoid CUDA OOM — only 3 forward passes needed
    device = torch.device("cpu")
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
    p.add_argument("--diag_cfg",      action="store_true",
                   help="CFG conditioning diagnostic: run inference at cfg_scale=0 "
                        "(unconditional) and cfg_scale=5 (strong conditioning) and "
                        "compare outputs. Reveals whether the model is actually using "
                        "context. If both outputs look identical, context is ignored.")
    p.add_argument("--S_churn",       type=float, default=40.0)
    p.add_argument("--num_steps",     type=int,   default=20)
    return p.parse_args()


def run_cfg_diagnostic(args):
    """
    CFG conditioning diagnostic.

    Runs the model at cfg_scale=0 (unconditional — context zeroed out)
    and cfg_scale=5 (strong conditioning) on the same sequence and
    computes the pixel-level difference between the two predictions.

    A well-conditioned model should produce significantly different outputs:
    - cfg_scale=0: predicts the marginal distribution (climatology)
    - cfg_scale=5: predicts sharpened, context-specific forecast

    If the two are nearly identical (mean absolute difference < 0.01
    in normalised space), the model is not using context effectively.
    """
    import torch, logging
    import numpy as np
    import matplotlib.pyplot as plt
    from model import UNet, EDMPrecond, MultiStepDenoiser, EDMSchedule, edm_sampler
    from dataset import METSATDataset

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.checkpoint, map_location=device)
    ca     = ckpt["args"]

    channels = ca["channels"]
    C        = len(channels)
    T_in     = ca["T_in"]
    T_out    = ca["T_out"]

    binary_li_ctx = ca.get("binary_li_ctx", False)
    C_ctx  = C + 1 if binary_li_ctx else C
    in_ch  = C + T_in * C_ctx + C

    unet    = UNet(in_ch, C, ca["base_channels"], tuple(ca["channel_mults"]),
                   ca["num_res_blocks"], tuple(ca["attn_resolutions"]),
                   dropout=0.0, emb_dim=ca["emb_dim"])
    precond = EDMPrecond(unet, sigma_data=ca.get("sigma_data", 1.0))
    model   = MultiStepDenoiser(precond, T_out, ca.get("dt_min", 10))

    state = ckpt.get("ema", ckpt["model"])
    model.load_state_dict(state)
    model.to(device).eval()

    schedule = EDMSchedule(sigma_data=ca.get("sigma_data", 1.0))

    # max_samples=10: only index 10 sequences — avoid scanning the full dataset
    # which is slow and unnecessary for a diagnostic on a single sequence.
    ds = METSATDataset(
        root           = args.data_dir,
        channel_list   = channels,
        T_in           = T_in,
        T_out          = T_out,
        img_size       = tuple(args.img_size),
        augment        = False,
        binary_li_ctx  = binary_li_ctx,
        max_samples    = 10,
    )

    # Use sequence index 0
    sample   = ds[0]
    ctx      = sample["context"].unsqueeze(0).to(device)   # (1, T_in, C_ctx, H, W)
    ch_mask  = sample["tgt_mask"][0].unsqueeze(0).to(device)  # (1, C)
    lead_idx = torch.zeros(1, dtype=torch.long, device=device)

    diffs = []
    n_trials = 3   # 3 trials is enough for the diagnostic

    logger.info("Running CFG diagnostic (%d trials)...", n_trials)
    for trial in range(n_trials):
        preds = {}
        for scale in [0.0, 5.0]:
            def denoiser_fn(x, sigma, _scale=scale, _ctx=ctx):
                cond   = model(x, sigma, _ctx, ch_mask, lead_idx)
                if _scale == 0.0:
                    return cond
                ctx_null = torch.zeros_like(_ctx)
                uncond = model(x, sigma, ctx_null, ch_mask, lead_idx)
                return uncond + _scale * (cond - uncond)

            pred = edm_sampler(
                denoiser_fn, (1, C, ctx.shape[-2], ctx.shape[-1]), device,
                num_steps = args.num_steps,
                sigma_min = schedule.sigma_data * 0.01,
                sigma_max = 80.0,
                S_churn   = args.S_churn,
            )
            preds[scale] = pred.cpu().numpy()

        diff = np.abs(preds[0.0] - preds[5.0]).mean()
        diffs.append(diff)
        logger.info("  Trial %d: mean |uncond - cond| = %.4f", trial+1, diff)

    mean_diff = np.mean(diffs)
    logger.info("")
    logger.info("=== CFG Diagnostic Result ===")
    logger.info("Mean |unconditional - conditional| = %.4f", mean_diff)
    if mean_diff < 0.01:
        logger.warning("DIAGNOSIS: Model is largely IGNORING context.")
        logger.warning("  cfg_scale=0 and cfg_scale=5 produce nearly identical outputs.")
        logger.warning("  The model has learned the marginal distribution, not p(future|context).")
    elif mean_diff < 0.05:
        logger.info("DIAGNOSIS: Weak context conditioning.")
        logger.info("  Some difference between unconditional and conditional predictions,")
        logger.info("  but context influence is limited.")
    else:
        logger.info("DIAGNOSIS: Context conditioning is working (diff=%.4f).", mean_diff)
        logger.info("  Model output changes significantly with context.")

    # Save visual comparison for the IR channel
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    ch_names = channels
    ir_idx   = ch_names.index("ir") if "ir" in ch_names else 0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"CFG Diagnostic — mean diff={mean_diff:.4f}", fontsize=11)
    axes[0].imshow(preds[0.0][0, ir_idx], cmap="gray")
    axes[0].set_title("cfg_scale=0 (unconditional)")
    axes[1].imshow(preds[5.0][0, ir_idx], cmap="gray")
    axes[1].set_title("cfg_scale=5 (conditioned)")
    diff_img = np.abs(preds[0.0][0, ir_idx] - preds[5.0][0, ir_idx])
    axes[2].imshow(diff_img, cmap="hot")
    axes[2].set_title(f"|diff| IR — mean={diff_img.mean():.4f}")
    for ax in axes: ax.axis("off")
    out_path = os.path.join(args.output_dir, "cfg_diagnostic.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved diagnostic plot → %s", out_path)


if __name__ == "__main__":
    args = parse_args()
    if args.diag_cfg:
        run_cfg_diagnostic(args)
    else:
        run_inference(args)
