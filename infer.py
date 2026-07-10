"""
Inference script: run ensemble forecast from a context window.

Usage:
    python infer.py \
        --checkpoint outputs/nonelbo_v1/best.pt \
        --data_dir   /path/to/data \
        --output_dir outputs/forecast \
        --n_members  20 --cfg_scale 1.5 --plot

    # CFG conditioning diagnostic (tests if model uses context)
    python infer.py \
        --checkpoint outputs/nonelbo_v1/best.pt \
        --data_dir   /path/to/data \
        --output_dir outputs/diag \
        --diag_cfg --gpu 0
"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from dataset import METSATDataset, denormalize
from model   import UNet, EDMPrecond, MultiStepDenoiser, EDMSchedule, edm_sampler, compute_in_ch
from evaluate import forecast, plot_forecast

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Model loader
# -------------------------------------------------------------------
def load_model(checkpoint: str, device: torch.device):
    ckpt      = torch.load(checkpoint, map_location=device)
    ckpt_args = ckpt["args"]
    channels  = ckpt["channels"]
    stats     = ckpt["stats"]
    C         = len(channels)
    T_in      = ckpt_args["T_in"]
    T_out     = ckpt_args["T_out"]
    dt_min    = ckpt_args["dt_min"]

    binary_li_ctx = ckpt_args.get("binary_li_ctx", False)
    ctx_channels  = ckpt_args.get("ctx_channels", None)
    in_ch = compute_in_ch(C, T_in, ctx_channels, binary_li_ctx)

    unet    = UNet(
        in_channels      = in_ch,
        out_channels     = C,
        base_channels    = ckpt_args["base_channels"],
        channel_mults    = tuple(ckpt_args["channel_mults"]),
        num_res_blocks   = ckpt_args["num_res_blocks"],
        attn_resolutions = tuple(ckpt_args["attn_resolutions"]),
        dropout          = 0.0,
        emb_dim          = ckpt_args["emb_dim"],
        img_size         = ckpt_args.get("img_size", [64, 64])[0],
    )
    precond = EDMPrecond(unet, sigma_data=ckpt_args.get("sigma_data", 1.0))
    model   = MultiStepDenoiser(precond, T_out=T_out, dt_min=dt_min)

    state = ckpt.get("ema", ckpt["model"])
    model.load_state_dict(state)
    model.to(device).eval()
    return model, channels, stats, ckpt_args, T_in, T_out, dt_min


# -------------------------------------------------------------------
# Main inference
# -------------------------------------------------------------------
def run_inference(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    logger.info(f"Device: {device}")

    model, channels, stats, ckpt_args, T_in, T_out, dt_min = \
        load_model(args.checkpoint, device)
    logger.info(f"Loaded: channels={channels}  T_in={T_in}  T_out={T_out}")

    binary_li_ctx = ckpt_args.get("binary_li_ctx", False)
    ctx_channels  = ckpt_args.get("ctx_channels", None)

    effective_T_out = T_out if args.use_gt else 1
    ds = METSATDataset(
        root          = args.data_dir,
        channel_list  = channels,
        T_in          = T_in,
        T_out         = effective_T_out,
        img_size      = tuple(args.img_size),
        stats         = stats,
        augment       = False,
        binary_li_ctx = binary_li_ctx,
        ctx_channels  = ctx_channels,
    )
    logger.info(f"Dataset: {len(ds)} sequences")
    os.makedirs(args.output_dir, exist_ok=True)

    for sample_idx in range(min(args.n_forecasts, len(ds))):
        item    = ds[sample_idx]
        ctx_np  = item["context"].numpy()
        mask_np = item["ctx_mask"][0].numpy()

        seq         = ds.valid_sequences[sample_idx]
        last_ctx_ts = seq[T_in - 1]
        stem        = f"forecast_{last_ctx_ts.strftime('%Y%m%dT%H%M%SZ')}"
        logger.info(f"[{sample_idx+1}] context ends {last_ctx_ts}")

        ens_norm = forecast(
            model, ctx_np, mask_np, device,
            n_members=args.n_members, cfg_scale=args.cfg_scale,
        )

        ens_denorm = np.zeros_like(ens_norm)
        ctx_denorm = np.zeros_like(ctx_np)
        for ci, ch in enumerate(channels):
            ens_denorm[:, :, ci] = denormalize(ens_norm[:, :, ci], stats, ch) \
                                   if ch in stats else ens_norm[:, :, ci]
            ctx_denorm[:, ci]    = denormalize(ctx_np[:, ci], stats, ch) \
                                   if ch in stats else ctx_np[:, ci]

        gt_denorm = None
        if args.use_gt and "target" in item:
            tgt_res  = item["target"].numpy()
            last_ctx = ctx_np[-1]
            tgt_abs  = tgt_res + last_ctx
            gt_denorm = np.zeros_like(tgt_abs)
            for ci, ch in enumerate(channels):
                gt_denorm[:, ci] = denormalize(tgt_abs[:, ci], stats, ch) \
                                   if ch in stats else tgt_abs[:, ci]

        npz_path = os.path.join(args.output_dir, f"{stem}.npz")
        np.savez_compressed(npz_path,
            ensemble=ens_denorm.astype(np.float32),
            context=ctx_denorm.astype(np.float32),
            channels=np.array(channels), dt_min=dt_min,
            last_ctx_time=str(last_ctx_ts),
            **({} if gt_denorm is None else {"ground_truth": gt_denorm.astype(np.float32)}))
        logger.info(f"  Saved {npz_path}")

        if args.plot:
            plot_forecast(
                context_np=ctx_denorm, ens_np=ens_denorm,
                channels=channels, gt_np=gt_denorm,
                steps_to_plot=args.steps_to_plot,
                save_path=os.path.join(args.output_dir, f"{stem}.png"),
            )
    logger.info("Done.")


# -------------------------------------------------------------------
# CFG conditioning diagnostic
# -------------------------------------------------------------------
def run_cfg_diagnostic(args):
    """
    Test whether the model is using context.

    Compares output at cfg_scale=0 (unconditional, context zeroed)
    vs cfg_scale=5 (strong conditioning) on the same input.
    If outputs are nearly identical the model ignores context.
    """
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    logger.info(f"CFG diagnostic on {device}")

    model, channels, stats, ckpt_args, T_in, T_out, dt_min = \
        load_model(args.checkpoint, device)

    binary_li_ctx = ckpt_args.get("binary_li_ctx", False)
    ctx_channels  = ckpt_args.get("ctx_channels", None)
    C             = len(channels)
    schedule      = EDMSchedule(sigma_data=ckpt_args.get("sigma_data", 1.0))

    ds = METSATDataset(
        root          = args.data_dir,
        channel_list  = channels,
        T_in          = T_in,
        T_out         = 1,
        img_size      = tuple(args.img_size),
        stats         = stats,
        augment       = False,
        binary_li_ctx = binary_li_ctx,
        ctx_channels  = ctx_channels,
        max_samples   = 10,
    )

    sample   = ds[0]
    ctx      = sample["context"].unsqueeze(0).to(device)       # (1, T_in, C_ctx, H, W)
    ch_mask  = sample["tgt_mask"][0].unsqueeze(0).to(device)   # (1, C)
    lead_idx = torch.zeros(1, dtype=torch.long, device=device)
    H, W     = ctx.shape[-2], ctx.shape[-1]

    diffs = []
    with torch.no_grad():
        for trial in range(3):
            results = {}
            for scale in [0.0, 5.0]:
                def denoiser_fn(x, sigma, _s=scale):
                    cond = model(x, sigma, ctx, ch_mask, lead_idx)
                    if _s == 0.0:
                        return cond
                    uncond = model(x, sigma, torch.zeros_like(ctx), ch_mask, lead_idx)
                    return uncond + _s * (cond - uncond)

                pred = edm_sampler(
                    denoiser_fn, (1, C, H, W), device,
                    num_steps=args.num_steps,
                    sigma_min=schedule.sigma_data * 0.01,
                    sigma_max=80.0,
                    S_churn=args.S_churn,
                )
                results[scale] = pred.detach().cpu().numpy()

            diff = float(np.abs(results[0.0] - results[5.0]).mean())
            diffs.append(diff)
            logger.info(f"  Trial {trial+1}: |uncond - cond| = {diff:.4f}")

    mean_diff = float(np.mean(diffs))
    logger.info("")
    logger.info(f"=== CFG Diagnostic: mean diff = {mean_diff:.4f} ===")
    if mean_diff < 0.01:
        logger.warning("DIAGNOSIS: Model IGNORES context (diff < 0.01)")
        logger.warning("  cfg_scale has no effect — model learned marginal distribution only.")
    elif mean_diff < 0.05:
        logger.warning(f"DIAGNOSIS: Weak context conditioning (diff = {mean_diff:.4f})")
    else:
        logger.info(f"DIAGNOSIS: Context conditioning works (diff = {mean_diff:.4f})")

    # Save visual comparison
    os.makedirs(args.output_dir, exist_ok=True)
    ir_idx = channels.index("ir") if "ir" in channels else 0

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"CFG Diagnostic  |  mean diff = {mean_diff:.4f}")
    axes[0].imshow(results[0.0][0, ir_idx], cmap="gray");  axes[0].set_title("cfg_scale=0 (unconditional)")
    axes[1].imshow(results[5.0][0, ir_idx], cmap="gray");  axes[1].set_title("cfg_scale=5 (conditioned)")
    diff_img = np.abs(results[0.0][0, ir_idx] - results[5.0][0, ir_idx])
    axes[2].imshow(diff_img, cmap="hot");                   axes[2].set_title(f"|diff|  mean={diff_img.mean():.4f}")
    for ax in axes: ax.axis("off")
    out = os.path.join(args.output_dir, "cfg_diagnostic.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--data_dir",      required=True)
    p.add_argument("--output_dir",    default="outputs/inference")
    p.add_argument("--gpu",           type=int,   default=0,
                   help="GPU index (use 1 if GPU 0 is occupied)")
    p.add_argument("--n_members",     type=int,   default=20)
    p.add_argument("--cfg_scale",     type=float, default=1.5)
    p.add_argument("--n_forecasts",   type=int,   default=10)
    p.add_argument("--img_size",      nargs=2, type=int, default=[64, 64])
    p.add_argument("--plot",          action="store_true")
    p.add_argument("--use_gt",        action="store_true")
    p.add_argument("--steps_to_plot", nargs="+", type=int, default=None)
    p.add_argument("--S_churn",       type=float, default=40.0)
    p.add_argument("--num_steps",     type=int,   default=20)
    p.add_argument("--diag_cfg",      action="store_true",
                   help="Run CFG conditioning diagnostic instead of inference")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.diag_cfg:
        run_cfg_diagnostic(args)
    else:
        run_inference(args)
