"""
Derive lead-time churn scaling from the VALIDATION set (not the test set).

Scientific rationale
--------------------
Per-lead spread-skill (Fortin et al. 2014) measures ensemble dispersion:
    SS(t) = ensemble_spread(t) / ensemble_mean_RMSE(t)
    SS = 1  -> perfectly calibrated dispersion
    SS < 1  -> UNDER-dispersed (overconfident)

If the ensemble is under-dispersed, its spread must be inflated by 1/SS(t)
to reach calibration. Since stochastic churn injects variance and spread
scales as sqrt(variance), the churn multiplier needed to reach SS=1 at
lead step t is approximately:
    m(t) = 1 / SS(t)          (on the standard-deviation scale)

We fit a linear profile   m(t) = 1 + churn_lead_scale * t/(T_out-1)
to the measured 1/SS(t) deficit via least squares, giving a SINGLE
derived churn_lead_scale — NOT tuned on any performance metric, and
measured on VALIDATION data (temporal tail of the training roots), so
the final test set is never touched.

Usage
-----
  python derive_churn_scale.py \
      --checkpoint outputs/nonelbo_v6_256_T36_LIctx/best.pt \
      --train_roots  <the 4 split/train roots> \
      --train_val_split 0.8 \
      --n_members 10 --num_steps 20 --cfg_scale 1.5 \
      --S_churn 60 --max_batches 40
"""
import argparse
import logging
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Reuse the exact model-loading, ensemble, and metric code from evaluate.py
from model import UNet, EDMPrecond, MultiStepDenoiser
from evaluate import generate_ensemble, spread_skill
from dataset import MultiRegionDataset
from torch.utils.data import DataLoader


def load_model(checkpoint, device):
    ckpt      = torch.load(checkpoint, map_location=device)
    ckpt_args = ckpt["args"]
    channels  = ckpt["channels"]
    stats     = ckpt["stats"]
    C     = len(channels)
    T_in  = ckpt_args["T_in"]; T_out = ckpt_args["T_out"]; dt_min = ckpt_args["dt_min"]

    binary_li_ctx = ckpt_args.get("binary_li_ctx", False)
    ctx_channels  = ckpt_args.get("ctx_channels", None)
    C_ctx_sel     = len(ctx_channels) if ctx_channels else C
    _li_in_ctx    = (ctx_channels is None) or ("li" in ctx_channels)
    C_ctx  = C_ctx_sel + 1 if (binary_li_ctx and _li_in_ctx) else C_ctx_sel
    in_ch  = C + T_in * C_ctx + C

    unet = UNet(in_channels=in_ch, out_channels=C,
                base_channels=ckpt_args["base_channels"],
                channel_mults=tuple(ckpt_args["channel_mults"]),
                num_res_blocks=ckpt_args["num_res_blocks"],
                attn_resolutions=tuple(ckpt_args["attn_resolutions"]),
                dropout=0.0, emb_dim=ckpt_args["emb_dim"],
                img_size=ckpt_args.get("img_size", [256, 256])[0])
    precond = EDMPrecond(unet, sigma_data=ckpt_args.get("sigma_data", 0.62))
    model   = MultiStepDenoiser(precond, T_out=T_out, dt_min=dt_min)
    state   = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt_args, channels, stats, T_in, T_out, dt_min


def build_val_loader(train_roots, channels, stats, T_in, T_out, img_size,
                     batch_size, num_workers, train_val_split,
                     binary_li_ctx, ctx_channels):
    """Validation split = temporal tail of the training roots (never trained on)."""
    ds = MultiRegionDataset(train_roots, channel_list=channels,
                            T_in=T_in, T_out=T_out, img_size=img_size,
                            stats=stats, augment=False,
                            binary_li_ctx=binary_li_ctx, ctx_channels=ctx_channels)
    stride = T_in + T_out
    total = 0
    for sub in ds.datasets:
        seqs = sub.valid_sequences
        n_train = max(1, int(len(seqs) * train_val_split))
        sub.valid_sequences = seqs[n_train:][::stride]   # tail, non-overlapping
        total += len(sub.valid_sequences)
    ds.lengths = [len(s.valid_sequences) for s in ds.datasets]
    ds.cumlen  = np.cumsum([0] + ds.lengths)
    logger.info(f"Validation sequences (temporal tail): {total}")
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train_roots", nargs="+", required=True)
    p.add_argument("--train_val_split", type=float, default=0.8)
    p.add_argument("--n_members", type=int, default=10)
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--cfg_scale", type=float, default=1.5)
    p.add_argument("--S_churn",   type=float, default=60.0)
    p.add_argument("--S_noise",   type=float, default=1.003)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=40,
                   help="Cap validation batches for a fast, stable estimate.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_args, channels, stats, T_in, T_out, dt_min = load_model(
        args.checkpoint, device)

    img_size = tuple(ckpt_args.get("img_size", [256, 256]))
    val_loader = build_val_loader(
        args.train_roots, channels, stats, T_in, T_out, img_size,
        args.batch_size, args.num_workers, args.train_val_split,
        ckpt_args.get("binary_li_ctx", False),
        ckpt_args.get("ctx_channels", None))

    # Accumulate spread-skill per lead step over validation batches
    ss_accum = [[] for _ in range(T_out)]
    n_done = 0
    for batch in val_loader:
        context  = batch["context"].to(device)
        last_ctx = batch["last_ctx"].to(device)
        target   = batch["target"].to(device)          # residuals
        ch_mask  = batch["tgt_mask"][:, 0].to(device)

        with torch.no_grad():
            ens = generate_ensemble(
                model, context, ch_mask, device,
                n_members=args.n_members, cfg_scale=args.cfg_scale,
                S_churn=args.S_churn, S_noise=args.S_noise,
                num_steps=args.num_steps,
                sigma_min=float(ckpt_args.get("sigma_min", 0.002)),
                sigma_max=float(ckpt_args.get("sigma_max", 80.0)),
            )  # (B, M, T_out, C, H, W) residuals

        ens_np = ens.cpu().numpy()
        tgt_np = target.cpu().numpy()
        B = ens_np.shape[0]
        for b in range(B):
            for t in range(T_out):
                ss_accum[t].append(spread_skill(ens_np[b, :, t], tgt_np[b, t]))

        n_done += 1
        if n_done >= args.max_batches:
            break

    ss = np.array([np.nanmean(v) if v else np.nan for v in ss_accum])

    # Derive churn_lead_scale: fit m(t)=1+scale*t/(T_out-1) to 1/SS(t)
    t_frac = np.arange(T_out) / max(T_out - 1, 1)
    deficit = 1.0 / ss                      # required std inflation
    # least squares for scale in  deficit = 1 + scale * t_frac
    # -> scale = sum((deficit-1)*t_frac) / sum(t_frac^2)
    mask = np.isfinite(deficit)
    scale = float(np.sum((deficit[mask] - 1.0) * t_frac[mask]) /
                  max(np.sum(t_frac[mask] ** 2), 1e-9))

    print("\n" + "=" * 60)
    print("VALIDATION spread-skill and derived churn scaling")
    print("=" * 60)
    print(f"{'Lead':>6}  {'spread_skill':>12}  {'1/SS (deficit)':>14}")
    print("-" * 40)
    for t in range(T_out):
        lead = (t + 1) * dt_min
        print(f"  +{lead:3d}m  {ss[t]:>12.3f}  {deficit[t]:>14.3f}")
    print("-" * 40)
    print(f"\n  Derived churn_lead_scale = {scale:.3f}")
    print(f"  (from validation spread-skill deficit; NOT tuned on test set)")
    print(f"\n  Apply to test evaluation with:")
    print(f"    python evaluate.py --config configs/evaluate.yaml \\")
    print(f"        --churn_lead_scale {scale:.3f}")


if __name__ == "__main__":
    main()
