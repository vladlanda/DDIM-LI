"""
Derive lead-time churn scaling from the VALIDATION set (not the test set).

Reads ALL parameters from the training config (configs/default.yaml) so the
data pipeline, channels, and — critically — the train/val split are
IDENTICAL to what was used during training. The validation sequences here
are exactly the ones the model never trained on.

Scientific rationale
--------------------
Per-lead spread-skill (Fortin et al. 2014):
    SS(t) = ensemble_spread(t) / ensemble_mean_RMSE(t)
    SS<1 => UNDER-dispersed (overconfident).
To reach SS=1 the ensemble std must be inflated by 1/SS(t). Churn injects
variance (std ~ sqrt(var)), so we fit
    m(t) = 1 + churn_lead_scale * t/(T_out-1)   ≈   1/SS(t)
by least squares -> a SINGLE derived churn_lead_scale, measured on
validation data, never tuned on the test set.

Usage
-----
  python derive_churn_scale.py --config configs/default.yaml
  # optional overrides: --max_batches 40 --n_members 10
"""
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from config import load_yaml
from model import UNet, EDMPrecond, MultiStepDenoiser
from evaluate import generate_ensemble, spread_skill
from dataset import MultiRegionDataset


def build_model_from_ckpt(ckpt, device):
    a  = ckpt["args"]
    ch = ckpt["channels"]
    C  = len(ch)
    binary_li_ctx = a.get("binary_li_ctx", False)
    ctx_channels  = a.get("ctx_channels", None)
    C_ctx_sel = len(ctx_channels) if ctx_channels else C
    li_in_ctx = (ctx_channels is None) or ("li" in ctx_channels)
    C_ctx  = C_ctx_sel + 1 if (binary_li_ctx and li_in_ctx) else C_ctx_sel
    in_ch  = C + a["T_in"] * C_ctx + C
    unet = UNet(in_channels=in_ch, out_channels=C,
                base_channels=a["base_channels"],
                channel_mults=tuple(a["channel_mults"]),
                num_res_blocks=a["num_res_blocks"],
                attn_resolutions=tuple(a["attn_resolutions"]),
                dropout=0.0, emb_dim=a["emb_dim"],
                img_size=a.get("img_size", [256, 256])[0])
    precond = EDMPrecond(unet, sigma_data=a.get("sigma_data", 0.62))
    model   = MultiStepDenoiser(precond, T_out=a["T_out"], dt_min=a["dt_min"])
    model.load_state_dict(ckpt.get("ema") or ckpt["model"])
    model.to(device).eval()
    return model


def build_val_split(cfg, ckpt_args, channels, stats, device):
    """
    Reproduce make_dataloaders' train/val split EXACTLY:
      - same train_roots, T_in, T_out, img_size, channels
      - per region: n_train = max(1, int(n * train_val_split))
      - val = valid_sequences[n_train:]  (identical slice, no reorder)
    valid_sequences ordering is deterministic (sorted timestamps), so this
    reproduces the held-out validation sequences byte-for-byte.

    CRITICAL: train_val_split, T_in, T_out are taken from the CHECKPOINT
    (ckpt_args) — the ground truth of what was actually used at training —
    NOT the current config, which may have been edited after training.
    """
    train_roots     = cfg["train_roots"]
    T_in            = ckpt_args["T_in"]                       # from checkpoint
    T_out           = ckpt_args["T_out"]                      # from checkpoint
    img_size        = tuple(ckpt_args.get("img_size", [256, 256]))
    train_val_split = ckpt_args.get("train_val_split",        # from checkpoint
                                    cfg.get("train_val_split", 0.8))
    binary_li_ctx   = ckpt_args.get("binary_li_ctx", False)
    ctx_channels    = ckpt_args.get("ctx_channels", None)
    logger.info(f"Split params from CHECKPOINT: T_in={T_in}, T_out={T_out}, "
                f"train_val_split={train_val_split}")

    ds = MultiRegionDataset(
        train_roots, channel_list=channels,
        T_in=T_in, T_out=T_out, img_size=img_size,
        stats=stats, augment=False,
        binary_li_ctx=binary_li_ctx, ctx_channels=ctx_channels,
    )

    total_val = 0
    for sub in ds.datasets:
        all_seqs = sub.valid_sequences               # deterministic order
        n        = len(all_seqs)
        n_train  = max(1, int(n * train_val_split))  # EXACT training boundary
        sub.valid_sequences = all_seqs[n_train:]     # val = tail, no reorder
        total_val += len(sub.valid_sequences)
        logger.info(f"  {sub.root}: n={n}, n_train={n_train}, "
                    f"val={len(sub.valid_sequences)}")

    ds.lengths = [len(s.valid_sequences) for s in ds.datasets]
    ds.cumlen  = np.cumsum([0] + ds.lengths)
    logger.info(f"Total validation sequences (matches training split): {total_val}")
    return ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to configs/default.yaml")
    # Optional overrides for speed / sampler settings (default: read from config)
    p.add_argument("--checkpoint",  default=None)
    p.add_argument("--n_members",   type=int,   default=None)
    p.add_argument("--num_steps",   type=int,   default=None)
    p.add_argument("--cfg_scale",   type=float, default=None)
    p.add_argument("--S_churn",     type=float, default=None)
    p.add_argument("--S_noise",     type=float, default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--max_batches", type=int,   default=40,
                   help="Cap validation batches for a fast, stable estimate.")
    p.add_argument("--val_subsample", type=int, default=8,
                   help="Take every Nth validation sequence for speed "
                        "(sampling only — does not change the split boundary).")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint: default to <output_dir>/best.pt from the config
    ckpt_path = args.checkpoint or f"{cfg['output_dir']}/best.pt"
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt["args"]
    channels  = ckpt["channels"]
    stats     = ckpt["stats"]
    T_out     = ckpt_args["T_out"]
    dt_min    = ckpt_args["dt_min"]

    model = build_model_from_ckpt(ckpt, device)

    # Sampler params: CLI override > config value
    def pick(name, default):
        v = getattr(args, name)
        return v if v is not None else cfg.get(name, default)
    n_members  = pick("n_members", 10)
    num_steps  = pick("num_steps", 20)
    cfg_scale  = pick("cfg_scale", 1.5)
    S_churn    = pick("S_churn",   60.0)
    S_noise    = pick("S_noise",   1.003)
    batch_size = pick("batch_size", 4)

    val_ds = build_val_split(cfg, ckpt_args, channels, stats, device)

    # Optional subsample for speed (sampling only — split boundary unchanged)
    if args.val_subsample > 1:
        for sub in val_ds.datasets:
            sub.valid_sequences = sub.valid_sequences[::args.val_subsample]
        val_ds.lengths = [len(s.valid_sequences) for s in val_ds.datasets]
        val_ds.cumlen  = np.cumsum([0] + val_ds.lengths)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    ss_accum = [[] for _ in range(T_out)]
    for i, batch in enumerate(val_loader):
        context  = batch["context"].to(device)
        target   = batch["target"].to(device)
        ch_mask  = batch["tgt_mask"][:, 0].to(device)
        with torch.no_grad():
            ens = generate_ensemble(
                model, context, ch_mask, device,
                n_members=n_members, cfg_scale=cfg_scale,
                S_churn=S_churn, S_noise=S_noise, num_steps=num_steps,
                sigma_min=float(ckpt_args.get("sigma_min", 0.002)),
                sigma_max=float(ckpt_args.get("sigma_max", 80.0)),
            )
        ens_np = ens.cpu().numpy(); tgt_np = target.cpu().numpy()
        for b in range(ens_np.shape[0]):
            for t in range(T_out):
                ss_accum[t].append(spread_skill(ens_np[b, :, t], tgt_np[b, t]))
        if i + 1 >= args.max_batches:
            break

    ss = np.array([np.nanmean(v) if v else np.nan for v in ss_accum])
    t_frac  = np.arange(T_out) / max(T_out - 1, 1)
    deficit = 1.0 / ss
    mask = np.isfinite(deficit)
    scale = float(np.sum((deficit[mask] - 1.0) * t_frac[mask]) /
                  max(np.sum(t_frac[mask] ** 2), 1e-9))

    print("\n" + "=" * 60)
    print("VALIDATION spread-skill -> derived churn scaling")
    print("=" * 60)
    print(f"{'Lead':>6}  {'spread_skill':>12}  {'1/SS':>8}")
    print("-" * 34)
    for t in range(T_out):
        print(f"  +{(t+1)*dt_min:3d}m  {ss[t]:>12.3f}  {deficit[t]:>8.3f}")
    print("-" * 34)
    print(f"\n  Derived churn_lead_scale = {scale:.3f}")
    print(f"  (from validation spread-skill deficit; test set untouched)\n")
    print(f"  Apply once to the test set:")
    print(f"    python evaluate.py --config configs/evaluate.yaml "
          f"--churn_lead_scale {scale:.3f}")


if __name__ == "__main__":
    main()
