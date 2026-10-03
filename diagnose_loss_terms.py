"""
Break the diffusion training loss into its terms for ONE batch at FIXED noise
levels, with the exact model, data and loss settings train.py uses.

Why: a smoke test showed a training loss of ~8.6e4 (a correctly scaled EDM
loss starts around 1). Every term scales with the network output, so the
breakdown tells which term (or which noise level) is responsible.

Usage (same config as training; untrained model unless --checkpoint):
  python diagnose_loss_terms.py --config configs/default.yaml --max_samples 50
"""
import sys

import torch

from config import parse_args
from dataset import make_dataloaders
from model import (EDMSchedule, asymmetric_li_loss, channel_weighted_mse, effective_li_weight,
                   neighbourhood_li_loss, spectral_loss)
from train import build_model


def main():
    ckpt_path = None
    if "--checkpoint" in sys.argv:
        i = sys.argv.index("--checkpoint")
        ckpt_path = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, stats = make_dataloaders(
        train_roots=args.train_roots, channel_list=args.channels, T_in=args.T_in, T_out=args.T_out,
        img_size=tuple(args.img_size), batch_size=args.batch_size, num_workers=0,
        max_samples=args.max_samples, binary_li_ctx=args.binary_li_ctx, ctx_channels=args.ctx_channels,
        augment_flip=False)
    channels = args.channels
    C = len(channels)
    li_idx = channels.index("li")
    model = build_model(C, args.T_in, args.T_out, args.dt_min, args).to(device).eval()
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck.get("ema") or ck["model"])
    schedule = EDMSchedule(P_mean=args.P_mean, P_std=args.P_std, sigma_min=args.sigma_min,
                           sigma_max=args.sigma_max, sigma_data=args.sigma_data)
    m, s = stats["li"]["mean"], stats["li"]["std"]
    thr = (args.li_event_threshold ** (1 / 3) - m) / s

    batch = next(iter(train_loader))
    context = batch["context"].to(device)
    target = batch["target"].to(device)
    tgt_mask = batch["tgt_mask"].to(device)
    last_ctx = batch["last_ctx"].to(device)
    B = target.shape[0]
    lead_idx = torch.zeros(B, dtype=torch.long, device=device)
    y = target[:, 0]
    ch_mask = tgt_mask[:, 0]
    dyn_w = effective_li_weight(batch["li_density"].to(device), base_weight=args.li_weight,
                                beta=args.li_weight_beta, ref_density=getattr(args, "li_weight_ref_density", 0.05))

    print(f"stats li: mean {m:.4f} std {s:.4f} | event threshold (normalised) {thr:.3f}")
    print(f"target residual y: ir std {y[:, 0].std():.3f}, li std {y[:, li_idx].std():.3f}, "
          f"li |max| {y[:, li_idx].abs().max():.2f} | context |max| {context.abs().max():.2f}")
    print(f"dynamic LI weight per sample: min {dyn_w.min():.1f} max {dyn_w.max():.1f}")
    print(f"\n{'sigma':>7} {'lambda':>10} {'|D-y| rms':>10} {'L_denoise':>11} {'L_asym':>10} "
          f"{'L_nbr':>10} {'L_spec':>10} {'weighted total':>15}")
    with torch.no_grad():
        for sig in [0.002, 0.01, 0.1, 0.5, 2.0, 10.0, 80.0]:
            sigma = torch.full((B,), sig, device=device)
            x = y + torch.randn_like(y) * sig
            pred = model(x, sigma, context, ch_mask, lead_idx)
            lw = schedule.edm_loss_weight(sigma)[:, None, None, None]
            Ld = channel_weighted_mse(pred * lw.sqrt(), y * lw.sqrt(), ch_mask, dyn_w).item()
            La = asymmetric_li_loss(pred, y, last_ctx, alpha=args.asym_alpha, li_idx=li_idx,
                                    norm_threshold=thr).item()
            Ln = neighbourhood_li_loss(pred, y, li_idx=li_idx, scales=args.nbr_scales).item()
            Ls = spectral_loss(pred, y, ch_mask, li_idx=li_idx).item()
            tot = Ld + args.asym_weight * La + args.nbr_weight * Ln + args.spectral_weight * Ls
            print(f"{sig:7.3f} {lw[0].item():10.1f} {(pred - y).pow(2).mean().sqrt().item():10.3f} "
                  f"{Ld:11.3f} {La:10.3f} {Ln:10.3f} {Ls:10.3f} {tot:15.3f}")
    print("\nTraining samples sigma ~ lognormal(P_mean, P_std); rows show which noise levels and terms dominate.")


if __name__ == "__main__":
    main()
