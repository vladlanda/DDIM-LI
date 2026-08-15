"""
Select and render example nowcast cases for the manuscript's Figure 4
(MANUSCRIPT_PLAN.md). Not a new inference pipeline -- reuses evaluate.py's
own checkpoint loading, generate_ensemble, and plot_forecast exactly
(same denormalization pattern as run_test_evaluation's --plot path), just
adds what was actually missing: CURATED case SELECTION rather than
plotting the first N sequences in test-set order.

evaluate.py --plot --max_plots N already renders this exact figure type
for the first N sequences -- if arbitrary/sequential examples are fine,
just use that directly and skip this script entirely. This script exists
because a manuscript figure should show *representative* cases (a clear
success, a genuinely uncertain/multimodal case, and honestly a failure
case -- see MANUSCRIPT_PLAN.md Figure 4), not whatever happens to sort
first.

Two-pass design, to avoid running expensive diffusion sampling on
candidates that get discarded:
  1. Cheap scan (no inference) over up to --n_scan sequences, ranking by
     ground-truth LI activity level (total positive pixels across the
     forecast horizon). Selects one low/near-zero, one moderate, and one
     high-activity candidate by default -- a reasonable, inspectable
     proxy for "boring", "typical", and "hard" cases without needing to
     run inference first. NOTE: this proxy cannot identify genuinely
     multimodal/uncertain cases (that requires seeing ensemble spread,
     which requires inference) -- if the paper specifically wants an
     uncertain-case panel, inspect a handful of --n_scan candidates'
     ensemble spread manually via --candidate_indices and iterate; this
     script's automatic selection is a reasonable default, not a
     guarantee of the most illustrative possible case.
  2. Full ensemble inference (generate_ensemble) + plotting (plot_forecast)
     on only the selected few candidates.

Outputs, per selected case:
  - A publication-quality PNG via plot_forecast directly (the primary
    deliverable -- same rendering already used by evaluate.py's own
    --plot path: context / ensemble-mean+spread / ground-truth rows).
  - An npz with the same case's raw arrays (context_ir, target_li,
    ensemble_li, all physical-space) for the notebook's Figure 4 cell,
    which just displays/re-confirms these rather than re-implementing
    the plotting logic.

Usage:
  python select_example_cases.py --config configs/evaluate.yaml \
      --checkpoint outputs/nature_256_T36_ir_li_only/best.pt \
      --output_dir manuscript/figures \
      --n_scan 60 --n_members 20
"""
import argparse
import logging
import os

import numpy as np
import torch

from config import load_yaml
from dataset import denormalize, make_test_loader
from evaluate import _li_to_physical, generate_ensemble, plot_forecast
from model import EDMPrecond, MultiStepDenoiser, UNet, compute_in_ch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_roots", nargs="+", default=None)
    p.add_argument("--output_dir", default="manuscript/figures")
    p.add_argument("--img_size", nargs=2, type=int, default=[256, 256])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--n_scan", type=int, default=60,
                   help="How many test sequences to cheaply scan (no "
                        "inference) before selecting cases. Larger = "
                        "better chance of finding good examples, but "
                        "linear in data-loading time only, not GPU time.")
    p.add_argument("--candidate_indices", nargs="+", type=int, default=None,
                   help="Skip automatic selection; render exactly these "
                        "sequence indices instead (0-indexed within the "
                        "n_scan scan). Use this if you've manually "
                        "identified a good uncertain/multimodal case by "
                        "inspecting a first pass's output.")

    p.add_argument("--n_members", type=int, default=20)
    p.add_argument("--cfg_scale", type=float, default=1.5)
    p.add_argument("--S_churn", type=float, default=40.0)
    p.add_argument("--S_noise", type=float, default=1.003)
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--li_event_threshold", type=float, default=5.0 / 255.0)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    if args.config is not None:
        cfg = load_yaml(args.config)
        for k, v in cfg.items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)
    if args.test_roots is None:
        p.error("--test_roots is required (set in CLI or --config)")
    return args


def load_model_and_data(args, device):
    """Same checkpoint-loading / model-construction pattern as
    evaluate.py's run_test_evaluation -- kept in sync deliberately, not
    reimplemented independently, so this script can never silently drift
    from how the real evaluation loads a checkpoint."""
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "args" not in ckpt:
        raise RuntimeError(
            "Checkpoint has no 'args' key -- this script assumes a "
            "current-format checkpoint (same requirement as evaluate.py)."
        )
    ckpt_args = ckpt["args"]
    channels = ckpt.get("channels", ckpt_args.get("channels"))
    stats = ckpt.get("stats")
    if stats is None:
        raise RuntimeError("Checkpoint has no 'stats' key -- cannot denormalise predictions.")

    T_in, T_out, dt_min = ckpt_args["T_in"], ckpt_args["T_out"], ckpt_args["dt_min"]
    C = len(channels)
    binary_li_ctx = ckpt_args.get("binary_li_ctx", False)
    ctx_channels = ckpt_args.get("ctx_channels", None)
    in_ch = compute_in_ch(C, T_in, ctx_channels, binary_li_ctx)

    unet = UNet(
        in_channels=in_ch, out_channels=C,
        base_channels=ckpt_args["base_channels"],
        channel_mults=tuple(ckpt_args["channel_mults"]),
        num_res_blocks=ckpt_args["num_res_blocks"],
        attn_resolutions=tuple(ckpt_args["attn_resolutions"]),
        dropout=0.0, emb_dim=ckpt_args["emb_dim"],
        img_size=ckpt_args.get("img_size", [64, 64])[0],
    )
    precond = EDMPrecond(unet, sigma_data=ckpt_args.get("sigma_data", 0.5))
    model = MultiStepDenoiser(precond, T_out=T_out, dt_min=dt_min)
    state = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    logger.info(f"Checkpoint loaded: {args.checkpoint}  channels={channels}  "
               f"T_in={T_in} T_out={T_out} dt={dt_min}min")

    test_loader = make_test_loader(
        test_roots=args.test_roots, channel_list=channels, stats=stats,
        T_in=T_in, T_out=T_out, img_size=tuple(args.img_size),
        batch_size=args.batch_size, num_workers=args.num_workers,
        binary_li_ctx=binary_li_ctx, ctx_channels=ctx_channels,
    )
    return model, test_loader, stats, channels, ctx_channels, T_out


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, test_loader, stats, channels, ctx_channels, T_out = load_model_and_data(args, device)
    li_idx = channels.index("li")

    # ---- Pass 1: cheap scan, no inference, rank by ground-truth LI activity ----
    logger.info(f"Scanning up to {args.n_scan} sequences for candidate selection...")
    candidates = []  # list of dicts: idx, activity, ctx_np, tgt_np, last_ctx_np
    seq_idx = 0
    for batch in test_loader:
        if seq_idx >= args.n_scan:
            break
        context = batch["context"].numpy()
        target = batch["target"].numpy()
        last_ctx = batch["last_ctx"].numpy()
        B = context.shape[0]
        for b in range(B):
            if seq_idx >= args.n_scan:
                break
            tgt_abs_li = target[b, :, li_idx] + last_ctx[b, li_idx][None]  # (T_out, H, W)
            obs_phys = _li_to_physical(tgt_abs_li, stats)
            activity = float((obs_phys >= args.li_event_threshold).sum())
            candidates.append({
                "idx": seq_idx, "activity": activity,
                "context": context[b].copy(), "target": target[b].copy(),
                "last_ctx": last_ctx[b].copy(),
            })
            seq_idx += 1
    logger.info(f"Scanned {len(candidates)} sequences. "
               f"Activity range: {min(c['activity'] for c in candidates):.0f} - "
               f"{max(c['activity'] for c in candidates):.0f} positive pixels")

    # ---- Select cases ----
    if args.candidate_indices is not None:
        selected = [c for c in candidates if c["idx"] in args.candidate_indices]
        names = [f"manual_{c['idx']}" for c in selected]
    else:
        by_activity = sorted(candidates, key=lambda c: c["activity"])
        nonzero = [c for c in by_activity if c["activity"] > 0]
        if len(nonzero) < 2:
            raise RuntimeError(
                f"Only {len(nonzero)} sequences with nonzero LI activity found in "
                f"the first {args.n_scan} scanned -- increase --n_scan."
            )
        low = nonzero[0]
        high = by_activity[-1]
        mid = nonzero[len(nonzero) // 2]
        selected = [low, mid, high]
        names = ["case_low_activity", "case_moderate_activity", "case_high_activity"]
        logger.info(f"Selected: low=idx{low['idx']} (activity={low['activity']:.0f}), "
                   f"moderate=idx{mid['idx']} (activity={mid['activity']:.0f}), "
                   f"high=idx{high['idx']} (activity={high['activity']:.0f})")

    if not selected:
        raise RuntimeError("No candidates matched --candidate_indices.")

    # ---- Pass 2: full ensemble inference + plotting, selected cases only ----
    npz_payload = {}
    for name, cand in zip(names, selected):
        logger.info(f"Running ensemble inference for '{name}' (sequence idx {cand['idx']})...")
        ctx_t = torch.from_numpy(cand["context"]).unsqueeze(0).to(device)     # (1,T_in,C,H,W)
        tgt_t = torch.from_numpy(cand["target"]).unsqueeze(0).to(device)      # (1,T_out,C,H,W)
        last_ctx_t = torch.from_numpy(cand["last_ctx"]).unsqueeze(0).to(device)  # (1,C,H,W)
        ch_mask = torch.ones(1, len(channels), device=device)  # full channel mask for a plain forecast

        ens = generate_ensemble(
            model, ctx_t, ch_mask, device,
            n_members=args.n_members, cfg_scale=args.cfg_scale,
            S_churn=args.S_churn, S_noise=args.S_noise, num_steps=args.num_steps,
        )  # (1, M, T_out, C, H, W) residuals

        last_ctx_np = last_ctx_t.cpu().numpy()
        ens_np = ens.cpu().numpy()[0] + last_ctx_np[:, None]            # (M,T_out,C,H,W) absolute-normalised
        tgt_np = tgt_t.cpu().numpy()[0] + last_ctx_np[0]                # (T_out,C,H,W) absolute-normalised
        ctx_np = cand["context"]                                        # (T_in,C_ctx,H,W) already absolute-normalised

        # Denormalise everything to physical units -- same per-channel
        # pattern as run_test_evaluation's --plot path.
        _ctx_chs = ctx_channels if ctx_channels else channels
        ctx_den = np.zeros_like(ctx_np)
        for ci, ch in enumerate(_ctx_chs):
            ctx_den[:, ci] = denormalize(ctx_np[:, ci], stats, ch) if ch in stats else ctx_np[:, ci]
        tgt_den = np.zeros_like(tgt_np)
        ens_den = np.zeros_like(ens_np)
        for ci, ch in enumerate(channels):
            fn = (lambda x, _ch=ch: denormalize(x, stats, _ch)) if ch in stats else (lambda x: x)
            tgt_den[:, ci] = fn(tgt_np[:, ci])
            for m in range(ens_np.shape[0]):
                ens_den[m, :, ci] = fn(ens_np[m, :, ci])

        png_path = os.path.join(args.output_dir, f"fig4_example_{name}.png")
        plot_forecast(context_np=ctx_den, ens_np=ens_den, channels=channels,
                      gt_np=tgt_den, save_path=png_path,
                      ctx_channels=_ctx_chs)
        logger.info(f"  -> {png_path}")

        # Also save physical-space LI arrays for the notebook's Figure 4
        # cell (context_ir kept in whatever units ctx_den has for "ir";
        # LI converted to the same physical/event-probability space used
        # throughout the rest of this project's evaluation).
        ir_idx_ctx = _ctx_chs.index("ir") if "ir" in _ctx_chs else 0
        npz_payload[f"{name}_context_ir"] = ctx_den[:, ir_idx_ctx].astype(np.float32)
        npz_payload[f"{name}_target_li"] = _li_to_physical(tgt_den[:, li_idx], stats).astype(np.float32)
        npz_payload[f"{name}_ensemble_li"] = np.stack([
            _li_to_physical(ens_den[m, :, li_idx], stats) for m in range(ens_den.shape[0])
        ]).astype(np.float32)

    npz_path = os.path.join(args.output_dir, "..", "example_cases.npz") \
        if os.path.basename(args.output_dir) == "figures" else \
        os.path.join(args.output_dir, "example_cases.npz")
    np.savez_compressed(npz_path, **npz_payload)
    logger.info(f"Saved -> {npz_path}")
    logger.info("Done. The PNGs above are the primary Figure 4 deliverable "
               "(via plot_forecast's own rendering); the npz is for the "
               "notebook's Figure 4 cell, which displays/re-confirms them "
               "rather than re-implementing the plotting.")


if __name__ == "__main__":
    main()
