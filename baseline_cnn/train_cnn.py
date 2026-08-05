"""
Train the deterministic CNN baseline.

Reuses dataset.py's make_dataloaders (same temporal train/val split, same
data pipeline) as the diffusion model, so training data is identical.
Single-GPU (no DDP) -- a deterministic single-forward-pass model is much
cheaper than the diffusion model's ensemble sampling, so this should train
fast enough without it. Can be extended to DDP later if needed.

Usage:
  python baseline_cnn/train_cnn.py --config configs/default.yaml \
      --epochs 150 --output_dir baseline_cnn/outputs/run1

  # With W&B logging (project inherited from configs/default.yaml's
  # wandb_project unless overridden):
  python baseline_cnn/train_cnn.py --config configs/default.yaml \
      --epochs 150 --output_dir baseline_cnn/outputs/run2 \
      --wandb_project DDIM-LI
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import make_dataloaders, denormalize
from config import load_yaml
from model_cnn import DeterministicCNN  # noqa: E402 (path inserted above)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _li_to_physical(arr, stats, ch="li"):
    """NumPy version -- kept for reference / used nowhere hot anymore."""
    if ch not in stats:
        return arr
    x = arr * stats[ch]["std"] + stats[ch]["mean"]
    if stats[ch].get("transform") == "cbrt":
        x = np.power(np.clip(x, 0.0, None), 3)
    return np.clip(x, 0.0, 1.0)


def _li_to_physical_torch(x: torch.Tensor, stats, ch: str = "li") -> torch.Tensor:
    """
    Pure-torch equivalent of _li_to_physical, staying entirely on-device.

    PERFORMANCE-CRITICAL FIX: the original per-batch training loop did
    tgt_abs[:, li_idx].detach().cpu().numpy() -> _li_to_physical (numpy)
    -> torch.from_numpy(...).to(device), once EVERY BATCH. That .cpu()
    call forces a hard CUDA synchronisation barrier -- the GPU pipeline
    (which normally runs many batches ahead of the CPU) has to fully
    drain before the transfer can happen, every single step. This is
    exactly the kind of thing that turns a ~1.4M-parameter, attention-
    free model (which should train fast) into something taking 1h/epoch.
    model.py's actual training_loss has NO such round-trip anywhere --
    confirms this was specific to this script, not an inherited cost.
    Fix: do the identical arithmetic with torch ops on the GPU tensor
    directly, no .cpu()/.numpy() at all.
    """
    if ch not in stats:
        return x
    out = x * stats[ch]["std"] + stats[ch]["mean"]
    if stats[ch].get("transform") == "cbrt":
        out = torch.clamp(out, min=0.0) ** 3
    return torch.clamp(out, 0.0, 1.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train_roots", nargs="+", default=None)
    p.add_argument("--packed_dirs", nargs="+", default=None,
                   help="Explicit override: output dirs from "
                        "preprocess_to_memmap.py, one per region, SAME "
                        "ORDER as --train_roots. Usually you want "
                        "--use_packed instead (auto-derives <root>/_packed "
                        "for each entry in --train_roots).")
    p.add_argument("--use_packed", action="store_true", default=False,
                   help="Auto-derive --packed_dirs as <root>/_packed for "
                        "every entry in --train_roots (the default output "
                        "location from preprocess_to_memmap.py). Avoids "
                        "manually typing out a second, parallel list that "
                        "could silently drift out of sync with --train_roots.")
    p.add_argument("--preload_to_ram", action="store_true", default=False,
                   help="Load each packed region's entire frames.dat/mask.dat "
                        "into a real in-RAM array at startup instead of using "
                        "np.memmap -- removes memmap/page-fault mechanics from "
                        "the hot path entirely. ONLY use if you've confirmed "
                        "the COMBINED size of all --train_roots regions "
                        "(check with `ls -la <region>/_packed/frames.dat`) "
                        "comfortably fits in available RAM (`free -h`) -- if "
                        "it doesn't fit, this can cause swapping and make "
                        "things WORSE, not better. Only safe with the default "
                        "'fork' multiprocessing start method (Linux default) "
                        "-- under 'spawn', each worker would redundantly "
                        "reload its own copy, multiplying RAM use.")
    p.add_argument("--channels", nargs="+", default=None)
    p.add_argument("--T_in", type=int, default=None)
    p.add_argument("--T_out", type=int, default=None)
    p.add_argument("--dt_min", type=int, default=None)
    p.add_argument("--img_size", nargs=2, type=int, default=None)
    p.add_argument("--binary_li_ctx", action="store_true", default=None)
    p.add_argument("--ctx_channels", nargs="+", default=None)
    p.add_argument("--train_val_split", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=150,
                   help="Ceiling on epochs. With ReduceLROnPlateau + "
                        "early stopping (see --early_stop_patience) this "
                        "is rarely reached in practice; it just bounds "
                        "worst-case wall-clock.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=None,
                   help="L2 regularization on Adam. Default (if unset) is "
                        "1e-4, matching Metzl et al. 2025's ResU-Net BNN "
                        "recipe exactly -- NOT inherited from "
                        "configs/default.yaml's weight_decay=0.0, which is "
                        "the diffusion model's value and is justified there "
                        "by EMA regularisation (see that config's comment) "
                        "-- a rationale that does not apply here, since "
                        "this CNN baseline has no EMA. Silently inheriting "
                        "0.0 would be the same class of bug already fixed "
                        "once for base_channels etc.; see _cnn_only_keys.")
    p.add_argument("--lr_patience", type=int, default=5,
                   help="ReduceLROnPlateau patience (epochs of no val_loss "
                        "improvement before dropping LR). Matches Metzl et "
                        "al. 2025 exactly.")
    p.add_argument("--lr_factor", type=float, default=0.1,
                   help="ReduceLROnPlateau LR multiplier on drop. Matches "
                        "Metzl et al. 2025 exactly.")
    p.add_argument("--lr_cooldown", type=int, default=3,
                   help="ReduceLROnPlateau cooldown after a drop before "
                        "patience counting resumes. Matches Metzl et al. "
                        "2025 exactly.")
    p.add_argument("--early_stop_patience", type=int, default=20,
                   help="Stop training if val_loss hasn't improved in this "
                        "many epochs. Set higher than lr_patience+"
                        "lr_cooldown (default 20 > 5+3=8) so at least one "
                        "LR drop gets a chance to produce improvement "
                        "before giving up. Not from the literature -- a "
                        "practical compute-budget safeguard, since best.pt "
                        "is already checkpointed by lowest val_loss "
                        "regardless of when/whether this triggers.")
    p.add_argument("--li_event_threshold", type=float, default=5.0/255.0)
    p.add_argument("--base_channels", type=int, default=None)
    p.add_argument("--channel_mults", nargs="+", type=int, default=None)
    p.add_argument("--num_res_blocks", type=int, default=None)
    p.add_argument("--attn_resolutions", nargs="+", type=int, default=None)
    p.add_argument("--emb_dim", type=int, default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume", nargs="?", const="__auto__", default=None,
                   help="Resume from a checkpoint. Bare --resume auto-uses "
                        "<output_dir>/latest.pt. Or give an explicit path: "
                        "--resume path/to/checkpoint.pt. Model weights "
                        "always resume exactly. Optimizer/scheduler state "
                        "resumes exactly too IF the checkpoint has it "
                        "(saved from this fix onward) -- older checkpoints "
                        "lack it, so Adam momentum and the "
                        "ReduceLROnPlateau plateau/cooldown counters "
                        "restart fresh in that case (logged either way).")
    p.add_argument("--wandb_project", type=str, default=None,
                   help="If set (or inherited from --config's wandb_project, "
                        "e.g. 'DDIM-LI'), logs to Weights & Biases. Run name "
                        "is the output_dir basename, matching train.py's "
                        "convention for the main diffusion model, so CNN-"
                        "baseline and diffusion-model runs are easy to tell "
                        "apart in the same wandb project. No-op if the "
                        "wandb package isn't installed.")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    # These keys are architecture hyperparameters for the DIFFUSION MODEL
    # in configs/default.yaml (base_channels=64, channel_mults=[1,2,3,4],
    # etc. -- ~22M-param scale). The generic merge loop below would
    # silently inherit them for this CNN baseline too, since it only
    # checks "is the arg still None", and these ARE present (non-None) in
    # the shared config. That defeats the lightweight-CNN switch entirely
    # -- confirmed via a real run reporting 21,768,641 params, an exact
    # match for the diffusion model's config, not the intended ~1.45M
    # lightweight config. Excluded here so the CNN baseline's own
    # defaults (or explicit --base_channels etc. CLI overrides) are what
    # actually apply, never a silent inheritance from the shared config.
    _cnn_only_keys = {"base_channels", "channel_mults", "num_res_blocks",
                      "attn_resolutions", "emb_dim", "weight_decay"}
    for k, v in cfg.items():
        if k in _cnn_only_keys:
            continue
        if hasattr(args, k) and getattr(args, k) is None:
            setattr(args, k, v)
    # Lightweight, literature-matched config (~1.4M params), NOT the
    # diffusion model's full-scale architecture. Chosen after benchmarking
    # against Metzl et al. 2025's reported ~1.6M-parameter BNN baseline and
    # LightningCast's operational-inference-oriented sizing: same NUMBER of
    # resolution levels (4) as our main model for structural consistency,
    # but far narrower channels and NO attention (attn_resolutions=())
    # -- both LightningCast and Metzl et al.'s BNN/AINN are plain
    # convolutional U-Nets, no attention. This is ~15x smaller than the
    # diffusion-model-scale config used in earlier smoke tests, and trains
    # far faster (no ensemble sampling either way, but far fewer FLOPs per
    # forward pass too).
    args.base_channels     = args.base_channels or 24
    args.channel_mults     = tuple(args.channel_mults or [1, 2, 2, 4])
    args.num_res_blocks    = args.num_res_blocks or 1
    args.attn_resolutions  = tuple(args.attn_resolutions if args.attn_resolutions is not None else [])
    args.emb_dim           = args.emb_dim or 64
    args.binary_li_ctx     = True if args.binary_li_ctx is None else args.binary_li_ctx
    args.train_val_split   = args.train_val_split or 0.8
    args.batch_size        = args.batch_size or 16
    # 1e-4, matching Metzl et al. 2025's ResU-Net BNN exactly -- see the
    # --weight_decay help text above for why this is NOT taken from
    # configs/default.yaml despite that file having its own weight_decay key.
    args.weight_decay      = args.weight_decay if args.weight_decay is not None else 1e-4

    if args.use_packed:
        if args.packed_dirs is not None:
            raise ValueError("Pass either --use_packed or an explicit "
                             "--packed_dirs, not both.")
        if args.train_roots is None:
            raise ValueError("--use_packed requires --train_roots to be set "
                             "(from CLI or --config) so packed dirs can be "
                             "derived from it.")
        args.packed_dirs = [os.path.join(r, "_packed") for r in args.train_roots]
        missing = [d for d in args.packed_dirs if not os.path.isdir(d)]
        if missing:
            raise FileNotFoundError(
                f"--use_packed derived {missing} but they don't exist. "
                f"Run preprocess_to_memmap.py --root <region> for each of "
                f"--train_roots first."
            )
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.packed_dirs is not None:
        from dataset_packed import make_dataloaders_packed
        logger.info(f"Using PACKED data loading: {args.packed_dirs}")
        train_loader, val_loader, stats = make_dataloaders_packed(
            train_packed_dirs=args.packed_dirs, channel_list=args.channels,
            T_in=args.T_in, T_out=args.T_out, dt_min=args.dt_min,
            batch_size=args.batch_size, num_workers=args.num_workers,
            stats_roots=args.train_roots,   # original JPEG roots, for stats
            train_val_split=args.train_val_split,
            binary_li_ctx=args.binary_li_ctx, ctx_channels=args.ctx_channels,
            preload_to_ram=args.preload_to_ram,
        )
    else:
        train_loader, val_loader, stats = make_dataloaders(
            train_roots=args.train_roots, channel_list=args.channels,
            T_in=args.T_in, T_out=args.T_out, img_size=tuple(args.img_size),
            batch_size=args.batch_size, num_workers=args.num_workers,
            train_val_split=args.train_val_split,
            binary_li_ctx=args.binary_li_ctx, ctx_channels=args.ctx_channels,
        )
    channels = args.channels
    li_idx = channels.index("li")
    C = len(channels)
    logger.info(f"channels={channels}  C={C}  T_in={args.T_in}  T_out={args.T_out}")

    model = DeterministicCNN(
        C=C, T_in=args.T_in, T_out=args.T_out, dt_min=args.dt_min,
        ctx_channels=args.ctx_channels, binary_li_ctx=args.binary_li_ctx,
        base_channels=args.base_channels, channel_mults=args.channel_mults,
        num_res_blocks=args.num_res_blocks, attn_resolutions=args.attn_resolutions,
        dropout=0.1, emb_dim=args.emb_dim, img_size=args.img_size[0],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"DeterministicCNN params: {n_params:,}")

    logger.info(f"weight_decay={args.weight_decay}  lr_patience={args.lr_patience}  "
               f"lr_factor={args.lr_factor}  lr_cooldown={args.lr_cooldown}  "
               f"early_stop_patience={args.early_stop_patience}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # ReduceLROnPlateau monitoring val_loss, matching Metzl et al. 2025's
    # ResU-Net recipe exactly (factor/patience/cooldown defaults above) --
    # NOT the diffusion model's fixed CosineAnnealingLR, which has no
    # relationship to when THIS model's val_loss actually plateaus.
    sched = ReduceLROnPlateau(opt, mode="min", factor=args.lr_factor,
                              patience=args.lr_patience, cooldown=args.lr_cooldown)

    start_epoch = 0
    best_val = float("inf")
    if args.resume is not None:
        ckpt_path = (os.path.join(args.output_dir, "latest.pt")
                    if args.resume == "__auto__" else args.resume)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--resume checkpoint not found: {ckpt_path}")
        logger.info(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
            logger.info("  Restored optimizer state (Adam momentum/variance) exactly.")
        else:
            logger.warning("  Checkpoint predates optimizer-state saving -- "
                          "Adam momentum restarts fresh. Model weights are "
                          "still resumed exactly.")
        if "sched" in ckpt:
            sched.load_state_dict(ckpt["sched"])
            logger.info(f"  Restored LR-scheduler state exactly "
                       f"(current LR={opt.param_groups[0]['lr']:.2e}).")
        else:
            logger.warning("  Checkpoint predates scheduler-state saving -- "
                          "ReduceLROnPlateau's plateau/cooldown counters "
                          "restart fresh at the original --lr.")
        # Prefer best.pt (if present, alongside this checkpoint's own dir)
        # for best_val, since latest.pt's val_loss is just its OWN epoch's
        # value, not necessarily the best seen so far.
        best_path = os.path.join(os.path.dirname(ckpt_path) or ".", "best.pt")
        if os.path.isfile(best_path):
            best_ckpt = torch.load(best_path, map_location="cpu")
            best_val = best_ckpt["val_loss"]
            logger.info(f"  best_val initialized from best.pt: "
                       f"{best_val:.4f} (epoch {best_ckpt['epoch']})")
        else:
            best_val = ckpt["val_loss"]
        logger.info(f"  Resuming at epoch {start_epoch}, best_val={best_val:.4f}. "
                   f"epochs_since_best resets to 0 (not saved historically -- "
                   f"conservative default, avoids stopping too early).")

    if HAS_WANDB and args.wandb_project:
        wandb_kwargs = dict(project=args.wandb_project,
                           name=os.path.basename(args.output_dir.rstrip("/")),
                           config=vars(args))
        if args.resume is not None:
            # Reuse the same run id (= output_dir basename) so resumed
            # training appends to the same W&B run instead of starting a
            # visually disconnected new one.
            wandb_kwargs["id"] = os.path.basename(args.output_dir.rstrip("/"))
            wandb_kwargs["resume"] = "allow"
        wandb.init(**wandb_kwargs)

    epochs_since_best = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        step_bar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        for batch in step_bar:
            # non_blocking=True + pin_memory=True (already set on the
            # DataLoader) lets the CPU->GPU transfer overlap with the GPU
            # still finishing the PREVIOUS step's compute, instead of the
            # transfer blocking the training loop.
            context  = batch["context"].to(device, non_blocking=True)
            target   = batch["target"].to(device, non_blocking=True)      # (B, T_out, C, H, W) residuals
            last_ctx = batch["last_ctx"].to(device, non_blocking=True)     # (B, C, H, W)
            B, T_out_b = context.shape[0], target.shape[1]

            lead_idx = torch.randint(0, T_out_b, (B,), device=device)
            tgt_abs = target[torch.arange(B), lead_idx] + last_ctx   # (B, C, H, W)

            # Binary LI ground truth in PHYSICAL space (same threshold used
            # throughout this project's evaluation scripts). Stays on GPU
            # (no .cpu()/.numpy() round-trip -- see _li_to_physical_torch).
            li_phys = _li_to_physical_torch(tgt_abs[:, li_idx], stats)
            li_bin = (li_phys >= args.li_event_threshold).float().unsqueeze(1)  # (B,1,H,W)

            logits = model(context, lead_idx)  # (B,1,H,W) RAW LOGITS
            # binary_cross_entropy_with_logits, NOT sigmoid()+binary_cross_entropy:
            # the latter is numerically unstable at logit magnitudes reached
            # routinely mid-training with this task's class imbalance -- verified
            # empirically (see model_cnn.py docstring) that gradients can vanish
            # to ~1e-10x correct, silently stalling learning without crashing.
            loss = F.binary_cross_entropy_with_logits(logits, li_bin)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            step_bar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{opt.param_groups[0]['lr']:.2e}")
        avg_train = total_loss / max(len(train_loader), 1)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                context  = batch["context"].to(device, non_blocking=True)
                target   = batch["target"].to(device, non_blocking=True)
                last_ctx = batch["last_ctx"].to(device, non_blocking=True)
                B, T_out_b = context.shape[0], target.shape[1]
                lead_idx = torch.randint(0, T_out_b, (B,), device=device)
                tgt_abs = target[torch.arange(B), lead_idx] + last_ctx
                li_phys = _li_to_physical_torch(tgt_abs[:, li_idx], stats)
                li_bin = (li_phys >= args.li_event_threshold).float().unsqueeze(1)
                logits = model(context, lead_idx)
                val_loss += F.binary_cross_entropy_with_logits(logits, li_bin).item()
        val_loss /= max(len(val_loader), 1)
        lr_before = opt.param_groups[0]["lr"]
        sched.step(val_loss)   # ReduceLROnPlateau -- must be stepped with the monitored metric
        lr_after = opt.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch}: train_loss={avg_train:.4f}  val_loss={val_loss:.4f}"
                   f"{'  (LR dropped ' + f'{lr_before:.2e} -> {lr_after:.2e})' if lr_after < lr_before else ''}")
        if HAS_WANDB and args.wandb_project:
            wandb.log({"epoch": epoch, "train_loss": avg_train,
                      "val_loss": val_loss, "lr": lr_after})

        torch.save({
            "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "epoch": epoch, "val_loss": val_loss,
            "args": vars(args), "channels": channels, "stats": stats,
        }, os.path.join(args.output_dir, "latest.pt"))

        if val_loss < best_val:
            best_val = val_loss
            epochs_since_best = 0
            torch.save({
                "model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "epoch": epoch, "val_loss": val_loss,
                "args": vars(args), "channels": channels, "stats": stats,
            }, os.path.join(args.output_dir, "best.pt"))
            logger.info(f"  New best val_loss={val_loss:.4f}")
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.early_stop_patience:
                logger.info(f"Early stopping: no val_loss improvement in "
                           f"{args.early_stop_patience} epochs "
                           f"(best={best_val:.4f}). best.pt is unaffected.")
                break

    if HAS_WANDB and args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
