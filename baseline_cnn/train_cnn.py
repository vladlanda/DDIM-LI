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
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import make_dataloaders, denormalize
from config import load_yaml
from model_cnn import DeterministicCNN  # noqa: E402 (path inserted above)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _li_to_physical(arr, stats, ch="li"):
    x = arr * stats[ch]["std"] + stats[ch]["mean"]
    if stats[ch].get("transform") == "cbrt":
        x = np.power(np.clip(x, 0.0, None), 3)
    return np.clip(x, 0.0, 1.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train_roots", nargs="+", default=None)
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
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--li_event_threshold", type=float, default=5.0/255.0)
    p.add_argument("--base_channels", type=int, default=None)
    p.add_argument("--channel_mults", nargs="+", type=int, default=None)
    p.add_argument("--num_res_blocks", type=int, default=None)
    p.add_argument("--attn_resolutions", nargs="+", type=int, default=None)
    p.add_argument("--emb_dim", type=int, default=None)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    for k, v in cfg.items():
        if hasattr(args, k) and getattr(args, k) is None:
            setattr(args, k, v)
    # Sensible fallbacks matching the diffusion model's architecture where
    # not overridden by config, so the CNN gets comparable capacity.
    args.base_channels     = args.base_channels or 64
    args.channel_mults     = tuple(args.channel_mults or [1, 2, 3, 4])
    args.num_res_blocks    = args.num_res_blocks or 2
    args.attn_resolutions  = tuple(args.attn_resolutions or [64, 32])
    args.emb_dim           = args.emb_dim or 256
    args.binary_li_ctx     = True if args.binary_li_ctx is None else args.binary_li_ctx
    args.train_val_split   = args.train_val_split or 0.8
    args.batch_size        = args.batch_size or 16
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        step_bar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        for batch in step_bar:
            context  = batch["context"].to(device)
            target   = batch["target"].to(device)      # (B, T_out, C, H, W) residuals
            last_ctx = batch["last_ctx"].to(device)     # (B, C, H, W)
            B, T_out_b = context.shape[0], target.shape[1]

            lead_idx = torch.randint(0, T_out_b, (B,), device=device)
            tgt_abs = target[torch.arange(B), lead_idx] + last_ctx   # (B, C, H, W)

            # Binary LI ground truth in PHYSICAL space (same threshold used
            # throughout this project's evaluation scripts).
            li_phys = torch.from_numpy(
                _li_to_physical(tgt_abs[:, li_idx].detach().cpu().numpy(), stats)
            ).to(device)
            li_bin = (li_phys >= args.li_event_threshold).float().unsqueeze(1)  # (B,1,H,W)

            pred = model(context, lead_idx)  # (B,1,H,W) sigmoid probability
            loss = F.binary_cross_entropy(pred, li_bin)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            step_bar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{sched.get_last_lr()[0]:.2e}")
        sched.step()
        avg_train = total_loss / max(len(train_loader), 1)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                context  = batch["context"].to(device)
                target   = batch["target"].to(device)
                last_ctx = batch["last_ctx"].to(device)
                B, T_out_b = context.shape[0], target.shape[1]
                lead_idx = torch.randint(0, T_out_b, (B,), device=device)
                tgt_abs = target[torch.arange(B), lead_idx] + last_ctx
                li_phys = torch.from_numpy(
                    _li_to_physical(tgt_abs[:, li_idx].cpu().numpy(), stats)
                ).to(device)
                li_bin = (li_phys >= args.li_event_threshold).float().unsqueeze(1)
                pred = model(context, lead_idx)
                val_loss += F.binary_cross_entropy(pred, li_bin).item()
        val_loss /= max(len(val_loader), 1)
        logger.info(f"Epoch {epoch}: train_loss={avg_train:.4f}  val_loss={val_loss:.4f}")

        torch.save({
            "model": model.state_dict(), "epoch": epoch, "val_loss": val_loss,
            "args": vars(args), "channels": channels, "stats": stats,
        }, os.path.join(args.output_dir, "latest.pt"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(), "epoch": epoch, "val_loss": val_loss,
                "args": vars(args), "channels": channels, "stats": stats,
            }, os.path.join(args.output_dir, "best.pt"))
            logger.info(f"  New best val_loss={val_loss:.4f}")


if __name__ == "__main__":
    main()
