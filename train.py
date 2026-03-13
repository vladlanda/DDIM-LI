"""
Training script for METSAT lightning nowcasting diffusion model.

Usage:
    python train.py --config configs/default.yaml

Or minimal CLI:
    python train.py \
        --train_roots datasets/central_africa_1 datasets/central_africa_2 datasets/central_africa_3 \
        --channels ir li ch0 ch1 \
        --T_in 6 --T_out 36 --epochs 200 --batch_size 4

NOTE on data splits:
  --train_roots   regions used for training; split internally into train/val
                  via --train_val_split (default 0.7/0.3, temporal split)
  --test_roots    held-out regions, never touched during training;
                  used only in evaluate.py after training is complete
"""

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from dataset import make_dataloaders
from model import (
    UNet, EDMPrecond, MultiStepDenoiser,
    EDMSchedule, edm_training_loss, edm_sampler,
)
from evaluate import evaluate_epoch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ===================================================================
# Model factory
# ===================================================================

def build_model(C: int, T_in: int, T_out: int, dt_min: int, args) -> MultiStepDenoiser:
    """Construct the full denoiser model."""
    # in_channels = noisy (C) + context (T_in*C) + mask (C)
    in_ch = C * (T_in + 2)

    unet = UNet(
        in_channels    = in_ch,
        out_channels   = C,
        base_channels  = args.base_channels,
        channel_mults  = tuple(args.channel_mults),
        num_res_blocks = args.num_res_blocks,
        attn_resolutions = tuple(args.attn_resolutions),
        dropout        = args.dropout,
        emb_dim        = args.emb_dim,
    )

    precond = EDMPrecond(unet, sigma_data=args.sigma_data)
    model   = MultiStepDenoiser(precond, T_out=T_out, dt_min=dt_min)
    return model


# ===================================================================
# EMA
# ===================================================================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, model: nn.Module):
        """Apply EMA weights to model (in-place)."""
        model.load_state_dict(self.shadow)


# ===================================================================
# Training loop
# ===================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ----- Data -----
    channels = args.channels
    C = len(channels)
    stat_path = os.path.join(args.output_dir, "channel_stats.json")
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader, stats = make_dataloaders(
        train_roots      = args.train_roots,
        channel_list     = channels,
        T_in             = args.T_in,
        T_out            = args.T_out,
        img_size         = tuple(args.img_size),
        batch_size       = args.batch_size,
        num_workers      = args.num_workers,
        stat_path        = stat_path,
        max_samples      = args.max_samples,
        train_val_split  = args.train_val_split,
    )

    # ----- Model -----
    model = build_model(C, args.T_in, args.T_out, args.dt_min, args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Model params: {n_params:.1f}M")

    ema      = EMA(model, decay=args.ema_decay)
    schedule = EDMSchedule(
        P_mean    = args.P_mean,
        P_std     = args.P_std,
        sigma_min = args.sigma_min,
        sigma_max = args.sigma_max,
    )

    # ----- Optimiser -----
    opt     = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched   = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler  = GradScaler(enabled=args.amp)

    # ----- Resume -----
    start_epoch = 0
    best_val    = float("inf")
    ckpt_path   = os.path.join(args.output_dir, "latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", best_val)
        logger.info(f"Resumed from epoch {start_epoch}")

    # ----- WandB -----
    if HAS_WANDB and args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args))

    # ----- Main loop -----
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            opt.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp):
                loss = edm_training_loss(
                    denoiser      = model,
                    schedule      = schedule,
                    batch         = batch,
                    device        = device,
                    cfg_drop_prob = args.cfg_drop_prob,
                    spectral_weight = args.spectral_weight,
                    li_weight     = args.li_weight,
                )

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(opt)
            scaler.update()
            ema.update(model)
            total_loss += loss.item()

            if step % 50 == 0:
                logger.info(f"Epoch {epoch} step {step}/{len(train_loader)}  "
                            f"loss={loss.item():.4f}")

        sched.step()
        avg_loss = total_loss / len(train_loader)
        elapsed  = time.time() - t0

        # ----- Validation -----
        val_metrics = {}
        if epoch % args.val_every == 0:
            val_metrics = evaluate_epoch(
                model, val_loader, schedule, device,
                stats=stats, channels=channels,
                num_samples=args.val_samples,
                dt_min=args.dt_min,
            )
            val_crps = val_metrics.get("crps_mean", float("inf"))

            if val_crps < best_val:
                best_val = val_crps
                torch.save({"model": model.state_dict(), "stats": stats,
                            "channels": channels, "args": vars(args)},
                           os.path.join(args.output_dir, "best.pt"))
                logger.info(f"  ↑ New best CRPS: {best_val:.4f}")

        # Save latest
        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "best_val": best_val,
        }, ckpt_path)

        log_dict = {"epoch": epoch, "train_loss": avg_loss,
                    "lr": sched.get_last_lr()[0], "elapsed_s": elapsed,
                    **val_metrics}
        logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}  time={elapsed:.1f}s  {val_metrics}")
        if HAS_WANDB and args.wandb_project:
            wandb.log(log_dict)

    logger.info("Training complete.")


# ===================================================================
# Arg parsing
# ===================================================================

if __name__ == "__main__":
    from config import parse_args, print_config
    args = parse_args()
    print_config(args)
    train(args)
