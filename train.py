"""
Training script for METSAT lightning nowcasting diffusion model.
Supports single-GPU and multi-GPU via DDP (torchrun).

Usage — single GPU:
    python train.py --config configs/default.yaml

Usage — 2 GPUs (DDP, recommended):
    torchrun --nproc_per_node=2 train.py --config configs/default.yaml

NOTE on data splits:
  --train_roots     regions used for training; split into train/val internally
                    via --train_val_split (default 0.7/0.3, temporal, no leakage)
  --test_roots      held-out regions, never touched during training;
                    used only in evaluate.py after training is complete

NOTE on batch_size:
  batch_size is PER GPU. With 2 GPUs the effective batch size is 2×batch_size.
  e.g. batch_size=4 → 8 samples per gradient step across both GPUs.
"""

import logging
import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast          # replaces deprecated torch.cuda.amp.*
from tqdm import tqdm
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
    EDMSchedule, edm_training_loss,
)
from evaluate import evaluate_epoch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ===================================================================
# DDP helpers
# ===================================================================

def setup_ddp():
    """Initialise the process group if launched via torchrun, else no-op."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank  = int(os.environ["LOCAL_RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, world_size, device


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(local_rank: int) -> bool:
    return local_rank == 0


def ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized()


# ===================================================================
# Model factory
# ===================================================================

def build_model(C: int, T_in: int, T_out: int, dt_min: int, args) -> MultiStepDenoiser:
    in_ch  = C * (T_in + 2)   # noisy + context + mask
    unet   = UNet(
        in_channels      = in_ch,
        out_channels     = C,
        base_channels    = args.base_channels,
        channel_mults    = tuple(args.channel_mults),
        num_res_blocks   = args.num_res_blocks,
        attn_resolutions = tuple(args.attn_resolutions),
        dropout          = args.dropout,
        emb_dim          = args.emb_dim,
    )
    precond = EDMPrecond(unet, sigma_data=args.sigma_data)
    return MultiStepDenoiser(precond, T_out=T_out, dt_min=dt_min)


# ===================================================================
# EMA  (always on CPU-mirrored fp32; only updated/saved on rank 0)
# ===================================================================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        # Store a plain copy of the unwrapped parameters
        self.shadow = {
            k: v.detach().float().cpu().clone()
            for k, v in self._unwrap(model).state_dict().items()
        }

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, DDP) else model

    @torch.no_grad()
    def update(self, model: nn.Module):
        src = self._unwrap(model).state_dict()
        for k in self.shadow:
            self.shadow[k].mul_(self.decay).add_(
                src[k].detach().float().cpu(), alpha=1 - self.decay
            )

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, sd: dict):
        self.shadow = {k: v.float().cpu() for k, v in sd.items()}


# ===================================================================
# Distributed DataLoader factory
# ===================================================================

def make_distributed_loaders(args, local_rank: int, world_size: int):
    """
    Wraps make_dataloaders to inject DistributedSampler when DDP is active.
    Each rank gets a non-overlapping shard of the data automatically.
    """
    channels  = args.channels
    stat_path = os.path.join(args.output_dir, "channel_stats.json")

    # Build datasets (all ranks do this; index-building is read-only)
    train_loader, val_loader, stats = make_dataloaders(
        train_roots     = args.train_roots,
        channel_list    = channels,
        T_in            = args.T_in,
        T_out           = args.T_out,
        img_size        = tuple(args.img_size),
        batch_size      = args.batch_size,        # per-GPU batch size
        num_workers     = args.num_workers,
        stat_path       = stat_path,
        max_samples     = args.max_samples,
        train_val_split = args.train_val_split,
    )

    if not ddp_active() or world_size == 1:
        return train_loader, val_loader, stats

    # Replace samplers with DistributedSampler
    from torch.utils.data import DataLoader

    train_sampler = DistributedSampler(
        train_loader.dataset,
        num_replicas = world_size,
        rank         = local_rank,
        shuffle      = True,
        drop_last    = True,
    )
    val_sampler = DistributedSampler(
        val_loader.dataset,
        num_replicas = world_size,
        rank         = local_rank,
        shuffle      = False,
        drop_last    = False,
    )

    train_loader = DataLoader(
        train_loader.dataset,
        batch_size  = args.batch_size,
        sampler     = train_sampler,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_loader.dataset,
        batch_size  = args.batch_size,
        sampler     = val_sampler,
        num_workers = args.num_workers,
        pin_memory  = True,
    )

    return train_loader, val_loader, stats


# ===================================================================
# Training loop
# ===================================================================

def train(args):
    local_rank, world_size, device = setup_ddp()
    main = is_main(local_rank)

    if main:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"World size: {world_size}  |  device: {device}")
        logger.info(f"Effective batch size: {args.batch_size * world_size} "
                    f"({args.batch_size} per GPU × {world_size} GPUs)")

    # ----- Data -----
    channels = args.channels
    C        = len(channels)
    train_loader, val_loader, stats = make_distributed_loaders(args, local_rank, world_size)

    # ----- Model -----
    model = build_model(C, args.T_in, args.T_out, args.dt_min, args).to(device)

    if main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        logger.info(f"Model params: {n_params:.1f}M")

    # Wrap with DDP — find_unused_parameters=False is faster when all params are used
    if ddp_active() and world_size > 1:
        model = DDP(
            model,
            device_ids          = [local_rank],
            output_device       = local_rank,
            find_unused_parameters = False,
        )

    # EMA lives only on rank 0 (no need to sync across GPUs)
    ema = EMA(model, decay=args.ema_decay) if main else None

    schedule = EDMSchedule(
        P_mean    = args.P_mean,
        P_std     = args.P_std,
        sigma_min = args.sigma_min,
        sigma_max = args.sigma_max,
    )

    # ----- Optimiser -----
    # Use the unwrapped model's parameters; DDP handles gradient sync automatically
    raw_model = model.module if isinstance(model, DDP) else model
    opt    = AdamW(raw_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched  = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler('cuda', enabled=args.amp)

    # ----- Resume -----
    start_epoch = 0
    best_val    = float("inf")
    ckpt_path   = os.path.join(args.output_dir, "latest.pt")

    if args.resume and os.path.exists(ckpt_path):
        # All ranks load the same checkpoint; map to their own device
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        if main and ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", best_val)
        if main:
            logger.info(f"Resumed from epoch {start_epoch}")

    # ----- WandB (rank 0 only) -----
    if main and HAS_WANDB and args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args))

    # ----- Main loop -----
    epoch_bar = tqdm(
        range(start_epoch, args.epochs),
        initial  = start_epoch,
        total    = args.epochs,
        desc     = "Epochs",
        unit     = "ep",
        disable  = not main,       # only rank 0 shows progress
        dynamic_ncols = True,
    )

    for epoch in epoch_bar:
        # Tell DistributedSampler which epoch we're on so shuffling differs
        if ddp_active() and hasattr(train_loader, "sampler"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        t0         = time.time()
        total_loss = 0.0

        step_bar = tqdm(
            train_loader,
            desc          = f"  Train E{epoch:03d}",
            unit          = "batch",
            leave         = False,
            disable       = not main,
            dynamic_ncols = True,
        )

        for step, batch in enumerate(step_bar):
            opt.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=args.amp):
                loss = edm_training_loss(
                    denoiser        = model,
                    schedule        = schedule,
                    batch           = batch,
                    device          = device,
                    cfg_drop_prob   = args.cfg_drop_prob,
                    spectral_weight = args.spectral_weight,
                    li_weight       = args.li_weight,
                )

            scaler.scale(loss).backward()
            # DDP automatically averages gradients across GPUs during backward

            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)

            scaler.step(opt)
            scaler.update()

            if main and ema is not None:
                ema.update(model)

            total_loss += loss.item()

            if main:
                step_bar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)

        sched.step()   # called after all opt.step()s in this epoch ✓
        avg_loss = total_loss / len(train_loader)
        elapsed  = time.time() - t0

        if main:
            epoch_bar.set_postfix(
                loss = f"{avg_loss:.4f}",
                lr   = f"{sched.get_last_lr()[0]:.2e}",
                refresh = False,
            )

        # ----- Validation (rank 0 only — avoid redundant ensemble generation) -----
        val_metrics = {}
        if main and epoch % args.val_every == 0:
            # Use unwrapped model for evaluation
            val_metrics = evaluate_epoch(
                raw_model, val_loader, schedule, device,
                stats       = stats,
                channels    = channels,
                num_samples = args.val_samples,
                dt_min      = args.dt_min,
            )
            val_crps = val_metrics.get("crps_mean", float("inf"))

            if val_crps < best_val:
                best_val = val_crps
                torch.save(
                    {"model": raw_model.state_dict(), "ema": ema.state_dict() if ema else {},
                     "stats": stats, "channels": channels, "args": vars(args)},
                    os.path.join(args.output_dir, "best.pt"),
                )
                logger.info(f"  ↑ New best CRPS: {best_val:.4f}")

        # ----- Checkpoint (rank 0 only) -----
        if main:
            torch.save({
                "epoch":     epoch,
                "model":     raw_model.state_dict(),
                "ema":       ema.state_dict() if ema else {},
                "opt":       opt.state_dict(),
                "sched":     sched.state_dict(),
                "best_val":  best_val,
            }, ckpt_path)

            log_dict = {"epoch": epoch, "train_loss": avg_loss,
                        "lr": sched.get_last_lr()[0], "elapsed_s": elapsed,
                        **val_metrics}
            logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}  "
                        f"time={elapsed:.1f}s  {val_metrics}")
            if HAS_WANDB and args.wandb_project:
                wandb.log(log_dict)

        # Barrier: make sure rank 0 finishes saving before other ranks proceed
        if ddp_active():
            dist.barrier()

    if main:
        logger.info("Training complete.")
    cleanup_ddp()


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    from config import parse_args, print_config
    args = parse_args()
    if is_main(int(os.environ.get("LOCAL_RANK", 0))):
        print_config(args)
    train(args)
