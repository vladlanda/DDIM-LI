"""
train_ar.py — DDP training for the autoregressive METSAT EDM model.

Structurally identical to train.py but uses:
  - ARDenoiser from model_ar.py (single-step prediction)
  - ar_training_loss from model_ar.py
  - fast_val_metrics_ar for cheap per-epoch validation

Usage — 2 GPUs (recommended):
    torchrun --nproc_per_node=2 train_ar.py \
        --config configs/default_ar.yaml

Usage — single GPU:
    python train_ar.py --config configs/default_ar.yaml
"""

import logging
import os
import time
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings(
    "ignore",
    message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
    category=UserWarning,
    module="torch.optim.lr_scheduler",
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from dataset import make_dataloaders
from model import UNet, EDMPrecond, EDMSchedule
from model_ar import ARPrecond, ARDenoiser, ar_training_loss
from evaluate import fast_val_metrics_ar   # AR-specific cheap val (no lead_idx)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class _TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


def _setup_logging():
    fmt     = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler = _TqdmLoggingHandler()
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


_setup_logging()
logger = logging.getLogger(__name__)


# ===================================================================
# DDP helpers (identical to train.py)
# ===================================================================

def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend   = "nccl",
        device_id = torch.device(f"cuda:{local_rank}"),
    )
    return local_rank, world_size, torch.device(f"cuda:{local_rank}")


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(local_rank): return local_rank == 0
def ddp_active(): return dist.is_available() and dist.is_initialized()


# ===================================================================
# EMA (identical to train.py)
# ===================================================================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow = {
            k: v.detach().float().cpu().clone()
            for k, v in self._unwrap(model).state_dict().items()
        }

    @staticmethod
    def _unwrap(model):
        return model.module if isinstance(model, DDP) else model

    @torch.no_grad()
    def update(self, model):
        src = self._unwrap(model).state_dict()
        for k in self.shadow:
            self.shadow[k].mul_(self.decay).add_(
                src[k].detach().float().cpu(), alpha=1 - self.decay
            )

    def state_dict(self): return self.shadow
    def load_state_dict(self, sd):
        self.shadow = {k: v.float().cpu() for k, v in sd.items()}


# ===================================================================
# Distributed data loaders
# ===================================================================

def make_distributed_loaders(args, local_rank, world_size):
    from torch.utils.data import DataLoader

    channels  = args.channels
    stat_path = os.path.join(args.output_dir, "channel_stats.json")

    train_loader, val_loader, stats = make_dataloaders(
        train_roots     = args.train_roots,
        channel_list    = channels,
        T_in            = args.T_in,
        T_out           = args.T_out,
        img_size        = tuple(args.img_size),
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        stat_path       = stat_path,
        max_samples     = args.max_samples,
        train_val_split = args.train_val_split,
    )

    if not ddp_active() or world_size == 1:
        return train_loader, val_loader, stats

    train_sampler = DistributedSampler(
        train_loader.dataset, num_replicas=world_size,
        rank=local_rank, shuffle=True, drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_loader.dataset, num_replicas=world_size,
        rank=local_rank, shuffle=False, drop_last=False,
    )
    train_loader = DataLoader(
        train_loader.dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_loader.dataset, batch_size=args.batch_size,
        sampler=val_sampler, num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, stats


# ===================================================================
# Model factory
# ===================================================================

def build_model(C, T_in, T_out, dt_min, args):
    # in_channels = C*(T_in + 2): noisy + context + mask  (same layout as direct model)
    in_ch = C * (T_in + 2)
    unet  = UNet(
        in_channels      = in_ch,
        out_channels     = C,
        base_channels    = args.base_channels,
        channel_mults    = tuple(args.channel_mults),
        num_res_blocks   = args.num_res_blocks,
        attn_resolutions = tuple(args.attn_resolutions),
        dropout          = args.dropout,
        emb_dim          = args.emb_dim,
    )
    precond = ARPrecond(unet, sigma_data=args.sigma_data)
    return ARDenoiser(precond, T_in=T_in, dt_min=dt_min)


# ===================================================================
# Training loop
# ===================================================================

def train(args):
    local_rank, world_size, device = setup_ddp()
    main = is_main(local_rank)

    if main:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"AR training | world_size={world_size} | device={device}")
        logger.info(f"Effective batch size: {args.batch_size * world_size}")

    channels = args.channels
    C        = len(channels)
    train_loader, val_loader, stats = make_distributed_loaders(args, local_rank, world_size)

    model = build_model(C, args.T_in, args.T_out, args.dt_min, args).to(device)

    if main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        logger.info(f"Model params: {n_params:.1f}M")

    if ddp_active() and world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    ema = EMA(model, decay=args.ema_decay) if main else None

    schedule = EDMSchedule(
        P_mean=args.P_mean, P_std=args.P_std,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
    )

    raw_model = model.module if isinstance(model, DDP) else model
    opt    = AdamW(raw_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler('cuda', enabled=args.amp)

    # ---- Resume / extend (identical logic to train.py) ----
    start_epoch = 0
    best_val    = float("inf")
    ckpt_path   = os.path.join(args.output_dir, "latest.pt")

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if main and ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])

        if args.extend:
            best_path = os.path.join(os.path.dirname(ckpt_path), "best.pt")
            if os.path.exists(best_path):
                best_ckpt = torch.load(best_path, map_location=device)
                raw_model.load_state_dict(best_ckpt["model"])
                if main and ema is not None and "ema" in best_ckpt:
                    ema.load_state_dict(best_ckpt["ema"])
                if "opt" in best_ckpt:
                    opt.load_state_dict(best_ckpt["opt"])
            start_epoch = 0
            best_val    = float("inf")
            for pg in opt.param_groups:
                pg["lr"] = args.lr
            import re
            base = args.output_dir.rstrip("/")
            m    = re.match(r"^(.*?)(_ext(\d+))?$", base)
            prev = int(m.group(3) or 0)
            args.output_dir = f"{m.group(1)}_ext{prev + 1}"
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = os.path.join(args.output_dir, "latest.pt")
            if main:
                logger.info(f"Extend mode → {args.output_dir}")
        else:
            start_epoch = ckpt["epoch"] + 1
            best_val    = ckpt.get("best_val", best_val)
            if main:
                logger.info(f"Resumed from epoch {start_epoch - 1}")
                if start_epoch >= args.epochs:
                    logger.warning("Nothing to do — increase --epochs")

    sched = CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01,
        last_epoch=start_epoch - 1,
    )
    if args.resume and not args.extend and os.path.exists(ckpt_path) and "sched" in ckpt:
        sched.load_state_dict(ckpt["sched"])

    if main and HAS_WANDB and args.wandb_project:
        wandb.init(
            project = args.wandb_project,
            name    = os.path.basename(args.output_dir.rstrip("/")),
            config  = vars(args),
        )

    # Baseline fast val for extend mode
    if args.extend:
        if main:
            logger.info("Extend mode: baseline validation ...")
        baseline = fast_val_metrics_ar(
            raw_model, val_loader, schedule, device,
            channels=channels, val_samples=args.val_samples,
        )
        if main:
            logger.info(f"  Baseline: {baseline}")
            if HAS_WANDB and args.wandb_project:
                wandb.log({"epoch": -1, "phase": "baseline", **baseline})

    # ---- Main loop ----
    epoch_bar = tqdm(
        range(start_epoch, args.epochs),
        initial=start_epoch, total=args.epochs,
        desc="Epochs", unit="ep",
        disable=not main, dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        if ddp_active() and hasattr(train_loader, "sampler"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        t0         = time.time()
        total_loss = 0.0

        step_bar = tqdm(
            train_loader, desc=f"  Train E{epoch:03d}",
            unit="batch", leave=False,
            disable=not main, dynamic_ncols=True,
        )

        for batch in step_bar:
            opt.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=args.amp):
                loss = ar_training_loss(
                    denoiser        = model,
                    schedule        = schedule,
                    batch           = batch,
                    device          = device,
                    cfg_drop_prob   = args.cfg_drop_prob,
                    spectral_weight = args.spectral_weight,
                    li_weight       = args.li_weight,
                )

            scaler.scale(loss).backward()

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

        sched.step()
        avg_loss = total_loss / len(train_loader)
        elapsed  = time.time() - t0

        if main:
            epoch_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{sched.get_last_lr()[0]:.2e}",
                refresh=False,
            )

        # ---- Validation + checkpoint ----
        if ddp_active():
            dist.barrier()

        val_metrics = fast_val_metrics_ar(
            raw_model, val_loader, schedule, device,
            channels=channels, val_samples=args.val_samples,
        )

        if main:
            val_loss = val_metrics.get("val_loss", float("inf"))
            if val_loss < best_val and val_loss < float("inf"):
                best_val = val_loss
                torch.save(
                    {"model": raw_model.state_dict(),
                     "ema":   ema.state_dict() if ema else {},
                     "opt":   opt.state_dict(),
                     "epoch": epoch, "best_val": best_val,
                     "stats": stats, "channels": channels,
                     "args":  vars(args)},
                    os.path.join(args.output_dir, "best.pt"),
                )
                logger.info(f"  ↑ New best val_loss: {best_val:.4f}")

            torch.save({
                "epoch": epoch, "model": raw_model.state_dict(),
                "ema":   ema.state_dict() if ema else {},
                "opt":   opt.state_dict(), "sched": sched.state_dict(),
                "best_val": best_val,
            }, ckpt_path)

            log_dict = {"epoch": epoch, "train_loss": avg_loss,
                        "lr": sched.get_last_lr()[0], "elapsed_s": elapsed,
                        **val_metrics}
            logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}  "
                        f"time={elapsed:.1f}s  {val_metrics}")
            if HAS_WANDB and args.wandb_project:
                wandb.log(log_dict)

        if ddp_active():
            dist.barrier()

    if main:
        logger.info("AR training complete.")
    cleanup_ddp()


# ===================================================================
# Entry point
# ===================================================================
if __name__ == "__main__":
    from config_ar import parse_args_ar, print_config
    args = parse_args_ar()
    if is_main(int(os.environ.get("LOCAL_RANK", 0))):
        print_config(args)
    train(args)