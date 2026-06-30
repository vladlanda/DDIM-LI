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

# Suppress the false-positive "step() before optimizer.step()" warning.
# Our sched.step() is always called after all opt.step()s in the epoch —
# PyTorch fires this warning when last_epoch is set on construction (resume)
# before the first forward pass of the new process, which is not a real bug.
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
from model import (
    UNet, EDMPrecond, MultiStepDenoiser,
    EDMSchedule, training_loss,
)
from evaluate import evaluate_epoch, fast_val_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class _TqdmLoggingHandler(logging.StreamHandler):
    """Routes all logging output through tqdm.write() so bars are not corrupted."""
    def emit(self, record: logging.LogRecord):
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
    root.handlers.clear()          # remove the basicConfig StreamHandler
    root.addHandler(handler)
    root.setLevel(logging.INFO)


_setup_logging()
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
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend   = "nccl",
        device_id = torch.device(f"cuda:{local_rank}"),  # silences barrier() warning
    )
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
    # Input channels:
    #   noisy residual   : C  (all data channels)
    #   context frames   : T_in * C_ctx
    #     C_ctx = len(ctx_channels) or C, +1 if binary_li_ctx and LI in ctx
    #   channel mask     : C
    C_ctx_sel = len(args.ctx_channels) if args.ctx_channels else C
    _li_in_ctx = (args.ctx_channels is None) or ("li" in args.ctx_channels)
    C_ctx  = C_ctx_sel + 1 if (args.binary_li_ctx and _li_in_ctx) else C_ctx_sel
    in_ch  = C + T_in * C_ctx + C
    unet   = UNet(
        in_channels      = in_ch,
        out_channels     = C,
        base_channels    = args.base_channels,
        channel_mults    = tuple(args.channel_mults),
        num_res_blocks   = args.num_res_blocks,
        attn_resolutions = tuple(args.attn_resolutions),
        dropout          = args.dropout,
        emb_dim          = args.emb_dim,
        img_size         = tuple(args.img_size)[0],
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
        train_roots       = args.train_roots,
        channel_list      = channels,
        T_in              = args.T_in,
        T_out             = args.T_out,
        img_size          = tuple(args.img_size),
        batch_size        = args.batch_size,        # per-GPU batch size
        num_workers       = args.num_workers,
        stat_path         = stat_path,
        max_samples       = args.max_samples,
        train_val_split   = args.train_val_split,
        oversample_factor  = args.oversample_factor,
        density_percentile = args.density_percentile,
        binary_li_ctx      = args.binary_li_ctx,
        ctx_channels       = args.ctx_channels,
    )

    if not ddp_active() or world_size == 1:
        return train_loader, val_loader, stats

    # Replace samplers with DistributedSampler for DDP.
    # NOTE: WeightedRandomSampler from make_dataloaders is replaced here,
    # so density-based oversampling is NOT active in multi-GPU training.
    # DistributedSampler handles data sharding but uses uniform sampling.
    # This is a known limitation — a distributed weighted sampler would
    # require a custom implementation. For now, the dynamic li_weight
    # (Cui et al. 2019) still operates per-sample inside training_loss.
    from torch.utils.data import DataLoader
    import logging as _log
    _log.getLogger(__name__).warning(
        "DDP mode: WeightedRandomSampler replaced by DistributedSampler. "
        "Density-based sequence oversampling is disabled. "
        "Dynamic li_weight (per-sample) remains active."
    )

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

    # Compute normalised threshold for asymmetric_li_loss from li_event_threshold.
    # li_event_threshold is in physical space [0,1]. Convert:
    #   physical → cbrt → z-score using per-dataset LI statistics
    # This ensures training threshold matches evaluation threshold exactly.
    _li_thresh = getattr(args, "li_event_threshold", 5.0/255.0)
    if "li" in channels and "li" in stats:
        _cbrt   = float(_li_thresh ** (1.0/3.0))
        _mean   = stats["li"]["mean"]
        _std    = stats["li"]["std"]
        asym_norm_threshold = (_cbrt - _mean) / _std
    else:
        asym_norm_threshold = 0.16  # fallback
    if main:
        logger.info(f"asym_norm_threshold = {asym_norm_threshold:.4f} "
                    f"(from li_event_threshold={_li_thresh:.5f})")

    # find_unused_parameters=True required because attention layers at specific
    # resolutions may not activate for every batch (e.g. when spatial dims
    # don't pass through attn_resolutions). Small performance cost is acceptable.
    if ddp_active() and world_size > 1:
        model = DDP(
            model,
            device_ids             = [local_rank],
            output_device          = local_rank,
            find_unused_parameters = False,  # all params used every step
        )

    # EMA lives only on rank 0 (no need to sync across GPUs)
    ema = EMA(model, decay=args.ema_decay) if main else None

    schedule = EDMSchedule(
        P_mean     = args.P_mean,
        P_std      = args.P_std,
        sigma_min  = args.sigma_min,
        sigma_max  = args.sigma_max,
        sigma_data = args.sigma_data,  # must match EDMPrecond sigma_data
    )

    # ----- Optimiser & scaler -----
    raw_model = model.module if isinstance(model, DDP) else model
    opt    = AdamW(raw_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler('cuda', enabled=args.amp)

    # ----- Resume (load before building scheduler) -----
    start_epoch = 0
    best_val    = float("inf")
    ckpt_path   = os.path.join(args.output_dir, "latest.pt")

    resume_mode = None   # 'recover' | 'extend' | None
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if main and ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", best_val)

        # Distinguish CRASH RECOVERY from intentional EXTENSION by comparing
        # the epoch we stopped at against the ORIGINAL epochs target stored
        # in the checkpoint's args.
        prev_epochs = ckpt.get("args", {}).get("epochs", args.epochs)
        if start_epoch < prev_epochs:
            resume_mode = "recover"   # stopped mid-run (e.g. power loss)
        else:
            resume_mode = "extend"    # previous run completed its full schedule

        if main:
            logger.info(f"Resumed from epoch {start_epoch - 1} "
                        f"(mode={resume_mode}, prev_epochs={prev_epochs}, "
                        f"target={args.epochs})")
            if start_epoch >= args.epochs:
                logger.warning(
                    f"start_epoch ({start_epoch}) >= args.epochs ({args.epochs}). "
                    "Nothing to do — increase --epochs to extend training."
                )

    # ── Build LR scheduler according to resume mode ──────────────────────
    #
    # RECOVERY (mid-run crash): continue the ORIGINAL cosine curve exactly.
    #   T_max = original total epochs, last_epoch positions on that curve.
    #   The LR resumes at the value it had when training stopped — NO spike.
    #
    # EXTEND (completed run, more epochs requested): warm restart.
    #   T_max = remaining new epochs, fresh cosine from args.lr → eta_min.
    #   Lets a converged model escape its minimum (Loshchilov & Hutter 2017).
    #
    # FRESH (no resume): standard full cosine over args.epochs.
    if resume_mode == "recover":
        prev_epochs = ckpt.get("args", {}).get("epochs", args.epochs)
        sched = CosineAnnealingLR(
            opt,
            T_max      = prev_epochs,        # ORIGINAL schedule length
            eta_min    = args.lr * 0.001,
            last_epoch = start_epoch - 1,    # continue from where we stopped
        )
        # Do NOT reset optimizer LR — we want the exact LR from the crash point.
        if main:
            logger.info(f"  Crash recovery: continuing cosine "
                        f"at epoch {start_epoch}/{prev_epochs}, "
                        f"LR={sched.get_last_lr()[0]:.3e}")
    else:
        # extend or fresh → warm restart from args.lr over remaining epochs
        for pg in opt.param_groups:
            pg["lr"] = args.lr
        remaining = args.epochs - start_epoch
        sched = CosineAnnealingLR(
            opt,
            T_max      = max(remaining, 1),
            eta_min    = args.lr * 0.001,
            last_epoch = -1,                 # fresh cosine from initial lr
        )
        if main and resume_mode == "extend":
            logger.info(f"  Extension: warm restart from LR={args.lr:.3e} "
                        f"over {remaining} new epochs")

    # ----- WandB (rank 0 only) -----
    if main and HAS_WANDB and args.wandb_project:
        wandb.init(project=args.wandb_project, name=os.path.basename(args.output_dir.rstrip("/")), config=vars(args))

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
                loss = training_loss(
                    denoiser             = model,
                    schedule             = schedule,
                    batch                = batch,
                    device               = device,
                    cfg_drop_prob        = args.cfg_drop_prob,
                    li_weight              = args.li_weight,
                    li_weight_beta         = args.li_weight_beta,
                    li_weight_ref_density  = getattr(args, "li_weight_ref_density", 0.05),
                    asym_weight            = args.asym_weight,
                    asym_alpha           = args.asym_alpha,
                    asym_norm_threshold  = asym_norm_threshold,
                    nbr_weight           = args.nbr_weight,
                    nbr_scales           = args.nbr_scales,
                    spectral_weight      = args.spectral_weight,
                    lead_time_weights    = args.lead_time_weights,
                    channels             = channels,
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
                step_bar.set_postfix(loss=f"{loss.item():.4f}",
                                     lr   = f"{sched.get_last_lr()[0]:.2e}",
                                      refresh=False)

        sched.step()   # called after all opt.step()s in this epoch ✓
        avg_loss = total_loss / len(train_loader)
        elapsed  = time.time() - t0

        if main:
            epoch_bar.set_postfix(
                loss = f"{avg_loss:.4f}",
                lr   = f"{sched.get_last_lr()[0]:.2e}",
                refresh = False,
            )

        # ----- Validation (rank 0 only, both ranks must wait) -----
        # Pattern: barrier → rank 0 does work → barrier.
        # Rank 1 does nothing between the two barriers — it just idles.
        # This is the only safe pattern for asymmetric work in DDP: both
        # ranks must enter and exit each barrier together, so all validation
        # (fast AND slow) must sit between one pair of barriers.
        # Putting a second validation call outside this pair is what caused
        # the previous NCCL timeout (NumelIn=1 ALLREDUCE).
        # ----- Validation (all ranks) + checkpoint (rank 0) -----
        # Both ranks run validation in parallel on their own val_loader shard.
        # fast_val_metrics / evaluate_epoch all_reduce results internally, so
        # only rank 0 receives the final dict; other ranks get {}.
        # Checkpoint and logging remain rank-0-only.
        # ONE barrier before, ONE after — rank 0 may be slower due to I/O.
        if ddp_active():
            dist.barrier()

        # Every epoch: cheap forward-pass denoising metrics (both ranks).
        val_metrics = fast_val_metrics(
            raw_model, val_loader, schedule, device,
            channels    = channels,
            val_samples = args.val_samples,
        )

        # Every val_every epochs: full probabilistic eval (both ranks).
        if args.slow_val and epoch % args.val_every == 0:
            slow_metrics = evaluate_epoch(
                raw_model, val_loader, schedule, device,
                stats       = stats,
                channels    = channels,
                n_members   = args.n_members,
                val_samples = args.val_samples,
                dt_min      = args.dt_min,
            )
            val_metrics.update(slow_metrics)

        # Checkpoint + logging — rank 0 only.
        if main:
            val_loss    = val_metrics.get("val_loss",    float("inf"))
            # Use val_loss directly as the checkpoint criterion.
            # val_loss already includes LI channel weighted at li_weight=20
            # via channel_weighted_mse in fast_val_metrics.
            # Adding val_li_mse separately would double-count LI.
            val_criterion = val_loss

            if val_criterion < best_val and val_criterion < float("inf"):
                best_val = val_criterion
                torch.save(
                    {"model":    raw_model.state_dict(),
                     "ema":      ema.state_dict() if ema else {},
                     "opt":      opt.state_dict(),
                     "sched":    sched.state_dict(),
                     "epoch":    epoch,
                     "best_val": best_val,
                     "stats":    stats,
                     "channels": channels,
                     "args":     vars(args)},
                    os.path.join(args.output_dir, "best.pt"),
                )
                logger.info(f"  ↑ New best val_loss={val_loss:.4f} (criterion={best_val:.4f})")

            torch.save({
                "epoch":    epoch,
                "model":    raw_model.state_dict(),
                "ema":      ema.state_dict() if ema else {},
                "opt":      opt.state_dict(),
                "sched":    sched.state_dict(),
                "best_val": best_val,
                "stats":    stats,
                "channels": channels,
                "args":     vars(args),
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