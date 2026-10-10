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
    EDMSchedule, training_loss, compute_in_ch,
    load_cnn_conditioner, load_init_weights,
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

def build_model(C: int, T_in: int, T_out: int, dt_min: int, args, stats=None,
                device="cpu") -> MultiStepDenoiser:
    cnn_path = getattr(args, "cnn_cond_checkpoint", None)
    in_ch = compute_in_ch(C, T_in, args.ctx_channels, args.binary_li_ctx, cnn_cond=bool(cnn_path))
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
    cnn = None
    if cnn_path:
        assert stats is not None, "CNN conditioning needs the training stats to check the CNN inputs"
        cnn = load_cnn_conditioner(cnn_path, device, args.channels, stats, T_in,
                                   args.binary_li_ctx, args.ctx_channels)
    return MultiStepDenoiser(precond, T_out=T_out, dt_min=dt_min, cnn=cnn)


# ===================================================================
# EMA  (always on CPU-mirrored fp32; only updated/saved on rank 0)
# ===================================================================

class EMA:
    """Exponential moving average of the weights, kept ON THE MODEL'S DEVICE.

    - Previously every step copied all parameters to the CPU and averaged there:
      a GPU sync plus ~87 MB transfer per step (21.8M params).
    - Ramp-up (warmup=True): effective decay min(decay, (1+n)/(10+n)) after n
      updates, so the EMA averages over roughly the last ~10% of training so far
      instead of staying dominated by the random initial weights for ~77k steps
      (it is validated, and selected, every epoch). The final EMA is unaffected
      once (1+n)/(10+n) exceeds the cap.
    - state_dict() returns CPU copies of the weights only (evaluation loads
      ckpt["ema"] directly as model weights); the update count is saved
      separately (ckpt["ema_updates"]).
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup: bool = True):
        self.decay       = decay
        self.warmup      = warmup
        self.num_updates = 0
        self.shadow = {k: v.detach().clone().float() if v.dtype.is_floating_point else v.detach().clone()
                       for k, v in self._unwrap(model).state_dict().items()}

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, DDP) else model

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        d   = self.current_decay()
        src = self._unwrap(model).state_dict()
        for k, s in self.shadow.items():
            v = src[k].detach()
            if s.dtype.is_floating_point:
                s.mul_(d).add_(v.to(s.dtype), alpha=1.0 - d)
            else:
                s.copy_(v)

    def state_dict(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict, num_updates: int = None):
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device, self.shadow[k].dtype))
        if num_updates is not None:
            self.num_updates = int(num_updates)


# ===================================================================
# Distributed DataLoader factory
# ===================================================================

class DistributedWeightedSampler(torch.utils.data.Sampler):
    """Weighted sampling with replacement, sharded across DDP ranks.

    Every rank draws the SAME global index sequence (generator seeded by
    seed + epoch) and keeps every world_size-th element, so ranks see disjoint
    shards of one weighted draw -- the multi-GPU equivalent of the single-GPU
    WeightedRandomSampler. Previously DDP replaced the weighted sampler by a
    uniform DistributedSampler, silently disabling density oversampling for the
    diffusion model (trained on 2 GPUs) while the CNN (1 GPU) kept it.
    """
    def __init__(self, weights, num_replicas: int, rank: int, seed: int = 0,
                 epoch_fraction: float = 1.0):
        self.weights      = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank         = rank
        self.seed         = seed
        self.epoch        = 0
        total             = max(num_replicas, int(round(len(self.weights) * float(epoch_fraction))))
        self.num_samples  = total // num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.multinomial(self.weights, self.num_samples * self.num_replicas,
                                replacement=True, generator=g)
        return iter(idx[self.rank::self.num_replicas].tolist())

    def __len__(self):
        return self.num_samples


def make_distributed_loaders(args, local_rank: int, world_size: int):
    """
    Wraps make_dataloaders to inject DistributedSampler when DDP is active.
    Each rank gets a non-overlapping shard of the data automatically.
    """
    channels  = args.channels
    stat_path = os.path.join(args.output_dir, "channel_stats.json")

    # Build datasets (all ranks do this; index-building is read-only)
    epoch_fraction = float(getattr(args, "epoch_fraction", 1.0) or 1.0)
    if getattr(args, "use_packed", False):
        # Preprocessed memory-mapped frames (preprocess_to_memmap.py): identical
        # values, no per-sample PNG decoding (84 files per sample), which left the
        # GPU idle ~75% of the time. Same split, stats, weighting and options.
        from dataset_packed import make_dataloaders_packed
        packed_dirs = [os.path.join(r, "_packed") for r in args.train_roots]
        missing = [d for d in packed_dirs if not os.path.isdir(d)]
        if missing:
            raise FileNotFoundError(f"use_packed: missing {missing}; run preprocess_to_memmap.py "
                                    f"--root <region> --channels {' '.join(channels)} for each training region")
        logging.getLogger(__name__).info(f"Using PACKED data loading: {packed_dirs}")
        train_loader, val_loader, stats = make_dataloaders_packed(
            train_packed_dirs  = packed_dirs,
            channel_list       = channels,
            T_in               = args.T_in,
            T_out              = args.T_out,
            dt_min             = args.dt_min,
            batch_size         = args.batch_size,
            num_workers        = args.num_workers,
            stats_roots        = args.train_roots,
            stat_path          = stat_path,
            max_samples        = args.max_samples,
            train_val_split    = args.train_val_split,
            oversample_factor  = args.oversample_factor,
            density_percentile = args.density_percentile,
            binary_li_ctx      = args.binary_li_ctx,
            ctx_channels       = args.ctx_channels,
            augment_flip       = bool(getattr(args, "augment_flip", False)),
            preload_to_ram     = bool(getattr(args, "preload_to_ram", False)),
            epoch_fraction     = epoch_fraction,
        )
    else:
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
            augment_flip       = bool(getattr(args, "augment_flip", False)),
            density_percentile = args.density_percentile,
            binary_li_ctx      = args.binary_li_ctx,
            ctx_channels       = args.ctx_channels,
            epoch_fraction     = epoch_fraction,
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
    _weights = getattr(train_loader.sampler, "weights", None)
    if _weights is not None:
        train_sampler = DistributedWeightedSampler(_weights, num_replicas=world_size,
                                                   rank=local_rank, epoch_fraction=epoch_fraction)
        _log.getLogger(__name__).info(
            "DDP mode: density oversampling kept via DistributedWeightedSampler.")
    else:
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

    # Global seeding (was absent: initial weights, the 1-GPU weighted sampler,
    # noise levels/noise and guidance dropout all came from unseeded generators,
    # so runs -- and ablations -- were not reproducible). Offset per rank so the
    # GPUs draw different noise. cuDNN kernels may still be non-deterministic.
    import random as _random
    import numpy as _np
    _seed = int(getattr(args, "seed", 0)) + local_rank
    _random.seed(_seed)
    _np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)

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
    model = build_model(C, args.T_in, args.T_out, args.dt_min, args, stats, device).to(device)
    if getattr(args, "cnn_cond_checkpoint", None) and args.cfg_drop_prob > 0:
        raise ValueError("cnn_cond_checkpoint with cfg_drop_prob > 0: the dropped (zeroed) context "
                         "would still feed the CNN; set cfg_drop_prob 0 (guidance is unused, cfg_scale 1).")
    # Fine-tune from an earlier run: start from its EMA weights. Only on a fresh start;
    # a --resume of THIS run loads latest.pt below and overrides it.
    if getattr(args, "init_from", None) and not (args.resume and os.path.exists(
            os.path.join(args.output_dir, "latest.pt"))):
        load_init_weights(model, args.init_from)
        if main:
            logger.info(f"Initialised from {args.init_from} (EMA weights; new input channels zero)")

    if main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        logger.info(f"Model params: {n_params:.1f}M")

    # Compute normalised threshold for asymmetric_li_loss from li_event_threshold.
    # li_event_threshold is in physical space [0,1]. Convert:
    #   physical → cbrt → z-score using per-dataset LI statistics
    # This ensures training threshold matches evaluation threshold exactly.
    _li_thresh = getattr(args, "li_event_threshold", 0.5 / 255.0)
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
    # EMA on EVERY rank: DDP keeps parameters identical after each step and the
    # update is deterministic, so all ranks hold identical shadows. That lets
    # validation run on the EMA weights on all ranks (the weights evaluation
    # loads), instead of the raw weights; only rank 0 saves it.
    ema = EMA(model, decay=args.ema_decay, warmup=bool(getattr(args, "ema_warmup", True)))

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
        if ema is not None and ckpt.get("ema"):
            ema.load_state_dict(ckpt["ema"], num_updates=ckpt.get("ema_updates"))
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
        # Only distributed samplers have set_epoch. torchrun with ONE process still
        # initialises a process group, but then uses the plain WeightedRandomSampler
        # (make_distributed_loaders returns early for world_size == 1).
        if hasattr(train_loader.sampler, "set_epoch"):
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

            if ema is not None:
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

        # Validate the EMA weights -- the weights evaluation loads (ckpt["ema"]) --
        # so the checkpoint is selected for the weights that are reported.
        # Swap them in, validate, restore the training weights.
        _swap_ema = ema is not None and getattr(args, "val_use_ema", True)
        if _swap_ema:
            _train_weights = {k: v.detach().clone() for k, v in raw_model.state_dict().items()}
            raw_model.load_state_dict(ema.state_dict())

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

        if _swap_ema:
            raw_model.load_state_dict(_train_weights)
            del _train_weights

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
                     "ema_updates": ema.num_updates if ema else 0,
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
                     "ema_updates": ema.num_updates if ema else 0,
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