"""
evaluate_dual.py — Dual-GPU evaluation for METSAT lightning nowcasting.

Identical functionality to evaluate.py but distributes the test set across
two GPUs using DDP, halving wall-clock time.

Usage:
    torchrun --nproc_per_node=2 evaluate_dual.py \\
        --checkpoint outputs/run1/best.pt \\
        --test_roots /media/.../central_africa_4 \\
        --output_dir outputs/eval_run1

    # Plot-only (no GPU needed — single process):
    python evaluate_dual.py \\
        --checkpoint outputs/run1/best.pt \\
        --output_dir outputs/eval_run1 \\
        --plot_only

How it works
------------
  - torchrun spawns 2 processes (rank 0 on GPU 0, rank 1 on GPU 1).
  - DistributedSampler shards the test set: each rank processes its half.
  - Both ranks run generate_ensemble() in parallel on their shard.
  - Scalar metrics (CRPS, RMSE, CSI, …) are accumulated as per-step sums
    and reduced with dist.all_reduce(SUM) → rank 0 divides to get the mean.
  - PR / calibration arrays are variable-length so each rank writes a temp
    npz, rank 0 concatenates them before plotting/saving.
  - All file output (JSON, CSV, npz, plots) is written by rank 0 only.
"""

import csv
import json
import logging
import os
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Reuse all metric functions, plot helpers, and _regenerate_plots from evaluate.py
from evaluate import (
    HAS_SKIMAGE, HAS_CUML,
    generate_ensemble,
    crps_energy, cloud_metrics, lightning_contingency,
    fss, brier_score, spread_skill,
    plot_forecast, _regenerate_plots,
    _pr_curve, _auc,
)

logger = logging.getLogger(__name__)


# ===================================================================
# DDP helpers (mirrors train.py)
# ===================================================================

def _setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        # Single-process fallback (e.g. --plot_only without torchrun)
        return 0, 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend   = "nccl",
        device_id = torch.device(f"cuda:{local_rank}"),
    )
    return local_rank, world_size, torch.device(f"cuda:{local_rank}")


def _cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_ddp():
    return dist.is_available() and dist.is_initialized()


# ===================================================================
# Per-step tensor accumulator with all_reduce
# ===================================================================

class _StepAccumulator:
    """
    Accumulates per-step sums and counts as GPU tensors so that a single
    all_reduce(SUM) aggregates across ranks.

    Usage:
        acc = _StepAccumulator(T_out, n_metrics, device)
        acc.add(t, [val1, val2, ...])   # called per (sample, step)
        means = acc.reduce()            # returns (T_out, n_metrics) numpy on rank 0
    """
    def __init__(self, T_out: int, n_metrics: int, device):
        self.T_out     = T_out
        self.n_metrics = n_metrics
        self.device    = device
        # [sum_m0, sum_m1, ..., count]  per step
        self.data  = torch.zeros(T_out, n_metrics + 1, device=device)

    def add(self, t: int, values):
        """Add a list of metric values for step t."""
        for i, v in enumerate(values):
            if v == v:   # nan check
                self.data[t, i]  += float(v)
        self.data[t, -1] += 1.0   # count

    def reduce(self):
        """
        all_reduce across ranks, return (T_out, n_metrics) numpy array on
        rank 0.  Non-rank-0 ranks get None.
        """
        if _is_ddp():
            dist.all_reduce(self.data, op=dist.ReduceOp.SUM)
        if _is_ddp() and dist.get_rank() != 0:
            return None
        counts = self.data[:, -1].clamp(min=1).cpu().numpy()           # (T_out,)
        sums   = self.data[:, :-1].cpu().numpy()                        # (T_out, n_metrics)
        return sums / counts[:, None]                                   # (T_out, n_metrics)


# ===================================================================
# Main dual-GPU evaluation
# ===================================================================

def run_dual_evaluation(args):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # ---- tqdm-safe logging ----
    class _TqdmHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                tqdm.write(self.format(record))
            except Exception:
                self.handleError(record)

    root_log = logging.getLogger()
    root_log.handlers.clear()
    h = _TqdmHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_log.addHandler(h)
    root_log.setLevel(logging.INFO)

    # ---- plot_only: no GPU needed, single process ----
    if args.plot_only:
        npz_path = os.path.join(args.output_dir, "plot_data.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"--plot_only requires {npz_path}\n"
                "Run evaluation once first (without --plot_only)."
            )
        logger.info(f"--plot_only: loading from {npz_path}")
        _regenerate_plots(npz_path, args)
        return {}

    # ---- DDP setup ----
    local_rank, world_size, device = _setup_ddp()
    main = (local_rank == 0)

    if main:
        logger.info(f"Dual-GPU evaluation: world_size={world_size}  device={device}")

    # ---- Load checkpoint (every rank loads independently) ----
    from model import UNet, EDMPrecond, MultiStepDenoiser, EDMSchedule

    ckpt      = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt["args"]
    channels  = ckpt["channels"]
    stats     = ckpt["stats"]
    T_in      = ckpt_args["T_in"]
    T_out     = ckpt_args["T_out"]
    dt_min    = ckpt_args["dt_min"]
    C         = len(channels)

    unet = UNet(
        in_channels      = C * (T_in + 2),
        out_channels     = C,
        base_channels    = ckpt_args["base_channels"],
        channel_mults    = tuple(ckpt_args["channel_mults"]),
        num_res_blocks   = ckpt_args["num_res_blocks"],
        attn_resolutions = tuple(ckpt_args["attn_resolutions"]),
        dropout          = 0.0,
        emb_dim          = ckpt_args["emb_dim"],
    )
    precond = EDMPrecond(unet, sigma_data=ckpt_args.get("sigma_data", 0.5))
    model   = MultiStepDenoiser(precond, T_out=T_out, dt_min=dt_min)
    state   = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state)
    model.to(device).eval()

    if main:
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"  channels={channels}  T_in={T_in}  T_out={T_out}  dt={dt_min}min")

    # ---- Test loader with DistributedSampler ----
    from dataset import make_test_loader

    # Build the full dataset first, then wrap with DistributedSampler
    full_loader = make_test_loader(
        test_roots   = args.test_roots,
        channel_list = channels,
        stats        = stats,
        T_in         = T_in,
        T_out        = T_out,
        img_size     = tuple(args.img_size),
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
    )
    sampler = DistributedSampler(
        full_loader.dataset,
        num_replicas = world_size,
        rank         = local_rank,
        shuffle      = False,
        drop_last    = False,
    )
    test_loader = DataLoader(
        full_loader.dataset,
        batch_size  = args.batch_size,
        sampler     = sampler,
        num_workers = args.num_workers,
        pin_memory  = True,
    )

    n_total = len(full_loader.dataset)
    n_local = len(test_loader.dataset)
    if main:
        logger.info(f"Test sequences: {n_total} total  "
                    f"(~{n_local} per rank with {world_size} GPUs)")

    # ---- Channel helpers ----
    li_idx     = channels.index("li") if "li" in channels else None
    cloud_chs  = [ch for ch in channels if ch != "li"]
    cloud_idxs = [channels.index(ch) for ch in cloud_chs]
    fss_scales = args.fss_scales

    # ---- Scalar accumulators (GPU tensors, all_reduced) ----
    # CRPS, spread_skill
    crps_acc = _StepAccumulator(T_out, 1, device)
    ss_acc   = _StepAccumulator(T_out, 1, device)

    # Per-channel cloud: RMSE, MAE, [SSIM]
    n_cloud_metrics = 2 + (1 if HAS_SKIMAGE else 0)
    cloud_acc = {ch: _StepAccumulator(T_out, n_cloud_metrics, device)
                 for ch in cloud_chs}

    # Lightning: CSI, POD, FAR, Brier, FSS-per-scale
    n_li_metrics = 4 + len(fss_scales)
    li_acc = _StepAccumulator(T_out, n_li_metrics, device) if li_idx is not None else None

    # PR / calibration: variable-length per step — saved to temp file,
    # rank 0 concatenates after the loop
    pr_steps  = list(range(T_out))
    pr_probs  = {t: [] for t in pr_steps}
    pr_labels = {t: [] for t in pr_steps}
    cal_probs  = {t: [] for t in pr_steps}
    cal_labels = {t: [] for t in pr_steps}

    if main:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.plot:
            os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    # ---- Evaluation loop ----
    seq_counter = 0
    bar_desc    = f"Eval rank{local_rank}"
    batch_bar   = tqdm(test_loader, desc=bar_desc, unit="batch",
                       dynamic_ncols=True, leave=True,
                       disable=not main)   # only rank 0 shows bar

    for batch in batch_bar:
        context  = batch["context"].to(device)
        target   = batch["target"].to(device)
        last_ctx = batch["last_ctx"].to(device)
        ch_mask  = batch["tgt_mask"][:, 0].to(device)

        ens = generate_ensemble(
            model, context, ch_mask, device,
            n_members = args.n_members,
            cfg_scale = args.cfg_scale,
        )

        last_ctx_np = last_ctx.cpu().numpy()
        ens_np  = ens.cpu().numpy() + last_ctx_np[:, None, None]
        tgt_np  = target.cpu().numpy() + last_ctx_np[:, None]
        ctx_np  = batch["context"].numpy()
        B       = ens_np.shape[0]

        for b in range(B):
            for t in range(T_out):
                ens_t    = ens_np[b, :, t]
                tgt_t    = tgt_np[b, t]
                ens_mean = ens_t.mean(axis=0)

                crps_acc.add(t, [crps_energy(ens_t, tgt_t)])
                ss_acc.add(t,   [spread_skill(ens_t, tgt_t)])

                if cloud_idxs:
                    cm = cloud_metrics(ens_mean, tgt_t, cloud_idxs, cloud_chs, stats)
                    for ch in cloud_chs:
                        if ch in cm:
                            vals = [cm[ch]["rmse"], cm[ch]["mae"]]
                            if HAS_SKIMAGE and "ssim" in cm[ch]:
                                vals.append(cm[ch]["ssim"])
                            cloud_acc[ch].add(t, vals)

                if li_idx is not None:
                    pred_prob = (ens_t[:, li_idx] > args.li_threshold).mean(axis=0)
                    obs_bin   = (tgt_t[li_idx] > args.li_threshold).astype(np.float32)

                    ct      = lightning_contingency(pred_prob, obs_bin)
                    bs      = brier_score(pred_prob, obs_bin)
                    fss_vals = [fss(pred_prob, obs_bin, scale=s) for s in fss_scales]
                    li_acc.add(t, [ct["csi"], ct["pod"], ct["far"], bs] + fss_vals)

                    if t in pr_steps:
                        flat_prob = pred_prob.ravel()
                        flat_lbl  = obs_bin.ravel()
                        stride    = max(1, len(flat_prob) // 4096)
                        pr_probs[t].append(flat_prob[::stride])
                        pr_labels[t].append(flat_lbl[::stride])
                        cal_probs[t].append(flat_prob[::stride])
                        cal_labels[t].append(flat_lbl[::stride])

            # Forecast plots — rank 0 only (it has its shard's sequences)
            if main and args.plot and seq_counter < args.max_plots:
                from dataset import denormalize as _denorm
                ctx_b   = ctx_np[b]
                ctx_den = np.zeros_like(ctx_b)
                ens_den = np.zeros_like(ens_np[b])
                tgt_den = np.zeros_like(tgt_np[b])
                for ci, ch in enumerate(channels):
                    fn = (lambda x, _ch=ch: _denorm(x, stats, _ch)) if ch in stats \
                         else (lambda x: x)
                    ctx_den[:, ci] = fn(ctx_b[:, ci])
                    tgt_den[:, ci] = fn(tgt_np[b, :, ci])
                    for m in range(ens_np.shape[1]):
                        ens_den[m, :, ci] = fn(ens_np[b, m, :, ci])
                png = os.path.join(args.output_dir, "plots",
                                   f"eval_{seq_counter:04d}.png")
                plot_forecast(context_np=ctx_den, ens_np=ens_den,
                              channels=channels, gt_np=tgt_den, save_path=png)

            seq_counter += 1

        if main:
            batch_bar.set_postfix(
                crps=f"{crps_acc.data[:, 0].mean().item() / max(crps_acc.data[:, -1].mean().item(), 1):.4f}",
                refresh=False,
            )

    # ---- Synchronise before aggregation ----
    if _is_ddp():
        dist.barrier()

    # ---- Reduce scalar metrics ----
    crps_means = crps_acc.reduce()   # (T_out, 1) or None on non-main
    ss_means   = ss_acc.reduce()
    cloud_means = {ch: cloud_acc[ch].reduce() for ch in cloud_chs}
    li_means   = li_acc.reduce() if li_acc is not None else None

    # ---- Reduce PR/calibration arrays (via temp files) ----
    # Each rank saves its local arrays; rank 0 loads and concatenates all ranks.
    if li_idx is not None:
        tmp_dir  = os.path.join(args.output_dir, "_pr_tmp")
        if main:
            os.makedirs(tmp_dir, exist_ok=True)
        if _is_ddp():
            dist.barrier()   # ensure tmp_dir exists before non-main writes

        tmp_path = os.path.join(tmp_dir, f"pr_rank{local_rank}.npz")
        pr_payload = {}
        for t in pr_steps:
            if pr_probs[t]:
                pr_payload[f"pr_prob_{t}"]   = np.concatenate(pr_probs[t])
                pr_payload[f"pr_label_{t}"]  = np.concatenate(pr_labels[t])
                pr_payload[f"cal_prob_{t}"]  = np.concatenate(cal_probs[t])
                pr_payload[f"cal_label_{t}"] = np.concatenate(cal_labels[t])
        np.savez_compressed(tmp_path, **pr_payload)

        if _is_ddp():
            dist.barrier()   # all ranks done writing

        if main:
            # Concatenate all rank files
            rank_files = [os.path.join(tmp_dir, f"pr_rank{r}.npz")
                          for r in range(world_size)]
            for t in pr_steps:
                parts_prob, parts_lbl, parts_cal_p, parts_cal_l = [], [], [], []
                for rf in rank_files:
                    if not os.path.exists(rf):
                        continue
                    d = np.load(rf, allow_pickle=True)
                    if f"pr_prob_{t}" in d:
                        parts_prob.append(d[f"pr_prob_{t}"])
                        parts_lbl.append(d[f"pr_label_{t}"])
                        parts_cal_p.append(d[f"cal_prob_{t}"])
                        parts_cal_l.append(d[f"cal_label_{t}"])
                if parts_prob:
                    pr_probs[t]   = [np.concatenate(parts_prob)]
                    pr_labels[t]  = [np.concatenate(parts_lbl)]
                    cal_probs[t]  = [np.concatenate(parts_cal_p)]
                    cal_labels[t] = [np.concatenate(parts_cal_l)]
            # Clean up temp files
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---- Rank 0: build per_step, summary, save, plot ----
    if not main:
        _cleanup_ddp()
        return {}

    def _m(arr, t, col): return float(arr[t, col]) if arr is not None else float("nan")

    lead_times = [(t + 1) * dt_min for t in range(T_out)]

    ch_unit = {}
    for ch in cloud_chs:
        is_cbrt = ch in stats and stats[ch].get("transform") == "cbrt"
        ch_unit[ch] = "norm" if is_cbrt else "K"

    per_step = []
    for t in range(T_out):
        row = {
            "lead_min":    lead_times[t],
            "crps":        _m(crps_means, t, 0),
            "spread_skill":_m(ss_means,   t, 0),
        }
        for ch in cloud_chs:
            cm = cloud_means.get(ch)
            row[f"rmse_{ch}"] = _m(cm, t, 0)
            row[f"mae_{ch}"]  = _m(cm, t, 1)
            if HAS_SKIMAGE and cm is not None and cm.shape[1] > 2:
                row[f"ssim_{ch}"] = _m(cm, t, 2)
        if li_means is not None:
            row.update({
                "csi":   _m(li_means, t, 0),
                "pod":   _m(li_means, t, 1),
                "far":   _m(li_means, t, 2),
                "brier": _m(li_means, t, 3),
                "fss":   _m(li_means, t, 4 + len(fss_scales) // 2),
            })
        per_step.append(row)

    fss_summary = {}
    if li_means is not None:
        for si, s in enumerate(fss_scales):
            fss_summary[f"fss_scale{s}"] = float(
                np.mean([_m(li_means, t, 4 + si) for t in range(T_out)])
            )

    def _mean_over_steps(key):
        vals = [r[key] for r in per_step if key in r and r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "checkpoint":  args.checkpoint,
        "test_roots":  args.test_roots,
        "n_sequences": n_total,
        "n_members":   args.n_members,
        "world_size":  world_size,
        "crps_mean":   _mean_over_steps("crps"),
        "crps_1h":     float(np.mean([per_step[t]["crps"] for t in range(min(6,  T_out))])),
        "crps_3h":     float(np.mean([per_step[t]["crps"] for t in range(min(18, T_out))])),
        "crps_6h":     per_step[-1]["crps"],
        "spread_skill":_mean_over_steps("spread_skill"),
    }
    for ch in cloud_chs:
        summary[f"rmse_{ch}_mean"] = _mean_over_steps(f"rmse_{ch}")
        summary[f"mae_{ch}_mean"]  = _mean_over_steps(f"mae_{ch}")
        if f"ssim_{ch}" in per_step[0]:
            summary[f"ssim_{ch}_mean"] = _mean_over_steps(f"ssim_{ch}")
    if li_means is not None:
        summary.update({
            "csi_mean":   _mean_over_steps("csi"),
            "csi_1h":     float(np.mean([per_step[t]["csi"] for t in range(min(6, T_out))])),
            "csi_6h":     per_step[-1]["csi"],
            "pod_mean":   _mean_over_steps("pod"),
            "far_mean":   _mean_over_steps("far"),
            "brier_mean": _mean_over_steps("brier"),
            **fss_summary,
        })

    # ---- Save JSON ----
    json_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary    -> {json_path}")

    # ---- Save CSV ----
    csv_path = os.path.join(args.output_dir, "metrics_per_step.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_step[0].keys())
        writer.writeheader()
        writer.writerows(per_step)
    logger.info(f"Per-step   -> {csv_path}")

    # ---- Save plot_data.npz ----
    def _fmean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    npz_path = os.path.join(args.output_dir, "plot_data.npz")
    npz_payload = {
        "per_step_json": np.array(json.dumps(per_step)),
        "lead_times":    np.array(lead_times),
        "channels":      np.array(channels),
        "cloud_chs":     np.array(cloud_chs),
        "fss_scales":    np.array(fss_scales),
        "T_out":         np.array(T_out),
        "dt_min":        np.array(dt_min),
        "pixel_size_km": np.array(args.pixel_size_km),
        "pr_steps":      np.array(pr_steps),
    }
    if li_means is not None:
        for si, s in enumerate(fss_scales):
            npz_payload[f"fss_s{s}"] = np.array(
                [_m(li_means, t, 4 + si) for t in range(T_out)]
            )
        for t in pr_steps:
            if pr_probs[t]:
                npz_payload[f"pr_prob_{t}"]   = np.concatenate(pr_probs[t])
                npz_payload[f"pr_label_{t}"]  = np.concatenate(pr_labels[t])
                npz_payload[f"cal_prob_{t}"]  = np.concatenate(cal_probs[t])
                npz_payload[f"cal_label_{t}"] = np.concatenate(cal_labels[t])
    np.savez_compressed(npz_path, **npz_payload)
    logger.info(f"Plot data  -> {npz_path}")

    # ---- All plots ----
    _regenerate_plots(npz_path, args)

    # ---- Print summary ----
    logger.info("\n" + "=" * 52)
    logger.info("  CLOUD METRICS")
    logger.info("=" * 52)
    for k in ["crps_mean", "crps_1h", "crps_3h", "crps_6h", "spread_skill"]:
        if k in summary:
            logger.info(f"  {k:<22s}: {summary[k]:.4f}")
    for ch in cloud_chs:
        for metric in ["rmse", "mae", "ssim"]:
            k = f"{metric}_{ch}_mean"
            if k in summary:
                logger.info(f"  {k:<22s}: {summary[k]:.4f}")
    if li_means is not None:
        logger.info("\n" + "=" * 52)
        logger.info("  LIGHTNING METRICS")
        logger.info("=" * 52)
        for k in ["csi_mean", "csi_1h", "csi_6h", "pod_mean", "far_mean", "brier_mean"]:
            if k in summary:
                logger.info(f"  {k:<22s}: {summary[k]:.4f}")
        for k in sorted(k for k in summary if k.startswith("fss_scale")):
            logger.info(f"  {k:<22s}: {summary[k]:.4f}")
    logger.info("=" * 52)

    _cleanup_ddp()
    return summary


# ===================================================================
# Entry point
# ===================================================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Dual-GPU evaluation for METSAT lightning nowcasting."
    )
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--test_roots",    required=True, nargs="+")
    p.add_argument("--output_dir",    default="outputs/evaluation")
    p.add_argument("--n_members",     type=int,   default=10)
    p.add_argument("--cfg_scale",     type=float, default=1.5)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--img_size",      nargs=2, type=int, default=[256, 256])
    p.add_argument("--li_threshold",  type=float, default=0.1)
    p.add_argument("--fss_scales",    nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--pixel_size_km", type=float, default=4.0)
    p.add_argument("--plot",          action="store_true")
    p.add_argument("--max_plots",     type=int,   default=20)
    p.add_argument("--plot_only",     action="store_true",
                   help="Skip inference — reload plot_data.npz and regenerate plots")
    args = p.parse_args()
    run_dual_evaluation(args)
