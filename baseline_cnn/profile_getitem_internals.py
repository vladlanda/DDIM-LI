"""
Line-level profiling INSIDE PackedMETSATDataset.__getitem__, on real data.

Two rounds of aggregate-level fixes (fancy-indexing, batched-reads) were
each verified correct but had zero measurable effect on real training
time, and a raw I/O benchmark confirmed storage is fast (~11GB/s random
access) while num_workers=0 was found to be SLOWER than num_workers=8
(ruling out inter-process transfer as the bottleneck, and confirming the
per-sample work is genuinely CPU-bound and parallelizable). This means
the cost is inside the per-sample Python/numpy computation itself, and
guessing at more aggregate-level fixes without knowing WHICH line is
slow risks repeating the last two rounds. This instruments every
meaningful step separately instead.

Usage:
  python baseline_cnn/profile_getitem_internals.py --config configs/default.yaml \
      --use_packed --n_samples 200
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_yaml
from dataset_packed import PackedMETSATDataset, CHANNEL_FILL_VALUES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n_samples", type=int, default=200)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    packed_dirs = [os.path.join(r, "_packed") for r in cfg["train_roots"]]
    return args, cfg, packed_dirs


def instrumented_getitem(ds, idx, timings):
    """Re-implements __getitem__ with wall-clock timing around every
    meaningful step, accumulating into `timings` (dict of lists, seconds)."""
    t = lambda: time.perf_counter()

    t0 = t()
    times = ds.valid_sequences[idx]
    timings["0_seq_lookup"].append(t() - t0)

    # ---- per-timestep load loop, but timed section-by-section ----
    t0 = t()
    row_reads = []
    channel_selects = []
    normalizations = []
    frames_list, masks_list = [], []
    for dt in times:
        ta = t()
        row = ds._row_of[dt]
        row_frame = ds._frames[row]
        row_mask  = ds._mask[row]
        row_reads.append(t() - ta)

        tb = t()
        raw_frame = row_frame[ds._packed_ch_idx]
        raw_mask  = row_mask[ds._packed_ch_idx]
        channel_selects.append(t() - tb)

        tc = t()
        h, w = ds.img_size
        frame = np.zeros((ds.C, h, w), dtype=np.float32)
        mask  = np.zeros(ds.C, dtype=np.float32)
        for i, ch in enumerate(ds.channel_list):
            if raw_mask[i] == 0:
                frame[i] = CHANNEL_FILL_VALUES.get(ch, 0.0)
                continue
            frame[i] = raw_frame[i].astype(np.float32) * (1.0 / 255.0)
            mask[i] = 1.0
        for i in range(ds.C):
            if mask[i] == 0.0:
                continue
            if ds._cbrt_mask[i]:
                frame[i] = np.cbrt(frame[i])
            frame[i] = (frame[i] - ds._norm_mean[i]) / ds._norm_std[i]
        normalizations.append(t() - tc)

        frames_list.append(frame)
        masks_list.append(mask)
    timings["1_row_reads_TOTAL"].append(sum(row_reads))
    timings["2_channel_select_TOTAL"].append(sum(channel_selects))
    timings["3_normalize_TOTAL"].append(sum(normalizations))
    timings["4_per_timestep_loop_TOTAL"].append(t() - t0)

    t0 = t()
    frames = np.stack(frames_list)
    masks  = np.stack(masks_list)
    timings["5_stack"].append(t() - t0)

    t0 = t()
    context = frames[:ds.T_in]
    target_abs = frames[ds.T_in:]
    last_ctx = context[-1:]
    target_residual = target_abs - last_ctx
    timings["6_split_residual"].append(t() - t0)

    t0 = t()
    li_idx = ds.channel_list.index("li") if "li" in ds.channel_list else None
    if li_idx is not None:
        li_abs = target_abs[:, li_idx]
        li_phys = li_abs * ds._norm_std[li_idx] + ds._norm_mean[li_idx]
        if ds._cbrt_mask[li_idx]:
            li_phys = np.power(np.clip(li_phys, 0.0, None), 3)
        li_density = float((li_phys >= 5.0/255.0).mean())
    timings["7_li_density"].append(t() - t0)

    t0 = t()
    ctx = context[:, ds.ctx_idx, :, :]
    _li_in_ctx = (li_idx is not None and li_idx in ds.ctx_idx)
    if ds.binary_li_ctx and _li_in_ctx:
        _li_ctx_pos = ds.ctx_idx.index(li_idx)
        li_norm = ctx[:, _li_ctx_pos]
        li_phys = li_norm * ds._norm_std[li_idx] + ds._norm_mean[li_idx]
        if ds._cbrt_mask[li_idx]:
            li_phys = np.power(np.clip(li_phys, 0.0, None), 3)
        li_bin = (li_phys >= 5.0/255.0).astype(np.float32)
        context_out = np.concatenate([ctx, li_bin[:, None, :, :]], axis=1)
    else:
        context_out = ctx
    timings["8_binary_li_ctx"].append(t() - t0)

    import torch
    t0 = t()
    result = {
        "context": torch.from_numpy(context_out),
        "target": torch.from_numpy(target_residual),
        "last_ctx": torch.from_numpy(last_ctx[0]),
    }
    timings["9_to_torch"].append(t() - t0)
    return result


def main():
    args, cfg, packed_dirs = parse_args()
    print(f"Loading packed dataset from {packed_dirs[0]} ...")
    ds = PackedMETSATDataset(
        packed_dirs[0], channel_list=cfg["channels"],
        T_in=cfg["T_in"], T_out=cfg["T_out"], dt_min=cfg["dt_min"],
        stats_root=cfg["train_roots"][0], augment=False,
        binary_li_ctx=cfg.get("binary_li_ctx", True),
        ctx_channels=cfg.get("ctx_channels"),
    )
    print(f"{len(ds)} sequences available. Profiling {args.n_samples} random samples...\n")

    rng = np.random.default_rng(0)
    idxs = rng.integers(0, len(ds), args.n_samples)
    timings = defaultdict(list)

    t_total0 = time.perf_counter()
    for idx in idxs:
        instrumented_getitem(ds, int(idx), timings)
    t_total1 = time.perf_counter()

    total_ms = (t_total1 - t_total0) / args.n_samples * 1000
    print(f"{'step':<30}{'mean ms/sample':>16}{'%% of total':>14}")
    print("-" * 60)
    for key in sorted(timings.keys()):
        vals = np.array(timings[key]) * 1000
        pct = 100 * vals.mean() / total_ms
        print(f"{key:<30}{vals.mean():>16.3f}{pct:>13.1f}%")
    print("-" * 60)
    print(f"{'TOTAL (measured)':<30}{total_ms:>16.3f}")


if __name__ == "__main__":
    main()
