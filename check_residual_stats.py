"""
check_residual_stats.py — per-channel residual and absolute frame statistics.

Usage:
    # Single folder
    python check_residual_stats.py /data/central_africa_1

    # Multiple folders — stats pooled across all of them
    python check_residual_stats.py /data/central_africa_1 /data/central_africa_2 \
                                   /data/central_africa_3 /data/central_africa_4

    # Custom channels / sequence budget
    python check_residual_stats.py /data/central_africa_1 /data/central_africa_2 \
        --channels ir li ch1 ch2 --n_seq 500


    python check_residual_stats.py \
        /home/vladlanda/Workplace/LI-DATASETS/full/central_africa_1 \
        /home/vladlanda/Workplace/LI-DATASETS/full/central_africa_2 \
        /home/vladlanda/Workplace/LI-DATASETS/full/central_africa_3 \
        --channels ir li ch0 ch1 \
        --T_in 18 \
        --T_out 6 \
        --n_seq 500
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from dataset import METSATDataset


def _accumulate(ds, channels, n_seq):
    """Return per-channel (sum, sum_sq, count) for residuals and absolute frames."""
    C = len(channels)
    res_sum  = np.zeros(C)
    res_sum2 = np.zeros(C)
    abs_sum  = np.zeros(C)
    abs_sum2 = np.zeros(C)
    count = 0

    indices = np.linspace(0, len(ds) - 1, min(n_seq, len(ds)), dtype=int)
    for i in indices:
        s = ds[i]
        res = s["target"].numpy()    # (T_out, C, H, W) residuals
        ctx = s["context"].numpy()   # (T_in,  C, H, W) absolute normalised
        for ci in range(C):
            rv = res[:, ci].ravel()
            res_sum[ci]  += rv.mean()
            res_sum2[ci] += (rv ** 2).mean()
            av = ctx[:, ci].ravel()
            abs_sum[ci]  += av.mean()
            abs_sum2[ci] += (av ** 2).mean()
        count += 1

    return res_sum, res_sum2, abs_sum, abs_sum2, count


def _stats(s, s2, n):
    mean = s / n
    std  = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
    return mean, std


def main():
    p = argparse.ArgumentParser(
        description="Per-channel residual statistics across dataset roots."
    )
    p.add_argument("roots",       nargs="+",
                   help="One or more dataset root directories")
    p.add_argument("--channels",  nargs="+", default=["ir", "li", "ch1", "ch2"])
    p.add_argument("--T_in",      type=int,  default=6)
    p.add_argument("--T_out",     type=int,  default=36)
    p.add_argument("--n_seq",     type=int,  default=300,
                   help="Sequences to sample per root (default: 300)")
    p.add_argument("--stat_path", default=None,
                   help="Pre-computed channel_stats.json (optional)")
    args = p.parse_args()

    C = len(args.channels)
    total_res_sum  = np.zeros(C)
    total_res_sum2 = np.zeros(C)
    total_abs_sum  = np.zeros(C)
    total_abs_sum2 = np.zeros(C)
    total_count    = 0

    for root in args.roots:
        print(f"\nLoading  {root} ...", flush=True)
        ds = METSATDataset(
            root         = root,
            channel_list = args.channels,
            T_in         = args.T_in,
            T_out        = args.T_out,
            stat_path    = args.stat_path,
            augment      = False,
        )
        n = min(args.n_seq, len(ds))
        print(f"  {len(ds)} sequences — sampling {n}", flush=True)

        rs, rs2, as_, as2, cnt = _accumulate(ds, args.channels, n)
        total_res_sum  += rs
        total_res_sum2 += rs2
        total_abs_sum  += as_
        total_abs_sum2 += as2
        total_count    += cnt

        # Per-root summary
        r_mean, r_std = _stats(rs, rs2, cnt)
        print(f"  {'Channel':<8}  {'res_mean':>9}  {'res_std':>8}")
        for ci, ch in enumerate(args.channels):
            print(f"  {ch:<8}  {r_mean[ci]:>9.4f}  {r_std[ci]:>8.4f}")

    # ---- Pooled summary ------------------------------------------------
    r_mean, r_std = _stats(total_res_sum,  total_res_sum2, total_count)
    a_mean, a_std = _stats(total_abs_sum,  total_abs_sum2, total_count)

    print(f"\n{'='*60}")
    print(f"POOLED STATISTICS  "
          f"({total_count} sequences across {len(args.roots)} root(s))")
    print(f"{'='*60}")

    print(f"\nResidual frames  (target - last_ctx):")
    print(f"  {'Channel':<8}  {'mean':>8}  {'std':>8}  {'sigma_data':>10}  note")
    print(f"  {'-'*56}")
    for ci, ch in enumerate(args.channels):
        note = "LARGE" if r_std[ci] > 1.5 else ("low" if r_std[ci] < 0.3 else "ok")
        print(f"  {ch:<8}  {r_mean[ci]:>8.4f}  {r_std[ci]:>8.4f}"
              f"  {r_std[ci]:>10.2f}  {note}")

    overall_std = float(np.sqrt(np.mean(r_std ** 2)))
    print(f"\n  RMS std across channels : {overall_std:.4f}")
    print(f"  Recommended sigma_data  : {overall_std:.2f}")

    print(f"\nAbsolute frames  (context, should be ~mean=0, std=1):")
    print(f"  {'Channel':<8}  {'mean':>8}  {'std':>8}")
    print(f"  {'-'*30}")
    for ci, ch in enumerate(args.channels):
        print(f"  {ch:<8}  {a_mean[ci]:>8.4f}  {a_std[ci]:>8.4f}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()