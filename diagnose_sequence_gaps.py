"""
Diagnose why build_valid_starts() finds few/zero valid T_in+T_out-length
consecutive sequences, given a healthy count of individual valid timesteps.

Directly measures the REAL gap structure of a region's index (fresh build,
bypassing any cache) rather than guessing:
  - total valid single timesteps (both required channels present)
  - gap histogram between consecutive valid timestamps
  - length distribution of maximal unbroken runs
  - how many runs of length >= seq_len actually exist, and the max run length

This tells us definitively whether "0 valid sequences" is a genuine
characteristic of the raw data (scattered missing timestamps making a long
unbroken run rare) or a bug (e.g. if the max run length is suspiciously
short relative to what the timestamp span should allow).

Usage:
  python diagnose_sequence_gaps.py --root /path/to/central_africa_1 \
      --T_in 36 --T_out 6 --dt_min 10
"""
import argparse
from pathlib import Path
from collections import Counter
import numpy as np

from dataset import build_index


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--T_in", type=int, default=36)
    p.add_argument("--T_out", type=int, default=6)
    p.add_argument("--dt_min", type=int, default=10)
    args = p.parse_args()
    seq_len = args.T_in + args.T_out

    # Fresh build, never touches any cache (matches the just-fixed default)
    index = build_index(args.root)
    times = sorted(index.keys())
    print(f"\nTotal valid single timesteps: {len(times)}")
    if len(times) < 2:
        print("Too few timesteps to analyse gaps.")
        return

    span_days = (times[-1] - times[0]).total_seconds() / 86400
    expected_if_perfect = int(span_days * 24 * 60 / args.dt_min)
    print(f"Timestamp span: {times[0]} -> {times[-1]}  ({span_days:.1f} days)")
    print(f"If every {args.dt_min}-min slot were present: ~{expected_if_perfect} timesteps")
    print(f"Actual / expected coverage: {100*len(times)/max(expected_if_perfect,1):.1f}%")

    # ---- gap histogram ----
    epochs = np.array([int(t.timestamp()) for t in times])
    gaps_min = np.diff(epochs) / 60.0
    print(f"\nGap histogram (minutes between consecutive VALID timestamps):")
    for lo, hi in [(0,11),(11,21),(21,31),(31,61),(61,121),(121,1e9)]:
        n = int(((gaps_min >= lo) & (gaps_min < hi)).sum())
        label = f"{lo:.0f}-{hi:.0f}" if hi < 1e9 else f">{lo:.0f}"
        print(f"  {label:>10} min: {n:>7}  ({100*n/len(gaps_min):.1f}%)")
    print(f"  median gap: {np.median(gaps_min):.1f} min "
          f"(expect ~{args.dt_min} if data were contiguous)")

    # ---- maximal unbroken run lengths ----
    dt_seconds = args.dt_min * 60
    gap_min_s, gap_max_s = dt_seconds - 60, dt_seconds + 60
    runs = []
    run_len = 1
    for g in np.diff(epochs):
        if gap_min_s <= g <= gap_max_s:
            run_len += 1
        else:
            runs.append(run_len)
            run_len = 1
    runs.append(run_len)
    runs = np.array(runs)

    print(f"\nMaximal unbroken runs (no missing {args.dt_min}-min slot within the run):")
    print(f"  number of runs        : {len(runs)}")
    print(f"  longest run           : {runs.max()} timesteps "
          f"({runs.max()*args.dt_min/60:.1f} hours)")
    print(f"  runs >= seq_len({seq_len}) : {int((runs >= seq_len).sum())}")
    print(f"  mean run length       : {runs.mean():.1f}")
    print(f"  median run length     : {np.median(runs):.1f}")

    # distribution
    print(f"\n  run-length distribution:")
    for lo, hi in [(1,5),(5,15),(15,seq_len),(seq_len,seq_len*2),(seq_len*2,int(1e9))]:
        n = int(((runs >= lo) & (runs < hi)).sum())
        total_ts_in_bucket = int(runs[(runs>=lo)&(runs<hi)].sum())
        label = f"{lo}-{hi}" if hi < 1e9 else f">={lo}"
        print(f"    length {label:>10}: {n:>6} runs, {total_ts_in_bucket:>7} timesteps")

    # expected number of valid sequences from these runs (sliding window)
    expected_seqs = int(np.sum(np.maximum(runs - seq_len + 1, 0)))
    print(f"\n  => expected valid sequences (sliding window over these runs): {expected_seqs}")
    print(f"     (should match build_valid_starts() exactly)")

    print("\nREADING THE RESULT:")
    print(f"  If longest run << {seq_len}, the data has frequent gaps shorter than")
    print(f"  {seq_len*args.dt_min} min apart -- REAL data sparsity, not a bug. Options:")
    print(f"  reduce T_in+T_out, or make sequence construction gap-tolerant.")
    print(f"  If longest run is long but 'runs >= seq_len' is still 0, or if the")
    print(f"  coverage % above looks wrong given what you see on disk, that points")
    print(f"  to a bug in index-building rather than genuine sparsity.")


if __name__ == "__main__":
    main()
