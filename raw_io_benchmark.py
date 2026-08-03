"""
Raw I/O benchmark, completely independent of dataset.py/dataset_packed.py.

Purpose: establish GROUND TRUTH for what your storage can actually do,
so we know whether the ~266ms/batch data-loading cost is a genuine
hardware/storage limit (in which case no further code change in this
project can fix it) or whether there's still a real software
inefficiency left to find.

Tests, on the packed frames.dat file directly:
  1. Sequential read throughput (read the whole file start to end).
  2. Random-access read throughput at REAL sample granularity (reading
     42-row chunks at random offsets, matching one training sample).

Usage:
  python raw_io_benchmark.py --packed_dir <region>/_packed \
      --seq_len 42 --n_reads 200
"""
import argparse
import os
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--packed_dir", required=True)
    p.add_argument("--seq_len", type=int, default=42,
                   help="T_in+T_out, rows read per random chunk (matches one sample)")
    p.add_argument("--n_reads", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    frames_path = os.path.join(args.packed_dir, "frames.dat")
    size_bytes = os.path.getsize(frames_path)
    size_gb = size_bytes / 1e9
    print(f"frames.dat size: {size_gb:.2f} GB")

    import json
    with open(os.path.join(args.packed_dir, "meta.json")) as f:
        meta = json.load(f)
    N, C, (h, w) = meta["N"], len(meta["channels"]), meta["img_size"]
    row_bytes = C * h * w
    print(f"N={N} rows, {C} channels, {h}x{w}, {row_bytes} bytes/row\n")

    # ---- Sequential read (raw file, no numpy/memmap machinery at all) ----
    print("--- Sequential read (raw file.read(), no memmap) ---")
    chunk = row_bytes * args.seq_len
    n_chunks = min(50, N // args.seq_len)
    t0 = time.time()
    with open(frames_path, "rb") as f:
        for _ in range(n_chunks):
            data = f.read(chunk)
    t1 = time.time()
    mb = n_chunks * chunk / 1e6
    print(f"  read {mb:.1f} MB sequentially in {t1-t0:.2f}s "
          f"-> {mb/(t1-t0):.0f} MB/s")

    # ---- Random access read (raw file.seek()+read(), matching one sample) ----
    print(f"\n--- Random access read (raw file.seek()+read(), "
          f"{args.seq_len}-row chunks) ---")
    rng = np.random.default_rng(0)
    max_start = N - args.seq_len
    starts = rng.integers(0, max_start, args.n_reads)
    t0 = time.time()
    with open(frames_path, "rb") as f:
        for s in starts:
            f.seek(int(s) * row_bytes)
            data = f.read(chunk)
    t1 = time.time()
    mb = args.n_reads * chunk / 1e6
    print(f"  read {mb:.1f} MB via {args.n_reads} random seeks in {t1-t0:.2f}s "
          f"-> {mb/(t1-t0):.0f} MB/s  ({(t1-t0)/args.n_reads*1000:.1f} ms/read)")

    print(f"\n=== READ THIS: ===")
    print(f"  If random-access throughput is CLOSE to sequential, your")
    print(f"  storage handles random reads well -- the bottleneck is")
    print(f"  elsewhere (Python/numpy overhead, DataLoader worker startup,")
    print(f"  something else) and further investigation is worthwhile.")
    print(f"  If random-access throughput is MUCH lower than sequential,")
    print(f"  that's a genuine storage/hardware characteristic (seek-heavy")
    print(f"  random I/O is slow on this device) that no Python-level code")
    print(f"  change in this project can fix -- the fix would need to be")
    print(f"  architectural (more RAM so data stays cached, faster storage,")
    print(f"  or a fundamentally different access pattern such as truly")
    print(f"  sequential epoch order instead of weighted random sampling).")


if __name__ == "__main__":
    main()
