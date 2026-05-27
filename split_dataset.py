"""
Chronological train/test split for METSAT lightning nowcasting dataset.

Scientific basis:
  - Data spans Oct 2024 – Jan 2026 (15 months, 4 regions).
  - Split date: 2025-11-01 00:00:00 UTC
    → Train: Oct 2024 – Oct 2025 (12 months, ~80%)
    → Test:  Nov 2025 – Jan 2026  ( 3 months, ~20%)
  - This gives one full annual cycle in training, and tests on the
    same OND convective season one year later (reproducible, season-matched).
  - A minimum gap of T_in+T_out = 24 frames × 10min = 4 hours between the
    last training frame and first test frame is guaranteed by the month boundary.
  - Consistent with MetNet (Sønderby et al. 2020), DGMR (Ravuri et al. 2021),
    and NowcastNet (Zhang et al. 2023) which all use fixed date cutoffs.

Usage:
  python split_dataset.py \
      --src_root /home/vladlanda/Workplace/LI-DATASETS/full \
      --dst_root /home/vladlanda/Workplace/LI-DATASETS/split \
      --split_date 2025-11-01 \
      --regions central_africa_1 central_africa_2 central_africa_3 central_africa_4 \
      --dry_run   # remove --dry_run to actually copy

Output structure:
  split/
    train/
      central_africa_1/  ...
      central_africa_2/  ...
      central_africa_3/  ...
      central_africa_4/  ...
    test/
      central_africa_1/  ...
      central_africa_2/  ...
      central_africa_3/  ...
      central_africa_4/  ...
"""

import argparse
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Filename pattern: {id}_{start}_{end}_{channel}.{ext}
# e.g. 53553_20251105T003006Z_20251105T003923Z_ir.jpg
# or   48474_20250923T182003Z_20250923T182935Z.wld  (no channel)
FNAME_RE = re.compile(
    r"^\d+_(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)(?:_\w+)?\.\w+$"
)

def parse_start_time(fname: str) -> datetime | None:
    m = FNAME_RE.match(fname)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )


def split_region(
    src_dir:    Path,
    train_dir:  Path,
    test_dir:   Path,
    split_dt:   datetime,
    dry_run:    bool,
    verbose:    bool,
) -> dict:
    files = list(src_dir.iterdir())
    stats = defaultdict(int)

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        t = parse_start_time(f.name)
        if t is None:
            # unrecognised filename — copy to train by default (e.g. .wld)
            dst = train_dir / f.name
            stats["unrecognised"] += 1
        elif t < split_dt:
            dst = train_dir / f.name
            stats["train"] += 1
        else:
            dst = test_dir / f.name
            stats["test"] += 1

        if not dry_run:
            shutil.copy2(f, dst)
        elif verbose:
            print(f"  {'DRY':4s}  {f.name}  →  {'train' if dst.parent == train_dir else 'test'}")

    return dict(stats)


def main():
    p = argparse.ArgumentParser(
        description="Chronological train/test split for METSAT dataset."
    )
    p.add_argument("--src_root",   required=True,
                   help="Root directory containing region folders")
    p.add_argument("--dst_root",   required=True,
                   help="Output root (will contain train/ and test/ subdirs)")
    p.add_argument("--split_date", default="2025-11-01",
                   help="Split date (YYYY-MM-DD). Files before this → train, "
                        "on or after → test. Default: 2025-11-01")
    p.add_argument("--regions",    nargs="+",
                   default=["central_africa_1", "central_africa_2",
                            "central_africa_3", "central_africa_4"],
                   help="Region folder names to process")
    p.add_argument("--dry_run",    action="store_true",
                   help="Print what would happen without copying files")
    p.add_argument("--verbose",    action="store_true",
                   help="Print each file decision (only useful with --dry_run "
                        "on a small sample)")
    args = p.parse_args()

    split_dt = datetime.strptime(args.split_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    print("=" * 60)
    print("METSAT chronological train/test split")
    print("=" * 60)
    print(f"  Source root : {src_root}")
    print(f"  Dest root   : {dst_root}")
    print(f"  Split date  : {split_dt.date()}  (files before → train, on/after → test)")
    print(f"  Regions     : {args.regions}")
    print(f"  Dry run     : {args.dry_run}")
    print()

    total_train = total_test = 0
    for region in args.regions:
        src_dir   = src_root / region
        train_dir = dst_root / "train" / region
        test_dir  = dst_root / "test"  / region

        if not src_dir.exists():
            print(f"  WARNING: {src_dir} does not exist — skipping")
            continue

        print(f"Processing {region} ...")
        stats = split_region(src_dir, train_dir, test_dir,
                             split_dt, args.dry_run, args.verbose)

        n_train = stats.get("train", 0)
        n_test  = stats.get("test",  0)
        n_unk   = stats.get("unrecognised", 0)
        total   = n_train + n_test + n_unk

        frac = n_test / max(total, 1) * 100
        print(f"  train: {n_train:7,}  test: {n_test:7,}  "
              f"unrecognised: {n_unk:5,}  "
              f"test fraction: {frac:.1f}%")

        # Estimate date range
        all_times = []
        for f in src_dir.iterdir():
            t = parse_start_time(f.name)
            if t:
                all_times.append(t)
        if all_times:
            print(f"  Date range: {min(all_times).date()} → {max(all_times).date()}")

        total_train += n_train
        total_test  += n_test
        print()

    print("=" * 60)
    print(f"TOTAL  train: {total_train:,}  test: {total_test:,}  "
          f"test fraction: {total_test / max(total_train+total_test, 1)*100:.1f}%")
    if args.dry_run:
        print()
        print("DRY RUN — no files were copied.")
        print("Remove --dry_run to execute the split.")
    print("=" * 60)


if __name__ == "__main__":
    main()
