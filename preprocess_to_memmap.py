"""
Pack a region's raw JPEG frames into a single memory-mapped binary array.

WHY: profiling showed data loading dominates training time even with
tuned num_workers on fast internal storage (29.4 min/epoch at best,
regressing at higher worker counts -- the signature of I/O contention,
not raw bandwidth limits). Root cause: __getitem__ opens+decodes
T_in+T_out (e.g. 42) timesteps x C channels = ~84 individual small JPEG
files PER TRAINING SAMPLE, sequentially, within one worker process. This
is inherently expensive regardless of drive speed -- every file open pays
OS/filesystem overhead, every JPEG pays decode cost, multiplied by every
sample, every epoch.

FIX: pack all of a region's valid frames into ONE contiguous, pre-decoded,
pre-resized uint8 array on disk (numpy memmap). Reading from a memmap is
just an OS page-cache-backed array slice -- no file-open syscall, no JPEG
decode, and critically, a memmap is automatically SHARED across all
DataLoader worker processes by the OS (unlike an in-process cache, which
would be redundantly duplicated per worker).

This does NOT modify dataset.py or the existing pipeline in any way --
it's a fully separate, parallel path. The existing JPEG-based loader
remains available and unchanged.

Output (written to <root>/_packed/):
  frames.dat   -- memmap, shape (N, C, H, W), dtype uint8, raw pixel values
  mask.dat     -- memmap, shape (N, C), dtype uint8, 1=channel present
  meta.json    -- channel_list, img_size, N, sorted ISO-format timestamps
                  (position in this list = row index into frames.dat/mask.dat)

Usage:
  python preprocess_to_memmap.py --root /path/to/central_africa_1 \
      --channels ir li --img_size 256 256
"""
import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import build_index, CHANNEL_FILL_VALUES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--channels", nargs="+", required=True,
                   help="Channel list to pack, e.g. ir li")
    p.add_argument("--img_size", nargs=2, type=int, default=[256, 256])
    p.add_argument("--out_dir", default=None,
                   help="Default: <root>/_packed")
    return p.parse_args()


def main():
    args = parse_args()
    h, w = args.img_size
    C = len(args.channels)
    out_dir = Path(args.out_dir or (Path(args.root) / "_packed"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the EXISTING, validated index-building logic -- same validity
    # rules (degeneracy check etc.) as the JPEG-based pipeline, so the set
    # of usable timesteps is identical between the two loading paths.
    index = build_index(args.root)
    sorted_times = sorted(index.keys())
    N = len(sorted_times)
    logger.info(f"Packing {N} timesteps x {C} channels from {args.root}")
    if N == 0:
        raise RuntimeError(f"No valid timesteps found in {args.root} -- "
                           f"nothing to pack.")

    frames_path = out_dir / "frames.dat"
    mask_path   = out_dir / "mask.dat"
    frames = np.memmap(frames_path, dtype=np.uint8, mode="w+", shape=(N, C, h, w))
    mask   = np.memmap(mask_path,   dtype=np.uint8, mode="w+", shape=(N, C))

    for row, dt in enumerate(tqdm(sorted_times, desc=f"Packing {Path(args.root).name}",
                                  unit="frame", dynamic_ncols=True)):
        chs = index[dt]
        for c, ch in enumerate(args.channels):
            if ch not in chs:
                # Store a placeholder -- the VALUE here is irrelevant. The
                # loader must special-case mask==0 by directly using
                # CHANNEL_FILL_VALUES and SKIPPING normalisation entirely,
                # exactly matching _load_frame's behaviour (which does NOT
                # apply cbrt/z-score to missing channels -- see the
                # 'if mask[i]==0.0: continue' guard in dataset.py). Trying
                # to bake a normalised fill value into this uint8 array
                # would be WRONG for missing channels regardless of what
                # value is stored, since the correct output skips
                # normalisation altogether, not just approximates it.
                frames[row, c] = 0
                mask[row, c] = 0
                continue
            img = Image.open(chs[ch]).convert("L")
            if img.size != (w, h):
                resize_method = Image.NEAREST if ch == "li" else Image.BILINEAR
                img = img.resize((w, h), resize_method)
            arr = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape(h, w)
            frames[row, c] = arr
            mask[row, c] = 1

    frames.flush()
    mask.flush()

    meta = {
        "channels": args.channels,
        "img_size": [h, w],
        "N": N,
        "timestamps": [t.isoformat() for t in sorted_times],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    size_gb = (frames_path.stat().st_size + mask_path.stat().st_size) / 1e9
    logger.info(f"Packed {N} timesteps -> {out_dir}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
