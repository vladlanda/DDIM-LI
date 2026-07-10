"""
Audit per-channel, per-month coverage AND degeneracy in the raw dataset.

Why: a channel file can EXIST but be all-black (~1.7 kB JPEG). build_index()
uses Path.exists(), so such a file is marked VALID and the model trains on a
constant plane labelled as real data.

Physical asymmetry exploited here:
  - an all-black IR frame is IMPOSSIBLE (Earth always emits in the IR)
      -> black ir* frame = missing/corrupt
  - an all-black LI frame is LEGITIMATE (no flashes in the window)
      -> never flag LI as degenerate

Strategy: stat() every file (cheap). Files under --size_bytes are candidates.
Open only a sample of them to confirm max==0, so we don't decode 1.4M JPEGs.

Usage:
  python audit_channel_coverage.py --root /path/to/unsplit \
      --channels ir li ch0 ch1 ch2 --size_bytes 3000
"""
import argparse, re, random
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

TS = re.compile(r"_(\d{8})T\d{6}Z_")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--channels", nargs="+",
                   default=["ir", "li", "ch0", "ch1", "ch2"])
    p.add_argument("--size_bytes", type=int, default=3000,
                   help="Files smaller than this are degeneracy candidates.")
    p.add_argument("--verify", type=int, default=40,
                   help="How many small files to actually open per channel.")
    args = p.parse_args()

    root = Path(args.root)
    regions = sorted([d for d in root.rglob("central_africa_*") if d.is_dir()])
    if not regions:
        regions = [root]

    for region in regions:
        # counts[(month, ch)] = [n_total, n_small]
        counts = defaultdict(lambda: [0, 0])
        small_paths = defaultdict(list)

        for f in region.glob("*.jpg"):
            m = TS.search(f.name)
            if not m:
                continue
            ch = f.stem.rsplit("_", 1)[-1]
            if ch not in args.channels:
                continue
            month = m.group(1)[:6]
            sz = f.stat().st_size
            counts[(month, ch)][0] += 1
            if sz < args.size_bytes:
                counts[(month, ch)][1] += 1
                small_paths[ch].append(f)

        months = sorted({k[0] for k in counts})
        print(f"\n=== {region.name} ===")
        hdr = f"{'month':>8}" + "".join(f"{c:>16}" for c in args.channels)
        print(hdr); print("-" * len(hdr))
        for mo in months:
            row = f"{mo:>8}"
            for ch in args.channels:
                n, nb = counts[(mo, ch)]
                # Never flag LI: an all-zero LI frame means "no flashes", which
                # is valid data. Only ir* channels can be physically all-black.
                if ch == "li" or nb == 0:
                    cell = f"{n}"
                else:
                    cell = f"{n} ({nb} blk)"
                row += f"{cell:>16}"
            print(row)

        # Verify that 'small' really means all-black, per channel
        print("\n  verification (opening a sample of small files):")
        for ch in args.channels:
            paths = small_paths[ch]
            if not paths:
                print(f"    {ch:>6}: no small files")
                continue
            sample = random.sample(paths, min(args.verify, len(paths)))
            allzero = sum(int(np.array(Image.open(q).convert("L")).max() == 0)
                          for q in sample)
            note = ""
            if ch == "li":
                note = "  <- LEGITIMATE for LI (no flashes)"
            print(f"    {ch:>6}: {allzero}/{len(sample)} sampled are all-zero{note}")

    print("\nREAD: for ir* channels, 'blk' counts are MISSING DATA masquerading")
    print("as present. Find the last month where ir123/ch2 has ~0 blk -> that is")
    print("your usable end date. Sporadic blk => must mask per-sample, not truncate.")

if __name__ == "__main__":
    main()
