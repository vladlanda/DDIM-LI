"""Compare the SAME timestamps in the old test set and the new test2 download.
usage: python data_prep/check_overlap_v2.py <old test root> <test2 root>"""
import re, sys
from pathlib import Path
import numpy as np
from PIL import Image

old_root, new_root = Path(sys.argv[1]), Path(sys.argv[2])
TS = re.compile(r"_(\d{8}T\d{6}Z)_\d{8}T\d{6}Z_(ir|li)\.png$")
load = lambda p: np.asarray(Image.open(p).convert("L"), dtype=np.float32)

def files(d):
    out = {}
    for f in d.glob("*.png"):
        m = TS.search(f.name)
        if m:
            out[(m.group(1)[:13], m.group(2))] = f   # key on start time to the minute
    return out

for region in sorted(p.name for p in new_root.iterdir() if p.is_dir()):
    o, n = files(old_root / region), files(new_root / region)
    common = sorted(k for k in o if k in n)
    ir = [k for k in common if k[1] == "ir"][::20]          # every 20th frame is plenty
    li = [k for k in common if k[1] == "li"]
    if not ir:
        print(f"{region}: no overlapping timestamps"); continue
    a = np.stack([load(o[k]) for k in ir]); b = np.stack([load(n[k]) for k in ir])
    q = [1, 10, 50, 90, 99]
    print(f"\n{region}: {len(li)} overlapping frames ({li[0][0]} .. {li[-1][0]}), IR checked on {len(ir)}")
    print(f"  IR mean old {a.mean():.1f} new {b.mean():.1f} | bias {(b - a).mean():+.2f} | "
          f"median |diff| {np.median(np.abs(b - a)):.1f} | corr {np.corrcoef(a.ravel(), b.ravel())[0, 1]:.4f}")
    print(f"  IR quantiles {q}: old {np.percentile(a, q).round(0)} new {np.percentile(b, q).round(0)}")
    lo = sum((load(o[k]) > 0).sum() for k in li); ln = sum((load(n[k]) > 0).sum() for k in li)
    print(f"  LI lightning pixels: old {lo} new {ln} (ratio {ln / max(lo, 1):.3f})")
