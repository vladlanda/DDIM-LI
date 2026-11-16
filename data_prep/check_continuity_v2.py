"""Daily IR / LI statistics across the test -> test2 boundary and the 2026 grid change.
A processing change (download tool, grey-value scaling, grid) shows up as a STEP in these
series on the change date; weather changes are gradual and differ between regions.

usage: python data_prep/check_continuity_v2.py <test root> <test2 root> [out.png]"""
import re, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

roots, out = [Path(a) for a in sys.argv[1:3]], (sys.argv[3] if len(sys.argv) > 3 else "continuity_v2.png")
TS = re.compile(r"^\d+_(\d{8})T(\d{2})(\d{2})\d{2}Z_\d{8}T\d{6}Z_ir\.png$")
regions = sorted({p.name for r in roots for p in r.iterdir() if p.is_dir()})
fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
grid_change = None
for region in regions:
    day = defaultdict(list)
    for root in roots:
        for f in (root / region).glob("*_ir.png"):
            m = TS.match(f.name)
            if m and m.group(3)[0] == "0":            # one frame per hour (minute 00-09)
                day[m.group(1)].append(f)
    rows = []
    for d in sorted(day):
        ir = np.concatenate([np.asarray(Image.open(f).convert("L")).ravel() for f in day[d]])
        li = np.concatenate([np.asarray(Image.open(str(f)[:-7] + "_li.png").convert("L")).ravel() for f in day[d]])
        wld = Path(str(day[d][0])[:-7] + ".wld")
        xs = float(wld.read_text().split()[0]) if wld.exists() else np.nan
        rows.append((datetime.strptime(d, "%Y%m%d"), *np.percentile(ir, [10, 50]), (li > 0).mean(), xs))
    t, p10, p50, lf, xs = map(np.array, zip(*rows))
    ax[0].plot(t, p50, lw=0.8, label=region); ax[1].plot(t, p10, lw=0.8); ax[2].plot(t, lf, lw=0.8)
    ch = np.flatnonzero(np.abs(np.diff(xs)) > 1e-6)
    if ch.size:
        grid_change = t[ch[0] + 1]
    print(f"{region}: {len(t)} days {t[0]:%Y-%m-%d}..{t[-1]:%Y-%m-%d}; pixel sizes {sorted(set(np.round(xs, 7).tolist()))}; "
          f"grid change {t[ch[0] + 1]:%Y-%m-%d}" if ch.size else f"{region}: {len(t)} days; one grid {xs[0]:.7f}")
for a, lab in zip(ax, ["IR daily median (grey)", "IR daily 10th pct (cold clouds)", "LI lightning-pixel fraction"]):
    a.set_ylabel(lab); a.axvline(datetime(2026, 1, 1), color="k", ls="--", lw=0.8)
    if grid_change is not None:
        a.axvline(grid_change, color="r", ls=":", lw=1)
ax[0].legend(ncol=4, fontsize=8); ax[0].set_title("black dashed: test -> test2; red dotted: IR grid change")
fig.tight_layout(); fig.savefig(out, dpi=120)
print("figure ->", out)
