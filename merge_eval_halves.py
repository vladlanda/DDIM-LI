"""
Merge two evaluation runs of the SAME model on complementary test subsets
(TEST_SEQ_EVERY=2 with TEST_SEQ_OFFSET=0 and =1, e.g. one per GPU) into one result.

    python merge_eval_halves.py <dir_offset0> <dir_offset1> <out_dir>

Merged sequence order = all sequences of the first run, then all of the second; their
sequence ids are shifted by the first run's sequence count. Merge EVERY model (and
build_test_metadata.py's output) with the same argument order, so the paired bootstrap
and analysis_regimes.py see identical sequence ids across models.

  *.npz   per-sample / per-sequence arrays (pr_*, cal_*, label_*, dist_*, seq_*) are
          concatenated exactly; ids shifted. Keys identical in both runs are kept; other
          summary keys (PR curves, per_step_json, FSS arrays) cannot be merged and are
          dropped -- recompute them from the merged samples.
  area_*.csv  recomputed exactly from the merged area_*.npz.
  other *.csv  numeric columns = sequence-count-weighted mean of the two runs: exact for
          means (CRPS, MAE, ...), a close approximation for pooled ratio scores (FSS,
          CSI/POD/FAR, spread/skill, PR-AUC). Headline AP/Brier come from the merged
          npz via bootstrap_pr_auc_ci.py / analysis_regimes.py, which are exact.
"""
import csv
import glob
import os
import re
import sys

import numpy as np

CONCAT = re.compile(r"^(pr_prob|pr_label|cal_prob|cal_label|label)_\d+$")
PER_SAMPLE = {"dist_recent_px", "dist_all_px", "seq_region", "seq_time_utc"}
IS_ID = lambda k: k == "seqid" or k.startswith("pr_seqid_")


def n_sequences(z):
    if "n_sequences" in z:
        return int(z["n_sequences"])
    ids = [z[k] for k in z if IS_ID(k) and z[k].size]
    return int(max(i.max() for i in ids)) + 1 if ids else 0


def merge_npz(a_path, b_path, out_path):
    a, b = dict(np.load(a_path, allow_pickle=True)), dict(np.load(b_path, allow_pickle=True))
    off = n_sequences(a)
    out, dropped = {}, []
    for k in a:
        if k not in b:
            dropped.append(k)
        elif IS_ID(k):
            out[k] = np.concatenate([a[k], b[k] + off]).astype(a[k].dtype)
        elif CONCAT.match(k) or k in PER_SAMPLE:
            out[k] = np.concatenate([a[k], b[k]])
        elif k == "n_sequences":
            out[k] = np.array(off + n_sequences(b))
        elif a[k].shape == b[k].shape and np.array_equal(a[k], b[k]):
            out[k] = a[k]
        else:
            dropped.append(k)
    np.savez_compressed(out_path, **out)
    return off, n_sequences(b), dropped


def merge_csv(a_path, b_path, out_path, na, nb):
    ra, rb = list(csv.DictReader(open(a_path))), list(csv.DictReader(open(b_path)))
    assert len(ra) == len(rb), f"{a_path}: {len(ra)} vs {len(rb)} rows"
    rows = []
    for x, y in zip(ra, rb):
        r = {}
        for k, v in x.items():
            try:
                fx, fy = float(v), float(y[k])
                r[k] = (na * fx + nb * fy) / (na + nb) if np.isfinite(fx) and np.isfinite(fy) else (
                    fx if np.isfinite(fx) else fy)
            except (TypeError, ValueError):
                r[k] = v                       # text column (model name, ...): keep
        rows.append(r)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ra[0])); w.writeheader(); w.writerows(rows)


def main(dir_a, dir_b, out_dir):
    from evaluate import save_area_outputs
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for pa in sorted(glob.glob(os.path.join(dir_a, "*.npz"))):
        name = os.path.basename(pa)
        pb = os.path.join(dir_b, name)
        if not os.path.exists(pb):
            print(f"  skip {name}: not in {dir_b}"); continue
        na, nb, dropped = merge_npz(pa, pb, os.path.join(out_dir, name))
        counts[name] = (na, nb)
        print(f"  {name}: {na} + {nb} sequences" + (f"  (dropped summaries: {', '.join(dropped)})" if dropped else ""))
    if len({v for v in counts.values() if v[0]}) > 1:
        print(f"  WARNING: files disagree on sequence counts {set(counts.values())} -- were both runs complete?")
    na, nb = max(counts.values(), key=lambda v: v[0] + v[1]) if counts else (1, 1)

    # area CSVs: recompute exactly from the merged samples (also rewrites their npz identically)
    areas = {}
    for p in glob.glob(os.path.join(out_dir, "area_*_r*px.npz")):
        m = re.match(r"area_(.+)_r(\d+)px\.npz$", os.path.basename(p))
        z = np.load(p)
        areas.setdefault(m.group(1), {})[int(m.group(2))] = {
            int(t): ([z[f"pr_prob_{t}"]], [z[f"pr_label_{t}"]], [z[f"pr_seqid_{t}"]]) for t in z["pr_steps"]}
    for name, store in areas.items():
        save_area_outputs(out_dir, name, store)
        print(f"  area_{name}.csv: recomputed from merged samples")

    for pa in sorted(glob.glob(os.path.join(dir_a, "*.csv"))):
        name = os.path.basename(pa)
        pb = os.path.join(dir_b, name)
        if name.startswith("area_") or not os.path.exists(pb):
            continue
        merge_csv(pa, pb, os.path.join(out_dir, name), na, nb)
        print(f"  {name}: sequence-weighted ({na}:{nb})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
