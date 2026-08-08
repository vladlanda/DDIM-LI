"""
Test the "ensemble coverage inflation" hypothesis from FINDINGS.md F4.

The FSS-vs-scale comparison (compare_diffusion_vs_cnn_fss.py) showed the
CNN baseline's advantage is large and PERFECTLY FLAT across spatial scale
at p>0.1, but shrinks and reverses at p>0.5. A flat-with-scale gap points
away from a positional/double-penalty explanation (which should shrink
with spatial tolerance at every threshold) and toward a coverage/
calibration explanation instead.

Mechanism under test: with n_members=10, the diffusion model's threshold
check at p>0.1 is exactly "at least 1 of 10 ensemble members predicted
lightning here" -- a generous, union-like criterion that could cover a
much larger area than a well-calibrated probability map, independent of
WHERE that area is. p>0.5 ("at least 5 of 10", majority vote) is a much
stricter criterion, much less prone to this inflation.

This script tests it directly, with data that already exists (no new
diffusion sampling run needed): both plot_data.npz and
baseline_cnn_pr_curves.npz already store the raw per-pixel pr_prob_{t}/
pr_label_{t} arrays used to build the PR curves (same schema, confirmed
via bootstrap_pr_auc_ci.py's load_run()). For each threshold, this
computes predicted positive-area fraction (mean(prob >= thr)) against
true positive-area fraction (mean(label)) for both models. If the
diffusion model's predicted fraction is far above the true fraction at
p>0.1 specifically (while the CNN's is closer to true), that confirms
coverage inflation as the mechanism.

Usage:
  python check_area_fraction_bias.py \
      --diffusion_npz outputs/nature_256_T36_ir_li_only/eval/plot_data.npz \
      --cnn_npz baseline_cnn/baseline_cnn_pr_curves.npz
"""
import argparse

import numpy as np


def load_run(npz_path):
    """Same schema/logic as bootstrap_pr_auc_ci.py's load_run()."""
    data = np.load(npz_path, allow_pickle=True)
    steps = sorted(int(t) for t in data["pr_steps"].tolist())
    out = {}
    for t in steps:
        pk, lk = f"pr_prob_{t}", f"pr_label_{t}"
        if pk in data and lk in data:
            out[t] = (data[pk], data[lk].astype(int))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diffusion_npz", required=True)
    p.add_argument("--cnn_npz", required=True)
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    args = p.parse_args()

    diff_run = load_run(args.diffusion_npz)
    cnn_run  = load_run(args.cnn_npz)
    steps = sorted(set(diff_run) & set(cnn_run))
    if not steps:
        raise ValueError("No overlapping lead steps between the two npz files")

    print(f"\n{'thr':>5} {'true area%':>11} "
         f"{'diff pred area%':>17} {'diff/true ratio':>16} "
         f"{'cnn pred area%':>16} {'cnn/true ratio':>15}")
    print("-" * 92)
    for thr in args.thresholds:
        true_fracs, diff_fracs, cnn_fracs = [], [], []
        for t in steps:
            d_prob, d_lbl = diff_run[t]
            c_prob, c_lbl = cnn_run[t]
            true_fracs.append(float(np.mean(d_lbl)))   # same obs both sides at same t
            diff_fracs.append(float(np.mean(d_prob >= thr)))
            cnn_fracs.append(float(np.mean(c_prob >= thr)))
        true_mean = float(np.mean(true_fracs))
        diff_mean = float(np.mean(diff_fracs))
        cnn_mean  = float(np.mean(cnn_fracs))
        diff_ratio = diff_mean / true_mean if true_mean > 0 else float("nan")
        cnn_ratio  = cnn_mean / true_mean if true_mean > 0 else float("nan")
        print(f"{thr:>5} {true_mean*100:>10.3f}% "
             f"{diff_mean*100:>16.3f}% {diff_ratio:>16.2f}x "
             f"{cnn_mean*100:>15.3f}% {cnn_ratio:>15.2f}x")

    print("\nRatio near 1.0x = well-calibrated coverage at that threshold.")
    print("Ratio >> 1.0x = over-covers (predicts positive over a much larger")
    print("area than truly occurs) -- consistent with ensemble union inflation")
    print("if this is much worse for diffusion than CNN specifically at p=0.1.")


if __name__ == "__main__":
    main()
