"""
Overlay model PR curves (solid) with persistence PR curves (dashed)
on a single plot, color-matched by lead time.

Requires:
  - model plot data:        <model_output_dir>/plot_data.npz   (from evaluate.py)
  - persistence PR curves:  <persist_output_dir>/persistence_pr_curves.npz
                            (from persistence_baseline.py)

Usage:
  python plot_pr_comparison.py \
      --model_npz   outputs/nonelbo_v6_256_T36_LIctx/plot_data.npz \
      --persist_npz outputs/persistence_baseline/persistence_pr_curves.npz \
      --out         pr_comparison.png
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


def load_model_pr(npz_path):
    """Extract per-step PR curves from the model's plot_data.npz."""
    data = np.load(npz_path, allow_pickle=True)
    pr_steps = data["pr_steps"].tolist()
    dt_min   = int(data["dt_min"]) if "dt_min" in data else 10
    curves = {}
    for t in pr_steps:
        # plot_data.npz stores raw probs/labels as pr_prob_{t} / pr_label_{t}
        if f"pr_prob_{t}" in data and f"pr_label_{t}" in data:
            prob = data[f"pr_prob_{t}"]
            lbl  = data[f"pr_label_{t}"]
            prec, rec, _ = precision_recall_curve(lbl, prob)
            curves[t] = (prec, rec, float(auc(rec, prec)))
    return curves, dt_min


def load_persist_pr(npz_path):
    """Extract persistence PR curves saved by persistence_baseline.py."""
    data = np.load(npz_path, allow_pickle=True)
    pr_steps = data["pr_steps"].tolist()
    dt_min   = int(data["dt_min"]) if "dt_min" in data else 10
    curves = {}
    for t in pr_steps:
        if f"prec_{t}" in data and f"rec_{t}" in data:
            prec = data[f"prec_{t}"]
            rec  = data[f"rec_{t}"]
            a    = float(data[f"auc_{t}"]) if f"auc_{t}" in data else float("nan")
            curves[t] = (prec, rec, a)
    return curves, dt_min


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_npz",   required=True)
    p.add_argument("--persist_npz", required=True)
    p.add_argument("--out", default="pr_comparison.png")
    args = p.parse_args()

    model_curves,   dt_min  = load_model_pr(args.model_npz)
    persist_curves, _       = load_persist_pr(args.persist_npz)

    steps  = sorted(model_curves.keys())
    # Color map matched to lead time (same scheme as the existing plot:
    # dark purple → dark red across lead times)
    cmap   = plt.cm.turbo
    colors = {t: cmap(0.12 + 0.76 * i / max(len(steps)-1, 1))
              for i, t in enumerate(steps)}

    fig, ax = plt.subplots(figsize=(8, 7))

    for t in steps:
        lead = (t + 1) * dt_min
        c    = colors[t]

        # Model — solid line
        prec_m, rec_m, auc_m = model_curves[t]
        ax.plot(rec_m, prec_m, color=c, lw=2.0, linestyle="-",
                label=f"+{lead}m model  AUC={auc_m:.2f}")

        # Persistence — dashed line, same color
        if t in persist_curves:
            prec_p, rec_p, auc_p = persist_curves[t]
            ax.plot(rec_p, prec_p, color=c, lw=1.6, linestyle="--",
                    alpha=0.75,
                    label=f"+{lead}m persist  AUC={auc_p:.2f}")

    ax.set_xlabel("Recall (POD)", fontsize=12)
    ax.set_ylabel("Precision (1 − FAR)", fontsize=12)
    ax.set_title("Precision–Recall: Model (solid) vs Persistence (dashed)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
