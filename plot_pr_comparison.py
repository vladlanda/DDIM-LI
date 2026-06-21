"""
Two-panel comparison figure:
  LEFT  — Precision-Recall curves        (model solid, persistence dashed)
  RIGHT — Reliability / calibration diagram (model solid, persistence dashed)

Both panels color-matched by lead time. All axes, ticks, labels and legend
text are bold.

Requires:
  - model plot data:        <model_output_dir>/plot_data.npz
  - persistence PR curves:  <persist_output_dir>/persistence_pr_curves.npz

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
from matplotlib import rcParams
from sklearn.metrics import precision_recall_curve, auc
from sklearn.calibration import calibration_curve


# ── Global bold styling ───────────────────────────────────────────────
rcParams["font.weight"]        = "bold"
rcParams["axes.labelweight"]   = "bold"
rcParams["axes.titleweight"]   = "bold"
rcParams["axes.linewidth"]     = 2.0
rcParams["xtick.major.width"]  = 2.0
rcParams["ytick.major.width"]  = 2.0


def _bold_ticks(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def load_model(npz_path):
    """Extract per-step PR + calibration data from model plot_data.npz."""
    data = np.load(npz_path, allow_pickle=True)
    pr_steps = data["pr_steps"].tolist()
    dt_min   = int(data["dt_min"]) if "dt_min" in data else 10
    pr, cal = {}, {}
    for t in pr_steps:
        if f"pr_prob_{t}" in data and f"pr_label_{t}" in data:
            prob = data[f"pr_prob_{t}"]
            lbl  = data[f"pr_label_{t}"]
            prec, rec, _ = precision_recall_curve(lbl, prob)
            pr[t] = (prec, rec, float(auc(rec, prec)))
        if f"cal_prob_{t}" in data and f"cal_label_{t}" in data:
            cprob = data[f"cal_prob_{t}"]
            clbl  = data[f"cal_label_{t}"]
            try:
                frac_pos, mean_pred = calibration_curve(
                    clbl, cprob, n_bins=10, strategy="uniform")
                cal[t] = (mean_pred, frac_pos)
            except Exception:
                pass
    return pr, cal, dt_min


def load_persist(npz_path):
    """Extract persistence PR (and calibration if present)."""
    data = np.load(npz_path, allow_pickle=True)
    pr_steps = data["pr_steps"].tolist()
    dt_min   = int(data["dt_min"]) if "dt_min" in data else 10
    pr, cal = {}, {}
    for t in pr_steps:
        if f"prec_{t}" in data and f"rec_{t}" in data:
            a = float(data[f"auc_{t}"]) if f"auc_{t}" in data else float("nan")
            pr[t] = (data[f"prec_{t}"], data[f"rec_{t}"], a)
        if f"cal_mean_{t}" in data and f"cal_frac_{t}" in data:
            cal[t] = (data[f"cal_mean_{t}"], data[f"cal_frac_{t}"])
    return pr, cal, dt_min


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_npz",   required=True)
    p.add_argument("--persist_npz", required=True)
    p.add_argument("--out", default="pr_comparison.png")
    args = p.parse_args()

    m_pr, m_cal, dt_min = load_model(args.model_npz)
    p_pr, p_cal, _      = load_persist(args.persist_npz)

    steps  = sorted(m_pr.keys())
    cmap   = plt.cm.turbo
    colors = {t: cmap(0.12 + 0.76 * i / max(len(steps)-1, 1))
              for i, t in enumerate(steps)}

    fig, (ax_pr, ax_cal) = plt.subplots(1, 2, figsize=(15, 7))

    # ── LEFT: Reliability / calibration diagram ──────────────────────
    ax_cal.plot([0, 1], [0, 1], "k--", lw=2.0, label="Perfect calibration")
    for t in steps:
        lead = (t + 1) * dt_min
        c    = colors[t]
        if t in m_cal:
            mean_pred, frac_pos = m_cal[t]
            ax_cal.plot(mean_pred, frac_pos, color=c, lw=2.4, marker="o",
                        markersize=5, linestyle="-",
                        label=f"+{lead}m model")
        if t in p_cal:
            mp, fp = p_cal[t]
            ax_cal.plot(mp, fp, color=c, lw=1.8, marker="s",
                        markersize=4, linestyle="--", alpha=0.7,
                        label=f"+{lead}m persist")

    ax_cal.set_xlabel("Mean Predicted Probability", fontsize=13, fontweight="bold")
    ax_cal.set_ylabel("Observed Frequency", fontsize=13, fontweight="bold")
    ax_cal.set_title("Reliability Diagram", fontsize=15, fontweight="bold")
    ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
    ax_cal.grid(True, alpha=0.3)
    leg1 = ax_cal.legend(fontsize=9, loc="upper left", framealpha=0.9)
    for txt in leg1.get_texts():
        txt.set_fontweight("bold")
    _bold_ticks(ax_cal)

    # ── RIGHT: Precision-Recall curves ───────────────────────────────
    for t in steps:
        lead = (t + 1) * dt_min
        c    = colors[t]
        prec_m, rec_m, auc_m = m_pr[t]
        ax_pr.plot(rec_m, prec_m, color=c, lw=2.4, linestyle="-",
                   label=f"+{lead}m model  AUC={auc_m:.2f}")
        if t in p_pr:
            prec_p, rec_p, auc_p = p_pr[t]
            ax_pr.plot(rec_p, prec_p, color=c, lw=1.8, linestyle="--",
                       alpha=0.7,
                       label=f"+{lead}m persist  AUC={auc_p:.2f}")

    ax_pr.set_xlabel("Recall (POD)", fontsize=13, fontweight="bold")
    ax_pr.set_ylabel("Precision (1 - FAR)", fontsize=13, fontweight="bold")
    ax_pr.set_title("Precision-Recall Curves", fontsize=15, fontweight="bold")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1)
    ax_pr.grid(True, alpha=0.3)
    leg2 = ax_pr.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.9)
    for txt in leg2.get_texts():
        txt.set_fontweight("bold")
    _bold_ticks(ax_pr)

    fig.suptitle("Model (solid) vs Persistence (dashed) - All 6 Lead Times",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
