"""
Compare the diffusion model vs the CNN baseline on FSS across spatial scale.

Purpose: diagnose FINDINGS.md F4 -- the CNN baseline beat the diffusion
model on raw pointwise PR-AUC at every lead time. The open question is
whether that's a genuine advantage or a "double penalty" scoring artifact
(a model hedging under real positional uncertainty scores deceptively
well pointwise; see FINDINGS.md F4 / E2 for the full reasoning). This
script answers it directly: if the CNN's edge shrinks or reverses as
spatial tolerance grows, that's evidence for the double-penalty
explanation. If it doesn't, the CNN has a real advantage needing a
different explanation.

Pulls the diffusion side from plot_data.npz (which already stores the
FULL fss_thr{thr}_s{s} grid across every scale, not just the single
mid-scale value that ends up in evaluate.py's metrics_per_step.csv --
see that CSV's `mid_s = fss_scales[len(fss_scales)//2]` line, which is
scale=8 by default, NOT pointwise -- a likely source of confusion if
comparing that CSV's bare fss_{thr} column against anything at a
different scale). Pulls the CNN side from baseline_cnn_metrics.csv's
fss_{thr}_scale{s} columns (added in the same commit as this script).

Usage:
  python compare_diffusion_vs_cnn_fss.py \
      --diffusion_npz outputs/nature_256_T36_ir_li_only/eval/plot_data.npz \
      --cnn_csv baseline_cnn/baseline_cnn_metrics.csv \
      --output_dir baseline_cnn
"""
import argparse
import csv
import os

import numpy as np


def load_diffusion_fss(npz_path, thresholds, scales):
    data = np.load(npz_path, allow_pickle=True)
    out = {}
    for thr in thresholds:
        for s in scales:
            key = f"fss_thr{thr}_s{s}"
            out[(thr, s)] = data[key] if key in data else None
    T_out         = int(data["T_out"])   if "T_out"   in data else None
    dt_min        = int(data["dt_min"])  if "dt_min"  in data else None
    pixel_size_km = float(data["pixel_size_km"]) if "pixel_size_km" in data else 4.0
    return out, T_out, dt_min, pixel_size_km


def load_cnn_fss(csv_path, thresholds, scales):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} has no rows")
    out = {}
    for thr in thresholds:
        for s in scales:
            col = f"fss_{thr}_scale{s}"
            out[(thr, s)] = (np.array([float(r[col]) for r in rows])
                             if col in rows[0] else None)
    lead_mins = [float(r["lead_min"]) for r in rows]
    return out, lead_mins


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diffusion_npz", required=True)
    p.add_argument("--cnn_csv", required=True)
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    p.add_argument("--scales", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--output_dir", default=".")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    diff_fss, T_out, dt_min, pixel_size_km = load_diffusion_fss(
        args.diffusion_npz, args.thresholds, args.scales)
    cnn_fss, lead_mins = load_cnn_fss(args.cnn_csv, args.thresholds, args.scales)

    missing = [k for k, v in diff_fss.items() if v is None] + \
              [k for k, v in cnn_fss.items() if v is None]
    if missing:
        print(f"WARNING: missing (threshold, scale) pairs, will show as NaN: {missing}")

    print(f"\n{'thr':>5} {'scale':>6} {'scale_km':>9} "
         f"{'diffusion':>12} {'CNN base':>12} {'delta (diff-cnn)':>18}  winner")
    print("-" * 78)
    for thr in args.thresholds:
        for s in args.scales:
            dv, cv = diff_fss[(thr, s)], cnn_fss[(thr, s)]
            d_mean = float(np.nanmean(dv)) if dv is not None else float("nan")
            c_mean = float(np.nanmean(cv)) if cv is not None else float("nan")
            delta = d_mean - c_mean
            scale_km = (2 * s + 1) * pixel_size_km
            winner = ("diffusion" if delta > 0.01 else
                     "cnn" if delta < -0.01 else "~tie")
            print(f"{thr:>5} {s:>6} {scale_km:>8.0f}km "
                 f"{d_mean:>12.4f} {c_mean:>12.4f} {delta:>+18.4f}  {winner}")

    # ── Plot: FSS vs scale, diffusion vs CNN, one panel per threshold ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "text.color": "black",
        "grid.color": "#cccccc", "grid.linestyle": "--", "grid.linewidth": 0.5,
        "legend.framealpha": 0.9, "legend.edgecolor": "#cccccc", "font.size": 9,
    })

    n_thr = len(args.thresholds)
    fig, axes = plt.subplots(1, n_thr, figsize=(5.5 * n_thr, 5), squeeze=False)
    fig.patch.set_facecolor("white")
    scale_km_axis = [(2 * s + 1) * pixel_size_km for s in args.scales]

    for col, thr in enumerate(args.thresholds):
        ax = axes[0, col]
        d_vals = [float(np.nanmean(diff_fss[(thr, s)]))
                 if diff_fss[(thr, s)] is not None else np.nan for s in args.scales]
        c_vals = [float(np.nanmean(cnn_fss[(thr, s)]))
                 if cnn_fss[(thr, s)] is not None else np.nan for s in args.scales]
        ax.plot(scale_km_axis, d_vals, color="#1f77b4", linewidth=2,
               marker="o", markersize=5, label="Diffusion model")
        ax.plot(scale_km_axis, c_vals, color="#d62728", linewidth=2,
               marker="s", markersize=5, label="CNN baseline")
        ax.axhline(0.5, color="#999999", linestyle="--", linewidth=1, label="FSS=0.5")
        ax.set_title(f"FSS  (p > {thr})", fontweight="bold")
        ax.set_xlabel("Scale (km)"); ax.set_ylabel("FSS")
        ax.set_ylim(-0.05, 1.05)
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("black"); sp.set_linewidth(0.8)
        ax.grid(True, color="#cccccc", linestyle="--", linewidth=0.5, zorder=0)
        ax.legend(fontsize=8)

    fig.suptitle("Diffusion Model vs CNN Baseline — FSS vs Spatial Scale "
                "(mean over lead times)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(args.output_dir, "diffusion_vs_cnn_fss_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nComparison plot -> {out_path}")


if __name__ == "__main__":
    main()
