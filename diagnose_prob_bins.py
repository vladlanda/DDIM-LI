"""
Per-probability-bin reliability diagnostic.

Tests the hypothesis: at long lead times, the model over-forecasts in
LOW-probability regions (a diffuse false-alarm haze) while remaining
well-calibrated in HIGH-probability regions.

For each lead time and each probability bin [a,b], computes:
  - n_pixels      : how many pixels fell in this bin
  - mean_pred     : mean predicted probability in the bin
  - obs_freq      : observed lightning frequency (fraction that verified)
  - reliability   : obs_freq - mean_pred
                    ( <0 => OVER-forecast / false-alarm haze
                      >0 => UNDER-forecast
                      ~0 => calibrated )
  - contribution  : fraction of all lightning-free-but-predicted mass here

Usage:
  python diagnose_prob_bins.py \
      --npz outputs/nonelbo_v6_256_T36_i12_LIctx/eval/plot_data.npz \
      --dt_min 10
"""
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--dt_min", type=int, default=10)
    p.add_argument("--leads", type=int, nargs="+", default=None,
                   help="Specific lead indices to show (default: all)")
    args = p.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    pr_steps = data["pr_steps"].tolist()
    if args.leads is not None:
        pr_steps = [t for t in pr_steps if t in args.leads]

    # Probability bins — finer at the low end where the haze lives
    edges = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0001])

    for t in pr_steps:
        lead = (t + 1) * args.dt_min
        if f"pr_prob_{t}" not in data:
            continue
        prob = data[f"pr_prob_{t}"].astype(np.float64)
        lbl  = data[f"pr_label_{t}"].astype(np.float64)

        print(f"\n=== +{lead} min ===")
        print(f"{'bin':>12}  {'n_pix':>9}  {'mean_pred':>9}  "
              f"{'obs_freq':>9}  {'reliab':>8}  {'verdict':>16}")
        print("-" * 72)

        total_fa = 0.0   # total false-alarm mass (predicted prob on non-events)
        bin_fa   = []
        for i in range(len(edges) - 1):
            a, b = edges[i], edges[i+1]
            m = (prob >= a) & (prob < b)
            n = int(m.sum())
            if n == 0:
                continue
            mean_pred = float(prob[m].mean())
            obs_freq  = float(lbl[m].mean())
            reliab    = obs_freq - mean_pred
            # false-alarm contribution: predicted prob mass on non-events
            fa_mass   = float((prob[m] * (1 - lbl[m])).sum())
            total_fa += fa_mass
            bin_fa.append(fa_mass)

            if reliab < -0.03:
                verdict = "OVER-forecast"
            elif reliab > 0.03:
                verdict = "under-forecast"
            else:
                verdict = "calibrated"
            print(f"  [{a:.2f},{b:.2f})  {n:>9,}  {mean_pred:>9.3f}  "
                  f"{obs_freq:>9.3f}  {reliab:>+8.3f}  {verdict:>16}")

        # Fraction of false-alarm mass coming from low-prob bins (<0.3)
        low_fa = sum(bin_fa[:4])  # bins below 0.3
        if total_fa > 0:
            print(f"  → {100*low_fa/total_fa:.0f}% of false-alarm mass is in "
                  f"prob<0.3 bins (the 'haze')")


if __name__ == "__main__":
    main()
