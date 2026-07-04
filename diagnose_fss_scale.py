"""
FSS-scale diagnostic: is the +60min error POSITIONAL or EXISTENCE?

Reads plot_data.npz (from evaluate.py), extracts FSS vs neighbourhood scale
at each lead time, and computes the "skillful scale" — the smallest
neighbourhood at which FSS crosses the useful-skill threshold
    FSS_useful = 0.5 + f0/2   (Roberts & Lean 2008)

Interpretation:
  - skillful scale SMALL / FSS rises steeply  -> POSITIONAL error
      (model right region, wrong exact pixels -> warp/advection helps)
  - skillful scale LARGE / FSS stays flat+low  -> EXISTENCE error
      (model misses whether lightning occurs -> needs convective-state info)

Usage:
  python diagnose_fss_scale.py \
      --npz outputs/nonelbo_v6_256_T36_LIctx/plot_data.npz \
      --pixel_km 4.0 --base_rate 0.06
"""
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--pixel_km",  type=float, default=4.0)
    p.add_argument("--base_rate", type=float, default=0.06,
                   help="Domain lightning base rate f0 (fraction of pixels).")
    p.add_argument("--prob_thr",  type=float, default=0.3,
                   help="Ensemble probability threshold to binarise at.")
    p.add_argument("--dt_min",    type=int,   default=10)
    args = p.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    fss_scales = data["fss_scales"].tolist()
    pr_steps   = data["pr_steps"].tolist()

    # Useful-skill threshold
    fss_target = 0.5 + args.base_rate / 2.0

    print(f"FSS-scale diagnostic")
    print(f"  base rate f0 = {args.base_rate}")
    print(f"  useful-skill threshold FSS = {fss_target:.3f}")
    print(f"  neighbourhood window = (2·scale+1) px, pixel = {args.pixel_km} km")
    print()

    header = f"{'Lead':>6} | " + " ".join(
        f"{(2*s+1)*args.pixel_km:>5.0f}km" for s in fss_scales)
    header += " | skillful_scale   error_type"
    print(header)
    print("-" * len(header))

    thr = args.prob_thr
    for t in pr_steps:
        lead = (t + 1) * args.dt_min
        fss_by_scale = []
        for s in fss_scales:
            key = f"fss_thr{thr}_s{s}"
            if key in data:
                arr = data[key]   # shape (T_out,) — FSS per lead step
                # Index the CURRENT lead step t, not the mean over all steps
                val = float(arr[t]) if arr.ndim >= 1 and arr.shape[0] > t else float(arr)
                fss_by_scale.append(val)
            else:
                fss_by_scale.append(np.nan)

        # Find skillful scale: smallest window where FSS >= target
        skillful_km = None
        for s, f in zip(fss_scales, fss_by_scale):
            if not np.isnan(f) and f >= fss_target:
                skillful_km = (2*s + 1) * args.pixel_km
                break

        # Classify error type from the curve
        f_small = fss_by_scale[0]                 # finest scale
        f_large = fss_by_scale[-1]                # coarsest scale
        rise    = f_large - f_small
        if skillful_km is not None and skillful_km <= 60:
            etype = "POSITIONAL (recovers at small scale)"
        elif rise > 0.25:
            etype = "POSITIONAL (steep rise with scale)"
        elif f_large < fss_target:
            etype = "EXISTENCE (never reaches skill)"
        else:
            etype = "MIXED"

        vals = " ".join(f"{f:>6.3f}" for f in fss_by_scale)
        sk   = f"{skillful_km:>6.0f}km" if skillful_km else "  none"
        print(f"  +{lead:3d}m | {vals} | {sk}   {etype}")

    print()
    print("READING THE RESULT:")
    print("  If +60min row reaches skill only at large scales (or 'none'),")
    print("  the error is EXISTENCE-dominated -> pursue convective-state /")
    print("  flash-rate-potential ideas, NOT optical flow.")
    print("  If +60min reaches skill at small scales, error is POSITIONAL")
    print("  -> advection / warp loss is the right lever.")


if __name__ == "__main__":
    main()
