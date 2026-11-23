"""
Fair baseline for the area-probability comparison: CNN neighbourhood-max probabilities
recalibrated with isotonic regression fitted on VALIDATION (per radius and lead), applied
to test, and compared with the diffusion ensemble's area probabilities on test.

    python recalibrate_area.py --val_dir <..._val_area> --cnn_test_dir <test cnn dir> \
        --diffusion_test_dir <merged diffusion dir> --out_dir <dir> [--n_boot 1000]

Writes area_cnn_nmax_recal_r*px.npz / area_cnn_nmax_recal.csv (same schema as the other
area files) and area_recal_comparison.csv: Brier and AP of diffusion, CNN-max and
recalibrated CNN-max, with a sequence-level paired bootstrap CI on
Brier(diffusion) - Brier(CNN-max recalibrated)  (negative = diffusion better).
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
from sklearn.isotonic import IsotonicRegression

from analysis_regimes import seq_bootstrap_mean
from bootstrap_pr_auc_ci import average_precision
from evaluate import save_area_outputs


def load(path):
    z = np.load(path)
    return {int(t): (z[f"pr_prob_{t}"], z[f"pr_label_{t}"].astype(int), z[f"pr_seqid_{t}"])
            for t in z["pr_steps"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_dir", required=True)
    p.add_argument("--cnn_test_dir", required=True)
    p.add_argument("--diffusion_test_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    store, rows = {}, []
    for vpath in sorted(glob.glob(os.path.join(args.val_dir, "area_cnn_nmax_r*px.npz")),
                        key=lambda q: int(re.search(r"_r(\d+)px", q).group(1))):
        hw = int(re.search(r"_r(\d+)px", vpath).group(1))
        val = load(vpath)
        cnn = load(os.path.join(args.cnn_test_dir, f"area_cnn_nmax_r{hw}px.npz"))
        dif = load(os.path.join(args.diffusion_test_dir, f"area_diffusion_r{hw}px.npz"))
        for t in sorted(val):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(val[t][0], val[t][1])
            cp, lbl, sid = cnn[t]
            dp, dlbl, dsid = dif[t]
            if not (np.array_equal(sid, dsid) and np.array_equal(lbl, dlbl)):
                raise ValueError(f"r{hw}px lead {t}: CNN and diffusion test samples are not aligned")
            rp = iso.predict(cp).astype(np.float32)
            store.setdefault(hw, {})[t] = ([rp], [lbl], [sid])
            se = lambda q: (np.clip(q, 0, 1) - lbl) ** 2
            b = seq_bootstrap_mean(se(dp), sid, args.n_boot, rng, paired=se(rp))
            base = lbl.mean()
            rows.append(dict(
                radius_km=round(hw * 3.14, 1), lead_min=10 * (t + 1), base_rate=base,
                brier_diffusion=b["point"], brier_cnn_nmax=se(cp).mean(), brier_cnn_recal=se(rp).mean(),
                diff_minus_recal=b["delta_point"], ci_lo=b["delta_ci_lo"], ci_hi=b["delta_ci_hi"],
                fbias_diffusion=dp.mean() / base, fbias_cnn_nmax=cp.mean() / base, fbias_cnn_recal=rp.mean() / base,
                ap_diffusion=average_precision(dp, lbl), ap_cnn_nmax=average_precision(cp, lbl),
                ap_cnn_recal=average_precision(rp, lbl)))
            r = rows[-1]
            print(f"r={r['radius_km']:5.1f} km +{r['lead_min']:2d}m  Brier diff {r['brier_diffusion']:.5f} "
                  f"cnn-max {r['brier_cnn_nmax']:.5f} recal {r['brier_cnn_recal']:.5f}  "
                  f"diff-recal {r['diff_minus_recal']:+.5f} [{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]"
                  f"{' *' if r['ci_hi'] < 0 or r['ci_lo'] > 0 else ''}")
    save_area_outputs(args.out_dir, "cnn_nmax_recal", store)
    with open(os.path.join(args.out_dir, "area_recal_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"-> {args.out_dir}/area_recal_comparison.csv  (* = CI excludes 0; negative = diffusion better)")


if __name__ == "__main__":
    main()
