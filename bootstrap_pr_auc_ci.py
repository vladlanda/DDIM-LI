"""
Sequence-level bootstrap confidence intervals for PR-AUC comparisons.

Why sequence-level, not pixel-level: pixels within one test sequence are
strongly spatially correlated (a single storm produces thousands of
correlated pixels). Resampling individual pixels drastically understates
uncertainty. This resamples whole SEQUENCES (with replacement) -- the
correct unit of statistical independence here -- exactly the discipline
used in the pre-registered channel-information tests earlier in this
project.

Requires each npz to contain, per lead step t:
  pr_prob_{t}, pr_label_{t}, pr_seqid_{t}   (all same length, aligned)
Produced by evaluate.py / persistence_baseline.py / optical_flow_baseline.py
after the seqid-tracking patch.

VALIDITY REQUIREMENT: "sequence i" must refer to the same physical test
scene across every npz being compared. This holds as long as all
evaluations used the same test_roots/T_in/T_out/img_size (shuffle=False,
deterministic sorted-timestamp ordering in make_test_loader) -- true for
every script in this repo run against the same checkpoint/config.

Usage:
  # Single-run CI (marginal, unpaired):
  python bootstrap_pr_auc_ci.py --npz outputs/.../eval/plot_data.npz \
      --label model --dt_min 10 --n_boot 1000

  # Paired comparison (model vs one or more baselines) — RECOMMENDED,
  # since paired resampling accounts for shared per-scene difficulty and
  # gives a tighter, more honest CI on the DELTA than two separate CIs:
  python bootstrap_pr_auc_ci.py \
      --npz outputs/.../eval/plot_data.npz \
      --baseline persistence_metrics.csv:outputs/persistence_baseline/persistence_pr_curves.npz \
      --baseline pysteps_li:pysteps_li/optical_flow_li_pr_curves.npz \
      --baseline pysteps_ir:pysteps_ir/optical_flow_ir_pr_curves.npz \
      --label model --dt_min 10 --n_boot 1000
"""
import argparse
import numpy as np
from sklearn.metrics import precision_recall_curve, auc


def pr_auc(prob, lbl):
    if lbl.sum() == 0 or lbl.sum() == lbl.size:
        return np.nan
    p, r, _ = precision_recall_curve(lbl, prob)
    return float(auc(r, p))


def load_run(npz_path):
    """Returns {t: (prob, lbl, seqid)} and the set of lead steps available."""
    data = np.load(npz_path, allow_pickle=True)
    steps = sorted(int(t) for t in data["pr_steps"].tolist())
    out = {}
    for t in steps:
        pk, lk, sk = f"pr_prob_{t}", f"pr_label_{t}", f"pr_seqid_{t}"
        if pk in data and lk in data and sk in data:
            out[t] = (data[pk], data[lk].astype(int), data[sk].astype(int))
    if not out:
        raise ValueError(
            f"{npz_path} has no pr_seqid_* arrays -- was it produced by a "
            f"version of the eval script WITHOUT the seqid-tracking patch? "
            f"Re-run the evaluation to regenerate this npz."
        )
    return out


def bootstrap_ci(prob, lbl, seqid, n_boot, rng, paired_with=None):
    """
    Resample sequence IDs with replacement (n draws from n unique sequences),
    then build the resampled pixel set by ACTUALLY DUPLICATING each drawn
    sequence's pixels according to how many times it was drawn -- this is
    the correct block bootstrap. PR-AUC is invariant to duplicating an
    entire pooled sample uniformly, but NOT to duplicating different
    sequences by different amounts (that changes the relative mix of
    easy/hard sequences pooled into one PR curve). A boolean-mask "include
    each drawn sequence once" shortcut was tried and found to be wrong by
    a large margin on a two-sequence synthetic check (0.20 AUC difference)
    -- it behaves like subsampling distinct sequences rather than a real
    weighted block bootstrap, and understates the true resampling variance.

    If paired_with is given (prob2, lbl2, seqid2), uses the SAME resampled
    sequence draw for both runs each iteration, returning the delta
    distribution directly (paired bootstrap) -- tighter and more honest
    than independently bootstrapping each run and subtracting percentiles.
    """
    uniq = np.unique(seqid)
    idx_by_seq = {s: np.where(seqid == s)[0] for s in uniq}
    point = pr_auc(prob, lbl)

    if paired_with is not None:
        prob2, lbl2, seqid2 = paired_with
        uniq2 = np.unique(seqid2)
        if not np.array_equal(np.sort(uniq), np.sort(uniq2)):
            raise ValueError(
                "Sequence ID sets differ between the two runs being paired "
                "-- they must come from evaluating the SAME test sequences "
                "(same test_roots/T_in/T_out/img_size) for a paired "
                "bootstrap to be valid."
            )
        idx_by_seq2 = {s: np.where(seqid2 == s)[0] for s in uniq2}

    vals, deltas = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_by_seq[s] for s in pick])
        v = pr_auc(prob[idx], lbl[idx])
        if np.isfinite(v):
            vals.append(v)
        if paired_with is not None:
            idx2 = np.concatenate([idx_by_seq2[s] for s in pick])
            v2 = pr_auc(prob2[idx2], lbl2[idx2])
            if np.isfinite(v) and np.isfinite(v2):
                deltas.append(v - v2)
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    result = {"point": point, "ci_lo": lo, "ci_hi": hi, "n_boot_valid": len(vals)}
    if paired_with is not None:
        dlo, dhi = (np.percentile(deltas, [2.5, 97.5]) if deltas else (np.nan, np.nan))
        result["delta_point"] = point - pr_auc(paired_with[0], paired_with[1])
        result["delta_ci_lo"] = dlo
        result["delta_ci_hi"] = dhi
        result["delta_n_boot_valid"] = len(deltas)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="Main run's npz (the model, usually)")
    p.add_argument("--label", default="model")
    p.add_argument("--baseline", action="append", default=[],
                   help="name:path.npz -- repeatable, one per baseline to "
                        "compare against (paired bootstrap on the delta).")
    p.add_argument("--dt_min", type=int, default=10)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    main_run = load_run(args.npz)

    baselines = {}
    for spec in args.baseline:
        name, path = spec.split(":", 1)
        baselines[name] = load_run(path)

    steps = sorted(main_run.keys())
    print("=" * 100)
    print(f"SEQUENCE-LEVEL BOOTSTRAP CI  (n_boot={args.n_boot}, resampling "
          f"whole test sequences, not pixels)")
    print("=" * 100)

    print(f"\n--- {args.label}: marginal PR-AUC with 95% CI ---")
    print(f"{'lead':>6}  {'PR-AUC':>8}  {'95% CI':>18}")
    print("-" * 40)
    for t in steps:
        prob, lbl, seqid = main_run[t]
        r = bootstrap_ci(prob, lbl, seqid, args.n_boot, rng)
        print(f"  +{(t+1)*args.dt_min:3d}m  {r['point']:>8.4f}  "
              f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")

    for name, base_run in baselines.items():
        print(f"\n--- {args.label} vs {name}: paired delta with 95% CI ---")
        print(f"{'lead':>6}  {name+'':>10}  {args.label:>10}  {'delta':>9}  "
              f"{'95% CI on delta':>20}  {'sig?':>5}")
        print("-" * 74)
        for t in steps:
            if t not in base_run:
                continue
            prob, lbl, seqid = main_run[t]
            bprob, blbl, bseqid = base_run[t]
            r = bootstrap_ci(prob, lbl, seqid, args.n_boot, rng,
                             paired_with=(bprob, blbl, bseqid))
            base_pt = r['point'] - r['delta_point']
            sig = "*" if (np.isfinite(r['delta_ci_lo']) and
                         (r['delta_ci_lo'] > 0 or r['delta_ci_hi'] < 0)) else " "
            print(f"  +{(t+1)*args.dt_min:3d}m  {base_pt:>10.4f}  {r['point']:>10.4f}  "
                  f"{r['delta_point']:>+9.4f}  "
                  f"[{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}]      {sig}")
    print("\n  * = 95% CI on the delta excludes zero (statistically significant)")


if __name__ == "__main__":
    main()
