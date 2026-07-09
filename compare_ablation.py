"""
Ablation comparison: does the auxiliary cooling task improve lightning skill,
and does the gain follow the pattern the MECHANISM predicts?

The hypothesis is not merely "PR-AUC goes up". It predicts a SPECIFIC pattern:
  - gain concentrates at LONG lead times (existence error dominates there)
  - gain is small at short lead (persistence already carries the answer)
  - gain concentrates at HIGH RECALL (the existence regime; cf. Song 2023,
    who found environmental info matters mainly above ~70% POD)

A uniform gain across all leads/recalls would instead indicate generic
regularisation — a weaker claim.

Outputs:
  1. PR-AUC per lead time, both models, delta and % change
  2. Partial PR-AUC restricted to high-recall (POD >= pod_min): the
     existence regime
  3. Bootstrap CI on the PR-AUC delta (is the gain significant?)

Usage:
  python compare_ablation.py \
      --baseline outputs/nature_baseline/eval/plot_data.npz \
      --treatment outputs/nature_aux_cool/eval/plot_data.npz \
      --dt_min 10 --pod_min 0.7
"""
import argparse
import numpy as np
from sklearn.metrics import precision_recall_curve, auc


def pr_auc(prob, lbl):
    if lbl.sum() == 0 or lbl.sum() == lbl.size:
        return np.nan
    prec, rec, _ = precision_recall_curve(lbl, prob)
    return float(auc(rec, prec))


def partial_pr_auc(prob, lbl, pod_min):
    """Mean precision over the high-recall (POD >= pod_min) region.

    Previous version integrated with auc() over the restricted recall span,
    which returns NaN whenever the curve has <2 distinct recall points above
    pod_min OR the span has zero width. With a coarse ensemble probability
    (only M+1 distinct values for M members) that happens often.

    We instead interpolate precision onto a fixed recall grid in
    [pod_min, 1] and average. This is well-defined for any monotone PR curve
    and comparable across models.
    """
    if lbl.sum() == 0 or lbl.sum() == lbl.size:
        return np.nan
    prec, rec, _ = precision_recall_curve(lbl, prob)
    order = np.argsort(rec)                    # ascending recall
    rec, prec = rec[order], prec[order]
    if rec.max() < pod_min:
        # The model never attains this recall at ANY threshold.
        # Precision there is undefined; report 0.0 (no skill in this regime)
        # rather than NaN, so the comparison remains interpretable.
        return 0.0
    grid = np.linspace(pod_min, min(1.0, rec.max()), 50)
    p_interp = np.interp(grid, rec, prec)
    return float(np.mean(p_interp))


def bootstrap_delta(pb, lb, pt, lt, n_boot=200, seed=0):
    """Bootstrap CI on (treatment - baseline) PR-AUC. Assumes the same
    pixel sample; resamples indices jointly."""
    rng = np.random.default_rng(seed)
    n = len(lb)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = pr_auc(pb[idx], lb[idx])
        b = pr_auc(pt[idx], lt[idx])
        if np.isfinite(a) and np.isfinite(b):
            deltas.append(b - a)
    if not deltas:
        return np.nan, np.nan
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline",  required=True, help="baseline plot_data.npz")
    p.add_argument("--treatment", required=True, help="aux-cool plot_data.npz")
    p.add_argument("--dt_min",  type=int,   default=10)
    p.add_argument("--pod_min", type=float, default=0.7,
                   help="Recall floor defining the 'existence regime'.")
    p.add_argument("--n_boot",  type=int,   default=200)
    args = p.parse_args()

    B = np.load(args.baseline,  allow_pickle=True)
    T = np.load(args.treatment, allow_pickle=True)
    steps = sorted(set(B["pr_steps"].tolist()) & set(T["pr_steps"].tolist()))

    print("=" * 84)
    print("ABLATION: auxiliary cooling task vs baseline")
    print("=" * 84)
    print(f"{'Lead':>6}  {'base':>7}  {'aux':>7}  {'delta':>8}  {'%':>7}"
          f"  {'95% CI on delta':>22}")
    print("-" * 84)

    deltas, rel_gains = [], []
    for t in steps:
        kb, lb_k = f"pr_prob_{t}", f"pr_label_{t}"
        if kb not in B or kb not in T:
            continue
        pb, lb = B[kb], B[lb_k].astype(int)
        pt, lt = T[kb], T[lb_k].astype(int)

        a = pr_auc(pb, lb)
        b = pr_auc(pt, lt)
        d = b - a
        pct = 100 * d / a if a > 0 else np.nan
        lo, hi = bootstrap_delta(pb, lb, pt, lt, n_boot=args.n_boot)
        sig = "*" if (np.isfinite(lo) and lo > 0) else " "
        deltas.append(d)
        rel_gains.append(pct)
        print(f"  +{(t+1)*args.dt_min:3d}m  {a:>7.3f}  {b:>7.3f}  "
              f"{d:>+8.3f}  {pct:>+6.1f}%  [{lo:+.3f}, {hi:+.3f}]{sig}")

    print("-" * 84)
    print("  * = 95% CI excludes zero (significant improvement)")

    # Existence regime: partial PR-AUC at high recall
    print(f"\n{'='*84}")
    print(f"EXISTENCE REGIME — mean precision at POD >= {args.pod_min}")
    print("(the mechanism predicts the gain concentrates HERE)")
    print("=" * 84)
    print(f"{'Lead':>6}  {'base':>7}  {'aux':>7}  {'delta':>8}  "
          f"{'max_rec_b':>9}  {'max_rec_a':>9}")
    print("-" * 62)
    for t in steps:
        kb, lb_k = f"pr_prob_{t}", f"pr_label_{t}"
        if kb not in B or kb not in T:
            continue
        lb, lt = B[lb_k].astype(int), T[lb_k].astype(int)
        a = partial_pr_auc(B[kb], lb, args.pod_min)
        b = partial_pr_auc(T[kb], lt, args.pod_min)
        # Max attainable recall tells us if a model can even reach this regime
        _, rb, _ = precision_recall_curve(lb, B[kb])
        _, rt, _ = precision_recall_curve(lt, T[kb])
        print(f"  +{(t+1)*args.dt_min:3d}m  {a:>7.3f}  {b:>7.3f}  {b-a:>+8.3f}  "
              f"{rb.max():>9.3f}  {rt.max():>9.3f}")
    print("\n  max_rec = highest recall attainable at any threshold. If a model")
    print("  cannot reach pod_min, its precision there is reported as 0.000.")

    # ── Interpretation guidance (no automated verdict) ─────────────────
    #
    # IMPORTANT: PR-AUC is a saturating, non-linear function of class
    # separation. Because baseline PR-AUC falls with lead time, the SAME
    # underlying skill improvement produces both a larger absolute delta AND
    # a larger relative gain at long lead — purely by construction. Neither
    # summary statistic cleanly separates "existence mechanism" from "generic
    # regularisation". So we report the evidence and leave the call to you.
    #
    # The DISCRIMINATING evidence is the EXISTENCE REGIME table above:
    # partial PR-AUC restricted to high recall (POD >= pod_min) measures skill
    # on precisely the axis the mechanism claims to improve — deciding whether
    # lightning occurs at all, rather than ranking easy negatives. Compare:
    #   gain in existence regime  >>  gain in full PR-AUC   -> supports mechanism
    #   gain roughly equal in both                          -> generic improvement
    print("\n" + "=" * 84)
    print("HOW TO READ THIS")
    print("=" * 84)
    print("  1. Per-lead table: is the gain significant (95% CI excludes 0)?")
    print("  2. EXISTENCE REGIME table is the discriminating evidence:")
    print("       gain there >> gain in full PR-AUC  -> supports the existence")
    print("         mechanism (cooling injects convective-existence information)")
    print("       gain similar in both               -> generic improvement, not")
    print("         the hypothesised mechanism; report honestly as such")
    print("  3. Do NOT read the lead-time profile alone as proof: PR-AUC")
    print("     saturates non-linearly, so a constant skill gain inflates both")
    print("     absolute and relative deltas wherever baseline skill is low.")
    print("  4. If neither table shows gain: negative result. Consider")
    print("     multi-window cooling or environmental (NWP/aerosol) conditioning.")

    if len(rel_gains) >= 3:
        print("\n  Summary (for reference, NOT a verdict):")
        print(f"    SHORT lead (+10,+20): delta={np.mean(deltas[:2]):+.4f}  "
              f"relative={np.nanmean(rel_gains[:2]):+.1f}%")
        print(f"    LONG  lead (last 2):  delta={np.mean(deltas[-2:]):+.4f}  "
              f"relative={np.nanmean(rel_gains[-2:]):+.1f}%")


if __name__ == "__main__":
    main()
