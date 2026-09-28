"""
Regime / calibration / diurnal / regional analysis of EXISTING predictions.
CPU only, no model inference -- joins build_test_metadata.py output onto the
PR samples each evaluation script already saved (see that script's docstring
for why the samples align, and note the label-equality assertion below).

Addresses the pre-submission review items:
  (1) Initiation vs continuation: PR-AUC split by whether any lightning was
      observed within --radius_km during the last 30 min of context.
      Persistence/optical flow cannot forecast initiation by construction.
  (2) Calibration of ALL probabilistic models, not only the diffusion model:
      Brier score, Brier skill score vs climatology, reliability curves.
  (3) Diurnal cycle: PR-AUC by time of day of the forecast issue time.
  (4) Per-region PR-AUC.
Every CI is a sequence-level bootstrap (whole test sequences resampled),
reusing bootstrap_pr_auc_ci.bootstrap_ci for PR-AUC so the method is
identical to the paper's headline table. Model-vs-reference deltas are
PAIRED (same resampled sequences for both models).

Usage:
  python analysis_regimes.py --metadata outputs/test_metadata.npz \
      --run diffusion:outputs/nature_256_T36_ir_li_only/eval_ens_50/plot_data.npz \
      --run persistence:outputs/persistence_baseline/persistence_pr_curves.npz \
      --run pysteps_li:pysteps_li/optical_flow_li_pr_curves.npz \
      --run pysteps_ir:pysteps_ir/optical_flow_ir_pr_curves.npz \
      --run cnn:baseline_cnn/baseline_cnn_pr_curves.npz \
      --run lightgbm:baseline_lightgbm/baseline_lightgbm_pr_curves.npz \
      --calib_models diffusion cnn lightgbm \
      --output_dir manuscript/analysis --n_boot 200
"""
import argparse
import csv
import os
from collections import OrderedDict

import numpy as np

from bootstrap_pr_auc_ci import METRICS, bootstrap_ci, load_run


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--run", action="append", required=True,
                   help="name:path to a PR npz. The FIRST --run is the reference "
                        "model that paired deltas are computed against.")
    p.add_argument("--calib_models", nargs="+", default=None,
                   help="Models whose scores are probabilities (calibration is "
                        "meaningless for persistence/optical-flow fields). "
                        "Default: all runs.")
    p.add_argument("--label_tolerance", type=float, default=1e-2,
                   help="Max fraction of pixels whose saved label may differ from "
                        "the metadata label (threshold-boundary rounding from "
                        "different normalisation stats). Above it: hard error.")
    p.add_argument("--radius_km", type=float, default=20.0)
    p.add_argument("--pixel_km", type=float, default=4.0)
    p.add_argument("--dt_min", type=int, default=10)
    p.add_argument("--hour_bin", type=int, default=3)
    p.add_argument("--region_lon", nargs="+", type=float, default=None,
                   help="Centre longitude per test region (same order as "
                        "test_roots) -> hours shown as LOCAL SOLAR time "
                        "(UTC + lon/15). Omit to use UTC.")
    p.add_argument("--metric", choices=["ap", "pr_auc"], default="ap",
                   help="ap = average precision (default, non-interpolated); "
                        "pr_auc = trapezoidal area, as in the original headline "
                        "table. Use the SAME metric everywhere in the paper.")
    p.add_argument("--n_boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="manuscript/analysis")
    return p.parse_args()


# ----------------------------------------------------------------- loading
def load_all(args):
    meta = dict(np.load(args.metadata, allow_pickle=True))
    T_out = int(meta["T_out"])
    runs = OrderedDict()
    for spec in args.run:
        name, path = spec.split(":", 1)
        runs[name] = load_run(path)
    n = meta["seqid"].size
    for name, run in runs.items():
        for t in range(T_out):
            if t not in run:
                raise ValueError(f"{name}: lead index {t} missing")
            prob, lbl, sid = run[t]
            if lbl.size != n or not np.array_equal(sid, meta["seqid"]):
                raise ValueError(
                    f"{name} lead {t}: sample count or sequence ids do NOT match the "
                    f"metadata ({lbl.size} vs {n} samples). The pixel alignment "
                    f"assumption is broken for this file -- was it produced with "
                    f"different test_roots / img_size / T_in / T_out?")
            ref_lbl = meta[f"label_{t}"]
            n_bad = int((lbl.astype(np.int8) != ref_lbl).sum())
            if n_bad:
                frac = n_bad / n
                msg = (f"{name} lead {t}: {n_bad} of {n} saved labels ({frac:.2e}) differ "
                       f"from the metadata labels (saved positives {int(lbl.sum())}, "
                       f"metadata positives {int(ref_lbl.sum())})")
                if frac > args.label_tolerance:
                    raise ValueError(msg + " -- above --label_tolerance: labels do NOT "
                                     "match the metadata; alignment assumption is broken.")
                print("  WARNING " + msg + " -- within tolerance (threshold-boundary "
                      "rounding); scoring against the common metadata labels.")
            # every model is scored against ONE ground truth
            run[t] = (prob, ref_lbl.astype(int), sid)
    print(f"Alignment OK: {len(runs)} runs x {T_out} leads x {n} pixels/lead match the metadata.")
    return meta, runs, T_out


# ---------------------------------------------------------- statistics
def seq_bootstrap_mean(values, seqid, n_boot, rng, paired=None):
    """Sequence-level bootstrap of a pixel-mean statistic (e.g. Brier).
    Fast: aggregates per sequence once, then resamples sequence weights."""
    uniq, inv = np.unique(seqid, return_inverse=True)
    s1 = np.bincount(inv, weights=values)
    cnt = np.bincount(inv).astype(float)
    s2 = np.bincount(inv, weights=paired) if paired is not None else None
    point = s1.sum() / cnt.sum()
    boots, dboots = [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, uniq.size, uniq.size), minlength=uniq.size)
        denom = (w * cnt).sum()
        v = (w * s1).sum() / denom
        boots.append(v)
        if s2 is not None:
            dboots.append(v - (w * s2).sum() / denom)
    out = {"point": point, "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5)}
    if s2 is not None:
        out["delta_point"] = point - s2.sum() / cnt.sum()
        out["delta_ci_lo"], out["delta_ci_hi"] = np.percentile(dboots, [2.5, 97.5])
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        fields = list(OrderedDict((k, None) for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}")


def fmt(x):
    return f"{x:.4f}" if np.isfinite(x) else "nan"


# ---------------------------------------------------------- (1) regimes
def regime_analysis(meta, runs, T_out, args, rng):
    r_px = args.radius_km / args.pixel_km
    regimes = {
        "initiation": meta["dist_recent_px"] > r_px,
        "continuation": meta["dist_recent_px"] <= r_px,
    }
    ref = next(iter(runs))
    rows = []
    for regime, mask in regimes.items():
        for t in range(T_out):
            _, lbl, sid = runs[ref][t]
            m_lbl, m_sid = lbl[mask], sid[mask]
            n_pos = int(m_lbl.sum())
            base = float(m_lbl.mean()) if m_lbl.size else np.nan
            for name, run in runs.items():
                prob = run[t][0][mask]
                row = {"regime": regime, "lead_min": (t + 1) * args.dt_min, "model": name,
                       "n_pixels": int(mask.sum()), "n_pos": n_pos, "base_rate": base}
                if n_pos == 0:
                    row.update(pr_auc=np.nan, ci_lo=np.nan, ci_hi=np.nan)
                elif name == ref:
                    r = bootstrap_ci(prob, m_lbl, m_sid, args.n_boot, rng, metric=METRICS[args.metric])
                    row.update(pr_auc=r["point"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"])
                else:
                    refprob = runs[ref][t][0][mask]
                    r = bootstrap_ci(refprob, m_lbl, m_sid, args.n_boot, rng,
                                     paired_with=(prob, m_lbl, m_sid), metric=METRICS[args.metric])
                    row.update(pr_auc=METRICS[args.metric](prob, m_lbl), ci_lo=np.nan, ci_hi=np.nan,
                               **{f"{ref}_minus_this": r["delta_point"],
                                  "delta_ci_lo": r["delta_ci_lo"], "delta_ci_hi": r["delta_ci_hi"]})
                rows.append(row)
            print(f"  {regime:12s} +{(t+1)*args.dt_min:3d}m  base={base:.4f}  " +
                  "  ".join(f"{r['model']}={fmt(r['pr_auc'])}" for r in rows[-len(runs):]))
    return rows


# ---------------------------------------------------------- (2) calibration
def calibration_analysis(runs, T_out, args, rng):
    from sklearn.calibration import calibration_curve
    names = args.calib_models or list(runs)
    ref = names[0]
    rows, curves = [], {}
    for t in range(T_out):
        _, lbl, sid = runs[ref][t]
        clim = float(lbl.mean())
        ref_se = (np.clip(runs[ref][t][0].astype(float), 0, 1) - lbl) ** 2
        for name in names:
            prob = np.clip(runs[name][t][0].astype(float), 0, 1)
            se = (prob - lbl) ** 2
            r = seq_bootstrap_mean(se, sid, args.n_boot, rng,
                                   paired=None if name == ref else ref_se)
            bs_clim = clim * (1 - clim)
            row = {"lead_min": (t + 1) * args.dt_min, "model": name,
                   "brier": r["point"], "brier_ci_lo": r["ci_lo"], "brier_ci_hi": r["ci_hi"],
                   "bss_vs_climatology": 1 - r["point"] / bs_clim if bs_clim > 0 else np.nan}
            if name != ref:  # delta = this - ref (Brier: lower is better)
                row.update(brier_minus_ref=r["delta_point"],
                           delta_ci_lo=r["delta_ci_lo"], delta_ci_hi=r["delta_ci_hi"])
            rows.append(row)
            try:
                frac, mean = calibration_curve(lbl, prob, n_bins=10, strategy="uniform")
                curves[(name, t)] = (mean, frac)
            except ValueError:
                pass
    return rows, curves


# ---------------------------------------------------------- (3)+(4) groups
def grouped_analysis(meta, runs, T_out, groups, label, args, rng):
    """PR-AUC per group (pooling all leads), paired delta vs the reference."""
    ref = next(iter(runs))
    seq_group = np.asarray(groups)
    pix_group = seq_group[meta["seqid"]]
    pooled = {name: tuple(np.concatenate([run[t][i] for t in range(T_out)]) for i in range(3))
              for name, run in runs.items()}
    pix_group_all = np.concatenate([pix_group] * T_out)
    rows = []
    for g in sorted(set(seq_group.tolist())):
        mask = pix_group_all == g
        _, lbl, sid = pooled[ref]
        m_lbl, m_sid = lbl[mask], sid[mask]
        if m_lbl.sum() == 0:
            continue
        for name in runs:
            prob = pooled[name][0][mask]
            if name == ref:
                r = bootstrap_ci(prob, m_lbl, m_sid, args.n_boot, rng, metric=METRICS[args.metric])
                extra = {"pr_auc": r["point"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]}
            else:
                r = bootstrap_ci(pooled[ref][0][mask], m_lbl, m_sid, args.n_boot, rng,
                                 paired_with=(prob, m_lbl, m_sid), metric=METRICS[args.metric])
                extra = {"pr_auc": METRICS[args.metric](prob, m_lbl), "ci_lo": np.nan, "ci_hi": np.nan,
                         f"{ref}_minus_this": r["delta_point"],
                         "delta_ci_lo": r["delta_ci_lo"], "delta_ci_hi": r["delta_ci_hi"]}
            rows.append({label: g, "model": name, "n_sequences": int(np.unique(m_sid).size),
                         "n_pos": int(m_lbl.sum()), "base_rate": float(m_lbl.mean()), **extra})
    return rows


def hour_groups(meta, args):
    hrs = np.array([int(s[11:13]) + int(s[14:16]) / 60 for s in meta["seq_time_utc"]])
    if args.region_lon:
        hrs = (hrs + np.asarray(args.region_lon)[meta["seq_region"]] / 15.0) % 24
    return (np.floor(hrs / args.hour_bin) * args.hour_bin).astype(int)


# ---------------------------------------------------------- figures
def make_figures(regime_rows, curves, calib_rows, diurnal_rows, region_rows, args, runs, T_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.linestyle": "--",
                         "grid.linewidth": 0.5, "grid.color": "#cccccc"})
    colors = dict(zip(runs, plt.cm.tab10.colors))
    od = args.output_dir

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, regime in zip(axes, ["initiation", "continuation"]):
        for name in runs:
            rr = [r for r in regime_rows if r["regime"] == regime and r["model"] == name]
            ax.plot([r["lead_min"] for r in rr], [r["pr_auc"] for r in rr], marker="o",
                    color=colors[name], label=name)
            if np.isfinite(rr[0]["ci_lo"]):
                ax.fill_between([r["lead_min"] for r in rr], [r["ci_lo"] for r in rr],
                                [r["ci_hi"] for r in rr], color=colors[name], alpha=0.15)
        base = [r["base_rate"] for r in regime_rows if r["regime"] == regime][::len(runs)]
        ax.plot([(t + 1) * args.dt_min for t in range(T_out)], base, "k:", label="base rate")
        ax.set_title(f"{regime.capitalize()} (activity within {args.radius_km:g} km, last 30 min: "
                     f"{'no' if regime == 'initiation' else 'yes'})", fontsize=9)
        ax.set_xlabel("Lead time (min)")
    axes[0].set_ylabel("Average precision" if args.metric == "ap" else "PR-AUC")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "fig7_initiation_vs_continuation.png"), dpi=300)
    plt.close(fig)

    names = args.calib_models or list(runs)
    show = sorted({0, T_out // 2 - 1 if T_out > 2 else 0, T_out - 1})
    fig, axes = plt.subplots(1, len(show) + 1, figsize=(4.4 * (len(show) + 1), 4.4))
    for ax, t in zip(axes, show):
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        for name in names:
            if (name, t) in curves:
                mean, frac = curves[(name, t)]
                ax.plot(mean, frac, marker="o", ms=3, color=colors[name], label=name)
        ax.set_title(f"Reliability, +{(t + 1) * args.dt_min} min")
        ax.set_xlabel("Forecast probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(fontsize=8)
    ax = axes[-1]
    for name in names:
        rr = [r for r in calib_rows if r["model"] == name]
        ax.errorbar([r["lead_min"] for r in rr], [r["brier"] for r in rr],
                    yerr=[[r["brier"] - r["brier_ci_lo"] for r in rr],
                          [r["brier_ci_hi"] - r["brier"] for r in rr]],
                    marker="o", capsize=3, color=colors[name], label=name)
    ax.set_title("Brier score (lower is better)")
    ax.set_xlabel("Lead time (min)")
    fig.tight_layout()
    fig.savefig(os.path.join(od, "fig8_calibration_all_models.png"), dpi=300)
    plt.close(fig)

    for rows, key, fname, xlabel in [
        (diurnal_rows, "hour", "figS_diurnal_skill.png",
         "Issue time (local solar h)" if args.region_lon else "Issue time (UTC h)"),
        (region_rows, "region", "figS_region_skill.png", "Region")]:
        if not rows:
            continue
        groups = sorted({r[key] for r in rows}, key=str)
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(groups) * len(runs) / 3), 4.5))
        width = 0.8 / len(runs)
        for i, name in enumerate(runs):
            vals = [next((r["pr_auc"] for r in rows if r[key] == g and r["model"] == name), np.nan)
                    for g in groups]
            ax.bar(np.arange(len(groups)) + i * width, vals, width, color=colors[name], label=name)
        ax.set_xticks(np.arange(len(groups)) + 0.4 - width / 2)
        ax.set_xticklabels([str(g) for g in groups], rotation=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(("Average precision" if args.metric == "ap" else "PR-AUC") + " (all leads pooled)")
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(os.path.join(od, fname), dpi=300)
        plt.close(fig)
    print(f"  figures -> {od}/fig7_*.png, fig8_*.png, figS_*.png")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    meta, runs, T_out = load_all(args)

    print("\n(1) Initiation vs continuation")
    regime_rows = regime_analysis(meta, runs, T_out, args, rng)
    write_csv(os.path.join(args.output_dir, "regime_initiation_continuation.csv"), regime_rows)

    print("\n(2) Calibration / Brier")
    calib_rows, curves = calibration_analysis(runs, T_out, args, rng)
    write_csv(os.path.join(args.output_dir, "calibration_brier.csv"), calib_rows)

    print("\n(3) Diurnal")
    diurnal_rows = grouped_analysis(meta, runs, T_out, hour_groups(meta, args), "hour", args, rng)
    write_csv(os.path.join(args.output_dir, "diurnal_skill.csv"), diurnal_rows)

    print("\n(4) Per region")
    names = meta["region_names"]
    region_rows = grouped_analysis(meta, runs, T_out, [str(names[i]) for i in meta["seq_region"]],
                                   "region", args, rng)
    write_csv(os.path.join(args.output_dir, "region_skill.csv"), region_rows)

    make_figures(regime_rows, curves, calib_rows, diurnal_rows, region_rows, args, runs, T_out)


if __name__ == "__main__":
    main()
