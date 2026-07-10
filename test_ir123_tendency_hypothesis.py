"""
ONE pre-registered test per channel. No sweeping.

HYPOTHESIS (fixed before running, same form for every --target_ch):
    The cooling TENDENCY of the target channel over the convective-
    development timescale (~2h, per the saturation found in
    diagnose_cooling_rate.py) carries EXISTENCE information about
    lightning at +40 to +60 min, BEYOND what ir105's own level and
    ir105's own tendency already explain.

    Run ONCE per channel (ir123, then ir87) with the SAME statistic,
    CI method, and decision rule. Not a sweep: each channel's test is
    independent and pre-registered on its own, before seeing that
    channel's result.

STATISTIC (fixed before running):
    Conditional |AUC-0.5| of d_ch2 = -(ir123[t0] - ir123[t0-12]) against
    block-level lightning occurrence at leads {40,50,60} min POOLED into one
    sample, conditioned on strata of (ir105 level x ir105 tendency x LI
    history in the last hour). One statistic. One 95% CI (sequence bootstrap,
    500 resamples). No other feature, lead, or regime is tested here.

CONTROLS (fixed before running):
    RANDOM(null)     — must sit at ~0.
    d_ir105(floor)   — ir105's OWN tendency, scored under the same
                       conditioning. Since ir105-tendency is one of the
                       conditioning variables, this is the leakage floor
                       (same logic as the ir105(ctrl) floor in the level-only
                       diagnostic).

DECISION RULE (fixed before running):
    SUPPORTED   if the 95% CI of d_ch2 lies entirely above the 95% CI of
                RANDOM(null) AND entirely above the point estimate of
                d_ir105(floor).
    NOT SUPPORTED otherwise. No alternative statistic will be substituted
                post hoc.

Usage:
  python test_ir123_tendency_hypothesis.py \
      --root /path/to/full/central_africa_2 --n_seq 200 --block 7 \
      --tend_lag 12 --n_ir_bins 20 --n_dir_bins 5 --n_boot 500
"""
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dataset import METSATDataset, compute_or_load_stats, denormalize

LI_THR = 5.0 / 255.0


def _li_phys(arr, stats):
    x = arr * stats["li"]["std"] + stats["li"]["mean"]
    if stats["li"].get("transform") == "cbrt":
        x = np.power(np.clip(x, 0, None), 3)
    return np.clip(x, 0, 1)


def block_reduce(a, b, how):
    H, W = a.shape
    H, W = (H // b) * b, (W // b) * b
    v = a[:H, :W].reshape(H // b, b, W // b, b)
    return v.max(axis=(1, 3)) if how == "max" else v.mean(axis=(1, 3))


def cond_info(x, y, strata, min_per_stratum=200):
    vals, wts = [], []
    for s in np.unique(strata):
        m = strata == s
        if m.sum() < min_per_stratum:
            continue
        ys = y[m]
        if ys.min() == ys.max():
            continue
        try:
            vals.append(roc_auc_score(ys, x[m]))
            wts.append(m.sum())
        except ValueError:
            continue
    if not vals:
        return np.nan
    return abs(float(np.average(vals, weights=wts)) - 0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--n_seq", type=int, default=200)
    p.add_argument("--block", type=int, default=7)
    p.add_argument("--T_in", type=int, default=36)
    p.add_argument("--T_out", type=int, default=6)
    p.add_argument("--dt_min", type=int, default=10)
    p.add_argument("--img_size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--tend_lag", type=int, default=12)
    p.add_argument("--n_ir_bins", type=int, default=20)
    p.add_argument("--n_dir_bins", type=int, default=5)
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--target_ch", choices=["ch2", "ch0", "ch1"], default="ch2",
                   help="Channel under test: ch2=ir123, ch0=ir87, ch1=ir97.")
    p.add_argument("--target_name", default=None,
                   help="Display name (default: inferred from --target_ch).")
    args = p.parse_args()
    _names = {"ch2": "ir123", "ch0": "ir87", "ch1": "ir97"}
    target_name = args.target_name or _names[args.target_ch]

    LEADS = [40, 50, 60]  # fixed by the hypothesis, minutes
    lead_steps = [t for t in range(args.T_out) if (t + 1) * args.dt_min in LEADS]
    assert lead_steps, "T_out too small for the +40/50/60min leads."

    chans = ["ir", "li", "ch0", "ch1", "ch2"]
    stats = compute_or_load_stats(args.root, chans, stat_path=f"{args.root}/stats.json")
    ds = METSATDataset(args.root, channel_list=chans, T_in=args.T_in, T_out=args.T_out,
                       dt_min=args.dt_min, img_size=tuple(args.img_size), stats=stats,
                       augment=False, binary_li_ctx=False, ctx_channels=None)
    stride = max(1, len(ds.valid_sequences) // args.n_seq)
    ds.valid_sequences = ds.valid_sequences[::stride][:args.n_seq]
    print(f"sequences: {len(ds.valid_sequences)}   leads pooled: {LEADS} min\n")

    ci = {c: chans.index(c) for c in chans}
    B = args.block
    rng = np.random.default_rng(0)

    Y, DCH2, DIR, IR_LVL, IR_TEND, LH, SID, RAND = ([] for _ in range(8))

    for i in tqdm(range(len(ds.valid_sequences)), desc="sequences"):
        b = ds[i]
        ctx = b["context"].numpy()
        tgt = b["target"].numpy() + b["last_ctx"].numpy()[None]
        t0, tp = ctx[-1], ctx[-1 - args.tend_lag]

        ir0  = denormalize(t0[ci["ir"]],  stats, "ir")
        irp  = denormalize(tp[ci["ir"]],  stats, "ir")
        tgt_0 = denormalize(t0[ci[args.target_ch]], stats, args.target_ch)
        tgt_p = denormalize(tp[ci[args.target_ch]], stats, args.target_ch)
        d_ir  = -(ir0 - irp)          # ir105 tendency (positive = cooled)
        d_ch2 = -(tgt_0 - tgt_p)      # target channel's tendency

        li_hist = np.zeros_like(ir0, dtype=bool)
        for k in range(1, 7):
            li_hist |= _li_phys(ctx[-k, ci["li"]], stats) >= LI_THR

        for t in lead_steps:
            y   = block_reduce((_li_phys(tgt[t, ci["li"]], stats) >= LI_THR).astype(float),
                               B, "max").ravel()
            Y.append(y)
            DCH2.append(block_reduce(d_ch2, B, "mean").ravel())
            DIR.append(block_reduce(d_ir, B, "mean").ravel())
            IR_LVL.append(block_reduce(ir0, B, "mean").ravel())
            IR_TEND.append(block_reduce(d_ir, B, "mean").ravel())
            LH.append(block_reduce(li_hist.astype(float), B, "max").ravel())
            SID.append(np.full(y.size, i))
            RAND.append(rng.standard_normal(y.size))

    Y, DCH2, DIR, IR_LVL, IR_TEND, LH, SID, RAND = (
        np.concatenate(v) for v in (Y, DCH2, DIR, IR_LVL, IR_TEND, LH, SID, RAND))

    def bins(x, nb):
        q = np.quantile(x, np.linspace(0, 1, nb + 1)[1:-1])
        return np.digitize(x, q)

    strata = ((bins(IR_LVL, args.n_ir_bins) * args.n_dir_bins
               + bins(IR_TEND, args.n_dir_bins)) * 2 + LH.astype(int))

    def boot_ci(x):
        uniq = np.unique(SID)
        vals = []
        for _ in range(args.n_boot):
            pick = rng.choice(uniq, size=uniq.size, replace=True)
            m = np.isin(SID, pick)
            if m.sum() < 500:
                continue
            v = cond_info(x[m], Y[m], strata[m])
            if np.isfinite(v):
                vals.append(v)
        pt = cond_info(x, Y, strata)
        if len(vals) < 20:
            return pt, np.nan, np.nan
        return pt, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    pt_target, lo_target, hi_target = boot_ci(DCH2)
    pt_floor,  lo_floor,  hi_floor  = boot_ci(DIR)
    pt_null,   lo_null,   hi_null   = boot_ci(RAND)

    print("=" * 70)
    print(f"PRE-REGISTERED TEST: d_{target_name} tendency, EXISTENCE, +40/50/60min")
    print("=" * 70)
    print(f"{'quantity':<22}{'point':>8}{'95% CI':>20}")
    print("-" * 50)
    print(f"{'d_'+target_name+' (target)':<22}{pt_target:>8.4f}   [{lo_target:.4f}, {hi_target:.4f}]")
    print(f"{'d_ir105 (floor)':<22}{pt_floor:>8.4f}   [{lo_floor:.4f}, {hi_floor:.4f}]")
    print(f"{'RANDOM (null)':<22}{pt_null:>8.4f}   [{lo_null:.4f}, {hi_null:.4f}]")
    print("-" * 50)

    supported = (np.isfinite(lo_target) and np.isfinite(hi_null) and np.isfinite(pt_floor)
                and lo_target > hi_null and lo_target > pt_floor)

    print(f"\nDECISION RULE: d_ch2's CI entirely above RANDOM's CI AND above d_ir105's point estimate")
    print(f"VERDICT: {'SUPPORTED' if supported else 'NOT SUPPORTED'}")
    if not supported:
        print(f"\n-> {target_name} tendency adds nothing beyond ir105 level+tendency+")
        print(f"   LI history in the existence regime at 40-60min. Do not add {target_name}.")
    else:
        print(f"\n-> {target_name} tendency carries existence information ir105 does not.")
        print("   This is the result to act on for this channel.")


if __name__ == "__main__":
    main()
