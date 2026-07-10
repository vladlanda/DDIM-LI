"""
Regime-resolved, CONDITIONAL channel information for lightning nowcasting.

Why this is not a repeat of LightningCast's channel ablation
------------------------------------------------------------
Cintineo et al. (2022) added/removed ABI bands and looked at total validation
skill. That conflates two very different failures:
  * a channel that is genuinely uninformative, and
  * a channel that is informative but REDUNDANT with a channel already used.
It also yields a single scalar, so it cannot say WHICH error a channel fixes.

Our error decomposition (FSS-by-threshold, positional-ceiling) showed lightning
skill fails in two separable regimes:
  EXISTENCE  — does this ~28 km region electrify at all?   (low-prob regime)
  POSITION   — given it electrifies, which pixel lights up? (high-prob regime)

So we ask, per channel, per lead time, TWO questions, each CONDITIONAL on what
ir105 + LI history already tell us:

  I_exist(X) : does X discriminate block-level occurrence, within strata of
               (ir105 level x LI-history state)?
  I_pos(X)   : GIVEN a block is active, does X discriminate WHICH pixel,
               within the same strata?

Measured as conditional AUC. 0.5 == adds nothing beyond the conditioning set.
A random feature is included as a null control; it must return ~0.5.

No model, no GPU — this is a property of the DATA.

Usage:
  python diagnose_channel_information.py \
      --root /path/to/full/central_africa_2 \
      --n_seq 200 --block 7
"""
import argparse, logging
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from dataset import METSATDataset, compute_or_load_stats, denormalize

LI_THR = 5.0 / 255.0


def _li_phys(arr, stats):
    x = arr * stats["li"]["std"] + stats["li"]["mean"]
    if stats["li"].get("transform") == "cbrt":
        x = np.power(np.clip(x, 0, None), 3)
    return np.clip(x, 0, 1)


def block_reduce(a, b, how):
    """Reduce (H,W) to (H//b, W//b) by max or mean."""
    H, W = a.shape
    H, W = (H // b) * b, (W // b) * b
    v = a[:H, :W].reshape(H // b, b, W // b, b)
    return v.max(axis=(1, 3)) if how == "max" else v.mean(axis=(1, 3))


def cond_auc(x, y, strata, min_per_stratum=200):
    """
    Stratified AUC: within each stratum compute AUC(x -> y), then average
    weighted by stratum size. Removes the information already carried by the
    stratifying variables (ir105 level, LI history).
    """
    vals, wts = [], []
    for s in np.unique(strata):
        m = strata == s
        if m.sum() < min_per_stratum:
            continue
        ys = y[m]
        if ys.min() == ys.max():          # need both classes
            continue
        try:
            vals.append(roc_auc_score(ys, x[m]))
            wts.append(m.sum())
        except ValueError:
            continue
    if not vals:
        return np.nan
    return float(np.average(vals, weights=wts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--n_seq", type=int, default=200)
    p.add_argument("--block", type=int, default=7, help="block size (7 px = 28 km)")
    p.add_argument("--T_in", type=int, default=36)
    p.add_argument("--T_out", type=int, default=6)
    p.add_argument("--dt_min", type=int, default=10)
    p.add_argument("--img_size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--n_ir_bins", type=int, default=20,
                   help="Strata for ir105. More bins = less leakage "
                        "of ir105 into the conditional AUC (0.58 at 5 "
                        "bins -> 0.53 at 20 -> 0.50 at 80).")
    p.add_argument("--pix_stride", type=int, default=7)
    args = p.parse_args()

    chans = ["ir", "li", "ch0", "ch1", "ch2"]
    stats = compute_or_load_stats(args.root, chans, stat_path=f"{args.root}/stats.json")
    ds = METSATDataset(args.root, channel_list=chans, T_in=args.T_in, T_out=args.T_out,
                       dt_min=args.dt_min, img_size=tuple(args.img_size), stats=stats,
                       augment=False, binary_li_ctx=False, ctx_channels=None)
    stride = max(1, len(ds.valid_sequences) // args.n_seq)
    ds.valid_sequences = ds.valid_sequences[::stride][:args.n_seq]
    logger.info(f"Using {len(ds.valid_sequences)} sequences from {args.root}")

    ci = {c: chans.index(c) for c in chans}
    B = args.block

    # accumulators: per lead, per feature -> lists
    # ir105(ctrl) is the CONDITIONING variable itself. Because stratification is
    # finite, some of ir105 leaks through; its conditional AUC is therefore the
    # LEAKAGE FLOOR. Any channel at or below it is redundant with ir105.
    feats = ["ir105(ctrl)", "ch0(ir87)", "ch1(ir97)", "ch2(ir123)",
             "BTD(105-123)", "BTD(105-87)", "BTD(105-97)", "RANDOM(null)"]
    acc = {t: {f: {"x_e": [], "x_p": []} for f in feats} for t in range(args.T_out)}
    # store the RAW conditioning variables; strata are built globally at report
    # time. Per-sequence quantile bins are NOT comparable across sequences and
    # badly under-condition (leakage floor 0.106 vs 0.028 in validation).
    acc_y = {t: {"y_e": [], "y_p": [],
                 "ir_e": [], "ir_p": [], "lh_e": [], "lh_p": []}
             for t in range(args.T_out)}

    rng = np.random.default_rng(0)

    for i in tqdm(range(len(ds.valid_sequences)), desc="sequences"):
        b = ds[i]
        ctx = b["context"].numpy()                     # (T_in, C, H, W) normalised
        tgt = b["target"].numpy() + b["last_ctx"].numpy()[None]   # absolute
        t0  = ctx[-1]

        # --- conditioning set: ir105 level, and LI history ---
        ir0 = denormalize(t0[ci["ir"]], stats, "ir")   # [0,1] pixel scale
        li_hist = np.zeros_like(ir0, dtype=bool)
        for k in range(1, 7):                          # last hour of LI
            li_hist |= _li_phys(ctx[-k, ci["li"]], stats) >= LI_THR

        # --- candidate features at t0 (physical [0,1] scale) ---
        d = {c: denormalize(t0[ci[c]], stats, c) for c in ["ir", "ch0", "ch1", "ch2"]}
        F = {
            "ir105(ctrl)":  d["ir"],
            "ch0(ir87)":    d["ch0"],
            "ch1(ir97)":    d["ch1"],
            "ch2(ir123)":   d["ch2"],
            "BTD(105-123)": d["ir"] - d["ch2"],
            "BTD(105-87)":  d["ir"] - d["ch0"],
            "BTD(105-97)":  d["ir"] - d["ch1"],
            "RANDOM(null)": rng.standard_normal(ir0.shape),
        }

        for t in range(args.T_out):
            y = (_li_phys(tgt[t, ci["li"]], stats) >= LI_THR)

            # ---------- EXISTENCE: block level ----------
            y_e   = block_reduce(y.astype(float), B, "max").ravel()
            ir_e  = block_reduce(ir0, B, "mean").ravel()
            lih_e = block_reduce(li_hist.astype(float), B, "max").ravel()
            acc_y[t]["y_e"].append(y_e)
            acc_y[t]["ir_e"].append(ir_e); acc_y[t]["lh_e"].append(lih_e)
            for f in feats:
                acc[t][f]["x_e"].append(block_reduce(F[f], B, "mean").ravel())

            # ---------- POSITION: pixels inside ACTIVE blocks ----------
            act = np.repeat(np.repeat(block_reduce(y.astype(float), B, "max"), B, 0), B, 1)
            Hc, Wc = act.shape
            sel = act.astype(bool)
            idx = np.zeros_like(sel); idx[::args.pix_stride, ::args.pix_stride] = True
            sel &= idx
            if sel.sum() < 50:
                continue
            yy   = y[:Hc, :Wc][sel]
            if yy.min() == yy.max():
                continue
            acc_y[t]["y_p"].append(yy.astype(float))
            acc_y[t]["ir_p"].append(ir0[:Hc, :Wc][sel])
            acc_y[t]["lh_p"].append(li_hist[:Hc, :Wc][sel].astype(float))
            for f in feats:
                acc[t][f]["x_p"].append(F[f][:Hc, :Wc][sel])

    # ---------------- report ----------------
    print("\n" + "=" * 96)
    print("CONDITIONAL channel information, resolved by ERROR REGIME")
    print(f"  conditioned on: ir105 level ({args.n_ir_bins} bins) x LI-in-last-hour (2 states)")
    print(f"  block = {args.block}px = {args.block*4} km    AUC 0.5 => adds NOTHING beyond conditioning set")
    print("=" * 96)

    def _strata(ir, lh, nb):
        """Global quantile bins of ir105 (comparable across sequences) x LI history."""
        q = np.quantile(ir, np.linspace(0, 1, nb + 1)[1:-1])
        return np.digitize(ir, q) * 2 + lh.astype(int)

    for regime, kx, ky, kir, klh, label in [
        ("EXISTENCE", "x_e", "y_e", "ir_e", "lh_e", "does the block electrify at all?"),
        ("POSITION",  "x_p", "y_p", "ir_p", "lh_p", "which pixel, given the block is active?"),
    ]:
        print(f"\n--- {regime}: {label} ---")
        print("    values are |AUC - 0.5| (informativeness). Direction is irrelevant:")
        print("    ir105 is ANTI-correlated with lightning (cold top = dark = flashes).")
        hdr = f"{'feature':<15}" + "".join(f"{(t+1)*args.dt_min:>9}m" for t in range(args.T_out))
        print(hdr); print("-" * len(hdr))
        for f in feats:
            row = f"{f:<15}"
            for t in range(args.T_out):
                if not acc_y[t][ky]:
                    row += f"{'--':>10}"; continue
                x  = np.concatenate(acc[t][f][kx])
                y  = np.concatenate(acc_y[t][ky])
                ir = np.concatenate(acc_y[t][kir])
                lh = np.concatenate(acc_y[t][klh])
                a  = cond_auc(x, y, _strata(ir, lh, args.n_ir_bins))
                row += f"{abs(a-0.5):>10.3f}" if np.isfinite(a) else f"{'--':>10}"
            print(row)

    print("\nREAD (in this order):")
    print("  1. RANDOM(null) must sit at ~0.000, else the estimator is biased.")
    print("  2. ir105(ctrl) is the LEAKAGE FLOOR (not 0). Finite stratification lets")
    print("     some ir105 through. Judge every channel against THIS number.")
    print("  3. A channel at or below the floor is REDUNDANT with ir105 + LI history.")
    print("     Its marginal skill is irrelevant: dropping it costs nothing.")
    print("  4. EXISTENCE vs POSITION: a channel may inform one regime and not the")
    print("     other. That asymmetry is the finding LightningCast's scalar")
    print("     ablation cannot express.")


if __name__ == "__main__":
    main()
