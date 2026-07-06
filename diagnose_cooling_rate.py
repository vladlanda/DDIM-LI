"""
Premise verification for the cooling-rate hypothesis (before any training).

Physical hypothesis (well-established in convective-initiation literature):
    Rapid cloud-top cooling (dT_b/dt < 0, i.e. IR brightness temperature
    dropping fast) indicates a strengthening updraft lofting ice — the
    precursor of charge separation and lightning. This should PRECEDE
    lightning initiation by minutes to tens of minutes.

This script quantifies, in YOUR data, whether that lead-lag signal exists:
  For each sequence:
    - cooling_rate = IR[last] - IR[last-k]   (in normalised units; negative
      brightness change = cooling; we flip sign so LARGER = faster cooling)
    - future_li    = lightning occurrence in target frame at lead t
  Then measure:
    (a) lightning rate in the top-decile cooling pixels vs the base rate
        -> "lift": how much more likely is lightning where it cooled fast
    (b) how lift decays with lead time (does cooling predict far ahead?)
    (c) correlation between cooling rate and future lightning probability

If cooling pixels show strong lift over base rate, the signal is real and
an auxiliary cooling-rate task is justified.

Usage:
  python diagnose_cooling_rate.py --config configs/default.yaml \
      --n_seq 300 --cool_lag 3
"""
import argparse, logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from config import load_yaml
from dataset import METSATDataset, compute_or_load_stats
try:
    from evaluate import _li_to_physical
except Exception:
    def _li_to_physical(arr, stats, ch="li"):
        if ch not in stats: return arr
        x = arr * stats[ch]["std"] + stats[ch]["mean"]
        if stats[ch].get("transform") == "cbrt":
            x = np.power(np.clip(x, 0.0, None), 3)
        return np.clip(x, 0.0, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n_seq", type=int, default=300)
    p.add_argument("--cool_lag", type=int, default=3,
                   help="Frames back over which to compute cooling rate "
                        "(3 = 30 min at dt=10).")
    p.add_argument("--li_event_threshold", type=float, default=5.0/255.0)
    p.add_argument("--roots", nargs="+", default=None,
                   help="Override roots (default: train_roots from config).")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    roots    = args.roots or cfg["train_roots"]
    channels = cfg["channels"] if isinstance(cfg.get("channels"), list) else ["ir","li","ch0","ch1"]
    T_in     = cfg["T_in"]; T_out = cfg["T_out"]; dt_min = cfg["dt_min"]
    img_size = tuple(cfg.get("img_size", [256, 256]))

    # Build a dataset over the first root (premise check doesn't need all data)
    stats = compute_or_load_stats(roots[0], channels,
                                  stat_path=f"{roots[0]}/stats.json")
    ds = METSATDataset(roots[0], channel_list=channels, T_in=T_in, T_out=T_out,
                       dt_min=dt_min, img_size=img_size, stats=stats,
                       augment=False, binary_li_ctx=False, ctx_channels=None)
    # subsample sequences
    stride = max(1, len(ds.valid_sequences) // args.n_seq)
    ds.valid_sequences = ds.valid_sequences[::stride][:args.n_seq]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    ir_idx = channels.index("ir")
    li_idx = channels.index("li")
    k = args.cool_lag

    # Accumulators per lead time
    base_rate      = [[] for _ in range(T_out)]   # overall LI rate
    hi_cool_rate   = [[] for _ in range(T_out)]   # LI rate in top-decile cooling
    corr_samples   = [[] for _ in range(T_out)]   # (cooling, future_li) pairs

    for batch in tqdm(loader, total=len(ds.valid_sequences), desc="Sequences"):
        ctx = batch["context"][0].numpy()    # (T_in, C, H, W) normalised
        tgt = batch["target"][0].numpy()     # (T_out, C, H, W) residuals
        last_ctx = ctx[-1]                   # (C, H, W)

        # Cooling rate over last k frames: IR[last] - IR[last-k].
        # In normalised IR, LOWER value = colder cloud top. Cooling = drop.
        # We define cooling = -(IR[last] - IR[last-k]) so larger = faster cool.
        if T_in <= k:
            continue
        ir_now  = ctx[-1,   ir_idx]
        ir_prev = ctx[-1-k, ir_idx]
        cooling = -(ir_now - ir_prev)        # (H, W), larger = cooled faster

        # top-decile cooling mask
        thr = np.quantile(cooling, 0.90)
        hot = cooling >= thr

        for t in range(T_out):
            # future LI in physical space (residual + last_ctx)
            li_abs  = tgt[t, li_idx] + last_ctx[li_idx]
            li_phys = _li_to_physical(li_abs, stats)
            li_bin  = (li_phys >= args.li_event_threshold).astype(np.float32)

            base_rate[t].append(float(li_bin.mean()))
            if hot.sum() > 0:
                hi_cool_rate[t].append(float(li_bin[hot].mean()))
            # subsample pixels for correlation
            flat_c = cooling.ravel()[::37]
            flat_l = li_bin.ravel()[::37]
            corr_samples[t].append((flat_c, flat_l))

    print("\n" + "=" * 66)
    print("COOLING-RATE PREMISE CHECK")
    print(f"  cooling computed over last {k} frames ({k*dt_min} min)")
    print("=" * 66)
    print(f"{'Lead':>6}  {'base_LI':>8}  {'LI|fast_cool':>13}  "
          f"{'lift':>6}  {'corr':>7}")
    print("-" * 52)
    for t in range(T_out):
        lead = (t + 1) * dt_min
        br   = np.mean(base_rate[t]) if base_rate[t] else np.nan
        hc   = np.mean(hi_cool_rate[t]) if hi_cool_rate[t] else np.nan
        lift = hc / br if br > 0 else np.nan
        # pooled correlation
        cs = np.concatenate([c for c, _ in corr_samples[t]])
        ls = np.concatenate([l for _, l in corr_samples[t]])
        corr = np.corrcoef(cs, ls)[0, 1] if cs.std() > 0 and ls.std() > 0 else np.nan
        print(f"  +{lead:3d}m  {br:>8.4f}  {hc:>13.4f}  {lift:>6.2f}  {corr:>7.3f}")

    print("-" * 52)
    print("\nREADING THE RESULT:")
    print("  lift = P(lightning | fast cooling) / P(lightning).")
    print("  lift >> 1 at short leads, decaying with lead time, confirms")
    print("  cooling-rate PRECEDES lightning -> auxiliary cooling-rate task")
    print("  is physically justified and should help existence prediction.")
    print("  lift ~ 1 (flat) would mean no predictive lead-lag signal.")


if __name__ == "__main__":
    main()
