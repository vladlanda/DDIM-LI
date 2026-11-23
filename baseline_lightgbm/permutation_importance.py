"""
Permutation importance of the LightGBM baseline's inputs on the TEST set: which physical
predictors carry the skill. Gain importance (printed at training) measures how often a
feature is used for splits; permutation importance measures how much test AP is LOST when
that feature's values are shuffled, which is what the paper can interpret physically.

    TEST_SEQ_EVERY=4 python baseline_lightgbm/permutation_importance.py \
        --model_dir <LightGBM run dir> --test_roots <4 test dirs> --output <csv>

Rows sampled: every --px_stride-th pixel per axis of every test sequence (env
TEST_SEQ_EVERY thins sequences), all 6 leads. Each feature (and each physical GROUP of
correlated features, shuffled jointly) is permuted --n_repeats times WITHIN each lead
(lead_idx is never permuted). Reported: AP drop (base AP - permuted AP), mean and std over
repeats, for ALL pixels and for INITIATION pixels (no lightning within 20 km in the last
30 min of context -- the same rule as build_test_metadata.py + analysis_regimes.py).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.ndimage import maximum_filter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import make_test_loader                                   # noqa: E402
from bootstrap_pr_auc_ci import average_precision                      # noqa: E402
from features import FEATURE_NAMES, compute_feature_maps, _li_to_physical  # noqa: E402

GROUPS = {
    "GROUP: all IR (cloud-top) features": [f for f in FEATURE_NAMES if f.startswith("ir_")],
    "GROUP: all LI (lightning history) features": [f for f in FEATURE_NAMES if f.startswith("li_")],
    "GROUP: IR cooling trends (30 min, 2 h)": ["ir_trend_30min", "ir_trend_2h"],
    "GROUP: IR current state (now, local mean/min)": ["ir_now", "ir_local_mean_9", "ir_local_min_9"],
}


def collect(args, meta):
    """Feature rows, labels and initiation flags on a sparse pixel grid of every test sequence."""
    stats, channels = meta["stats"], meta["channels"]
    li = channels.index("li")
    thr, T_out = meta["li_event_threshold"], meta["T_out"]
    loader = make_test_loader(test_roots=args.test_roots, channel_list=channels, stats=stats,
                              T_in=meta["T_in"], T_out=T_out, img_size=(256, 256),
                              batch_size=8, num_workers=args.num_workers,
                              binary_li_ctx=meta.get("binary_li_ctx", True),
                              ctx_channels=meta.get("ctx_channels"))
    S, r_px = args.px_stride, int(np.floor(args.radius_km / args.pixel_km))
    X, y, init = [[] for _ in range(T_out)], [[] for _ in range(T_out)], [[] for _ in range(T_out)]
    names = [f for f in FEATURE_NAMES if f != "lead_idx"]
    for batch in tqdm(loader, desc="features", unit="batch"):
        ctx, tgt, last = batch["context"].numpy(), batch["target"].numpy(), batch["last_ctx"].numpy()
        for b in range(ctx.shape[0]):
            li_ctx = _li_to_physical(ctx[b, :, li], stats)
            f = compute_feature_maps(ctx[b, :, 0], li_ctx, thr)
            cols = np.stack([f[n][::S, ::S].ravel() for n in names], axis=1)
            recent = (li_ctx[-args.recent_frames:] >= thr).any(axis=0)
            # chessboard distance > r_px  <=>  nothing active in the (2 r_px + 1)^2 square
            ini = ~maximum_filter(recent, size=2 * r_px + 1, mode="constant", cval=False)[::S, ::S].ravel()
            for t in range(T_out):
                obs = _li_to_physical(tgt[b, t, li] + last[b, li], stats) >= thr
                X[t].append(np.c_[cols, np.full(len(cols), t)])
                y[t].append(obs[::S, ::S].ravel())
                init[t].append(ini)
    order = names + ["lead_idx"]
    perm = [order.index(n) for n in FEATURE_NAMES]                    # model column order
    return ([np.concatenate(a)[:, perm].astype(np.float32) for a in X],
            [np.concatenate(a).astype(np.int8) for a in y], [np.concatenate(a) for a in init])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--test_roots", nargs="+", required=True)
    p.add_argument("--output", default="permutation_importance.csv")
    p.add_argument("--px_stride", type=int, default=8, help="every k-th pixel per axis (8 -> 1024 px/image)")
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--radius_km", type=float, default=20.0)
    p.add_argument("--pixel_km", type=float, default=3.14)
    p.add_argument("--recent_frames", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    import lightgbm as lgb

    meta = json.load(open(os.path.join(args.model_dir, "meta.json")))
    assert meta["feature_names"] == FEATURE_NAMES
    booster = lgb.Booster(model_file=os.path.join(args.model_dir, "model.txt"))
    X, y, init = collect(args, meta)
    rng = np.random.default_rng(args.seed)
    print(f"{sum(len(v) for v in y):,} rows; initiation share of events: "
          f"{sum((v & i).sum() for v, i in zip(y, init)) / max(1, sum(v.sum() for v in y)):.1%}")

    def ap_all(Xs):
        pr = [booster.predict(x) for x in Xs]
        return (average_precision(np.concatenate(pr), np.concatenate(y)),
                average_precision(np.concatenate([q[i] for q, i in zip(pr, init)]),
                                  np.concatenate([v[i] for v, i in zip(y, init)])))

    base_all, base_ini = ap_all(X)
    print(f"base AP: all {base_all:.4f}, initiation {base_ini:.4f}")
    items = [(n, [n]) for n in FEATURE_NAMES if n != "lead_idx"] + list(GROUPS.items())
    rows = []
    for name, feats in tqdm(items, desc="permuting", unit="feature"):
        idx = [FEATURE_NAMES.index(f) for f in feats]
        drops = []
        for _ in range(args.n_repeats):
            Xp = []
            for x in X:                                   # shuffle within each lead
                x = x.copy()
                x[:, idx] = x[np.ix_(rng.permutation(len(x)), idx)]   # one row order for the whole group
                Xp.append(x)
            a, i = ap_all(Xp)
            drops.append((base_all - a, base_ini - i))
        d = np.array(drops)
        rows.append(dict(feature=name, drop_all=d[:, 0].mean(), drop_all_sd=d[:, 0].std(),
                         drop_initiation=d[:, 1].mean(), drop_initiation_sd=d[:, 1].std(),
                         base_ap_all=base_all, base_ap_initiation=base_ini))
    rows.sort(key=lambda r: -r["drop_all"])
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{'feature':48s} {'AP drop (all)':>15s} {'AP drop (initiation)':>22s}")
    for r in rows:
        print(f"{r['feature']:48s} {r['drop_all']:+.4f} ± {r['drop_all_sd']:.4f}   "
              f"{r['drop_initiation']:+.4f} ± {r['drop_initiation_sd']:.4f}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
