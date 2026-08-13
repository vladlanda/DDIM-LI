"""
Train the LightGBM baseline. See baseline_lightgbm/features.py's docstring
for the feature design and scope rationale (in the methodological spirit
of Song et al. 2023, not a literal reproduction of their coarser protocol
-- see FINDINGS.md F1/F3, PAPER_TODO.md Phase 3).

Reuses dataset.py's make_dataloaders (same train/val split, same stats
computation, same channels/T_in/T_out/img_size as the diffusion model and
CNN baseline configs) -- NOT a separate data pipeline, so the same
sequences and the same normalization stats are used everywhere in this
project, which matters for the paired bootstrap CI comparison later
(bootstrap_pr_auc_ci.py's validity requirement: every npz being compared
must use the same test_roots/T_in/T_out/img_size).

Single model amortized across all 6 lead times via a `lead_idx` feature,
NOT one model per lead time -- same reasoning as the CNN baseline's
lead-time conditioning (see FINDINGS.md C7): holding structure constant
across our own baselines isolates model CLASS as the controlled variable
in our internal comparisons, which matters more here than matching any
one piece of external literature's exact protocol.

Class imbalance handled via LightGBM's native `is_unbalance=True` (not
manual pixel oversampling) -- the idiomatic LightGBM approach, avoids
introducing a second, undocumented imbalance-handling mechanism alongside
the CNN baseline's WeightedRandomSampler oversampling.

Usage:
  python baseline_lightgbm/train_lightgbm.py --config configs/default.yaml \
      --output_dir baseline_lightgbm/outputs/run1
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import make_dataloaders
from config import load_yaml
from features import FEATURE_NAMES, _li_to_physical, compute_feature_maps, feature_maps_to_matrix  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train_roots", nargs="+", default=None)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--T_in", type=int, default=None)
    p.add_argument("--T_out", type=int, default=None)
    p.add_argument("--dt_min", type=int, default=None)
    p.add_argument("--img_size", nargs=2, type=int, default=None)
    p.add_argument("--channels", nargs="+", default=None)
    p.add_argument("--binary_li_ctx", type=lambda x: x.lower() == "true", default=None)
    p.add_argument("--ctx_channels", nargs="+", default=None)
    p.add_argument("--li_event_threshold", type=float, default=5.0 / 255.0,
                   help="Matches the CNN baseline's default exactly -- see "
                        "that script's help text for the physical-space "
                        "rationale (cbrt preserves zero exactly).")

    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--train_val_split", type=float, default=None)
    p.add_argument("--max_train_batches", type=int, default=400,
                   help="Number of TRAINING batches to draw sequences from "
                        "for feature extraction. Not 'epochs' -- LightGBM "
                        "trains once on a fixed tabular set, this just "
                        "bounds how many sequences contribute to it.")
    p.add_argument("--max_val_batches", type=int, default=80)
    p.add_argument("--pixels_per_image_lead", type=int, default=1500,
                   help="Random pixels subsampled per (sequence, lead_idx) "
                        "pair for the TRAINING set (uniform random, not "
                        "class-stratified -- imbalance handled by "
                        "is_unbalance=True instead).")
    p.add_argument("--val_pixels_per_image_lead", type=int, default=5000,
                   help="Same subsampling for the VALIDATION set, at a "
                        "larger default for a more precise early-stopping "
                        "signal. Deliberately still bounded, NOT full-image "
                        "-- at default --max_val_batches with 256x256 "
                        "images and T_out=6, full-image extraction would be "
                        "~10-20GB for the validation matrix alone (a real "
                        "bug caught during testing, not a hypothetical: see "
                        "FINDINGS.md), which is impractical on most "
                        "machines. Evaluation (evaluate_lightgbm.py) still "
                        "always uses every pixel of the TEST set -- that's "
                        "a one-time cost per sequence, not accumulated "
                        "across hundreds of validation batches during "
                        "training.")

    p.add_argument("--num_boost_round", type=int, default=2000)
    p.add_argument("--early_stopping_rounds", type=int, default=50)
    p.add_argument("--num_leaves", type=int, default=63)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--min_child_samples", type=int, default=100)
    p.add_argument("--feature_fraction", type=float, default=0.8)
    p.add_argument("--bagging_fraction", type=float, default=0.8)
    p.add_argument("--bagging_freq", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _resolve_config(args):
    cfg = load_yaml(args.config)
    for k in ["train_roots", "T_in", "T_out", "dt_min", "img_size", "channels",
              "binary_li_ctx", "ctx_channels", "batch_size", "train_val_split"]:
        if getattr(args, k) is None and k in cfg:
            setattr(args, k, cfg[k])
    assert args.train_roots, "--train_roots not set and not in config"
    args.img_size = tuple(args.img_size) if args.img_size else (256, 256)
    args.batch_size = args.batch_size or 4
    args.train_val_split = args.train_val_split or 0.7
    # Resolve to the actual effective value now, not at the call site --
    # otherwise meta.json could record binary_li_ctx=null while training
    # actually used True (make_dataloaders' own default), and
    # evaluate_lightgbm.py reading that null back would silently diverge
    # from what was actually trained on.
    args.binary_li_ctx = True if args.binary_li_ctx is None else args.binary_li_ctx
    return args


def extract_dataset(loader, max_batches, li_idx, li_event_threshold,
                    pixels_per_image_lead, rng, is_train, stats):
    """Iterate `loader`, extracting a tabular (X, y) feature/label dataset.

    ALWAYS subsamples pixels_per_image_lead random pixels per (sequence,
    lead) pair, for both train and validation -- full-image extraction
    (H*W pixels x T_out leads x hundreds of sequences) was tested and
    found to reach ~10-20GB for the validation set alone at realistic
    defaults, before this function was changed to subsample validation
    too (see --val_pixels_per_image_lead's help text). Use a larger
    pixels_per_image_lead for validation than training if you want a
    more precise early-stopping signal -- that's a size knob, not a
    full-vs-subsampled distinction anymore.
    """
    X_parts, y_parts = [], []
    n_seq = 0
    t0 = time.time()
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        context  = batch["context"].numpy()   # (B, T_in, C, H, W)
        target   = batch["target"].numpy()    # (B, T_out, C, H, W)
        last_ctx = batch["last_ctx"].numpy()  # (B, C, H, W)
        B, T_out = context.shape[0], target.shape[1]
        H, W = context.shape[-2], context.shape[-1]

        for b in range(B):
            ir_ctx = context[b, :, 0]                          # (T_in, H, W)
            li_ctx_phys = _li_to_physical(context[b, :, li_idx], stats)  # (T_in, H, W)
            # Computed ONCE per sequence -- does not depend on lead_idx,
            # reused for all T_out lead times below (see features.py's
            # feature_maps_to_matrix docstring for why this matters).
            feats = compute_feature_maps(ir_ctx, li_ctx_phys, li_event_threshold)

            for t in range(T_out):
                tgt_abs_li = target[b, t, li_idx] + last_ctx[b, li_idx]
                obs_phys = _li_to_physical(tgt_abs_li, stats)
                y_full = (obs_phys >= li_event_threshold).astype(np.float32)  # (H, W)

                n_pix = min(pixels_per_image_lead, H * W)
                flat_idx = rng.choice(H * W, size=n_pix, replace=False)
                pixel_indices = np.stack([flat_idx // W, flat_idx % W], axis=1)
                y = y_full[pixel_indices[:, 0], pixel_indices[:, 1]]

                X = feature_maps_to_matrix(feats, lead_idx=t, pixel_indices=pixel_indices)
                X_parts.append(X)
                y_parts.append(y)
            n_seq += 1

        if bi % 20 == 0:
            logger.info(f"  [{'train' if is_train else 'val'}] batch {bi}/{max_batches}, "
                       f"{n_seq} sequences so far, {time.time()-t0:.0f}s elapsed")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    logger.info(f"[{'train' if is_train else 'val'}] final dataset: "
               f"X={X.shape}, y={y.shape}, positive_frac={y.mean():.4f}")
    return X, y


def main():
    if lgb is None:
        raise ImportError("lightgbm is not installed. pip install lightgbm --break-system-packages")

    args = parse_args()
    args = _resolve_config(args)
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    logger.info(f"Building dataloaders from {len(args.train_roots)} train_roots")
    train_loader, val_loader, stats = make_dataloaders(
        train_roots=args.train_roots, channel_list=args.channels,
        T_in=args.T_in, T_out=args.T_out, img_size=args.img_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
        train_val_split=args.train_val_split,
        binary_li_ctx=args.binary_li_ctx,
        ctx_channels=args.ctx_channels,
    )
    li_idx = args.channels.index("li")
    logger.info(f"Extracting TRAIN features (up to {args.max_train_batches} batches, "
               f"{args.pixels_per_image_lead} px/image/lead)...")
    X_train, y_train = extract_dataset(
        train_loader, args.max_train_batches, li_idx, args.li_event_threshold,
        args.pixels_per_image_lead, rng, is_train=True, stats=stats,
    )
    logger.info(f"Extracting VAL features (up to {args.max_val_batches} batches, "
               f"{args.val_pixels_per_image_lead} px/image/lead)...")
    X_val, y_val = extract_dataset(
        val_loader, args.max_val_batches, li_idx, args.li_event_threshold,
        args.val_pixels_per_image_lead, rng, is_train=False, stats=stats,
    )

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_set)

    params = dict(
        objective="binary", metric=["binary_logloss", "auc"],
        num_leaves=args.num_leaves, learning_rate=args.learning_rate,
        min_child_samples=args.min_child_samples,
        feature_fraction=args.feature_fraction, bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq, is_unbalance=True, seed=args.seed, verbose=-1,
    )
    logger.info(f"Training LightGBM: {params}")
    booster = lgb.train(
        params, train_set, num_boost_round=args.num_boost_round,
        valid_sets=[train_set, val_set], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=True),
                  lgb.log_evaluation(period=50)],
    )

    model_path = os.path.join(args.output_dir, "model.txt")
    booster.save_model(model_path, num_iteration=booster.best_iteration)
    logger.info(f"Saved model -> {model_path} (best_iteration={booster.best_iteration})")

    meta = dict(
        stats=stats, channels=args.channels, T_in=args.T_in, T_out=args.T_out,
        dt_min=args.dt_min, img_size=list(args.img_size),
        binary_li_ctx=args.binary_li_ctx, ctx_channels=args.ctx_channels,
        li_event_threshold=args.li_event_threshold, feature_names=FEATURE_NAMES,
        best_iteration=booster.best_iteration, best_score=booster.best_score,
        args=vars(args),
    )
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Saved metadata -> {meta_path}")

    importances = sorted(zip(FEATURE_NAMES, booster.feature_importance(importance_type="gain")),
                         key=lambda kv: -kv[1])
    logger.info("Feature importances (gain):")
    for name, imp in importances:
        logger.info(f"  {name:30s} {imp:>12.1f}")


if __name__ == "__main__":
    main()
