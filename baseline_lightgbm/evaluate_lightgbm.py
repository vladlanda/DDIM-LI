"""
Evaluate the LightGBM baseline. Mirrors baseline_cnn/evaluate_cnn.py's
structure and output schema exactly (npz with pr_prob_t/pr_label_t/
pr_seqid_t per lead step, CSV with csi/pod/far/fss_{thr}_scale{s}/pr_auc
per lead), so bootstrap_pr_auc_ci.py and compare_diffusion_vs_cnn_fss.py
work against it unchanged, with a `--baseline lightgbm:...` flag -- no new
comparison tooling needed. Reuses evaluate_cnn.py's _make_plots and
_li_to_physical directly (imported, not duplicated) so all three
baselines' figures share the identical journal-style theme.

Usage:
  python baseline_lightgbm/evaluate_lightgbm.py --config configs/evaluate.yaml \
      --model_dir baseline_lightgbm/outputs/run1 \
      --output_dir baseline_lightgbm
"""
import argparse
import csv
import json
import logging
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import make_test_loader
from config import load_yaml
from features import FEATURE_NAMES, compute_feature_maps, feature_maps_to_matrix  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baseline_cnn"))
from evaluate_cnn import _make_plots, _li_to_physical  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from sklearn.metrics import precision_recall_curve as _pr_curve
from sklearn.calibration import calibration_curve as _cal_curve
from sklearn.metrics import auc as _auc

try:
    from evaluate import lightning_skill_curve, fss
except ImportError:
    lightning_skill_curve = fss = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--model_dir", required=True,
                   help="Directory containing model.txt + meta.json from train_lightgbm.py")
    p.add_argument("--test_roots", nargs="+", default=None)
    p.add_argument("--output_dir", default="baseline_lightgbm")
    p.add_argument("--img_size", nargs=2, type=int, default=[256, 256])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--fss_prob_thresholds", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    p.add_argument("--fss_scales", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--pixel_size_km", type=float, default=4.0)
    args = p.parse_args()
    if args.config is not None:
        cfg = load_yaml(args.config)
        for k, v in cfg.items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)
    if args.test_roots is None:
        p.error("--test_roots is required (set in CLI or --config)")
    return args


def main():
    if lgb is None:
        raise ImportError("lightgbm is not installed. pip install lightgbm --break-system-packages")

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.model_dir, "meta.json")) as f:
        meta = json.load(f)
    booster = lgb.Booster(model_file=os.path.join(args.model_dir, "model.txt"))
    stats = meta["stats"]
    channels = meta["channels"]
    li_idx = channels.index("li")
    T_in, T_out, dt_min = meta["T_in"], meta["T_out"], meta["dt_min"]
    li_event_threshold = meta["li_event_threshold"]
    feature_names = meta["feature_names"]
    assert feature_names == FEATURE_NAMES, (
        "meta.json's feature_names doesn't match the current features.py "
        "FEATURE_NAMES -- the model was trained with a different feature "
        "set than this script would extract. Re-train or check out the "
        "matching features.py version before evaluating."
    )
    logger.info(f"Loaded LightGBM model from {args.model_dir} "
               f"(best_iteration={meta.get('best_iteration')})")

    test_loader = make_test_loader(
        test_roots=args.test_roots, channel_list=channels, stats=stats,
        T_in=T_in, T_out=T_out, img_size=tuple(args.img_size),
        batch_size=args.batch_size, num_workers=args.num_workers,
        binary_li_ctx=meta.get("binary_li_ctx", True),
        ctx_channels=meta.get("ctx_channels"),
    )
    logger.info(f"Test sequences: {len(test_loader.dataset)}")

    lead_times = [(t + 1) * dt_min for t in range(T_out)]
    pr_probs  = {t: [] for t in range(T_out)}
    pr_labels = {t: [] for t in range(T_out)}
    pr_seqids = {t: [] for t in range(T_out)}
    skill_by_step = [[] for _ in range(T_out)]
    fss_by_thr_scale_step = {
        thr: {s: [[] for _ in range(T_out)] for s in args.fss_scales}
        for thr in args.fss_prob_thresholds
    }
    seq_counter = 0

    for batch in tqdm(test_loader, desc="LightGBM baseline eval", dynamic_ncols=True):
        context  = batch["context"].numpy()
        target   = batch["target"].numpy()
        last_ctx = batch["last_ctx"].numpy()
        B = context.shape[0]
        H, W = context.shape[-2], context.shape[-1]

        for b in range(B):
            ir_ctx = context[b, :, 0]
            li_ctx_phys = _li_to_physical(context[b, :, li_idx], stats)
            # Computed ONCE per sequence, reused for all T_out predictions
            # below -- same efficiency reasoning as train_lightgbm.py.
            feats = compute_feature_maps(ir_ctx, li_ctx_phys, li_event_threshold)

            for t in range(T_out):
                X = feature_maps_to_matrix(feats, lead_idx=t, pixel_indices=None)
                pred_prob = booster.predict(X).reshape(H, W).astype(np.float32)

                tgt_abs_li = target[b, t, li_idx] + last_ctx[b, li_idx]
                obs_phys = _li_to_physical(tgt_abs_li, stats)
                obs_bin = (obs_phys >= li_event_threshold).astype(np.float32)

                p, o = pred_prob, obs_bin
                if lightning_skill_curve is not None:
                    skill_by_step[t].append(lightning_skill_curve(p, o))
                for thr in args.fss_prob_thresholds:
                    if fss is not None:
                        p_bin_thr = (p >= thr).astype(float)
                        for s in args.fss_scales:
                            fss_by_thr_scale_step[thr][s][t].append(fss(p_bin_thr, o, scale=s))

                stride = max(1, o.size // 4096)
                flat_p = p.ravel()[::stride]
                flat_o = o.ravel()[::stride]
                pr_probs[t].append(flat_p)
                pr_labels[t].append(flat_o)
                pr_seqids[t].append(np.full(flat_p.shape, seq_counter + b, dtype=np.int32))
        seq_counter += B

    auc_by_step, pr_curves = {}, {}
    for t in range(T_out):
        if not pr_probs[t]:
            continue
        all_prob = np.concatenate(pr_probs[t]).astype(np.float32)
        all_lbl  = np.concatenate(pr_labels[t]).astype(np.int32)
        prec, rec, _ = _pr_curve(all_lbl, all_prob)
        auc_by_step[t] = float(_auc(rec, prec))
        pr_curves[t] = (prec.astype(np.float32), rec.astype(np.float32))

    npz_payload = {"pr_steps": np.array(list(pr_curves.keys())), "dt_min": np.array(dt_min),
                   "n_sequences": np.array(seq_counter)}
    cal_curves = {}
    for t, (prec, rec) in pr_curves.items():
        npz_payload[f"prec_{t}"] = prec
        npz_payload[f"rec_{t}"] = rec
        npz_payload[f"auc_{t}"] = np.array(auc_by_step.get(t, float("nan")))
        try:
            all_prob = np.concatenate(pr_probs[t]).astype(np.float32)
            all_lbl  = np.concatenate(pr_labels[t]).astype(np.int32)
            frac_pos, mean_pred = _cal_curve(all_lbl, all_prob, n_bins=10, strategy="uniform")
            npz_payload[f"cal_mean_{t}"] = mean_pred.astype(np.float32)
            npz_payload[f"cal_frac_{t}"] = frac_pos.astype(np.float32)
            cal_curves[t] = (mean_pred.astype(np.float32), frac_pos.astype(np.float32))
        except Exception as e:
            logger.warning(f"Calibration failed at step {t}: {e}")
        npz_payload[f"pr_prob_{t}"] = np.concatenate(pr_probs[t]).astype(np.float32)
        npz_payload[f"pr_label_{t}"] = np.concatenate(pr_labels[t]).astype(np.int32)
        npz_payload[f"pr_seqid_{t}"] = np.concatenate(pr_seqids[t]).astype(np.int32)

    pr_npz = os.path.join(args.output_dir, "baseline_lightgbm_pr_curves.npz")
    np.savez_compressed(pr_npz, **npz_payload)
    logger.info(f"PR + calibration curves -> {pr_npz}")

    _mean = lambda lst: float(np.mean(lst)) if lst else float("nan")
    per_step = []
    for t in range(T_out):
        row = {"lead_min": lead_times[t]}
        if skill_by_step[t]:
            for thr in args.fss_prob_thresholds:
                csi_at = [float(np.interp(thr, sc["thresholds"], sc["csi"]))
                         for sc in skill_by_step[t] if len(sc["thresholds"])]
                pod_at = [float(np.interp(thr, sc["thresholds"], sc["pod"]))
                         for sc in skill_by_step[t] if len(sc["thresholds"])]
                far_at = [float(np.interp(thr, sc["thresholds"], sc["far"]))
                         for sc in skill_by_step[t] if len(sc["thresholds"])]
                row[f"csi_{thr}"] = _mean(csi_at)
                row[f"pod_{thr}"] = _mean(pod_at)
                row[f"far_{thr}"] = _mean(far_at)
            csi_max_vals = []
            for sc in skill_by_step[t]:
                if len(sc["csi"]):
                    csi_max_vals.append(sc["csi"][int(np.argmax(sc["csi"]))])
            row["csi_max"] = _mean(csi_max_vals)
        for thr in args.fss_prob_thresholds:
            for s in args.fss_scales:
                row[f"fss_{thr}_scale{s}"] = _mean(fss_by_thr_scale_step[thr][s][t])
        if t in auc_by_step:
            row["pr_auc"] = auc_by_step[t]
        per_step.append(row)

    csv_path = os.path.join(args.output_dir, "baseline_lightgbm_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_step[0].keys())
        writer.writeheader()
        writer.writerows(per_step)
    logger.info(f"Saved -> {csv_path}")

    # Reuses evaluate_cnn.py's _make_plots verbatim, just with this
    # baseline's own filename prefix and display name (both parameterized
    # in _make_plots specifically so this reuse doesn't mislabel LightGBM's
    # plots as "CNN Baseline" / baseline_cnn_*.png).
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    _make_plots(plot_dir, T_out, dt_min, lead_times, per_step,
               pr_curves, auc_by_step, cal_curves, args.fss_prob_thresholds,
               args.fss_scales, args.pixel_size_km,
               filename_prefix="baseline_lightgbm", display_name="LightGBM Baseline")

    print("\n=== LightGBM Baseline — Summary ===")
    print(f"{'Lead':>8}  {'PR-AUC':>8}  {'CSI_max':>8}")
    for row in per_step:
        print(f"  +{int(row['lead_min']):3d}min  "
             f"{row.get('pr_auc', float('nan')):>8.3f}  "
             f"{row.get('csi_max', float('nan')):>8.3f}")


if __name__ == "__main__":
    main()
