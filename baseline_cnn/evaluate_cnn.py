"""
Evaluate the deterministic CNN baseline.

Scope note: this model outputs a single-channel LI probability (a genuine
classifier output, sigmoid of BCE training -- more principled than the
ad-hoc "clip physical field to [0,1]" pseudo-probability persistence/
optical-flow baselines use). It does NOT predict a physical field for any
channel, so CRPS/spread_skill/RMSE_ir/SSIM_ir are not meaningful here and
are omitted -- consistent with how LightningCast/Metzl et al. and similar
literature baselines report results (LI-classification metrics only).

Output schema (CSV + npz with pr_seqid) matches persistence_baseline.py /
optical_flow_baseline.py for direct bootstrap-CI comparability via
bootstrap_pr_auc_ci.py.

Usage:
  python baseline_cnn/evaluate_cnn.py --config configs/evaluate.yaml \
      --checkpoint baseline_cnn/outputs/run1/best.pt \
      --output_dir baseline_cnn
"""
import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import make_test_loader
from config import load_yaml
from model_cnn import DeterministicCNN  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from sklearn.metrics import precision_recall_curve as _pr_curve
from sklearn.calibration import calibration_curve as _cal_curve
from sklearn.metrics import auc as _auc

try:
    from evaluate import lightning_skill_curve, fss
except ImportError:
    lightning_skill_curve = fss = None


def _li_to_physical(arr, stats, ch="li"):
    if ch not in stats:
        return arr
    x = arr * stats[ch]["std"] + stats[ch]["mean"]
    if stats[ch].get("transform") == "cbrt":
        x = np.power(np.clip(x, 0.0, None), 3)
    return np.clip(x, 0.0, 1.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_roots", nargs="+", default=None)
    p.add_argument("--packed_dirs", nargs="+", default=None,
                   help="Explicit override: output dirs from "
                        "preprocess_to_memmap.py, one per test region, SAME "
                        "ORDER as --test_roots. Usually you want "
                        "--use_packed instead.")
    p.add_argument("--use_packed", action="store_true", default=False,
                   help="Auto-derive --packed_dirs as <root>/_packed for "
                        "every entry in --test_roots.")
    p.add_argument("--output_dir", default="baseline_cnn")
    p.add_argument("--img_size",   nargs=2, type=int, default=[256, 256])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--li_event_threshold", type=float, default=5.0/255.0)
    p.add_argument("--fss_prob_thresholds", nargs="+", type=float,
                   default=[0.1, 0.3, 0.5])
    args = p.parse_args()
    if args.config is not None:
        cfg = load_yaml(args.config)
        for k, v in cfg.items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)
    if args.test_roots is None:
        p.error("--test_roots is required (set in CLI or --config)")
    if args.use_packed:
        if args.packed_dirs is not None:
            raise ValueError("Pass either --use_packed or an explicit "
                             "--packed_dirs, not both.")
        args.packed_dirs = [os.path.join(r, "_packed") for r in args.test_roots]
        missing = [d for d in args.packed_dirs if not os.path.isdir(d)]
        if missing:
            raise FileNotFoundError(
                f"--use_packed derived {missing} but they don't exist. "
                f"Run preprocess_to_memmap.py --root <region> for each of "
                f"--test_roots first."
            )
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt["args"]
    stats = ckpt["stats"]
    channels = ckpt["channels"]
    C = len(channels)
    li_idx = channels.index("li")
    T_in, T_out, dt_min = ckpt_args["T_in"], ckpt_args["T_out"], ckpt_args["dt_min"]

    model = DeterministicCNN(
        C=C, T_in=T_in, T_out=T_out, dt_min=dt_min,
        ctx_channels=ckpt_args.get("ctx_channels"),
        binary_li_ctx=ckpt_args.get("binary_li_ctx", True),
        base_channels=ckpt_args["base_channels"],
        channel_mults=tuple(ckpt_args["channel_mults"]),
        num_res_blocks=ckpt_args["num_res_blocks"],
        attn_resolutions=tuple(ckpt_args["attn_resolutions"]),
        dropout=0.0, emb_dim=ckpt_args["emb_dim"],
        img_size=ckpt_args.get("img_size", args.img_size)[0],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    if args.packed_dirs is not None:
        from dataset_packed import make_test_loader_packed
        logger.info(f"Using PACKED data loading: {args.packed_dirs}")
        test_loader = make_test_loader_packed(
            test_packed_dirs=args.packed_dirs, channel_list=channels,
            T_in=T_in, T_out=T_out, dt_min=dt_min,
            batch_size=args.batch_size, num_workers=args.num_workers,
            stats=stats, stats_roots=args.test_roots,
            binary_li_ctx=ckpt_args.get("binary_li_ctx", True),
            ctx_channels=ckpt_args.get("ctx_channels"),
        )
    else:
        test_loader = make_test_loader(
            test_roots=args.test_roots, channel_list=channels, stats=stats,
            T_in=T_in, T_out=T_out, img_size=tuple(args.img_size),
            batch_size=args.batch_size, num_workers=args.num_workers,
            binary_li_ctx=ckpt_args.get("binary_li_ctx", True),
            ctx_channels=ckpt_args.get("ctx_channels"),
        )
    logger.info(f"Test sequences: {len(test_loader.dataset)}")

    lead_times = [(t + 1) * dt_min for t in range(T_out)]
    pr_probs  = {t: [] for t in range(T_out)}
    pr_labels = {t: [] for t in range(T_out)}
    pr_seqids = {t: [] for t in range(T_out)}
    skill_by_step = [[] for _ in range(T_out)]
    fss_by_step = {thr: [[] for _ in range(T_out)] for thr in args.fss_prob_thresholds}
    seq_counter = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="CNN baseline eval", dynamic_ncols=True):
            context  = batch["context"].to(device, non_blocking=True)
            target   = batch["target"].numpy()
            last_ctx = batch["last_ctx"].numpy()
            B = context.shape[0]

            for t in range(T_out):
                lead_idx = torch.full((B,), t, device=device, dtype=torch.long)
                logits = model(context, lead_idx)  # (B,1,H,W) RAW LOGITS
                pred_prob = torch.sigmoid(logits)[:, 0].cpu().numpy()  # (B,H,W) probability

                tgt_abs_li = target[:, t, li_idx] + last_ctx[:, li_idx]
                obs_phys = _li_to_physical(tgt_abs_li, stats)
                obs_bin  = (obs_phys >= args.li_event_threshold).astype(np.float32)

                for b in range(B):
                    p, o = pred_prob[b], obs_bin[b]
                    if lightning_skill_curve is not None:
                        skill_by_step[t].append(lightning_skill_curve(p, o))
                    for thr in args.fss_prob_thresholds:
                        if fss is not None:
                            fss_by_step[thr][t].append(
                                fss((p >= thr).astype(float), o, scale=1))

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
    for t, (prec, rec) in pr_curves.items():
        npz_payload[f"prec_{t}"] = prec
        npz_payload[f"rec_{t}"]  = rec
        npz_payload[f"auc_{t}"]  = np.array(auc_by_step.get(t, float("nan")))
        try:
            all_prob = np.concatenate(pr_probs[t]).astype(np.float32)
            all_lbl  = np.concatenate(pr_labels[t]).astype(np.int32)
            frac_pos, mean_pred = _cal_curve(all_lbl, all_prob, n_bins=10, strategy="uniform")
            npz_payload[f"cal_mean_{t}"] = mean_pred.astype(np.float32)
            npz_payload[f"cal_frac_{t}"] = frac_pos.astype(np.float32)
        except Exception as e:
            logger.warning(f"Calibration failed at step {t}: {e}")
        npz_payload[f"pr_prob_{t}"]  = np.concatenate(pr_probs[t]).astype(np.float32)
        npz_payload[f"pr_label_{t}"] = np.concatenate(pr_labels[t]).astype(np.int32)
        npz_payload[f"pr_seqid_{t}"] = np.concatenate(pr_seqids[t]).astype(np.int32)

    pr_npz = os.path.join(args.output_dir, "baseline_cnn_pr_curves.npz")
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
            row[f"fss_{thr}"] = _mean(fss_by_step[thr][t])
        if t in auc_by_step:
            row["pr_auc"] = auc_by_step[t]
        per_step.append(row)

    csv_path = os.path.join(args.output_dir, "baseline_cnn_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_step[0].keys())
        writer.writeheader()
        writer.writerows(per_step)
    logger.info(f"Saved -> {csv_path}")

    print("\n=== Deterministic CNN Baseline — Summary ===")
    print(f"{'Lead':>8}  {'PR-AUC':>8}  {'CSI_max':>8}")
    for row in per_step:
        print(f"  +{int(row['lead_min']):3d}min  "
              f"{row.get('pr_auc', float('nan')):>8.3f}  "
              f"{row.get('csi_max', float('nan')):>8.3f}")


if __name__ == "__main__":
    main()
