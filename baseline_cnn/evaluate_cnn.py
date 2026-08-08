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

Also saves standalone, journal-style white-theme plots (CNN-only, not
overlaid with the main model or other baselines -- see FINDINGS.md /
PAPER_TODO.md for that separate, larger gap):
  baseline_cnn_skill_curves.png       — CSI/POD/FAR/PR-AUC vs lead time
  baseline_cnn_precision_recall.png   — PR curves, one line per lead time
  baseline_cnn_calibration.png        — reliability diagram per lead time
  baseline_cnn_fss_vs_scale.png       — FSS vs spatial scale (matches
                                         evaluate.py's fss_vs_scale.png,
                                         same [1,2,4,8,16,32] pixel scales
                                         by default) -- the key diagnostic
                                         for whether a pointwise-metric
                                         (PR-AUC/CSI) advantage survives
                                         spatial tolerance, see FINDINGS.md.

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
    p.add_argument("--preload_to_ram", action="store_true", default=False,
                   help="Load each packed test region fully into RAM instead "
                        "of np.memmap. See train_cnn.py --preload_to_ram for "
                        "the full caveats (RAM size, fork vs spawn).")
    p.add_argument("--output_dir", default="baseline_cnn")
    p.add_argument("--img_size",   nargs=2, type=int, default=[256, 256])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--li_event_threshold", type=float, default=5.0/255.0)
    p.add_argument("--fss_prob_thresholds", nargs="+", type=float,
                   default=[0.1, 0.3, 0.5])
    p.add_argument("--fss_scales", nargs="+", type=int,
                   default=[1, 2, 4, 8, 16, 32],
                   help="FSS neighbourhood half-widths in pixels, matching "
                        "evaluate.py/configs/evaluate.yaml exactly -- was "
                        "previously hardcoded to scale=1 only (pointwise, "
                        "no spatial tolerance), unlike the main model's "
                        "multi-scale evaluation. Needed to test whether a "
                        "baseline's pointwise-metric advantage survives "
                        "spatial tolerance (see FINDINGS.md).")
    p.add_argument("--pixel_size_km", type=float, default=4.0)
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


def _make_plots(output_dir, T_out, dt_min, lead_times, per_step,
                pr_curves, auc_by_step, cal_curves, fss_prob_thresholds,
                fss_scales, pixel_size_km):
    """Save journal-style white-theme PNGs for the CNN baseline, matching
    evaluate.py's plotting theme/palette (same rcParams, same turbo
    lead-time colormap, same axis-styling helper) so figures compare
    cleanly side by side. Standalone (CNN-only) plots -- NOT an overlay
    with the main model or other baselines; see FINDINGS.md/PAPER_TODO.md
    for the separate multi-baseline comparison-figure gap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm

    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "black",
        "axes.labelcolor":   "black",
        "xtick.color":       "black",
        "ytick.color":       "black",
        "text.color":        "black",
        "grid.color":        "#cccccc",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "#cccccc",
        "font.size":         9,
    })

    def _styled_ax(ax):
        ax.set_facecolor("white")
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        for s in ax.spines.values():
            s.set_edgecolor("black")
            s.set_linewidth(0.8)
        ax.grid(True, color="#cccccc", linestyle="--", linewidth=0.5, zorder=0)

    lt = lead_times
    lt_cmap = mcm.get_cmap("turbo")
    lt_colors = [lt_cmap(i / max(T_out - 1, 1)) for i in range(T_out)]

    def _lt_legend(ax, labeled_steps, ncol=2, loc="best", extra_handles=None):
        handles = list(extra_handles or [])
        for t in labeled_steps:
            handles.append(plt.Line2D([0], [0], color=lt_colors[t], linewidth=2,
                                      label=f"+{(t+1)*dt_min}m"))
        ax.legend(handles=handles, fontsize=7.5, ncol=ncol, loc=loc,
                  framealpha=0.9, handlelength=1.4, columnspacing=0.8, handletextpad=0.4)

    # ── Figure 1: CSI / POD / FAR / PR-AUC vs lead time ─────────────
    thr_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f"]
    thr_colors  = {thr: thr_palette[i % len(thr_palette)]
                   for i, thr in enumerate(fss_prob_thresholds)}

    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 9))
    fig1.patch.set_facecolor("white")
    ax_csi, ax_pod, ax_far, ax_aucp = axes1.flatten()

    for thr in fss_prob_thresholds:
        c, lbl = thr_colors[thr], f"p>{thr}"
        ax_csi.plot(lt, [r.get(f"csi_{thr}", float("nan")) for r in per_step],
                    color=c, linewidth=1.5, label=lbl)
        ax_pod.plot(lt, [r.get(f"pod_{thr}", float("nan")) for r in per_step],
                    color=c, linewidth=1.5, label=lbl)
        ax_far.plot(lt, [r.get(f"far_{thr}", float("nan")) for r in per_step],
                    color=c, linewidth=1.5, label=lbl)
    ax_aucp.plot(lt, [r.get("pr_auc", float("nan")) for r in per_step],
                color="#9467bd", linewidth=1.5, marker="o", markersize=4)

    for ax, title, ylabel, ylim in [
        (ax_csi,  "CSI vs Lead Time",    "CSI",    (0, 1)),
        (ax_pod,  "POD vs Lead Time",    "POD",    (0, 1)),
        (ax_far,  "FAR vs Lead Time",    "FAR",    (0, 1)),
        (ax_aucp, "PR-AUC vs Lead Time", "PR-AUC", (0, 1)),
    ]:
        ax.set_title(title); ax.set_xlabel("Lead time (min)"); ax.set_ylabel(ylabel)
        if ylim: ax.set_ylim(*ylim)
        _styled_ax(ax)
    for ax in [ax_csi, ax_pod, ax_far]:
        ax.legend(fontsize=8, framealpha=0.9)

    fig1.suptitle("CNN Baseline — Lightning Detection Skill", fontsize=12, fontweight="bold")
    fig1.tight_layout()
    skill_path = os.path.join(output_dir, "baseline_cnn_skill_curves.png")
    fig1.savefig(skill_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    logger.info(f"Skill curves    -> {skill_path}")

    # ── Figure 2: Precision-Recall curves, one line per lead time ──
    fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
    fig2.patch.set_facecolor("white")
    for t, (prec, rec) in pr_curves.items():
        ax2.plot(rec, prec, color=lt_colors[t], linewidth=1.2, alpha=0.9)
    ax2.set_xlabel("Recall (POD)")
    ax2.set_ylabel("Precision (1 \u2212 FAR)")
    ax2.set_title("CNN Baseline \u2014 Precision-Recall Curves", fontweight="bold")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    _styled_ax(ax2)
    handles_pr = [
        plt.Line2D([0], [0], color=lt_colors[t], linewidth=2,
                   label=f"+{(t+1)*dt_min}m  AUC={auc_by_step.get(t, float('nan')):.2f}")
        for t in sorted(pr_curves.keys())
    ]
    ax2.legend(handles=handles_pr, fontsize=7.5, ncol=2, loc="lower left",
              framealpha=0.9, handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
    fig2.tight_layout()
    pr_path = os.path.join(output_dir, "baseline_cnn_precision_recall.png")
    fig2.savefig(pr_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    logger.info(f"PR curves       -> {pr_path}")

    # ── Figure 3: Reliability / calibration diagram ─────────────────
    fig3, ax3 = plt.subplots(figsize=(6, 5.5))
    fig3.patch.set_facecolor("white")
    diag_line = plt.Line2D([0], [0], color="black", linestyle="--",
                           linewidth=1.2, label="Perfect calibration")
    ax3.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2, zorder=5)
    for t, (mean_pred, frac_pos) in cal_curves.items():
        ax3.plot(mean_pred, frac_pos, color=lt_colors[t], linewidth=1.2,
                 marker="o", markersize=3, alpha=0.9)
    ax3.set_xlabel("Mean Predicted Probability")
    ax3.set_ylabel("Observed Frequency")
    ax3.set_title("CNN Baseline \u2014 Reliability Diagram", fontweight="bold")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    _styled_ax(ax3)
    _lt_legend(ax3, sorted(cal_curves.keys()), loc="upper left", extra_handles=[diag_line])
    fig3.tight_layout()
    cal_path = os.path.join(output_dir, "baseline_cnn_calibration.png")
    fig3.savefig(cal_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    logger.info(f"Calibration     -> {cal_path}")

    # ── Figure 4: FSS vs spatial scale ───────────────────────────────
    # The key diagnostic for whether a pointwise-metric advantage (raw
    # PR-AUC/CSI at scale=1) survives spatial tolerance, or is a "double
    # penalty" artifact of scoring a genuinely spatially-uncertain field
    # pointwise. Top row: one line per lead time. Bottom row: mean over
    # lead times, one line per threshold -- mirrors evaluate.py exactly
    # so the two fss_vs_scale.png figures are directly comparable.
    thr_palette3 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    thr_colors3  = {thr: thr_palette3[i % len(thr_palette3)]
                    for i, thr in enumerate(fss_prob_thresholds)}
    n_thr = len(fss_prob_thresholds)
    scale_km = [(2 * s + 1) * pixel_size_km for s in fss_scales]

    fig4, axes4 = plt.subplots(2, n_thr, figsize=(5.5 * n_thr, 9), squeeze=False)
    fig4.patch.set_facecolor("white")

    for col, thr in enumerate(fss_prob_thresholds):
        ax_top, ax_bot = axes4[0, col], axes4[1, col]

        for t in range(T_out):
            fss_vals = [per_step[t].get(f"fss_{thr}_scale{s}", float("nan"))
                       for s in fss_scales]
            ax_top.plot(scale_km, fss_vals, color=lt_colors[t],
                       linewidth=1.0, marker="o", markersize=2, alpha=0.8)
        ax_top.axhline(0.5, color="#d62728", linestyle="--", linewidth=1.5)
        ax_top.set_title(f"FSS  (p > {thr})", fontweight="bold")
        ax_top.set_xlabel("Scale (km)"); ax_top.set_ylabel("FSS")
        ax_top.set_ylim(-0.05, 1.05)
        _styled_ax(ax_top)
        skill_line = plt.Line2D([0], [0], color="#d62728", linestyle="--",
                                linewidth=1.5, label="FSS=0.5")
        _lt_legend(ax_top, list(range(T_out)), ncol=3, loc="lower right",
                  extra_handles=[skill_line])

        fss_mean = [float(np.mean([per_step[t].get(f"fss_{thr}_scale{s}", np.nan)
                                   for t in range(T_out)]))
                   for s in fss_scales]
        ax_bot.plot(scale_km, fss_mean, color=thr_colors3[thr],
                   linewidth=2.0, marker="o", markersize=5, label=f"Mean (p>{thr})")
        ax_bot.axhline(0.5, color="#d62728", linestyle="--", linewidth=1.5, label="FSS=0.5")
        ax_bot.set_title(f"FSS mean over lead times  (p > {thr})", fontweight="bold")
        ax_bot.set_xlabel("Scale (km)"); ax_bot.set_ylabel("FSS")
        ax_bot.set_ylim(-0.05, 1.05)
        ax_bot.legend(fontsize=9)
        _styled_ax(ax_bot)

    fig4.suptitle(f"CNN Baseline — FSS vs Spatial Scale — {T_out} lead times",
                 fontsize=12, fontweight="bold")
    fig4.tight_layout()
    fss_path = os.path.join(output_dir, "baseline_cnn_fss_vs_scale.png")
    fig4.savefig(fss_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig4)
    logger.info(f"FSS vs scale    -> {fss_path}")


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
            preload_to_ram=args.preload_to_ram,
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
    fss_by_thr_scale_step = {
        thr: {s: [[] for _ in range(T_out)] for s in args.fss_scales}
        for thr in args.fss_prob_thresholds
    }
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
                            p_bin_thr = (p >= thr).astype(float)
                            for s in args.fss_scales:
                                fss_by_thr_scale_step[thr][s][t].append(
                                    fss(p_bin_thr, o, scale=s))

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
        npz_payload[f"rec_{t}"]  = rec
        npz_payload[f"auc_{t}"]  = np.array(auc_by_step.get(t, float("nan")))
        try:
            all_prob = np.concatenate(pr_probs[t]).astype(np.float32)
            all_lbl  = np.concatenate(pr_labels[t]).astype(np.int32)
            frac_pos, mean_pred = _cal_curve(all_lbl, all_prob, n_bins=10, strategy="uniform")
            npz_payload[f"cal_mean_{t}"] = mean_pred.astype(np.float32)
            npz_payload[f"cal_frac_{t}"] = frac_pos.astype(np.float32)
            cal_curves[t] = (mean_pred.astype(np.float32), frac_pos.astype(np.float32))
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
            for s in args.fss_scales:
                row[f"fss_{thr}_scale{s}"] = _mean(fss_by_thr_scale_step[thr][s][t])
        if t in auc_by_step:
            row["pr_auc"] = auc_by_step[t]
        per_step.append(row)

    csv_path = os.path.join(args.output_dir, "baseline_cnn_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_step[0].keys())
        writer.writeheader()
        writer.writerows(per_step)
    logger.info(f"Saved -> {csv_path}")

    _make_plots(args.output_dir, T_out, dt_min, lead_times, per_step,
               pr_curves, auc_by_step, cal_curves, args.fss_prob_thresholds,
               args.fss_scales, args.pixel_size_km)

    print("\n=== Deterministic CNN Baseline — Summary ===")
    print(f"{'Lead':>8}  {'PR-AUC':>8}  {'CSI_max':>8}")
    for row in per_step:
        print(f"  +{int(row['lead_min']):3d}min  "
              f"{row.get('pr_auc', float('nan')):>8.3f}  "
              f"{row.get('csi_max', float('nan')):>8.3f}")


if __name__ == "__main__":
    main()
