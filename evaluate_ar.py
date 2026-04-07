"""
evaluate_ar.py — Test-set evaluation for the AR METSAT diffusion model.

Identical metric suite to evaluate.py but generates forecasts via
autoregressive rollout instead of direct multi-step prediction.

Usage:
    python evaluate_ar.py \\
        --checkpoint outputs/run_ar/best.pt \\
        --test_roots /media/.../central_africa_4 \\
        --output_dir outputs/eval_ar \\
        --T_ar 36 --n_members 10

    # Re-plot from saved data (no GPU needed):
    python evaluate_ar.py \\
        --checkpoint outputs/run_ar/best.pt \\
        --output_dir outputs/eval_ar \\
        --plot_only
"""

import csv
import json
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

# Reuse all metric functions and plot helpers from evaluate.py
from evaluate import (
    _li_to_physical,
    HAS_SKIMAGE,
    crps_energy, cloud_metrics, lightning_skill_curve,
    fss, spread_skill,
    plot_forecast, _regenerate_plots,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Load AR model from checkpoint
# ===================================================================

def _load_ar_model(checkpoint: str, device: torch.device):
    from model import UNet, EDMPrecond, EDMSchedule
    from model_ar import ARPrecond, ARDenoiser

    ckpt      = torch.load(checkpoint, map_location=device)
    ckpt_args = ckpt["args"]
    channels  = ckpt["channels"]
    stats     = ckpt["stats"]
    T_in      = ckpt_args["T_in"]
    T_out     = ckpt_args["T_out"]
    dt_min    = ckpt_args["dt_min"]
    C         = len(channels)

    unet    = UNet(
        in_channels      = C * (T_in + 2),
        out_channels     = C,
        base_channels    = ckpt_args["base_channels"],
        channel_mults    = tuple(ckpt_args["channel_mults"]),
        num_res_blocks   = ckpt_args["num_res_blocks"],
        attn_resolutions = tuple(ckpt_args["attn_resolutions"]),
        dropout          = 0.0,
        emb_dim          = ckpt_args["emb_dim"],
    )
    precond = ARPrecond(unet, sigma_data=ckpt_args.get("sigma_data", 0.5))
    model   = ARDenoiser(precond, T_in=T_in, dt_min=dt_min)
    state   = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model, channels, stats, T_in, T_out, dt_min


# ===================================================================
# AR ensemble generation
# ===================================================================

@torch.no_grad()
def generate_ar_ensemble(
    model,                          # ARDenoiser
    context_abs: torch.Tensor,      # (B, T_in, C, H, W) absolute normalised
    ch_mask:     torch.Tensor,      # (B, C)
    device:      torch.device,
    T_ar:        int,
    n_members:   int = 10,
    num_steps:   int = 20,
    cfg_scale:   float = 1.5,
) -> torch.Tensor:                  # (B, M, T_ar, C, H, W) absolute normalised
    """
    Generate an ensemble of AR forecasts.

    Each member is an independent AR rollout — stochasticity comes from
    the diffusion sampling (different noise realisations each member).
    CFG is applied at each denoising step.
    """
    from model import edm_sampler

    B = context_abs.shape[0]
    C = context_abs.shape[2]
    H = context_abs.shape[3]
    W = context_abs.shape[4]

    members = []
    member_bar = tqdm(
        range(n_members),
        desc="    AR ensemble", unit="member",
        dynamic_ncols=True, leave=False,
    )

    for _ in member_bar:
        ctx = context_abs.clone()   # rolling window

        preds = []
        for step in range(T_ar):
            last_abs = ctx[:, -1]

            def denoiser_fn(x, sigma,
                            _ctx=ctx.clone(), _mask=ch_mask):
                cond   = model(x, sigma, _ctx, _mask)
                uncond = model(x, sigma, torch.zeros_like(_ctx), _mask)
                return uncond + cfg_scale * (cond - uncond)

            pred_residual = edm_sampler(
                denoiser_fn, (B, C, H, W), device,
                num_steps = num_steps,
                sigma_min = model.precond.sigma_data * 0.01,
                sigma_max = 80.0,
            )
            pred_abs = pred_residual + last_abs
            preds.append(pred_abs)
            ctx = torch.cat([ctx[:, 1:], pred_abs.unsqueeze(1)], dim=1)

        members.append(torch.stack(preds, dim=1))   # (B, T_ar, C, H, W)

    return torch.stack(members, dim=1)  # (B, M, T_ar, C, H, W)


# ===================================================================
# Main evaluation
# ===================================================================

def run_ar_evaluation(args):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    class _TqdmHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                tqdm.write(self.format(record))
            except Exception:
                self.handleError(record)

    root_log = logging.getLogger()
    root_log.handlers.clear()
    h = _TqdmHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_log.addHandler(h)
    root_log.setLevel(logging.INFO)

    # ---- plot_only ----
    if args.plot_only:
        npz_path = os.path.join(args.output_dir, "plot_data.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"--plot_only requires {npz_path}\n"
                "Run evaluation once first."
            )
        _regenerate_plots(npz_path, args)
        return {}

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- Load model ----
    model, channels, stats, T_in, T_out, dt_min = _load_ar_model(args.checkpoint, device)
    T_ar = args.T_ar
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"  channels={channels}  T_in={T_in}  T_ar={T_ar}  dt={dt_min}min")

    # ---- Test loader ----
    from dataset import make_test_loader
    test_loader = make_test_loader(
        test_roots   = args.test_roots,
        channel_list = channels,
        stats        = stats,
        T_in         = T_in,
        T_out        = T_out,
        img_size     = tuple(args.img_size),
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
    )
    logger.info(f"Test sequences: {len(test_loader.dataset)}")

    li_idx     = channels.index("li") if "li" in channels else None
    cloud_chs  = [ch for ch in channels if ch != "li"]
    cloud_idxs = [channels.index(ch) for ch in cloud_chs]
    fss_scales = args.fss_scales

    # ---- Accumulators ----
    crps_by_step = [[] for _ in range(T_ar)]
    ss_by_step   = [[] for _ in range(T_ar)]
    rmse_by_ch_step = {ch: [[] for _ in range(T_ar)] for ch in cloud_chs}
    mae_by_ch_step  = {ch: [[] for _ in range(T_ar)] for ch in cloud_chs}
    ssim_by_ch_step = {ch: [[] for _ in range(T_ar)] for ch in cloud_chs}
    fss_prob_thresholds = args.fss_prob_thresholds
    skill_by_step = [[] for _ in range(T_ar)]
    brier_by_step = [[] for _ in range(T_ar)]   # kept for backward compat
    fss_by_thr_scale_step = {
        thr: {s: [[] for _ in range(T_ar)] for s in fss_scales}
        for thr in fss_prob_thresholds
    }
    pr_steps  = list(range(T_ar))
    pr_probs  = {t: [] for t in pr_steps}
    pr_labels = {t: [] for t in pr_steps}
    cal_probs  = {t: [] for t in pr_steps}
    cal_labels = {t: [] for t in pr_steps}

    os.makedirs(args.output_dir, exist_ok=True)
    if args.plot:
        os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    seq_counter = 0
    batch_bar   = tqdm(test_loader, desc="Evaluating", unit="batch",
                       dynamic_ncols=True, leave=True)

    for batch in batch_bar:
        # Context is absolute normalised frames (from dataset)
        context_abs = batch["context"].to(device)           # (B, T_in, C, H, W)
        target_res  = batch["target"].to(device)            # (B, T_out, C, H, W) residuals
        last_ctx    = batch["last_ctx"].to(device)          # (B, C, H, W)
        ch_mask     = batch["tgt_mask"][:, 0].to(device)

        # Ground-truth absolute frames for the first T_ar steps
        target_abs = target_res + last_ctx[:, None]         # (B, T_out, C, H, W)
        target_abs = target_abs[:, :T_ar]                   # (B, T_ar, C, H, W)

        # AR ensemble generation
        ens = generate_ar_ensemble(
            model, context_abs, ch_mask, device,
            T_ar      = T_ar,
            n_members = args.n_members,
            cfg_scale = args.cfg_scale,
        )  # (B, M, T_ar, C, H, W) absolute normalised

        ens_np = ens.cpu().numpy()
        tgt_np = target_abs.cpu().numpy()
        ctx_np = batch["context"].numpy()
        B      = ens_np.shape[0]

        for b in range(B):
            for t in range(T_ar):
                ens_t    = ens_np[b, :, t]
                tgt_t    = tgt_np[b, t]
                ens_mean = ens_t.mean(axis=0)

                crps_by_step[t].append(crps_energy(ens_t, tgt_t))
                ss_by_step[t].append(spread_skill(ens_t, tgt_t))

                if cloud_idxs:
                    cm = cloud_metrics(ens_mean, tgt_t, cloud_idxs, cloud_chs, stats)
                    for ch in cloud_chs:
                        if ch in cm:
                            rmse_by_ch_step[ch][t].append(cm[ch]["rmse"])
                            mae_by_ch_step[ch][t].append(cm[ch]["mae"])
                            if "ssim" in cm[ch]:
                                ssim_by_ch_step[ch][t].append(cm[ch]["ssim"])

                if li_idx is not None:
                    obs_phys  = _li_to_physical(tgt_t[li_idx], stats)
                    ens_phys  = np.stack([_li_to_physical(ens_t[m, li_idx], stats)
                                          for m in range(ens_t.shape[0])])
                    obs_bin   = (obs_phys > 0).astype(np.float32)
                    pred_prob = (ens_phys > 0).mean(axis=0).astype(np.float32)

                    sk = lightning_skill_curve(pred_prob, obs_bin)
                    skill_by_step[t].append(sk)
                    brier_by_step[t].append(sk["brier"])
                    for thr in fss_prob_thresholds:
                        for s in fss_scales:
                            fss_by_thr_scale_step[thr][s][t].append(
                                fss(pred_prob, obs_bin, scale=s)
                            )
                    if t in pr_steps:
                        flat_prob = pred_prob.ravel()
                        flat_lbl  = obs_bin.ravel()
                        stride    = max(1, len(flat_prob) // 4096)
                        pr_probs[t].append(flat_prob[::stride])
                        pr_labels[t].append(flat_lbl[::stride])
                        cal_probs[t].append(flat_prob[::stride])
                        cal_labels[t].append(flat_lbl[::stride])

            if args.plot and seq_counter < args.max_plots:
                from dataset import denormalize as _denorm
                ctx_b   = ctx_np[b]
                ctx_den = np.zeros_like(ctx_b)
                ens_den = np.zeros_like(ens_np[b])
                tgt_den = np.zeros_like(tgt_np[b])
                for ci, ch in enumerate(channels):
                    fn = (lambda x, _ch=ch: _denorm(x, stats, _ch)) if ch in stats \
                         else (lambda x: x)
                    ctx_den[:, ci] = fn(ctx_b[:, ci])
                    tgt_den[:, ci] = fn(tgt_np[b, :, ci])
                    for m in range(ens_np.shape[1]):
                        ens_den[m, :, ci] = fn(ens_np[b, m, :, ci])
                png = os.path.join(args.output_dir, "plots", f"eval_{seq_counter:04d}.png")
                plot_forecast(context_np=ctx_den, ens_np=ens_den,
                              channels=channels, gt_np=tgt_den, save_path=png)

            seq_counter += 1

        live_crps = float(np.mean([np.mean(v) for v in crps_by_step if v]))
        if li_idx is not None and skill_by_step[0]:
            live_csi = float(np.mean([
                float(np.interp(0.5, sc["thresholds"][::-1], sc["csi"][::-1]))
                for step_list in skill_by_step for sc in step_list
                if len(sc["thresholds"]) > 0
            ]))
        else:
            live_csi = float("nan")
        batch_bar.set_postfix(crps=f"{live_crps:.4f}", csi=f"{live_csi:.3f}", refresh=False)

    # ---- Aggregate ----
    def _mean(lst): return float(np.mean(lst)) if lst else float("nan")

    lead_times = [(t + 1) * dt_min for t in range(T_ar)]
    ch_unit    = {ch: ("norm" if ch in stats and stats[ch].get("transform") == "cbrt"
                        else "K")
                  for ch in cloud_chs}

    per_step = []
    for t in range(T_ar):
        row = {"lead_min": lead_times[t],
               "crps":     _mean(crps_by_step[t]),
               "spread_skill": _mean(ss_by_step[t])}
        for ch in cloud_chs:
            row[f"rmse_{ch}"] = _mean(rmse_by_ch_step[ch][t])
            row[f"mae_{ch}"]  = _mean(mae_by_ch_step[ch][t])
            if ssim_by_ch_step[ch][t]:
                row[f"ssim_{ch}"] = _mean(ssim_by_ch_step[ch][t])
        if li_idx is not None:
            row["brier"] = _mean(brier_by_step[t])
            if skill_by_step[t]:
                for thr in fss_prob_thresholds:
                    mid_s = fss_scales[len(fss_scales) // 2]
                    row[f"fss_{thr}"] = _mean(fss_by_thr_scale_step[thr][mid_s][t])
                for thr in fss_prob_thresholds:
                    row[f"csi_{thr}"] = float(np.mean([
                        float(np.interp(thr, sc["thresholds"][::-1], sc["csi"][::-1]))
                        for sc in skill_by_step[t]]))
                    row[f"pod_{thr}"] = float(np.mean([
                        float(np.interp(thr, sc["thresholds"][::-1], sc["pod"][::-1]))
                        for sc in skill_by_step[t]]))
                    row[f"far_{thr}"] = float(np.mean([
                        float(np.interp(thr, sc["thresholds"][::-1], sc["far"][::-1]))
                        for sc in skill_by_step[t]]))
        per_step.append(row)

    summary = {
        "checkpoint":  args.checkpoint,
        "test_roots":  args.test_roots,
        "n_sequences": seq_counter,
        "n_members":   args.n_members,
        "T_ar":        T_ar,
        "model_type":  "autoregressive",
        "crps_mean":   _mean([r["crps"] for r in per_step]),
        "crps_1h":     _mean([r["crps"] for r in per_step[:6]]),
        "crps_3h":     _mean([r["crps"] for r in per_step[:18]]),
        "crps_6h":     per_step[-1]["crps"],
        "spread_skill":_mean([r["spread_skill"] for r in per_step]),
    }
    for ch in cloud_chs:
        summary[f"rmse_{ch}_mean"] = _mean([r[f"rmse_{ch}"] for r in per_step])
        summary[f"mae_{ch}_mean"]  = _mean([r[f"mae_{ch}"]  for r in per_step])
        ssim_vals = [r.get(f"ssim_{ch}", float("nan")) for r in per_step]
        if not all(np.isnan(ssim_vals)):
            summary[f"ssim_{ch}_mean"] = _mean(ssim_vals)
    if li_idx is not None:
        fss_summary = {}
        for thr in fss_prob_thresholds:
            for s in fss_scales:
                fss_summary[f"fss_thr{thr}_scale{s}"] = _mean(
                    [_mean(fss_by_thr_scale_step[thr][s][t]) for t in range(T_ar)]
                )
        summary.update({"brier_mean": _mean([r["brier"] for r in per_step]),
                         **fss_summary})

    # ---- Save outputs ----
    json_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary    -> {json_path}")

    csv_path = os.path.join(args.output_dir, "metrics_per_step.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_step[0].keys())
        writer.writeheader()
        writer.writerows(per_step)
    logger.info(f"Per-step   -> {csv_path}")

    npz_path = os.path.join(args.output_dir, "plot_data.npz")
    npz_payload = {
        "per_step_json": np.array(json.dumps(per_step)),
        "lead_times":    np.array(lead_times),
        "channels":      np.array(channels),
        "cloud_chs":     np.array(cloud_chs),
        "fss_scales":    np.array(fss_scales),
        "T_out":         np.array(T_ar),
        "dt_min":        np.array(dt_min),
        "pixel_size_km": np.array(args.pixel_size_km),
        "pr_steps":      np.array(pr_steps),
    }
    if li_idx is not None:
        npz_payload["fss_prob_thresholds"] = np.array(fss_prob_thresholds)
        for thr in fss_prob_thresholds:
            for s in fss_scales:
                npz_payload[f"fss_thr{thr}_s{s}"] = np.array(
                    [_mean(fss_by_thr_scale_step[thr][s][t]) for t in range(T_ar)]
                )
        for t in pr_steps:
            if pr_probs[t]:
                npz_payload[f"pr_prob_{t}"]   = np.concatenate(pr_probs[t])
                npz_payload[f"pr_label_{t}"]  = np.concatenate(pr_labels[t])
                npz_payload[f"cal_prob_{t}"]  = np.concatenate(cal_probs[t])
                npz_payload[f"cal_label_{t}"] = np.concatenate(cal_labels[t])
    np.savez_compressed(npz_path, **npz_payload)
    logger.info(f"Plot data  -> {npz_path}")

    _regenerate_plots(npz_path, args)

    # ---- Print summary ----
    logger.info("\n" + "=" * 52)
    logger.info("  AR MODEL — CLOUD METRICS")
    logger.info("=" * 52)
    for k in ["crps_mean", "crps_1h", "crps_3h", "crps_6h", "spread_skill"]:
        if k in summary:
            logger.info(f"  {k:<22s}: {summary[k]:.4f}")
    if li_idx is not None:
        logger.info("\n" + "=" * 52)
        logger.info("  AR MODEL — LIGHTNING METRICS")
        logger.info("=" * 52)
        if "brier_mean" in summary:
            logger.info(f"  {'brier_mean':<22s}: {summary['brier_mean']:.4f}")
        logger.info("  (CSI/POD/FAR: see skill_curves_lightning.png)")
    logger.info("=" * 52)
    return summary


# ===================================================================
# Entry point
# ===================================================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="AR model evaluation for METSAT nowcasting.")
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--test_roots",    nargs="+", default=None)
    p.add_argument("--output_dir",    default="outputs/eval_ar")
    p.add_argument("--T_ar",          type=int,   default=36,
                   help="Number of AR rollout steps (default=36 = 6h at 10min)")
    p.add_argument("--n_members",     type=int,   default=10)
    p.add_argument("--cfg_scale",     type=float, default=1.5)
    p.add_argument("--batch_size",    type=int,   default=2,
                   help="Smaller than direct model — AR rollout uses more GPU memory")
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--img_size",      nargs=2, type=int, default=[256, 256])
    p.add_argument("--fss_prob_thresholds", nargs="+", type=float,
                   default=[0.1, 0.3, 0.5],
                   help="Ensemble probability thresholds for FSS.")
    p.add_argument("--gpu",           type=int,   default=0)
    p.add_argument("--fss_scales",    nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--pixel_size_km", type=float, default=4.0)
    p.add_argument("--plot",          action="store_true")
    p.add_argument("--max_plots",     type=int,   default=20)
    p.add_argument("--plot_only",     action="store_true")
    args = p.parse_args()
    run_ar_evaluation(args)
