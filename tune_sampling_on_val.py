"""
Pick diffusion sampling settings (cfg_scale x S_churn) on VALIDATION data only.

Uses exactly the chronological train/val split of training (first
train_val_split of each region's sequences -> train, T_in+T_out gap, rest -> val),
then non-overlapping sequences (stride T_in+T_out, as in make_test_loader) and an
even subsample of --n_per_region per region. The test set is never touched.

Every setting sees the same sequences and the same noise (seed reset per setting).

Note on S_churn: edm_sampler uses gamma = min(S_churn/num_steps, sqrt(2)-1), so with
num_steps=20 any S_churn >= 8.28 is identical (fully saturated). The default grid
therefore spans 0 (deterministic ODE) .. 8.3 (max).

Per lead and setting it reports, at the event threshold (AFA >= 1):
  ap         average precision of the ensemble probability (higher better)
  brier      Brier score (lower better)
  freq_bias  mean forecast probability / observed base rate (1 = unbiased;
             >1 = over-forecasting, e.g. denoiser haze)
  p_neg      mean probability on observed-no-lightning pixels (haze)
  crps       fair CRPS of LI in physical units (lower better)
  ss         pooled spread/skill of LI (1 = calibrated spread)

Usage:
  python tune_sampling_on_val.py --checkpoint outputs/v2_ir_li/best.pt \
      --output outputs/v2_ir_li/tune_sampling_val.csv
  # optional: --train_roots ... (if paths differ from those stored in the checkpoint)
"""
import argparse
import csv
import itertools
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bootstrap_pr_auc_ci import average_precision
from dataset import MultiRegionDataset
from evaluate import _li_to_physical, crps_energy, generate_ensemble
from select_example_cases import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def val_loader(ckpt_args, stats, channels, roots, n_per_region, batch_size, num_workers):
    T_in, T_out = ckpt_args["T_in"], ckpt_args["T_out"]
    ds = MultiRegionDataset(
        roots, channel_list=channels, T_in=T_in, T_out=T_out,
        img_size=tuple(ckpt_args.get("img_size", [256, 256])), stats=stats, augment=False,
        binary_li_ctx=ckpt_args.get("binary_li_ctx", False),
        ctx_channels=ckpt_args.get("ctx_channels", None),
    )
    split = ckpt_args["train_val_split"]
    for d in ds.datasets:
        n_train = max(1, int(len(d.valid_sequences) * split))          # same as make_dataloaders
        val = d.valid_sequences[n_train + T_in + T_out:][::T_in + T_out]
        idx = np.linspace(0, len(val) - 1, min(n_per_region, len(val))).round().astype(int)
        d.valid_sequences = [val[i] for i in idx]
        log.info(f"  {d.root}: {len(d.valid_sequences)} val sequences used")
    ds.lengths = [len(d) for d in ds.datasets]
    ds.cumlen = np.cumsum([0] + ds.lengths)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _nmax(a, hw):
    """Neighbourhood maximum over a (2hw+1)^2 square on the last two axes."""
    from scipy.ndimage import maximum_filter
    k = 2 * hw + 1
    return maximum_filter(a, size=(1,) * (a.ndim - 2) + (k, k), mode="constant", cval=0)


def _p_indep(p, hw):
    """P(any event in the square) if pixels were INDEPENDENT with probabilities p."""
    from scipy.ndimage import uniform_filter
    k = 2 * hw + 1
    logq = np.log1p(-np.clip(p, 0, 1 - 1e-6))
    size = (1,) * (p.ndim - 2) + (k, k)          # never filter across the batch axis
    return 1.0 - np.exp(uniform_filter(logq, size=size, mode="constant", cval=0.0) * k * k)


AREA_STRIDE = 4   # ponytail: neighbourhood fields are smooth; keep every 4th pixel per axis (16x less memory)


def _area_rows(model, area, T_out):
    rows = []
    for hw, per_lead in area.items():
        for t in range(T_out):
            prob, lbl = per_lead[t]
            rows.append(dict(model=model, radius_km=round(hw * 3.14, 1), lead_min=10 * (t + 1),
                             **_scores([prob], [lbl])))
    return rows


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _scores(prob, lbl):
    """AP / Brier / freq. bias / mean prob on observed negatives for one lead."""
    pr, lb = np.concatenate(prob), np.concatenate(lbl).astype(np.float32)
    base = lb.mean()
    return dict(ap=average_precision(pr, lb), brier=float(((np.clip(pr, 0, 1) - lb) ** 2).mean()),
                freq_bias=float(np.clip(pr, 0, 1).mean() / base) if base > 0 else np.nan,
                p_neg=float(np.clip(pr, 0, 1)[lb == 0].mean()), base_rate=float(base))


def score_baselines(args, batches, stats, channels, li, T_out, device):
    """CNN (sigmoid probability) and persistence on the SAME val batches as the diffusion grid.
    Persistence AP ranks by the raw last-frame LI value (as persistence_baseline.py);
    its Brier/bias use the binary forecast (last frame >= threshold)."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_cnn"))
    from model_cnn import DeterministicCNN

    ck = torch.load(args.cnn_checkpoint, map_location=device)
    ca = ck["args"]
    # The CNN must see identically normalised inputs: same channels, stats and context layout.
    assert ck["channels"] == channels, (ck["channels"], channels)
    for ch in channels:
        for k in ("mean", "std"):
            assert abs(ck["stats"][ch][k] - stats[ch][k]) < 1e-6, f"stats differ: {ch}.{k}"
    cnn = DeterministicCNN(
        C=len(channels), T_in=ca["T_in"], T_out=ca["T_out"], dt_min=ca["dt_min"],
        ctx_channels=ca.get("ctx_channels"), binary_li_ctx=ca.get("binary_li_ctx", True),
        base_channels=ca["base_channels"], channel_mults=tuple(ca["channel_mults"]),
        num_res_blocks=ca["num_res_blocks"], attn_resolutions=tuple(ca["attn_resolutions"]),
        dropout=0.0, emb_dim=ca["emb_dim"], img_size=ca.get("img_size", [256, 256])[0],
    ).to(device)
    cnn.load_state_dict(ck["model"]); cnn.eval()

    thr = args.li_event_threshold
    out = {m: ([[] for _ in range(T_out)], [[] for _ in range(T_out)])
           for m in ("cnn", "persistence_raw", "persistence_bin")}
    area_models = ("cnn_nmax", "cnn_indep", "persistence_bin")
    area = {m: {hw: [([], []) for _ in range(T_out)] for hw in args.area_hw} for m in area_models}
    S = AREA_STRIDE
    with torch.no_grad():
        for batch in tqdm(batches, desc="CNN + persistence", unit="batch"):
            ctx = batch["context"].to(device)
            last = batch["last_ctx"].numpy()
            tgt = _li_to_physical(batch["target"].numpy()[:, :, li] + last[:, None, li], stats)
            last_phys = _li_to_physical(last[:, li], stats)
            for t in range(T_out):
                lead = torch.full((ctx.shape[0],), t, device=device, dtype=torch.long)
                p_cnn = torch.sigmoid(cnn(ctx, lead))[:, 0].cpu().numpy()
                y = (tgt[:, t] >= thr).ravel()
                for m, pr in (("cnn", p_cnn), ("persistence_raw", last_phys),
                              ("persistence_bin", (last_phys >= thr).astype(np.float32))):
                    out[m][0][t].append(pr.ravel()); out[m][1][t].append(y)
                for hw in args.area_hw:
                    lab = _nmax((tgt[:, t] >= thr).astype(np.uint8), hw)[:, ::S, ::S].ravel()
                    for m, pr in (("cnn_nmax", _nmax(p_cnn, hw)), ("cnn_indep", _p_indep(p_cnn, hw)),
                                  ("persistence_bin", _nmax((last_phys >= thr).astype(np.float32), hw))):
                        area[m][hw][t][0].append(pr[:, ::S, ::S].ravel()); area[m][hw][t][1].append(lab)

    path = args.output.replace(".csv", "") + "_baselines.csv"
    rows = []
    for m, (prob, lbl) in out.items():
        for t in range(T_out):
            rows.append(dict(model=m, lead_min=10 * (t + 1), **_scores(prob[t], lbl[t])))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    diff = {}
    if os.path.exists(args.output):
        for r in csv.DictReader(open(args.output)):
            if (float(r["cfg_scale"]), float(r["S_churn"])) == tuple(args.compare):
                diff[int(r["lead_min"])] = r
    get = lambda m, lead, k: next(float(r[k]) for r in rows if r["model"] == m and r["lead_min"] == lead)
    print(f"\nSame {sum(len(b['context']) for b in batches)} val sequences, event = LI >= {thr:.6f}")
    print(f"{'lead':>5} | {'AP diff':>8} {'AP cnn':>7} {'AP pers':>7} {'diff-cnn':>8} | "
          f"{'Br diff':>8} {'Br cnn':>8} | {'fb diff':>7} {'fb cnn':>6}")
    for t in range(T_out):
        lead = 10 * (t + 1)
        d = diff.get(lead)
        ad = float(d["ap"]) if d else np.nan
        print(f"{lead:4d}m | {ad:8.4f} {get('cnn', lead, 'ap'):7.4f} {get('persistence_raw', lead, 'ap'):7.4f} "
              f"{ad - get('cnn', lead, 'ap'):+8.4f} | {float(d['brier']) if d else np.nan:8.5f} "
              f"{get('cnn', lead, 'brier'):8.5f} | {float(d['freq_bias']) if d else np.nan:7.2f} "
              f"{get('cnn', lead, 'freq_bias'):6.2f}")
    print(f"diffusion = cfg {args.compare[0]}, churn {args.compare[1]} from {args.output}; "
          f"baselines -> {path}")

    if not args.area_hw:
        return
    stem = args.output.replace(".csv", "")
    arows = [r for m in area_models for r in _area_rows(m, {hw: [(np.concatenate(a), np.concatenate(b))
             for a, b in area[m][hw]] for hw in args.area_hw}, T_out)]
    _write(stem + "_baselines_area.csv", arows)
    dfile = stem + "_area.csv"
    drows = [r for r in csv.DictReader(open(dfile))] if os.path.exists(dfile) else []
    key = f"diffusion_cfg{args.compare[0]}_churn{args.compare[1]}"
    print("\nAREA probability: event = any lightning in a square of half-width r around the pixel")
    print(f"{'r km':>5} {'lead':>5} | {'AP diff':>8} {'AP cnn_nmax':>11} {'AP cnn_indep':>12} {'AP pers':>7} | "
          f"{'Br diff':>8} {'Br nmax':>8} {'Br indep':>8} | {'base':>5}")
    for hw in args.area_hw:
        rk = round(hw * 3.14, 1)
        for t in range(T_out):
            lead = 10 * (t + 1)
            g = lambda m, k: next(float(r[k]) for r in arows
                                  if r["model"] == m and r["radius_km"] == rk and r["lead_min"] == lead)
            d = next((r for r in drows if r["model"] == key and float(r["radius_km"]) == rk
                      and int(r["lead_min"]) == lead), None)
            print(f"{rk:5.1f} {lead:4d}m | {float(d['ap']) if d else np.nan:8.4f} {g('cnn_nmax', 'ap'):11.4f} "
                  f"{g('cnn_indep', 'ap'):12.4f} {g('persistence_bin', 'ap'):7.4f} | "
                  f"{float(d['brier']) if d else np.nan:8.5f} {g('cnn_nmax', 'brier'):8.5f} "
                  f"{g('cnn_indep', 'brier'):8.5f} | {g('cnn_nmax', 'base_rate'):5.3f}")
    print(f"area rows -> {stem}_baselines_area.csv (diffusion: {dfile})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train_roots", nargs="+", default=None)
    p.add_argument("--output", default="tune_sampling_val.csv")
    p.add_argument("--cfg_scales", type=float, nargs="*", default=[1.0, 1.5, 2.0],
                   help="pass with no values to skip the diffusion grid (baselines only)")
    p.add_argument("--S_churns", type=float, nargs="+", default=[0.0, 4.0, 8.3])
    p.add_argument("--n_members", type=int, default=10)
    p.add_argument("--n_per_region", type=int, default=24)
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--S_noise", type=float, default=1.003)
    p.add_argument("--li_event_threshold", type=float, default=0.5 / 255.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cnn_checkpoint", default=None,
                   help="also score the CNN baseline + persistence on the same val sequences")
    p.add_argument("--area_hw", type=int, nargs="*", default=[2, 3, 6],
                   help="neighbourhood half-widths in pixels (3.14 km) for area-probability scores; "
                        "empty = off")
    p.add_argument("--compare", type=float, nargs=2, default=[1.0, 8.3], metavar=("CFG", "CHURN"),
                   help="diffusion setting shown next to the baselines")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, ckpt_args, stats, channels = load_model(args, device)
    li = channels.index("li")
    T_out = ckpt_args["T_out"]
    loader = val_loader(ckpt_args, stats, channels, args.train_roots or ckpt_args["train_roots"],
                        args.n_per_region, args.batch_size, args.num_workers)
    batches = list(tqdm(loader, desc="Loading val subset", unit="batch"))  # load once, reuse per setting
    thr = args.li_event_threshold

    if args.cnn_checkpoint:
        score_baselines(args, batches, stats, channels, li, T_out, device)
    grid = list(itertools.product(args.cfg_scales, args.S_churns))
    if not grid:
        return

    fields = ["cfg_scale", "S_churn", "lead_min", "ap", "brier", "freq_bias", "p_neg", "crps", "ss",
              "base_rate", "minutes"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        area_all = []
        bar = tqdm(total=len(grid) * len(batches), desc="Tuning (settings x batches)",
                   unit="batch", dynamic_ncols=True)
        for cfg, churn in grid:
            bar.set_postfix(cfg=cfg, churn=churn)
            torch.manual_seed(args.seed)         # common random numbers across settings
            t0 = time.time()
            prob = [[] for _ in range(T_out)]; lbl = [[] for _ in range(T_out)]
            crps = [[] for _ in range(T_out)]
            var_sum = np.zeros(T_out); mse_sum = np.zeros(T_out); n_px = np.zeros(T_out)
            area = {hw: [([], []) for _ in range(T_out)] for hw in args.area_hw}
            M = args.n_members
            for batch in batches:
                with torch.no_grad():
                    ens = generate_ensemble(
                        model, batch["context"].to(device), batch["tgt_mask"][:, 0].to(device),
                        device, n_members=M, num_steps=args.num_steps, cfg_scale=cfg,
                        S_churn=churn, S_noise=args.S_noise,
                        sigma_min=float(ckpt_args.get("sigma_min", 0.002)),
                        sigma_max=float(ckpt_args.get("sigma_max", 80.0)))
                last = batch["last_ctx"].numpy()
                ens = _li_to_physical(ens.cpu().numpy()[:, :, :, li] + last[:, None, None, li], stats)
                tgt = _li_to_physical(batch["target"].numpy()[:, :, li] + last[:, None, li], stats)
                for b in range(ens.shape[0]):
                    for t in range(T_out):
                        e, y = ens[b, :, t], tgt[b, t]                      # (M,H,W), (H,W)
                        prob[t].append((e >= thr).mean(0).ravel())
                        lbl[t].append((y >= thr).ravel())
                        crps[t].append(crps_energy(e, y))
                        var_sum[t] += np.var(e, axis=0, ddof=1).sum() * (M + 1) / M
                        mse_sum[t] += ((e.mean(0) - y) ** 2).sum()
                        n_px[t] += y.size
                        for hw in args.area_hw:   # members' own spatial structure -> area probability
                            S = AREA_STRIDE
                            area[hw][t][0].append(_nmax((e >= thr).astype(np.uint8), hw).mean(0)[::S, ::S].ravel())
                            area[hw][t][1].append(_nmax((y >= thr).astype(np.uint8)[None], hw)[0, ::S, ::S].ravel())
                bar.update(1)
            minutes = (time.time() - t0) / 60
            for t in range(T_out):
                pr, lb = np.concatenate(prob[t]), np.concatenate(lbl[t]).astype(np.float32)
                base = lb.mean()
                row = dict(cfg_scale=cfg, S_churn=churn, lead_min=10 * (t + 1),
                           ap=average_precision(pr, lb), brier=float(((pr - lb) ** 2).mean()),
                           freq_bias=float(pr.mean() / base) if base > 0 else np.nan,
                           p_neg=float(pr[lb == 0].mean()), crps=float(np.mean(crps[t])),
                           ss=float(np.sqrt(var_sum[t] / n_px[t]) / np.sqrt(mse_sum[t] / n_px[t])),
                           base_rate=float(base), minutes=round(minutes, 1))
                w.writerow(row); f.flush()
            if args.area_hw:
                area_all.extend(_area_rows(f"diffusion_cfg{cfg}_churn{churn}", {hw: [
                    (np.concatenate(a), np.concatenate(b)) for a, b in area[hw]] for hw in args.area_hw}, T_out))
                _write(args.output.replace(".csv", "") + "_area.csv", area_all)
            bar.write(f"cfg={cfg} churn={churn}: {minutes:.1f} min")
        bar.close()

    # Summary: lead-averaged scores per setting
    rows = list(csv.DictReader(open(args.output)))
    print(f"\n{'cfg':>5} {'churn':>6} {'AP':>7} {'Brier':>8} {'fbias':>6} {'p_neg':>8} {'CRPS':>9} {'SS':>5}")
    for cfg, churn in itertools.product(args.cfg_scales, args.S_churns):
        r = [x for x in rows if float(x["cfg_scale"]) == cfg and float(x["S_churn"]) == churn]
        m = lambda k: np.mean([float(x[k]) for x in r])
        print(f"{cfg:5.2f} {churn:6.1f} {m('ap'):7.4f} {m('brier'):8.5f} {m('freq_bias'):6.2f} "
              f"{m('p_neg'):8.5f} {m('crps'):9.6f} {m('ss'):5.2f}")
    print(f"\nper-lead rows -> {args.output}")


if __name__ == "__main__":
    main()
