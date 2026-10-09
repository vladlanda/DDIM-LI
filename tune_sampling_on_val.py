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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train_roots", nargs="+", default=None)
    p.add_argument("--output", default="tune_sampling_val.csv")
    p.add_argument("--cfg_scales", type=float, nargs="+", default=[1.0, 1.5, 2.0])
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
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, ckpt_args, stats, channels = load_model(args, device)
    li = channels.index("li")
    T_out = ckpt_args["T_out"]
    loader = val_loader(ckpt_args, stats, channels, args.train_roots or ckpt_args["train_roots"],
                        args.n_per_region, args.batch_size, args.num_workers)
    batches = list(tqdm(loader, desc="Loading val subset", unit="batch"))  # load once, reuse per setting
    thr = args.li_event_threshold

    fields = ["cfg_scale", "S_churn", "lead_min", "ap", "brier", "freq_bias", "p_neg", "crps", "ss",
              "base_rate", "minutes"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        grid = list(itertools.product(args.cfg_scales, args.S_churns))
        bar = tqdm(total=len(grid) * len(batches), desc="Tuning (settings x batches)",
                   unit="batch", dynamic_ncols=True)
        for cfg, churn in grid:
            bar.set_postfix(cfg=cfg, churn=churn)
            torch.manual_seed(args.seed)         # common random numbers across settings
            t0 = time.time()
            prob = [[] for _ in range(T_out)]; lbl = [[] for _ in range(T_out)]
            crps = [[] for _ in range(T_out)]
            var_sum = np.zeros(T_out); mse_sum = np.zeros(T_out); n_px = np.zeros(T_out)
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
