"""
Quick diagnostic: is training slow because of DATA LOADING or COMPUTE?

Times three things separately, on real data/config, for a bounded number
of batches (cheap, no need to run anywhere near a full epoch):
  1. Pure data loading (iterate the DataLoader, touch nothing else).
  2. Data loading + moving to GPU (.to(device)).
  3. Full step: data + GPU transfer + forward + backward + optimizer step.

Whichever of these dominates tells us where to focus. If (1) alone is
already slow, the fix is in the data pipeline (num_workers, disk I/O,
JPEG decode cost), not the model -- no model-side change can help.

Usage:
  python baseline_cnn/profile_bottleneck.py --config configs/default.yaml \
      --n_batches 20
"""
import argparse
import sys
import os
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import make_dataloaders
from config import load_yaml
from model_cnn import DeterministicCNN, compute_in_ch_deterministic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n_batches", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--packed_dirs", nargs="+", default=None,
                   help="Explicit override: if given, profiles the PACKED "
                        "memmap loader instead of the JPEG loader (same "
                        "order as train_roots in the config). Usually you "
                        "want --use_packed instead.")
    p.add_argument("--use_packed", action="store_true", default=False,
                   help="Auto-derive --packed_dirs as <root>/_packed for "
                        "every entry in the config's train_roots.")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    for k, v in cfg.items():
        if hasattr(args, k) and getattr(args, k) is None:
            setattr(args, k, v)
    if args.use_packed:
        if args.packed_dirs is not None:
            raise ValueError("Pass either --use_packed or an explicit "
                             "--packed_dirs, not both.")
        args.packed_dirs = [os.path.join(r, "_packed") for r in cfg["train_roots"]]
        missing = [d for d in args.packed_dirs if not os.path.isdir(d)]
        if missing:
            raise FileNotFoundError(
                f"--use_packed derived {missing} but they don't exist. "
                f"Run preprocess_to_memmap.py --root <region> first."
            )
    return args, cfg


def main():
    args, cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type != "cuda":
        print("WARNING: no CUDA available in this environment -- timings "
              "below won't reflect real GPU behaviour, run this on the "
              "actual training machine.")

    num_workers = args.num_workers if args.num_workers is not None else cfg.get("num_workers", 4)
    batch_size  = args.batch_size  if args.batch_size  is not None else cfg.get("batch_size", 16)
    print(f"num_workers={num_workers}  batch_size={batch_size}  n_batches={args.n_batches}")

    if args.packed_dirs is not None:
        from dataset_packed import make_dataloaders_packed
        print(f"Profiling PACKED loader: {args.packed_dirs}")
        train_loader, _, stats = make_dataloaders_packed(
            train_packed_dirs=args.packed_dirs, channel_list=cfg["channels"],
            T_in=cfg["T_in"], T_out=cfg["T_out"], dt_min=cfg["dt_min"],
            batch_size=batch_size, num_workers=num_workers,
            stats_roots=cfg["train_roots"],
            train_val_split=cfg.get("train_val_split", 0.8),
            binary_li_ctx=cfg.get("binary_li_ctx", True),
            ctx_channels=cfg.get("ctx_channels"),
        )
    else:
        print(f"Profiling JPEG loader")
        train_loader, _, stats = make_dataloaders(
            train_roots=cfg["train_roots"], channel_list=cfg["channels"],
            T_in=cfg["T_in"], T_out=cfg["T_out"], img_size=tuple(cfg["img_size"]),
            batch_size=batch_size, num_workers=num_workers,
            train_val_split=cfg.get("train_val_split", 0.8),
            binary_li_ctx=cfg.get("binary_li_ctx", True),
            ctx_channels=cfg.get("ctx_channels"),
        )
    channels = cfg["channels"]; C = len(channels)

    # ---- Stage 1: pure data loading ----
    it = iter(train_loader)
    t0 = time.time()
    for _ in range(args.n_batches):
        batch = next(it)
    t1 = time.time()
    per_batch_load = (t1 - t0) / args.n_batches
    print(f"\n[1] Pure data loading:        {per_batch_load*1000:.1f} ms/batch  "
          f"({t1-t0:.2f}s total for {args.n_batches} batches)")

    # ---- Stage 2: data loading + GPU transfer ----
    it = iter(train_loader)
    t0 = time.time()
    for _ in range(args.n_batches):
        batch = next(it)
        context = batch["context"].to(device)
        target = batch["target"].to(device)
        last_ctx = batch["last_ctx"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.time()
    per_batch_transfer = (t1 - t0) / args.n_batches
    print(f"[2] Data load + GPU transfer: {per_batch_transfer*1000:.1f} ms/batch")

    # ---- Stage 3: full step (data + transfer + forward + backward + opt) ----
    model = DeterministicCNN(
        C=C, T_in=cfg["T_in"], T_out=cfg["T_out"], dt_min=cfg["dt_min"],
        ctx_channels=cfg.get("ctx_channels"), binary_li_ctx=cfg.get("binary_li_ctx", True),
        base_channels=24, channel_mults=(1,2,2,4), num_res_blocks=1,
        attn_resolutions=(), dropout=0.1, emb_dim=64, img_size=cfg["img_size"][0],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    li_idx = channels.index("li")

    it = iter(train_loader)
    t0 = time.time()
    for _ in range(args.n_batches):
        batch = next(it)
        context = batch["context"].to(device)
        target = batch["target"].to(device)
        last_ctx = batch["last_ctx"].to(device)
        B, T_out_b = context.shape[0], target.shape[1]
        lead_idx = torch.randint(0, T_out_b, (B,), device=device)
        tgt_abs = target[torch.arange(B), lead_idx] + last_ctx
        li_bin = (tgt_abs[:, li_idx] > 0).float().unsqueeze(1)  # rough, just for timing
        logits = model(context, lead_idx)
        loss = F.binary_cross_entropy_with_logits(logits, li_bin)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.time()
    per_batch_full = (t1 - t0) / args.n_batches
    print(f"[3] Full step (data+model):  {per_batch_full*1000:.1f} ms/batch")

    print(f"\n=== DIAGNOSIS ===")
    compute_cost = per_batch_full - per_batch_transfer
    print(f"  data loading alone:      {per_batch_load*1000:>8.1f} ms/batch")
    print(f"  + GPU transfer:          {(per_batch_transfer-per_batch_load)*1000:>8.1f} ms/batch")
    print(f"  + model fwd/bwd/opt:     {compute_cost*1000:>8.1f} ms/batch")
    print(f"  = full step:             {per_batch_full*1000:>8.1f} ms/batch")
    print(f"  (stage-to-stage deltas can be noisy with few batches/separate")
    print(f"   loader iterators -- the [1] vs [3] comparison below is the")
    print(f"   reliable verdict; increase --n_batches if deltas look odd)")
    if per_batch_load > 0.5 * per_batch_full:
        print(f"\n  -> DATA LOADING dominates ({100*per_batch_load/per_batch_full:.0f}% of step "
              f"time). The model is not the bottleneck -- num_workers, disk "
              f"I/O, or JPEG decode cost is. Increasing num_workers or "
              f"checking disk/storage speed is the next thing to try, not "
              f"further model changes.")
    else:
        print(f"\n  -> COMPUTE dominates ({100*compute_cost/per_batch_full:.0f}% of step time). "
              f"Worth profiling the model forward/backward specifically.")

    n_batches_per_epoch = len(train_loader)
    est_epoch_min = n_batches_per_epoch * per_batch_full / 60
    print(f"\n  batches/epoch: {n_batches_per_epoch}  "
          f"-> estimated epoch time at this rate: {est_epoch_min:.1f} min")


if __name__ == "__main__":
    main()
