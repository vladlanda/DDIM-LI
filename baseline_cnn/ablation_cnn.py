"""
Input-ablation attribution of the CNN baseline on the TEST set: where its skill comes
from, for all pixels and for storm initiation.

    python baseline_cnn/ablation_cnn.py --checkpoint <CNN best.pt> --test_roots <4 dirs> \
        --output <csv> [--batch_size 32]

Context layout per frame: [ir, li, li_binary] (binary_li_ctx). Variants, each applied to
the whole 6 h context of every test sequence:
  none          unchanged (reference)
  ir_swap       IR history taken from ANOTHER test sequence (half a batch away, same
                region) -- removes the information in this scene's cloud tops while
                keeping inputs realistic
  li_swap       lightning history (li + li_binary) taken from another sequence
  ir_frozen     every IR frame replaced by the CURRENT frame -- keeps where cold cloud
                tops are now, removes how they evolved (cooling, growth, motion)
  li_frozen     every lightning frame replaced by the current one -- keeps lightning
                now, removes its 6 h history
AP is computed per lead on every 4th pixel per axis (4,096 px/image), for all pixels and
for initiation pixels (no lightning within 20 km in the last 30 min of the ORIGINAL
context -- the rule of build_test_metadata.py + analysis_regimes.py).
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bootstrap_pr_auc_ci import average_precision            # noqa: E402
from dataset import make_test_loader                         # noqa: E402
from evaluate import _li_to_physical                         # noqa: E402
from model import load_cnn_conditioner                       # noqa: E402

IR, LI, LIB = 0, 1, 2      # context channels per frame


def ablate(ctx, variant):
    """ctx: (B, T_in, C, H, W) torch tensor -> ablated copy."""
    B = ctx.shape[0]
    x = ctx.clone()
    other = torch.roll(torch.arange(B), B // 2)          # partner sequence in the batch
    if variant == "ir_swap":
        x[:, :, IR] = ctx[other, :, IR]
    elif variant == "li_swap":
        x[:, :, LI:] = ctx[other, :, LI:]
    elif variant == "ir_frozen":
        x[:, :, IR] = ctx[:, -1:, IR]
    elif variant == "li_frozen":
        x[:, :, LI:] = ctx[:, -1:, LI:]
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_roots", nargs="+", required=True)
    p.add_argument("--output", default="cnn_ablation.csv")
    p.add_argument("--variants", nargs="+", default=["none", "ir_swap", "li_swap", "ir_frozen", "li_frozen"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--px_stride", type=int, default=4)
    p.add_argument("--radius_km", type=float, default=20.0)
    p.add_argument("--pixel_km", type=float, default=3.14)
    p.add_argument("--recent_frames", type=int, default=3)
    p.add_argument("--li_event_threshold", type=float, default=0.5 / 255.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    a, stats, channels = ck["args"], ck["stats"], ck["channels"]
    assert channels[:2] == ["ir", "li"] and a.get("binary_li_ctx", True) and a.get("ctx_channels") is None, \
        "ablation assumes context = [ir, li, li_binary] per frame"
    cnn = load_cnn_conditioner(args.checkpoint, device, channels, stats, a["T_in"],
                               a.get("binary_li_ctx", True), a.get("ctx_channels")).eval()
    loader = make_test_loader(test_roots=args.test_roots, channel_list=channels, stats=stats,
                              T_in=a["T_in"], T_out=a["T_out"], img_size=(256, 256),
                              batch_size=args.batch_size, num_workers=args.num_workers,
                              binary_li_ctx=True, ctx_channels=None)
    T_out, S, thr = a["T_out"], args.px_stride, args.li_event_threshold
    r_px = int(np.floor(args.radius_km / args.pixel_km))
    li = channels.index("li")
    prob = {v: [[] for _ in range(T_out)] for v in args.variants}
    lbl, ini = [[] for _ in range(T_out)], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="CNN ablation", unit="batch"):
            ctx = batch["context"]
            tgt, last = batch["target"].numpy(), batch["last_ctx"].numpy()
            recent = ctx[:, -args.recent_frames:, LIB].numpy().any(axis=1)            # (B,H,W)
            ini.append(~maximum_filter(recent, size=(1, 2 * r_px + 1, 2 * r_px + 1),
                                       mode="constant", cval=False)[:, ::S, ::S].reshape(-1))
            for t in range(T_out):
                obs = _li_to_physical(tgt[:, t, li] + last[:, li], stats) >= thr
                lbl[t].append(obs[:, ::S, ::S].reshape(-1))
            for v in args.variants:
                x = ablate(ctx, v).to(device)
                for t in range(T_out):
                    lead = torch.full((x.shape[0],), t, dtype=torch.long, device=device)
                    pr = torch.sigmoid(cnn(x, lead).float())[:, 0, ::S, ::S].cpu().numpy()
                    prob[v][t].append(pr.reshape(-1))

    ini = np.concatenate(ini)
    rows = []
    for v in args.variants:
        for t in range(T_out):
            pr, lb = np.concatenate(prob[v][t]), np.concatenate(lbl[t])
            rows.append(dict(variant=v, lead_min=10 * (t + 1), ap_all=average_precision(pr, lb),
                             ap_initiation=average_precision(pr[ini], lb[ini]),
                             base_rate_initiation=float(lb[ini].mean())))
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    ref = {(r["lead_min"]): r for r in rows if r["variant"] == "none"}
    print(f"\n{'variant':10s} " + " ".join(f"{f'+{10*(t+1)}m all/init':>17s}" for t in range(T_out)))
    for v in args.variants:
        rr = [r for r in rows if r["variant"] == v]
        print(f"{v:10s} " + " ".join(f"{r['ap_all']:.3f}/{r['ap_initiation']:.3f}".rjust(17) for r in rr))
    if "none" in args.variants:
        print("\nAP change vs none (all / initiation):")
        for v in args.variants:
            if v == "none":
                continue
            rr = [r for r in rows if r["variant"] == v]
            print(f"{v:10s} " + " ".join(
                f"{r['ap_all'] - ref[r['lead_min']]['ap_all']:+.3f}/{r['ap_initiation'] - ref[r['lead_min']]['ap_initiation']:+.3f}".rjust(17)
                for r in rr))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
