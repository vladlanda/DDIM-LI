"""
Positional-ceiling diagnostic: how much PR-AUC is recoverable if we forgive
small spatial displacement at each lead time?

Motivation
----------
FSS analysis showed the +60min error is largely POSITIONAL (skillful at
~36km but not pixel-exact). PR-AUC is pixel-exact, so positional jitter
caps it. This script measures the CEILING: if the model's predicted
probability field were allowed to match observations within a small
neighbourhood, how high would PR-AUC go?

Three PR-AUC variants per lead time:
  1. exact          — standard pixel-wise PR-AUC (baseline)
  2. pooled(r)      — max-pool BOTH pred and obs over (2r+1) window before
                      PR-AUC. Forgives sub-window displacement. This is the
                      neighbourhood-relaxed skill.
  3. best-shift     — shift the pred field by the single global (dy,dx) in
                      [-R,R] that maximises overlap with obs, then PR-AUC.
                      Isolates pure translation error (bulk advection).

Interpretation:
  - If pooled/best-shift PR-AUC at +60min >> exact and reaches ~0.7,
    the skill IS there and a motion-following / warp formulation could
    unlock it -> retraining justified.
  - If even relaxed PR-AUC stays < 0.7, then 0.7 is not achievable from
    this signal and we should stop chasing it.

Usage
-----
  python diagnose_positional_ceiling.py --config configs/evaluate.yaml \
      --n_seq 60 --n_members 10 --pool_radii 1 2 3 --max_shift 4
"""
import argparse, logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from sklearn.metrics import precision_recall_curve, auc
from scipy.ndimage import maximum_filter, shift as ndi_shift

from config import load_yaml
from model import UNet, EDMPrecond, MultiStepDenoiser
from evaluate import generate_ensemble, _li_to_physical
from dataset import make_test_loader


def build_model_from_ckpt(ckpt, device):
    a  = ckpt["args"]; ch = ckpt["channels"]; C = len(ch)
    binary_li_ctx = a.get("binary_li_ctx", False)
    ctx_channels  = a.get("ctx_channels", None)
    C_ctx_sel = len(ctx_channels) if ctx_channels else C
    li_in_ctx = (ctx_channels is None) or ("li" in ctx_channels)
    C_ctx  = C_ctx_sel + 1 if (binary_li_ctx and li_in_ctx) else C_ctx_sel
    in_ch  = C + a["T_in"] * C_ctx + C
    unet = UNet(in_channels=in_ch, out_channels=C,
                base_channels=a["base_channels"],
                channel_mults=tuple(a["channel_mults"]),
                num_res_blocks=a["num_res_blocks"],
                attn_resolutions=tuple(a["attn_resolutions"]),
                dropout=0.0, emb_dim=a["emb_dim"],
                img_size=a.get("img_size", [256, 256])[0])
    precond = EDMPrecond(unet, sigma_data=a.get("sigma_data", 0.62))
    model   = MultiStepDenoiser(precond, T_out=a["T_out"], dt_min=a["dt_min"])
    model.load_state_dict(ckpt.get("ema") or ckpt["model"])
    model.to(device).eval()
    return model


def pr_auc(prob, lbl):
    """PR-AUC on flattened 2D fields. prob, lbl same shape."""
    p = prob.ravel().astype(np.float32)
    l = lbl.ravel().astype(np.int32)
    if l.sum() == 0 or l.sum() == l.size:
        return np.nan
    prec, rec, _ = precision_recall_curve(l, p)
    return float(auc(rec, prec))


def best_shift_prauc(prob, lbl, max_shift):
    """PR-AUC maximised over integer global shifts of prob in [-R,R]^2."""
    best = -1.0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = ndi_shift(prob, (dy, dx), order=0, mode="constant", cval=0.0)
            a = pr_auc(shifted, lbl)
            if not np.isnan(a) and a > best:
                best = a
    return best if best >= 0 else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n_seq", type=int, default=60,
                   help="Number of test sequences to sample.")
    p.add_argument("--n_members", type=int, default=10)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--cfg_scale", type=float, default=None)
    p.add_argument("--S_churn",   type=float, default=None)
    p.add_argument("--pool_radii", type=int, nargs="+", default=[1, 2, 3],
                   help="Neighbourhood radii (px) to test for pooled PR-AUC.")
    p.add_argument("--max_shift", type=int, default=4,
                   help="Max global shift (px) for best-shift PR-AUC.")
    p.add_argument("--li_event_threshold", type=float, default=5.0/255.0)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or f"{cfg['output_dir']}/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt["args"]; channels = ckpt["channels"]; stats = ckpt["stats"]
    T_out = ckpt_args["T_out"]; dt_min = ckpt_args["dt_min"]
    li_idx = channels.index("li")
    model = build_model_from_ckpt(ckpt, device)

    def pick(name, d):
        v = getattr(args, name)
        return v if v is not None else cfg.get(name, d)
    num_steps = pick("num_steps", 20)
    cfg_scale = pick("cfg_scale", 1.5)
    S_churn   = pick("S_churn", 60.0)

    test_loader = make_test_loader(
        test_roots=cfg["test_roots"], channel_list=channels, stats=stats,
        T_in=ckpt_args["T_in"], T_out=T_out,
        img_size=tuple(ckpt_args.get("img_size", [256, 256])),
        batch_size=1, num_workers=4,
        binary_li_ctx=ckpt_args.get("binary_li_ctx", False),
        ctx_channels=ckpt_args.get("ctx_channels", None),
    )

    # Accumulate 2D prob/label fields per lead step
    fields = {t: [] for t in range(T_out)}   # list of (prob2d, lbl2d)
    n_done = 0
    for batch in tqdm(test_loader, total=args.n_seq, desc="Test seqs", unit="seq"):
        context  = batch["context"].to(device)
        target   = batch["target"].to(device)
        last_ctx = batch["last_ctx"].to(device)
        ch_mask  = batch["tgt_mask"][:, 0].to(device)
        with torch.no_grad():
            ens = generate_ensemble(
                model, context, ch_mask, device,
                n_members=args.n_members, cfg_scale=cfg_scale,
                S_churn=S_churn, S_noise=1.003, num_steps=num_steps,
                sigma_min=float(ckpt_args.get("sigma_min", 0.002)),
                sigma_max=float(ckpt_args.get("sigma_max", 80.0)),
            )  # (1, M, T_out, C, H, W) residuals
        ens_abs = ens[0].cpu().numpy() + last_ctx[0, None, None].cpu().numpy()
        tgt_abs = target[0].cpu().numpy() + last_ctx[0, None].cpu().numpy()
        for t in range(T_out):
            obs_phys = _li_to_physical(tgt_abs[t, li_idx], stats)
            ens_phys = np.stack([_li_to_physical(ens_abs[m, t, li_idx], stats)
                                 for m in range(ens_abs.shape[0])])
            prob = (ens_phys >= args.li_event_threshold).mean(axis=0)   # 2D
            lbl  = (obs_phys >= args.li_event_threshold).astype(float)  # 2D
            fields[t].append((prob, lbl))
        n_done += 1
        if n_done >= args.n_seq:
            break

    # Compute the three PR-AUC variants per lead step
    print("\n" + "=" * 78)
    print("POSITIONAL CEILING — PR-AUC under neighbourhood relaxation")
    print("=" * 78)
    hdr = f"{'Lead':>6}  {'exact':>7}"
    for r in args.pool_radii:
        hdr += f"  {'pool'+str((2*r+1)*4)+'km':>10}"
    hdr += f"  {'best_shift':>10}"
    print(hdr)
    print("-" * len(hdr))

    for t in range(T_out):
        lead = (t + 1) * dt_min
        probs = [f[0] for f in fields[t]]
        lbls  = [f[1] for f in fields[t]]

        # exact: concat all sequences
        exact = pr_auc(np.stack(probs), np.stack(lbls))

        row = f"  +{lead:3d}m  {exact:>7.3f}"

        # pooled at each radius
        for r in args.pool_radii:
            size = 2 * r + 1
            pooled_p, pooled_l = [], []
            for prob, lbl in zip(probs, lbls):
                pooled_p.append(maximum_filter(prob, size=size, mode="constant"))
                pooled_l.append(maximum_filter(lbl,  size=size, mode="constant"))
            row += f"  {pr_auc(np.stack(pooled_p), np.stack(pooled_l)):>10.3f}"

        # best-shift: per-sequence optimal shift, then aggregate
        shifted_p, kept_l = [], []
        for prob, lbl in zip(probs, lbls):
            # find best shift for this sequence
            best_a, best_field = -1, prob
            for dy in range(-args.max_shift, args.max_shift + 1):
                for dx in range(-args.max_shift, args.max_shift + 1):
                    s = ndi_shift(prob, (dy, dx), order=0, mode="constant", cval=0.0)
                    a = pr_auc(s, lbl)
                    if not np.isnan(a) and a > best_a:
                        best_a, best_field = a, s
            shifted_p.append(best_field); kept_l.append(lbl)
        row += f"  {pr_auc(np.stack(shifted_p), np.stack(kept_l)):>10.3f}"

        print(row)

    print("-" * len(hdr))
    print("\nREADING THE RESULT:")
    print("  Compare +60min 'exact' vs 'pool' / 'best_shift':")
    print("  - If relaxed PR-AUC jumps to ~0.7+, the skill is present but")
    print("    positionally jittered -> motion-following/warp training is")
    print("    justified (retraining worth the compute).")
    print("  - If relaxed PR-AUC stays < 0.7, that ceiling is not reachable")
    print("    from this signal -> stop chasing 0.7, report the horizon.")


if __name__ == "__main__":
    main()
