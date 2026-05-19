"""
config.py — YAML + argparse bridge.

How it works:
  1. A --config flag is parsed first (before full argparse)
  2. The YAML file is loaded and fed into argparse as defaults
  3. Any remaining CLI flags override the YAML values
  4. Result is a single flat argparse.Namespace used everywhere

This means:
  python train.py --config configs/default.yaml              # pure YAML
  python train.py --config configs/default.yaml --lr 5e-5   # YAML + override
  python train.py --lr 1e-4 --epochs 100                    # pure CLI (no YAML)
"""

import argparse
import sys
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ===================================================================
# Argument definitions  (single source of truth)
# ===================================================================

def add_arguments(parser: argparse.ArgumentParser):
    """Add all arguments to a parser. Used by both train.py and infer.py."""

    # Config file (special: handled before the rest)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file. CLI args override YAML.")

    # ---- Data ----
    parser.add_argument("--train_roots",     nargs="+", default=None,
                        help="Regions used for training (split into train/val internally).")
    parser.add_argument("--test_roots",      nargs="+", default=None,
                        help="Held-out regions for final evaluation only. Never seen during training.")
    parser.add_argument("--train_val_split", type=float, default=0.7,
                        help="Fraction of each region's sequences used for training. "
                             "Remainder becomes the validation set. Split is temporal (not random).")
    parser.add_argument("--channels",    nargs="+", default=["ir", "li", "ch1", "ch2"])
    parser.add_argument("--T_in",        type=int,   default=6)
    parser.add_argument("--T_out",       type=int,   default=36)
    parser.add_argument("--dt_min",      type=int,   default=10)
    parser.add_argument("--img_size",    nargs=2,    type=int, default=[256, 256])
    parser.add_argument("--max_samples", type=int,   default=None,
                        help="Limit sequences per dataset split (None = full dataset). "
                             "Use a small number (e.g. 50) for a quick smoke test.")

    # ---- Model ----
    parser.add_argument("--base_channels",     type=int,   default=128)
    parser.add_argument("--channel_mults",     nargs="+",  type=int, default=[1, 2, 3, 4])
    parser.add_argument("--num_res_blocks",    type=int,   default=2)
    parser.add_argument("--attn_resolutions",  nargs="+",  type=int, default=[16, 8])
    parser.add_argument("--dropout",           type=float, default=0.1)
    parser.add_argument("--emb_dim",           type=int,   default=512)
    parser.add_argument("--sigma_data",        type=float, default=0.5)

    # ---- EDM noise schedule ----
    parser.add_argument("--P_mean",    type=float, default=-1.2)
    parser.add_argument("--P_std",     type=float, default=1.2)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)

    # ---- Training ----
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--batch_size",      type=int,   default=4,
                        help="Per-GPU batch size. Effective batch = batch_size × num_gpus.")
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--grad_clip",       type=float, default=1.0)
    parser.add_argument("--ema_decay",       type=float, default=0.9999)
    parser.add_argument("--amp",             type=_bool, default=True)
    parser.add_argument("--num_workers",     type=int,   default=4,
                        help="DataLoader workers PER GPU.")
    parser.add_argument("--cfg_drop_prob",   type=float, default=0.15)
    # -- Channel weighting (Cui et al. 2019) --
    parser.add_argument("--li_weight",        type=float, default=30.0,
                        help="Base LI channel upweight. Dynamic per-sample scaling applied on top.")
    parser.add_argument("--li_weight_beta",   type=float, default=0.9999,
                        help="Beta for Cui et al. 2019 effective number formula.")
    # -- Asymmetric FP/FN loss (Gao et al. 2022) --
    parser.add_argument("--asym_weight",     type=float, default=1.0,
                        help="Weight for asymmetric LI loss term.")
    parser.add_argument("--asym_alpha",      type=float, default=0.3,
                        help="FP cost relative to FN. 0.3 = FP penalised at 30%% of FN. "
                             "Anneal toward 1.0 during training to balance.")
    # -- Neighbourhood spatial loss (Zhang et al. 2023) --
    parser.add_argument("--nbr_weight",      type=float, default=0.5,
                        help="Weight for neighbourhood spatial consistency loss.")
    parser.add_argument("--nbr_scales",      nargs="+", type=int, default=[5, 11],
                        help="Kernel sizes for neighbourhood avg_pool (pixels). "
                             "k=5 ≈ 20km, k=11 ≈ 44km at 4km/pixel.")
    # -- Spectral loss (cloud channels only) --
    parser.add_argument("--spectral_weight", type=float, default=0.1,
                        help="FFT magnitude loss weight. Applied to IR/cloud channels only.")
    # -- Stratified sampling --
    parser.add_argument("--oversample_factor",  type=float, default=5.0,
                        help="High-density LI sequences oversampled N×.")
    parser.add_argument("--density_percentile", type=float, default=75.0,
                        help="Sequences above this LI density percentile are oversampled.")
    # -- Binary LI context channel --
    parser.add_argument("--ctx_channels",    nargs="+",  default=None,
                        help="Channels to include in context. Default=None uses all. "
                             "Example: --ctx_channels ir ch0 ch1 excludes LI from context.")
    parser.add_argument("--binary_li_ctx",   type=_bool, default=True,
                        help="Add a binary (>=1/255) LI channel to context frames. "
                             "Gives the model an explicit spatial prior on where "
                             "lightning was occurring. Following Ravuri et al. 2021.")
    parser.add_argument("--n_members",       type=int,   default=10)
    parser.add_argument("--cfg_scale",       type=float, default=1.5)

    # ---- Validation / logging ----
    parser.add_argument("--slow_val",    type=_bool, default=True,
                        help="Enable full probabilistic eval (ODE solver) every "
                             "val_every epochs. Set false to run only fast_val_metrics.")
    parser.add_argument("--val_every",   type=int, default=5,
                        help="Run full ensemble validation every N epochs.")
    parser.add_argument("--val_samples", type=int, default=-1,
                        help="Batches to use for both fast and slow validation. "
                             "-1 = full val set; >0 = random sample of that many batches.")
    parser.add_argument("--output_dir",   type=str, default="outputs/run1")
    parser.add_argument("--resume",       type=_bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="")

    return parser


def _bool(v):
    """argparse-compatible boolean type (handles 'true'/'false' strings from YAML)."""
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("true", "yes", "1"):
        return True
    if str(v).lower() in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v!r}")


# ===================================================================
# YAML loader
# ===================================================================

def load_yaml(path: str) -> dict:
    if not HAS_YAML:
        raise ImportError("PyYAML is required for YAML config support. "
                          "Install it with: pip install pyyaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _flatten_yaml(cfg: dict) -> dict:
    """
    YAML values are already flat in our config, but this handles any
    type coercions needed to match argparse expectations.
    e.g. yaml booleans → Python bools (yaml handles this automatically),
    yaml lists → Python lists (also automatic).
    """
    flat = {}
    for k, v in cfg.items():
        if k == "config":
            continue  # don't re-inject the config path
        flat[k] = v
    return flat


# ===================================================================
# Main parse function
# ===================================================================

def parse_args(argv=None) -> argparse.Namespace:
    """
    Parse arguments with YAML → argparse defaults → CLI override chain.

    Priority (highest to lowest):
      1. CLI arguments explicitly passed
      2. Values from YAML config file (--config path)
      3. argparse defaults defined in add_arguments()
    """
    if argv is None:
        argv = sys.argv[1:]

    # ---- Step 1: peek at --config before full parse ----
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre.parse_known_args(argv)

    # ---- Step 2: build full parser ----
    parser = argparse.ArgumentParser(
        description="METSAT Lightning Nowcasting — EDM Diffusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)

    # ---- Step 3: load YAML and set as defaults ----
    if pre_args.config is not None:
        cfg_path = pre_args.config
        if not Path(cfg_path).exists():
            parser.error(f"Config file not found: {cfg_path}")
        yaml_cfg = load_yaml(cfg_path)
        flat_cfg = _flatten_yaml(yaml_cfg)

        # Validate YAML keys against known arguments
        known_keys = {a.dest for a in parser._actions}
        unknown = set(flat_cfg.keys()) - known_keys
        if unknown:
            print(f"[config.py] WARNING: unknown YAML keys (ignored): {unknown}",
                  file=sys.stderr)

        parser.set_defaults(**flat_cfg)

    # ---- Step 4: parse everything (CLI overrides YAML defaults) ----
    args = parser.parse_args(argv)
    return args


# ===================================================================
# Pretty-print config
# ===================================================================

def print_config(args: argparse.Namespace):
    """Print the resolved config in a readable format."""
    print("\n" + "=" * 55)
    print("  Resolved configuration")
    print("=" * 55)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:<22} = {v}")
    print("=" * 55 + "\n")