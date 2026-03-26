"""
config_ar.py — YAML + argparse bridge for the autoregressive model.

Extends config.py with AR-specific arguments (T_ar).
All existing arguments from config.py are inherited.
"""

import argparse
import sys
from pathlib import Path

from config import _bool, load_yaml, _flatten_yaml, print_config   # reuse helpers

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def add_arguments_ar(parser: argparse.ArgumentParser):
    """Add all arguments including AR-specific ones."""
    from config import add_arguments
    add_arguments(parser)

    # AR-specific
    parser.add_argument("--T_ar", type=int, default=36,
                        help="Number of autoregressive steps to unroll at "
                             "evaluation / inference time (default=36 = 6h at 10min steps).")
    return parser


def parse_args_ar():
    """Parse CLI args with YAML support for the AR model."""
    # First pass: get --config path
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre.parse_known_args()

    parser = argparse.ArgumentParser(
        description="METSAT AR diffusion model — training / evaluation"
    )
    add_arguments_ar(parser)

    defaults = {}
    if pre_args.config:
        yaml_cfg = load_yaml(pre_args.config)
        defaults = _flatten_yaml(yaml_cfg)

    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining)
    args.config = pre_args.config
    return args
