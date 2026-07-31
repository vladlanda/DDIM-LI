"""
Thin wrapper: optical-flow baseline, PRIMARY variant (flow_source=li).

This is the standard nowcasting-literature convention: derive the motion
field from the SAME field being forecast (LI), exactly as precipitation
pySTEPS baselines derive flow from radar reflectivity and advect radar
reflectivity. This is what a reviewer expects by default when "optical
flow baseline" is claimed for a lightning forecast.

Run from anywhere; results are saved into THIS directory regardless of
cwd. All CLI args from optical_flow_baseline.py are supported and pass
through unchanged.

Usage (from repo root or from inside pysteps_li/):
  python pysteps_li/run.py --config configs/evaluate.yaml
  # or, if run from inside this directory:
  python run.py --config ../configs/evaluate.yaml
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _REPO_ROOT)

from optical_flow_baseline import parse_args, run_optical_flow_evaluation

if __name__ == "__main__":
    args = parse_args()
    args.flow_source = "li"                 # fixed for this variant
    if args.output_dir == "outputs/optical_flow_baseline":  # user didn't override
        args.output_dir = _THIS_DIR          # save results into pysteps_li/
    run_optical_flow_evaluation(args)
