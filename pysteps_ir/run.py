"""
Thin wrapper: optical-flow baseline, SECONDARY/robustness variant
(flow_source=ir).

Derives the motion field from the denser, more reliable IR channel and
applies it to advect LI -- gives the optical-flow method its strongest
reasonable chance (direct precedent: severe-convection nowcasting papers
derive one motion field from the primary/densest field and apply it to
advect several other target fields).

Run from anywhere; results are saved into THIS directory regardless of
cwd. All CLI args from optical_flow_baseline.py are supported and pass
through unchanged.

Usage (from repo root or from inside pysteps_ir/):
  python pysteps_ir/run.py --config configs/evaluate.yaml
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
    args.flow_source = "ir"                  # fixed for this variant
    if args.output_dir == "outputs/optical_flow_baseline":  # user didn't override
        args.output_dir = _THIS_DIR           # save results into pysteps_ir/
    run_optical_flow_evaluation(args)
