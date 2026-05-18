#!/usr/bin/env bash
# ============================================================
#  METSAT Lightning Nowcasting — Evaluation
# ============================================================
#
# Usage:
#   ./evaluate.sh                          # standard evaluation
#   ./evaluate.sh --S_churn 80             # override any yaml param
#   ./evaluate.sh --plot_only              # regenerate plots from existing npz
#
# Workflow for temperature scaling:
#   1. ./evaluate.sh                                      # produces plot_data.npz
#   2. ./evaluate.sh --fit_temperature <out>/plot_data.npz \
#                    --temperature_path <out>/temperatures.json
#   3. ./evaluate.sh --temperature_path <out>/temperatures.json
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-configs/evaluate.yaml}"

echo ""
echo "=================================================="
echo "  METSAT Evaluation"
echo "=================================================="
echo "  Config  : $CONFIG"
echo "  Extra   : $*"
echo "=================================================="
echo ""

python evaluate.py --config "$CONFIG" "$@"
