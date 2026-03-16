#!/usr/bin/env bash
# ============================================================
# extend.sh — continue training from a finished run with a
#             fresh LR scheduler (cosine restart from scratch).
#
# What it does
# ------------
#   Loads weights + optimiser state from <OUTPUT_DIR>/latest.pt,
#   resets the epoch counter to 0 and starts a brand-new cosine
#   annealing cycle for EXTRA_EPOCHS epochs.
#   Results are saved to <OUTPUT_DIR>_ext1/  (then _ext2, _ext3 …)
#   so the original run is never overwritten.
#
# Usage
# -----
#   chmod +x extend.sh
#   ./extend.sh                               # 50 extra epochs, default dirs/lr
#   ./extend.sh --output_dir outputs/run2     # extend a different run
#   ./extend.sh --epochs 100 --lr 5e-5        # longer phase, lower peak lr
#   ./extend.sh --epochs 30 --max_samples 50  # quick smoke-test
#
# All extra arguments are forwarded to train.py after the fixed flags,
# so any train.py option can be overridden on the command line.
# ============================================================

# ---------- tuneable knobs ----------
NUM_GPUS=2
CONFIG="configs/default.yaml"

# Directory that contains the latest.pt to load from.
# Override with:  ./extend.sh --output_dir outputs/myrun
OUTPUT_DIR="outputs/run1"

# Number of additional epochs for this phase.
# Override with:  ./extend.sh --epochs 100
EXTRA_EPOCHS=50

# Peak LR for the new cosine cycle.
# Override with:  ./extend.sh --lr 5e-5
LR="1e-4"

# OMP threads — match launch.sh
TOTAL_CORES=$(nproc)
OMP_THREADS=$(( TOTAL_CORES / NUM_GPUS ))
# ------------------------------------

export OMP_NUM_THREADS=${OMP_THREADS}
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=3600

# ---- parse --output_dir / --epochs / --lr overrides from $@ ----
# We preview them here just for the banner; the real values are
# passed through to torchrun unchanged via "$@".
DISPLAY_DIR=${OUTPUT_DIR}
DISPLAY_EPOCHS=${EXTRA_EPOCHS}
DISPLAY_LR=${LR}

args=("$@")
for (( i=0; i<${#args[@]}; i++ )); do
    case "${args[$i]}" in
        --output_dir) DISPLAY_DIR="${args[$((i+1))]}" ;;
        --epochs)     DISPLAY_EPOCHS="${args[$((i+1))]}" ;;
        --lr)         DISPLAY_LR="${args[$((i+1))]}" ;;
    esac
done

echo "================================================"
echo "  METSAT Lightning Nowcasting — Extend Training"
echo "================================================"
echo "  GPUs           : ${NUM_GPUS}"
echo "  OMP_NUM_THREADS: ${OMP_THREADS}  (${TOTAL_CORES} cores / ${NUM_GPUS})"
echo "  Config         : ${CONFIG}"
echo "  Source run     : ${DISPLAY_DIR}/latest.pt"
echo "  Extra epochs   : ${DISPLAY_EPOCHS}"
echo "  Peak LR        : ${DISPLAY_LR}"
echo "  Output dir     : ${DISPLAY_DIR}_ext{N}  (auto-incremented)"
echo "  Extra args     : $@"
echo "================================================"
echo ""

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    train.py \
    --config   ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --epochs   ${EXTRA_EPOCHS} \
    --lr       ${LR} \
    --resume   true \
    --extend   true \
    "$@"
