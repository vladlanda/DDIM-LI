#!/usr/bin/env bash
# ============================================================
# extend.sh — continue training from a finished run with a
#             fresh LR scheduler (cosine restart from scratch).
#
# What it does
# ------------
#   Loads model + optimiser from best.pt, resets the epoch
#   counter to 0, runs one baseline validation pass (logged to
#   wandb as epoch=-1), then trains for EXTRA_EPOCHS epochs
#   with a brand-new cosine annealing cycle.
#
#   Results are saved to <OUTPUT_DIR>_ext1/  (then _ext2, ...)
#   so the original run is never overwritten.
#
# Why best.pt and not latest.pt?
#   At the end of a cosine cycle the LR is near eta_min (~lr*0.01).
#   best.pt holds the strongest generalising weights, not just the
#   most recent ones.
#
# Usage
# -----
#   chmod +x extend.sh
#   ./extend.sh                               # defaults below
#   ./extend.sh --output_dir outputs/run2     # extend a different run
#   ./extend.sh --epochs 100 --lr 5e-5        # longer phase, lower lr
#   ./extend.sh --epochs 30 --max_samples 50  # quick smoke-test
#
#   All extra arguments are forwarded to train.py unchanged.
# ============================================================

# ---------- tuneable knobs ----------
NUM_GPUS=2
CONFIG="configs/default.yaml"

# Directory that contains best.pt / latest.pt to extend from.
OUTPUT_DIR="outputs/run_1h_dim128"

# Number of additional epochs for this phase.
EXTRA_EPOCHS=21

# Peak LR for the new cosine cycle.
LR="1e-3"

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

# ---- parse overrides from $@ for the banner ----
args=("$@")
i=0
while [ $i -lt ${#args[@]} ]; do
    case "${args[$i]}" in
        --output_dir) OUTPUT_DIR="${args[$((i+1))]}"; i=$(( i+2 )) ;;
        --epochs)     EXTRA_EPOCHS="${args[$((i+1))]}"; i=$(( i+2 )) ;;
        --lr)         LR="${args[$((i+1))]}"; i=$(( i+2 )) ;;
        *)            i=$(( i+1 )) ;;
    esac
done

BEST_PT="${OUTPUT_DIR}/best.pt"

# Compute the _extN dir name (mirrors the regex in train.py)
BASE=$(echo "${OUTPUT_DIR}" | sed 's:/*$::')
if [[ "${BASE}" =~ ^(.*_ext)([0-9]+)$ ]]; then
    EXT_DIR="${BASH_REMATCH[1]}$(( BASH_REMATCH[2] + 1 ))"
else
    EXT_DIR="${BASE}_ext1"
fi

echo "================================================"
echo "  METSAT Lightning Nowcasting — Extend Training"
echo "================================================"
echo "  GPUs            : ${NUM_GPUS}"
echo "  OMP_NUM_THREADS : ${OMP_THREADS}  (${TOTAL_CORES} cores / ${NUM_GPUS})"
echo "  Config          : ${CONFIG}"
echo "  Source weights  : ${BEST_PT}"
echo "  Extra epochs    : ${EXTRA_EPOCHS}"
echo "  Peak LR         : ${LR}"
echo "  Output dir      : ${EXT_DIR}"
echo "  Baseline val    : logged to wandb as epoch=-1 before training"
echo "  Extra args      : $@"
echo "================================================"
echo ""

# ---- Sanity check ----
if [ ! -f "${BEST_PT}" ]; then
    echo "ERROR: ${BEST_PT} not found."
    echo "       Run training first, or set --output_dir to the correct run."
    exit 1
fi

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    train.py \
    --config     ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --epochs     ${EXTRA_EPOCHS} \
    --lr         ${LR} \
    --resume     true \
    --extend     true \
    "$@"

EXIT=$?
if [ ${EXIT} -ne 0 ]; then
    echo "ERROR: Training failed (exit ${EXIT})."
    exit ${EXIT}
fi

echo ""
echo "================================================"
echo "  Extension complete."
echo "  Checkpoints : ${EXT_DIR}/"
echo ""
echo "  To chain another extension:"
echo "  ./extend.sh --output_dir ${EXT_DIR} --epochs N"
echo "================================================"