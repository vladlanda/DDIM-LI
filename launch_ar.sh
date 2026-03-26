#!/usr/bin/env bash
# ============================================================
# launch_ar.sh — DDP training launcher for AR nowcasting model
#
# Usage:
#   chmod +x launch_ar.sh
#   ./launch_ar.sh                          # uses configs/default_ar.yaml
#   ./launch_ar.sh --max_samples 50         # quick smoke test
#   ./launch_ar.sh --resume true            # resume from latest checkpoint
# ============================================================

NUM_GPUS=2
CONFIG="configs/default_ar.yaml"
TOTAL_CORES=$(nproc)
OMP_THREADS=$(( TOTAL_CORES / NUM_GPUS ))

export OMP_NUM_THREADS=${OMP_THREADS}
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=3600

echo "================================================"
echo "  METSAT AR Nowcasting — DDP Training"
echo "================================================"
echo "  GPUs           : ${NUM_GPUS}"
echo "  OMP_NUM_THREADS: ${OMP_THREADS}  (${TOTAL_CORES} cores / ${NUM_GPUS})"
echo "  Config         : ${CONFIG}"
echo "  Extra args     : $@"
echo "================================================"
echo ""

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29502 \
    train_ar.py \
    --config ${CONFIG} \
    "$@"
