#!/usr/bin/env bash
# ============================================================
# launch.sh — DDP training launcher for 2 × RTX 5090
#
# Usage:
#   chmod +x launch.sh
#   ./launch.sh                          # uses configs/default.yaml
#   ./launch.sh --max_samples 50         # quick smoke test
#   ./launch.sh --resume true            # resume from latest checkpoint
# ============================================================

# ---------- tuneable knobs ----------
NUM_GPUS=2
CONFIG="configs/default.yaml"
# OMP threads per process: total physical cores / NUM_GPUS
# Adjust to match your CPU (run `nproc` to check)
TOTAL_CORES=$(nproc)
OMP_THREADS=$(( TOTAL_CORES / NUM_GPUS ))
# ------------------------------------

export OMP_NUM_THREADS=${OMP_THREADS}

# Improves NCCL performance on PCIe (no NVLink)
export NCCL_P2P_DISABLE=0          # keep P2P on — PCIe P2P is still faster than host
export NCCL_IB_DISABLE=1           # no InfiniBand on a workstation
export NCCL_SOCKET_IFNAME=lo       # use loopback for inter-process signalling

# Reduces fragmentation in PyTorch's CUDA allocator (replaces deprecated PYTORCH_CUDA_ALLOC_CONF)
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "================================================"
echo "  METSAT Lightning Nowcasting — DDP Training"
echo "================================================"
echo "  GPUs           : ${NUM_GPUS}"
echo "  OMP_NUM_THREADS: ${OMP_THREADS}  (${TOTAL_CORES} cores ÷ ${NUM_GPUS})"
echo "  Config         : ${CONFIG}"
echo "  Extra args     : $@"
echo "================================================"
echo ""

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    train.py \
    --config ${CONFIG} \
    "$@"
