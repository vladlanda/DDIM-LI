#!/usr/bin/env bash
# ============================================================
# kill_training.sh — kill all running training processes
# Usage: ./kill_training.sh
# ============================================================

echo "Looking for training processes..."

# Collect PIDs
PIDS=$(pgrep -f "torchrun|train.py|python.*train" | tr '\n' ' ')

if [ -z "$PIDS" ]; then
    echo "No training processes found."
else
    echo "Found PIDs: $PIDS"

    # Graceful kill first
    pkill -f "torchrun"
    pkill -f "train.py"
    sleep 2

    # Force kill anything still alive
    pkill -9 -f "torchrun"
    pkill -9 -f "train.py"

    echo "Processes killed."
fi

# Show GPU status after kill
echo ""
echo "GPU status:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
    || echo "(nvidia-smi not available)"