#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=00:19:00
#SBATCH --job-name=mace-4100-pair-retry
#SBATCH --output=logs/%x-%j.log
source runtime.sh
export CUBLAS_WORKSPACE_CONFIG=:16:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sha256sum --quiet -c provenance/transfer.sha256
sha256sum --quiet -c provenance/capacity-native-recovery.sha256
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader | tee "$MACE_RESULT_ROOT/gpu.csv"
export PYTHONPATH="$PWD/compatibility"
export TRITON_CACHE_DIR="$PWD/cache/receiver-csr"
printf '%s\n' "$PYTORCH_CUDA_ALLOC_CONF" > "$MACE_RESULT_ROOT/allocator.txt"
printf '%s\n' native-graph > "$MACE_RESULT_ROOT/baseline.txt"
reference=../mace-ictc-o2-hp-gbu2-20261120-r1-6e9330b/paired-315635/receipts/n4096-tiled-eager.pt
printf '8ec28d57ac4062faa48ed9bc02245c95a5da205e3b03d7456c73f4c4063f627d  %s\n' "$reference" | sha256sum --quiet -c -
cp -- "$reference" "$MACE_RESULT_ROOT/receipts/"
gate() {
    "$mace_python" -u capacity_check.py native-graph &&
    "$mace_python" -u capacity_check.py receiver-csr-partition-graph &&
    touch "$MACE_RESULT_ROOT/qualified.ok"
}
stage() {
    local step=$1 previous=$2
    "$mace_python" -u measure_force.py "$step" --case n4096 \
        --variant receiver-csr-partition-graph --wait-for "$MACE_RESULT_ROOT/$previous.ok"
}
(gate || { touch "$MACE_RESULT_ROOT/failed"; exit 1; }) > "$MACE_RESULT_ROOT/qualify.log" 2>&1 &
gate_pid=$!
(stage trace qualified || { touch "$MACE_RESULT_ROOT/failed"; exit 1; }) > "$MACE_RESULT_ROOT/trace.log" 2>&1 &
trace_pid=$!
(stage time trace || { touch "$MACE_RESULT_ROOT/failed"; exit 1; }) > "$MACE_RESULT_ROOT/time.log" 2>&1 &
time_pid=$!
(stage memory time || { touch "$MACE_RESULT_ROOT/failed"; exit 1; }) > "$MACE_RESULT_ROOT/memory.log" 2>&1 &
memory_pid=$!
wait "$gate_pid" "$trace_pid" "$time_pid" "$memory_pid"
test -f "$MACE_RESULT_ROOT/memory.ok"
