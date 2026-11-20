#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:17:00
#SBATCH --job-name=mace-receiver-csr
#SBATCH --output=logs/%x-%j.log
source runtime.sh
export CUBLAS_WORKSPACE_CONFIG=:16:8
sha256sum --quiet -c provenance/transfer.sha256
sha256sum --quiet -c provenance/receiver-csr.sha256
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader | tee "$MACE_RESULT_ROOT/gpu.csv"
export PYTHONPATH="$PWD/compatibility"
export TRITON_CACHE_DIR="$PWD/cache/receiver-csr"
make_gate() {
    "$mace_python" -u check_receiver.py &&
    "$mace_python" -u qualify_force.py --mode receiver-csr-graph &&
    touch "$MACE_RESULT_ROOT/qualified.ok"
}
stage() {
    local step=$1 previous=$2
    "$mace_python" -u measure_force.py "$step" --wait-for "$MACE_RESULT_ROOT/$previous.ok"
}
(make_gate || { touch "$MACE_RESULT_ROOT/failed"; exit 1; }) > "$MACE_RESULT_ROOT/qualify.log" 2>&1 &
gate_pid=$!
(stage trace qualified || { touch "$MACE_RESULT_ROOT/failed"; exit 1; }) > "$MACE_RESULT_ROOT/trace.log" 2>&1 &
trace_pid=$!
(stage time trace || { touch "$MACE_RESULT_ROOT/failed"; exit 1; }) > "$MACE_RESULT_ROOT/time.log" 2>&1 &
time_pid=$!
(stage memory time || { touch "$MACE_RESULT_ROOT/failed"; exit 1; }) > "$MACE_RESULT_ROOT/memory.log" 2>&1 &
memory_pid=$!
wait "$gate_pid" "$trace_pid" "$time_pid" "$memory_pid"
test -f "$MACE_RESULT_ROOT/memory.ok"
