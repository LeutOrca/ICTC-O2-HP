#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=26G
#SBATCH --time=00:10:00
#SBATCH --job-name=mace-h100-two-selectors
#SBATCH --output=logs/%x-%j.log
source runtime.sh
export CUBLAS_WORKSPACE_CONFIG=:16:8
printf "CUBLAS_WORKSPACE_CONFIG=%s\n" "$CUBLAS_WORKSPACE_CONFIG"
sha256sum --quiet -c provenance/transfer.sha256
sha256sum --quiet -c provenance/benchmark.sha256
sha256sum --quiet -c provenance/bulk-source.sha256
sha256sum --quiet -c provenance/hotspot.sha256
sha256sum --quiet -c provenance/capture-both.sha256
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader | tee "$MACE_RESULT_ROOT/gpu.csv"
export PYTHONPATH="$PWD/compatibility"
anchor="$MACE_RESULT_ROOT"
stage() {
    local name=$1
    shift
    if ! "$mace_python" capture_both_runner.py "$name" --qualification "$anchor"         --signal "$anchor/$name.ok" "$@" > "$anchor/$name.log" 2>&1; then
        touch "$anchor/$name.failed"
        return 1
    fi
}
stage qualify &
qualify_pid=$!
stage trace --wait-for "$anchor/qualify.ok" &
trace_pid=$!
stage time --wait-for "$anchor/trace.ok" &
time_pid=$!
stage memory --wait-for "$anchor/time.ok" &
memory_pid=$!
set +e
for pid in "$qualify_pid" "$trace_pid" "$time_pid" "$memory_pid"; do wait "$pid"; done
set -e
if test -e "$anchor/memory.ok"; then
    "$mace_python" audit_capture_both.py "$SLURM_JOB_ID"
fi
if test -e "$anchor/trace.ok"; then
    "$mace_python" audit_traces.py "$anchor"
fi
test -f "$anchor/AUDIT.json"
