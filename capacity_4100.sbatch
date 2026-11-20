#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=58G
#SBATCH --time=00:37:00
#SBATCH --job-name=mace-capacity-4100
#SBATCH --output=logs/%x-%j.log
source runtime.sh
export CUBLAS_WORKSPACE_CONFIG=:16:8
sha256sum --quiet -c provenance/transfer.sha256
sha256sum --quiet -c provenance/capacity-4100.sha256
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader | tee "$MACE_RESULT_ROOT/gpu.csv"
export PYTHONPATH="$PWD/compatibility"
export TRITON_CACHE_DIR="$PWD/cache/receiver-csr"
"$mace_python" -u qualify_force.py --mode native-graph-tiled > "$MACE_RESULT_ROOT/tiled-calibration.log" 2>&1
check() {
    "$mace_python" -u capacity_check.py "$1" > "$MACE_RESULT_ROOT/$1.log" 2>&1
}
check reference
baseline=native-graph
if check native-graph; then
    :
else
    status=$?
    if [[ $status != 42 ]]; then
        exit "$status"
    fi
    baseline=native-graph-tiled
    check "$baseline"
fi
printf '%s\n' "$baseline" > "$MACE_RESULT_ROOT/baseline.txt"
check receiver-csr-graph
for step in trace time memory; do
    "$mace_python" -u measure_force.py "$step" --baseline "$baseline" --case n4096 > "$MACE_RESULT_ROOT/$step.log" 2>&1
done
