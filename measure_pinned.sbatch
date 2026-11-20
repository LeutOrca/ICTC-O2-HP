#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:18:00
#SBATCH --job-name=mace-pinned-inputs
#SBATCH --output=logs/%x-%j.log
source runtime.sh
export CUBLAS_WORKSPACE_CONFIG=:16:8
sha256sum --quiet -c provenance/transfer.sha256
sha256sum --quiet -c provenance/pinned-inputs.sha256
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader | tee "$MACE_RESULT_ROOT/gpu.csv"
export PYTHONPATH="$PWD/compatibility"
export TRITON_CACHE_DIR="$PWD/cache/receiver-csr"
"$mace_python" -u qualify_force.py --case b8 --mode receiver-csr-graph --mode receiver-csr-pinned-graph > "$MACE_RESULT_ROOT/qualify.log" 2>&1
for step in trace time memory; do
    "$mace_python" -u measure_force.py "$step" --case b8 --compare-prior receiver-csr-graph \
        --variant receiver-csr-pinned-graph > "$MACE_RESULT_ROOT/$step.log" 2>&1
done
