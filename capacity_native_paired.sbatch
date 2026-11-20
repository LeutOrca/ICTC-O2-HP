#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --job-name=mace-native-4100-pair
#SBATCH --output=logs/%x-%j.log
source runtime.sh
export CUBLAS_WORKSPACE_CONFIG=:16:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sha256sum --quiet -c provenance/transfer.sha256
sha256sum --quiet -c provenance/capacity-native-paired.sha256
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader | tee "$MACE_RESULT_ROOT/gpu.csv"
export PYTHONPATH="$PWD/compatibility"
export TRITON_CACHE_DIR="$PWD/cache/receiver-csr"
printf '%s\n' "$PYTORCH_CUDA_ALLOC_CONF" > "$MACE_RESULT_ROOT/allocator.txt"
printf '%s\n' native-graph > "$MACE_RESULT_ROOT/baseline.txt"
for mode in reference native-graph receiver-csr-graph receiver-csr-partition-graph; do
    "$mace_python" -u capacity_check.py "$mode" > "$MACE_RESULT_ROOT/$mode.log" 2>&1
done
for step in trace time memory; do
    "$mace_python" -u measure_force.py "$step" --case n4096 \
        --compare-prior receiver-csr-graph --variant receiver-csr-partition-graph > "$MACE_RESULT_ROOT/$step.log" 2>&1
done
