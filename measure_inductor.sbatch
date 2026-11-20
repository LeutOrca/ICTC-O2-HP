#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:13:00
#SBATCH --job-name=mace-b8-inductor
#SBATCH --output=logs/%x-%j.log
source runtime.sh
export CUBLAS_WORKSPACE_CONFIG=:16:8
sha256sum --quiet -c provenance/transfer.sha256
sha256sum --quiet -c provenance/inductor-b8.sha256
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader | tee "$MACE_RESULT_ROOT/gpu.csv"
export PYTHONPATH="$PWD/compatibility"
export TRITON_CACHE_DIR="$PWD/cache/receiver-csr"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor" TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
"$mace_python" -u qualify_force.py --case b8 --mode receiver-csr-partition-graph \
    --mode receiver-csr-inductor-graph > "$MACE_RESULT_ROOT/qualify.log" 2>&1
for step in trace time memory; do
    "$mace_python" -u measure_force.py "$step" --case b8 --compare-prior receiver-csr-partition-graph \
        --variant receiver-csr-inductor-graph > "$MACE_RESULT_ROOT/$step.log" 2>&1
done
