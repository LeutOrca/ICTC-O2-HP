#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:06:00
#SBATCH --job-name=mace-native-allocator
#SBATCH --output=logs/%x-%j.log
source runtime.sh
export CUBLAS_WORKSPACE_CONFIG=:16:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sha256sum --quiet -c provenance/transfer.sha256
sha256sum --quiet -c provenance/capacity-allocator.sha256
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader | tee "$MACE_RESULT_ROOT/gpu.csv"
export PYTHONPATH="$PWD/compatibility"
reference=../mace-ictc-o2-hp-gbu2-20261120-r1-6e9330b/paired-315635/receipts/n4096-tiled-eager.pt
printf '8ec28d57ac4062faa48ed9bc02245c95a5da205e3b03d7456c73f4c4063f627d  %s\n' "$reference" | sha256sum --quiet -c -
cp -- "$reference" "$MACE_RESULT_ROOT/receipts/"
printf '%s\n' "$PYTORCH_CUDA_ALLOC_CONF" > "$MACE_RESULT_ROOT/allocator.txt"
"$mace_python" -u capacity_check.py native-graph
