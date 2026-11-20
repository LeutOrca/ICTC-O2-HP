#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=00:15:00
#SBATCH --job-name=hpx-own-h100
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
study_root=${HPX_STUDY_ROOT:?Set the immutable study root}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:16:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PWD/cache/triton"
export HPX_MACE_ENTRY="$study_root/../mace-ictc-o2-hp-gbu2-20261120-r1-78231f3"
export MACE_SOURCE="$HPX_MACE_ENTRY/frozen"
export MACE_RESULT_ROOT="$PWD/results/mace-${SLURM_JOB_ID}"
export PYTHONPATH="$HPX_MACE_ENTRY/compatibility"
py=/public_hw/home/phy_bjxiao/RNA-Au-database/envs/mace_gpu/bin/python
export LD_LIBRARY_PATH="${py%/bin/python}/lib/python3.11/site-packages/nvidia/cuda_cupti/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p logs results "$MACE_RESULT_ROOT/receipts"
sha256sum --quiet -c provenance/input.sha256
printf 'd4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a  %s\n' \
    "$MACE_SOURCE/models/pretrained/native.model" | sha256sum --quiet -c -
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader > "results/gpu-${SLURM_JOB_ID}.txt"
"$py" -u "$@"
