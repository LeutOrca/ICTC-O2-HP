#!/bin/bash
#SBATCH -p GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=o2-approved-a100-fallback
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 EQX_USE_TF32=0 EQX_USE_CUDA_GRAPH=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:16:8
export TRITON_CACHE_DIR="$PWD/cache/a100-fallback"
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u validation/a100_fallback.py "$1"
