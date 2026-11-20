#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:08:00
#SBATCH --job-name=hpx-own-matrix
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=9.0
export TRITON_CACHE_DIR="$PWD/cache/triton" TORCH_EXTENSIONS_DIR="$PWD/cache/extensions"
export CUDA_HOME="$PWD/../o2-hpx-opt-20261117-r1/toolchain/cuda_host_v3"
export LD_LIBRARY_PATH="/public_hw/home/phy_bjxiao/RNA-Au-database/envs/mace_gpu/lib/python3.11/site-packages/nvidia/cuda_cupti/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python_bin=/public_hw/home/phy_bjxiao/RNA-Au-database/envs/mace_gpu/bin/python
mkdir -p results cache
sha256sum --quiet -c provenance/input.sha256
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader > results/gpu.txt
"$python_bin" -m py_compile matrix_kernels.py cuda_matrix.py qualify_matrix.py
"$python_bin" -u qualify_matrix.py triton "results/triton-${SLURM_JOB_ID}.json"
"$python_bin" -u qualify_matrix.py cuda "results/cuda-${SLURM_JOB_ID}.json"
