#!/bin/bash
#SBATCH -p GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=hpx-own-kernel-a100
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.0
export CXX=/usr/bin/g++-11 CC=/usr/bin/gcc-11
export CUDA_HOME="$PWD/../o2-hpx-opt-20261117-r1/toolchain/cuda_host_v2"
export LIBRARY_PATH="$CUDA_HOME/lib64"
export TRITON_CACHE_DIR="$PWD/cache/a100-triton"
export TORCH_EXTENSIONS_DIR="$PWD/cache/a100-extensions"
py=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
mkdir -p results cache
sha256sum --quiet -c provenance/a100-input.sha256
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader > "results/a100-gpu-${SLURM_JOB_ID}.txt"
srun "$py" -m py_compile matrix_kernels.py frozen_so2.py cuda_matrix.py qualify_matrix.py qualify_so2.py qualify_mace.py
srun "$py" -u qualify_matrix.py triton "results/triton-${SLURM_JOB_ID}.json"
srun "$py" -u qualify_so2.py triton "results/so2-triton-${SLURM_JOB_ID}.json"
srun "$py" -u qualify_matrix.py cuda "results/cuda-${SLURM_JOB_ID}.json"
srun "$py" -u qualify_so2.py cuda "results/so2-cuda-${SLURM_JOB_ID}.json"
export HPX_V3_UPSTREAM="$PWD/../o2-hpx-equiformerv3-oam-20261119-r1/upstream/equiformer_v3"
srun "$py" -u qualify_norm.py "results/norm-${SLURM_JOB_ID}.json"
