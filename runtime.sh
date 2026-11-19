#!/bin/bash
source /etc/profile
set -euo pipefail
export MACE_RUN="$PWD" MACE_SOURCE="$PWD/frozen"
export MACE_RESULT_ROOT="$PWD/paired-${SLURM_JOB_ID}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export NVIDIA_TF32_OVERRIDE=0 EQX_USE_TF32=0 EQX_USE_CUDA_GRAPH=0 MACE_USE_CUEQ_CG=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export TRITON_CACHE_DIR="$PWD/cache/triton" TORCH_EXTENSIONS_DIR="$PWD/cache/extensions"
export XDG_CACHE_HOME="$PWD/cache/xdg" MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=9.0
export CUDA_HOME="$PWD/../o2-hpx-opt-20261117-r1/toolchain/cuda_host_v3"
mace_python=/public_hw/home/phy_bjxiao/RNA-Au-database/envs/mace_gpu/bin/python
export LD_LIBRARY_PATH="${mace_python%/bin/python}/lib/python3.11/site-packages/nvidia/cuda_cupti/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$MACE_RESULT_ROOT"/{receipts,traces} "$XDG_CACHE_HOME/torch/kernels"
