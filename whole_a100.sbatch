#!/bin/bash
#SBATCH -p GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=hpx-own-whole-a100
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
study_root=${HPX_STUDY_ROOT:?Set the immutable study root}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 EQV3_DETERMINISTIC=1
export TRITON_CACHE_DIR="$PWD/cache/a100-triton"
export TORCH_EXTENSIONS_DIR="$PWD/cache/a100-extensions"
export HPX_MACE_ENTRY="$study_root/../mace-ictc-o2-hp-gbu2-20261120-r1"
export MACE_SOURCE="$study_root/../mace-ictc-o2-hp-gbu2-20261119-r1/frozen"
export MACE_RESULT_ROOT="$PWD/results/mace-${SLURM_JOB_ID}"
export HPX_V3_ENTRY="$study_root/../o2-hpx-equiformerv3-oam-20261119-r2"
export HPX_V3_UPSTREAM="$study_root/../o2-hpx-equiformerv3-oam-20261119-r1/upstream/equiformer_v3"
export PYTHONPATH="$HPX_V3_ENTRY/env-vendor:$HPX_V3_ENTRY/deps:$HPX_V3_UPSTREAM:$HPX_V3_UPSTREAM/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LD_LIBRARY_PATH="/public/home/sps-xia/bjxiao/miniconda3/envs/o2/lib/python3.11/site-packages/nvidia/cuda_cupti/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
py=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
mkdir -p "$MACE_RESULT_ROOT/receipts"
sha256sum --quiet -c provenance/whole-input.sha256
if test "$1" = mace; then
    srun "$py" -u qualify_mace.py b8 --output "$MACE_RESULT_ROOT/b8.json"
    srun "$py" -u qualify_mace.py n512 --output "$MACE_RESULT_ROOT/n512.json"
else
    srun "$py" -u qualify_v3.py --suite mptrj-b1 --output "results/v3-b1-${SLURM_JOB_ID}.json"
    srun "$py" -u qualify_v3.py --suite mptrj-b8 --output "results/v3-b8-${SLURM_JOB_ID}.json"
fi
