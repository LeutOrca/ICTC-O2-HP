#!/bin/bash
#SBATCH -p GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=hpx-own-stage
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
study_root=${HPX_STUDY_ROOT:?Set the immutable study root}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 EQV3_DETERMINISTIC=1
export TRITON_CACHE_DIR="$PWD/cache/triton"
export HPX_MACE_ENTRY="$study_root/../mace-ictc-o2-hp-gbu2-20261120-r1"
export MACE_SOURCE="$study_root/../mace-ictc-o2-hp-gbu2-20261119-r1/frozen"
export MACE_RESULT_ROOT="$PWD/results/mace-${SLURM_JOB_ID}"
export HPX_V3_ENTRY="$study_root/../o2-hpx-equiformerv3-oam-20261119-r2"
export HPX_V3_UPSTREAM="$study_root/../o2-hpx-equiformerv3-oam-20261119-r1/upstream/equiformer_v3"
export PYTHONPATH="$HPX_V3_ENTRY/env-vendor:$HPX_V3_ENTRY/deps:$HPX_V3_UPSTREAM:$HPX_V3_UPSTREAM/src"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LD_LIBRARY_PATH="/public/home/sps-xia/bjxiao/miniconda3/envs/o2/lib/python3.11/site-packages/nvidia/cuda_cupti/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
py=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
mkdir -p logs results "$MACE_RESULT_ROOT/receipts"
sha256sum --quiet -c provenance/input.sha256
printf 'job=%s node=%s stage=%s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$*"
if test "$1" = paired-v3; then
    bash pair_v3.sh "$2" "$3"
elif test "$1" = paired-layer; then
    srun "$py" -u qualify_layer.py results/so2-layer.json
    for stage in time memory; do
        for repeat in 0 1 2 3 4; do
            roles=(native own)
            if (( repeat % 2 )); then roles=(own native); fi
            for mode in forward input_vjp; do
                for role in "${roles[@]}"; do
                    srun "$py" -u measure_layer.py "$stage" "$role" "$mode" \
                        "results/$stage-$mode-$role-$repeat.json" --repeat "$repeat"
                done
            done
        done
    done
    for mode in forward input_vjp; do
        for role in native own; do
            srun "$py" -u measure_layer.py trace "$role" "$mode" \
                "results/trace-$mode-$role.json"
        done
    done
else
    srun "$py" -u "$@"
fi
