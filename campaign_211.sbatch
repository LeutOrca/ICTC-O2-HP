#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --job-name=o2-cue013-paired
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
source h100_env.sh
runtime_root=/public_hw/home/phy_bjxiao/o2-benchmark/environments/torch211-ac7e624
test -f "$runtime_root/READY"
export O2_RUNTIME=211 O2_ROLES=native,cue-current O2_RESULT_SUFFIX=-211
export O2_CROSS_REFERENCE_ROOT="$remote_runs/o2-hp-github-whole-20261121-r1-9753ed3/results/campaign"
export PYTHONPATH="$runtime_root:$HPX_MACE_ENTRY/compatibility"
export LD_LIBRARY_PATH="$runtime_root/nvidia/cuda_cupti/lib:$LD_LIBRARY_PATH"
export TORCH_EXTENSIONS_DIR="$PWD/cache/extensions-211" TRITON_CACHE_DIR="$PWD/cache/triton-211"
cases=(b1 b8 n512 n4096)
"$py" -u campaign.py --case "${cases[$SLURM_ARRAY_TASK_ID]}"
