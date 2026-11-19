#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:08:00
#SBATCH --job-name=mace-h100-determ
#SBATCH --output=logs/%x-%j.log
source runtime.sh
sha256sum --quiet -c provenance/transfer.sha256
sha256sum --quiet -c provenance/bulk-source.sha256
sha256sum --quiet -c provenance/deterministic.sha256
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader | tee "$MACE_RESULT_ROOT/gpu.csv"
export PYTHONPATH="$PWD/compatibility"
"$mace_python" deterministic_check.py
