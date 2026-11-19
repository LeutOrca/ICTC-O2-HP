#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:17:00
#SBATCH --job-name=mace-h100-pairs
#SBATCH --output=logs/%x-%j.log
source runtime.sh
sha256sum --quiet -c provenance/transfer.sha256
sha256sum --quiet -c provenance/benchmark.sha256
sha256sum --quiet -c provenance/bulk-source.sha256
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader | tee "$MACE_RESULT_ROOT/gpu.csv"
export PYTHONPATH="$PWD/compatibility"
"$mace_python" bulk_runner.py qualify
"$mace_python" bulk_runner.py time --qualification "$MACE_RESULT_ROOT"
"$mace_python" bulk_runner.py memory --qualification "$MACE_RESULT_ROOT"
"$mace_python" audit_pairs.py "$SLURM_JOB_ID"
