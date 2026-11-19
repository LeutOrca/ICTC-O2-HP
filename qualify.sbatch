#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --job-name=mace-h100-ef
#SBATCH --output=logs/%x-%j.log
source runtime.sh
sha256sum --quiet -c provenance/transfer.sha256
sha256sum --quiet -c provenance/checkpoint.sha256
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader | tee "$MACE_RESULT_ROOT/gpu.csv"
"$mace_python" -c 'import torch, triton, matscipy; print(torch.__version__, triton.__version__, matscipy.__version__)'
for point in b1 b8 n512 n4096; do
    for backend in native hp-scalar hp-local-o2 hp-uvu; do
        if ! "$mace_python" paired_runner.py --stage qualify --backend "$backend" --case "$point"; then
            printf 'qualify\t%s\t%s\n' "$backend" "$point" | tee -a "$MACE_RESULT_ROOT/errors.tsv"
        fi
    done
done
if ! find "$MACE_RESULT_ROOT/receipts" -name '*-native-*-qualify-0.json' -print -quit | grep -q .; then
    exit 1
fi
