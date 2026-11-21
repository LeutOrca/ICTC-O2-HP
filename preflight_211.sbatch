#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=o2-cue013-preflight
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
source h100_env.sh
runtime_root=/public_hw/home/phy_bjxiao/o2-benchmark/environments/torch211-ac7e624
archive=/public_hw/home/phy_bjxiao/o2-benchmark/environments/torch211-cu128-ac7e624.tgz
printf 'ac7e624692f9a191d984a406d4700e2adac746b127f35887ac2e171dd78c02ab  %s\n' "$archive" | sha256sum --quiet -c -
if [ ! -f "$runtime_root/READY" ]; then
    mkdir "$runtime_root"
    tar -xzf "$archive" -C "$runtime_root"
    touch "$runtime_root/READY"
fi
export O2_RUNTIME=211
export PYTHONPATH="$runtime_root:$HPX_MACE_ENTRY/compatibility"
export LD_LIBRARY_PATH="$runtime_root/nvidia/cuda_cupti/lib:$LD_LIBRARY_PATH"
export TORCH_EXTENSIONS_DIR="$PWD/cache/extensions-211" TRITON_CACHE_DIR="$PWD/cache/triton-211"
"$py" -u inventory.py "provenance/environment-h100-211-${SLURM_JOB_ID}.json"
set +e
"$py" -u whole.py ref native b1 fixed results/preflight/211-native-b1.json
"$py" -u whole.py build cue-current b1 fixed results/preflight/211-cue-current-build.json
if [ "$?" = 0 ]; then
    "$py" -u whole.py qualify cue-current b1 fixed results/preflight/211-cue-current-b1.json --reference results/preflight/211-native-b1.pt
fi
