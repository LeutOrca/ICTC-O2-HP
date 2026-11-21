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
base_site="${py%/bin/python}/lib/python3.11/site-packages"
"$py" -S prepare_runtime.py "$base_site" "$PWD/deps-211" "$PWD/provenance/dependency-view-211.json"
export PYTHONPATH="$runtime_root:$PWD/deps-211:$HPX_MACE_ENTRY/compatibility"
export O2_CLEAN_RUNTIME=1
export O2_NATIVE_GRAPH=original
mkdir -p build-tools
ninja_wheels=(build-wheels/ninja-1.13.0-*.whl)
test "${#ninja_wheels[@]}" = 1
"$py" -S -m zipfile -e "${ninja_wheels[0]}" build-tools
export PATH="$PWD/build-tools/ninja/data/bin:$PATH"
chmod u+x build-tools/ninja/data/bin/ninja
export PATH="$runtime_root/ninja/data/bin:$PATH"
export LD_LIBRARY_PATH="$runtime_root/nvidia/cuda_cupti/lib:$LD_LIBRARY_PATH"
export TORCH_EXTENSIONS_DIR="$PWD/cache/extensions-211" TRITON_CACHE_DIR="$PWD/cache/triton-211"
"$py" -S -u inventory.py "provenance/environment-h100-211-${SLURM_JOB_ID}.json"
set +e
"$py" -S -u whole.py ref native b1 fixed results/preflight/211-native-b1.json
for role in eqx-current cue-current; do
    "$py" -S -u whole.py build "$role" b1 fixed "results/preflight/211-${role}-build.json"
    if [ "$?" = 0 ]; then
        "$py" -S -u whole.py qualify "$role" b1 fixed "results/preflight/211-${role}-b1.json" --reference results/preflight/211-native-b1.pt
    fi
done
