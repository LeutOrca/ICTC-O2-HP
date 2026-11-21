#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:04:00
#SBATCH --job-name=v3-native-sass
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
tool=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/lib/python3.11/site-packages/triton/backends/nvidia/bin/cuobjdump
lib=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/lib/python3.11/site-packages/nvidia/cublas/lib/libcublas.so.12
mkdir -p results provenance
sha256sum "$lib" > provenance/cublas.sha256
"$tool" --gpu-architecture sm_80 --dump-elf-symbols "$lib" > results/NATIVE-SYMBOLS.txt
rg 'gemmSN_(TN|NN)_kernel' results/NATIVE-SYMBOLS.txt > results/SO2-SYMBOLS.txt
head -12 results/SO2-SYMBOLS.txt
