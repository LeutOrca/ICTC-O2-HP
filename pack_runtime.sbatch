#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=o2-current-runtime-pack
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
runtime_site="$PWD/../o2-hpx-opt-20261117-r1/environments/torch211-cu128/lib/python3.11/site-packages"
tar -I 'pigz -1 -p 4' -cf environment211.tgz -C "$runtime_site" .
sha256sum environment211.tgz > provenance/environment211.sha256
