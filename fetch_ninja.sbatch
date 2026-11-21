#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=o2-eqx-build-tool
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
mkdir -p build-wheels
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy /public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -m pip download --no-deps --retries 0 --timeout 20 --index-url https://pypi.org/simple ninja==1.13.0 -d build-wheels
sha256sum build-wheels/* > provenance/ninja.sha256
