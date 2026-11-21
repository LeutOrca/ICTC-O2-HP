#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=o2-whole-audit-plot
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
py=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
"$py" aggregate.py
"$py" ../../figures/o2-hp-github-whole-20261121-r1/whole_network.py
"$py" ../../figures/o2-hp-github-whole-20261121-r1/whole_memory.py
pdffonts ../../figures/o2-hp-github-whole-20261121-r1/Whole_MACE_speed_throughput.pdf
pdffonts ../../figures/o2-hp-github-whole-20261121-r1/Whole_MACE_peak_memory.pdf
