#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:05:00
#SBATCH --job-name=o2-full-failure-audit
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i /public/home/sps-xia/bjxiao/.ssh/id_rsa_GBU -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o ConnectTimeout=8 -p 18022 phy_bjxiao@10.10.4.99 'source /etc/profile; hostname; squeue -h -u phy_bjxiao -o "%i|%T|%j|%R"' > provenance/GBU2-CPU-PROBE.txt 2>&1 || true
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python inspect_failures.py
