#!/bin/bash
# Control-plane only. Submit one bounded H100 diagnostic after connectivity returns.
set -euo pipefail
cd "$(dirname "$0")"
frozen=$(cat provenance/DIAGNOSTIC-GIT.txt)
test "$(git rev-parse HEAD)" = "$frozen"
sha256sum -c provenance/DIAGNOSTIC-PAYLOAD.sha256
short=${frozen:0:7}
payload="provenance/diagnostic-${short}.tgz"
payload_sha=$(sha256sum "$payload" | cut -d ' ' -f 1)
remote_dir="/public_hw/home/phy_bjxiao/o2-benchmark/runs/o2-hp-full-parity-repair-20261122-r1-${short}"
ssh_opts=(-o BatchMode=yes -o IdentitiesOnly=yes -i /public/home/sps-xia/bjxiao/.ssh/id_rsa_GBU
          -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o ConnectTimeout=8 -p 18022)
remote=phy_bjxiao@10.10.4.99
ssh "${ssh_opts[@]}" "$remote" 'source /etc/profile; hostname; squeue -h -u phy_bjxiao -o "%i|%T|%j|%R"' > provenance/DIAGNOSTIC-QUEUE.txt
ssh "${ssh_opts[@]}" "$remote" "set -e
test ! -e '$remote_dir'
mkdir -p '$remote_dir'"
scp -o BatchMode=yes -o IdentitiesOnly=yes -i /public/home/sps-xia/bjxiao/.ssh/id_rsa_GBU \
    -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o ConnectTimeout=8 -P 18022 \
    "$payload" "$remote:$remote_dir/payload.tgz"
ssh "${ssh_opts[@]}" "$remote" "source /etc/profile
set -euo pipefail
cd '$remote_dir'
printf '%s  %s\\n' '$payload_sha' payload.tgz | sha256sum -c -
tar -xzf payload.tgz
sha256sum -c provenance/DIAGNOSTIC-SOURCE.sha256 > provenance/DEPLOYMENT-CHECK.txt
mkdir -p logs results cache
job_id=\$(sbatch --parsable diagnose.sbatch --order 4 --role eqx-o2 --seed 20261003 --reference-start 8192 --reference-rows 4096)
printf '%s\\n' \"\$job_id\" > provenance/SUBMITTED-JOB.txt
printf '%s\\n' \"\$job_id\"" | tee provenance/SUBMITTED-JOB.txt
