#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH -c 2
#SBATCH --mem=2G
#SBATCH -t 00:04:00
#SBATCH -J mace-h100-restore
#SBATCH -o logs/%x-%j.log
set -euo pipefail
input_root=$PWD
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
baseline=1b65a8c5f462cbe35232ddc7f0f2d6466ad2aeed
for bundle in snapshots/*.bundle; do
    name=${bundle##*/}
    sha256sum -c "provenance/${name%.bundle}-bundle.sha256"
    destination="$tmp/${name%.bundle}"
    git init -q "$destination"
    head=$(git bundle list-heads "$bundle" | awk 'NR==1 {print $2}')
    git -C "$destination" fetch -q "$input_root/$bundle" "$head"
    git -C "$destination" checkout -q --detach FETCH_HEAD
    git -C "$destination" merge-base --is-ancestor "$baseline" HEAD
    git -C "$destination" fsck --no-dangling
    git -C "$destination" log -1 --format="%H %s"
done
