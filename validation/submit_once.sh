#!/bin/bash
# Run from an immutable cohort root. A repeated transport request returns the
# recorded job rather than submitting another GPU allocation.
set -euo pipefail
token=$1
shift
[[ $token =~ ^[a-zA-Z0-9_-]+$ ]] || exit 64
mkdir -p submissions logs
exec 9>"submissions/$token.lock"
flock -x 9
if [[ -s submissions/$token.job ]]; then
    cat "submissions/$token.job"
    exit 0
fi
name="o2g1-$token"
if [[ -e submissions/$token.requested ]]; then
    echo "Submission may already exist; reconcile scheduler for $name before retry" >&2
    exit 65
fi
date -u +%FT%TZ >"submissions/$token.requested"
printf '%q ' "$@" >"submissions/$token.command"
printf '\n' >>"submissions/$token.command"
job=$(sbatch --parsable --job-name="$name" "$@")
[[ $job =~ ^[0-9]+$ ]] || exit 66
printf '%s\n' "$job" >"submissions/$token.job.tmp"
mv "submissions/$token.job.tmp" "submissions/$token.job"
cat "submissions/$token.job"
