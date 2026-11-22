"""Qualify bounded-build candidates, then same-card paired timing/memory.

Every role is built and qualified in a separate CUDA-clean process. A failed
input VJP cannot be accepted on the strength of a passing forward receipt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from fork_runner import run as fork_run

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("qualify", "measure"))
    parser.add_argument("family", choices=("uvu", "radial"))
    parser.add_argument("orders", type=int, nargs="+")
    args = parser.parse_args()
    result = ROOT / "results" / (args.phase + "-bounded-" + args.family)
    result.mkdir(parents=True, exist_ok=True)
    outcomes = []

    def run(stage, order, role, metric="forward_vjp", **options):
        path = result / f"{stage}-L{order}-full-{role}-{metric}-r{options.get('repeat', 0)}.json"
        command = [sys.executable, "-S", "-u", str(ROOT / "benchmark.py"), stage,
                   args.family, str(order), "full", "16384", str(path),
                   "--role", role, "--metric", metric]
        if stage == "qualify":
            command.extend(("--reference-chunk", "4096"))
        for key, value in options.items():
            command.extend(("--" + key, str(value)))
        limit = {"build": 900, "qualify": 1500, "time": 1500, "memory": 600}[stage]
        tick = time.monotonic()
        code = fork_run(command, path.with_suffix(".log"), limit)
        elapsed = time.monotonic() - tick
        data = json.loads(path.read_text()) if path.exists() else {}
        if code == 124 or not path.exists():
            data.update(stage=stage, family=args.family, order=order, parity="full", rows=16384,
                        role=role, metric=metric, passed=False,
                        status="timeout" if code == 124 else "process-failure")
            path.write_text(json.dumps(data, indent=2))
        good = data.get("passed", False)
        if stage == "qualify":
            good = role in data.get("accepted_by_metric", {}).get("forward_vjp", [])
        outcomes.append(dict(stage=stage, order=order, role=role, metric=metric,
                             receipt=str(path.relative_to(ROOT)), passed=good,
                             process_elapsed_s=elapsed, code=code))
        print(stage, args.family, order, role, metric, data.get("status"), good, round(elapsed, 2), flush=True)
        return path, data, good

    gates = []
    for order in args.orders:
        if args.phase == "qualify":
            roles = ("public-repair",) if args.family == "uvu" else ("public-repair", "public-unfused-repair")
        else:
            roles = ("public-repair", "eqx-o2") if args.family == "uvu" else (
                "public-repair", "public-unfused-repair", "eqx-o2-repair")
        qualifications = []
        for role in roles:
            _, _, good = run("build", order, role)
            if good:
                path, data, good = run("qualify", order, role)
                if good:
                    qualifications.append((path, data))
        if len(qualifications) == len(roles):
            gates.append((order, qualifications))
    all_passed = len(gates) == len(args.orders)
    # No performance work until every requested numerical gate has passed.
    if all_passed and args.phase == "measure":
        for order, qualifications in gates:
            merged = dict(qualifications[0][1])
            merged.update(role=None, accepted=list(roles),
                accepted_by_metric={m: list(roles) for m in ("forward", "forward_vjp")},
                numerics={k: v for _, g in qualifications for k, v in g["numerics"].items()},
                component_qualifications=[hashlib.sha256(p.read_bytes()).hexdigest() for p, _ in qualifications])
            qpath = result / f"qualify-L{order}-full-merged.json"
            qpath.write_text(json.dumps(merged, indent=2))
            for repeat in range(5):
                _, _, good = run("time", order, "all", qualification=qpath, repeat=repeat)
                all_passed &= good
            for metric in ("forward", "forward_vjp"):
                for role in roles:
                    _, _, good = run("memory", order, role, metric, qualification=qpath)
                    all_passed &= good
    record = dict(phase=args.phase, family=args.family, orders=args.orders,
                  job=os.getenv("SLURM_JOB_ID"), all_passed=all_passed, outcomes=outcomes,
                  original_fp32_gates=True, network_training_paused=True)
    (result / ("SUMMARY-" + "-".join(map(str, args.orders)) + ".json")).write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    if not all_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
