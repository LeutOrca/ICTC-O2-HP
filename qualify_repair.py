"""Numerical-only gate for all eight true failures. No timing before acceptance."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    result = ROOT / "results" / "numerical-repair"
    result.mkdir(parents=True, exist_ok=True)
    targets = [(n, "eqx-o2-repair") for n in range(4, 9)] + [
               (n, "public-unfused-repair") for n in range(4, 7)]
    outcomes = []
    from fork_runner import run
    for order, role in targets:
        for stage in ("build", "qualify"):
            path = result / f"{stage}-L{order}-full-{role}.json"
            command = [sys.executable, "-S", "-u", str(ROOT / "benchmark.py"), stage,
                       "radial", str(order), "full", "16384", str(path),
                       "--role", role, "--metric", "forward_vjp", "--reference-chunk", "4096"]
            code = run(command, path.with_suffix(".log"), 900 if stage == "build" else 1500)
            data = json.loads(path.read_text()) if path.exists() else {}
            if code == 124:
                data.update(passed=False, status="timeout", stage=stage, role=role,
                            family="radial", order=order, parity="full", rows=16384)
                path.write_text(json.dumps(data, indent=2))
            accepted = role in data.get("accepted_by_metric", {}).get("forward_vjp", [])
            good = data.get("passed", False) if stage == "build" else accepted
            print(order, role, stage, data.get("status", "no-receipt"), good, flush=True)
            if not good:
                outcomes.append(dict(order=order, role=role, passed=False, stage=stage, receipt=str(path)))
                break
            if stage == "qualify":
                outcomes.append(dict(order=order, role=role, passed=True, receipt=str(path)))
    record = dict(stage="numerical-repair-summary", all_eight_passed=len(outcomes)==8 and all(r["passed"] for r in outcomes),
                  configuration=os.environ["O2_RADIAL_ADJOINT"], outcomes=outcomes,
                  job=os.environ.get("SLURM_JOB_ID"), timing_not_run=True)
    (result / "SUMMARY.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    if not record["all_eight_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
