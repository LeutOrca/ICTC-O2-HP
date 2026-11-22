"""Isolated fresh Triton-cache construction receipts for the 15 old timeouts."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    targets = [("uvu", n, "public-repair") for n in (6, 7, 8)]
    targets += [("radial", n, "public-repair") for n in (5, 6, 7, 8)]
    targets += [("radial", n, "public-unfused-repair") for n in (7, 8)]
    output = ROOT / "results/cold-build"
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for family, order, role in targets:
        stem = f"{family}-L{order}-full-{role}"
        path = output / (stem + ".json")
        cache = ROOT / "cold-cache" / os.environ["SLURM_JOB_ID"] / stem
        assert not cache.exists(), str(cache)
        env = dict(os.environ, TRITON_CACHE_DIR=str(cache), O2_PROCESS_MODE="exec")
        command = [sys.executable, "-S", "-u", str(ROOT / "benchmark.py"), "build", family,
                   str(order), "full", "16384", str(path), "--role", role, "--metric", "forward_vjp"]
        tick = time.monotonic()
        with path.with_suffix(".log").open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                code = proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = 124
        elapsed = time.monotonic() - tick
        data = json.loads(path.read_text()) if path.exists() else {}
        good = code == 0 and data.get("builds", {}).get(role, {}).get("passed", False)
        receipts.append(dict(family=family, order=order, role=role, passed=good,
                             elapsed_s=elapsed, code=code, limit_s=900,
                             initially_absent_triton_cache=str(cache), receipt=str(path.relative_to(ROOT))))
        print(json.dumps(receipts[-1]), flush=True)
    record = dict(passed=all(r["passed"] for r in receipts), job=os.environ["SLURM_JOB_ID"],
                  scope="forward plus all input VJPs; fresh Triton cache per case; separate process",
                  covered_old_timeout_slots=15, cases=receipts)
    (output / "SUMMARY.json").write_text(json.dumps(record, indent=2))
    if not record["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
