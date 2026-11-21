"""Continue missing paired curves; reuse frozen caches, never old numerical gates."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("family", choices=("tp", "uvu", "radial"))
    p.add_argument("rows", type=int)
    p.add_argument("start", type=int)
    p.add_argument("--parity", choices=("canonical", "full"))
    p.add_argument("--memory", action="store_true")
    a = p.parse_args()
    result = ROOT / "results" / f"{a.family}-{a.rows}"
    result.mkdir(parents=True, exist_ok=True)
    attempts = []

    def run(stage, order, parity, timeout=1200, **options):
        path = result / f"{stage}-L{order}-{parity}-{options.get('role','all')}-{options.get('metric','both')}-r{options.get('repeat', 0)}.json"
        command = [sys.executable, "-S", "-u", str(ROOT / "benchmark.py"), stage,
            a.family, str(order), parity, str(a.rows), str(path)]
        for key, value in options.items():
            command.extend(("--"+key, str(value)))
        tick = time.monotonic()
        with path.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = 124
        if code == 124 or not path.exists():
            data = json.loads(path.read_text()) if path.exists() else {}
            data.update(stage=stage, family=a.family, order=order, parity=parity, rows=a.rows,
                        status="timeout" if code == 124 else "process-failure", passed=False, returncode=code,
                        timeout_s=timeout)
            path.write_text(json.dumps(data, indent=2))
        data = json.loads(path.read_text())
        attempts.append(dict(stage=stage, order=order, parity=parity, returncode=code,
            wall_s=time.monotonic()-tick, status=data["status"]))
        (result / "attempts.json").write_text(json.dumps(attempts, indent=2))
        print(a.family, order, parity, stage, options.get("repeat", ""), data["status"], data.get("accepted"), flush=True)
        return path, data

    for order in range(a.start, 9):
        for parity in ((a.parity,) if a.parity else ("canonical", "full")):
            _, build = run("build", order, parity, timeout=600)
            if not build["passed"]:
                continue
            q, gate = run("qualify", order, parity)
            if not gate["passed"]:
                continue
            for repeat in range(5):
                run("time", order, parity, qualification=q, repeat=repeat)
            if a.memory:
                for metric in ("forward", "forward_vjp"):
                    for role in gate["accepted_by_metric"][metric]:
                        run("memory", order, parity, qualification=q, role=role, metric=metric)
            # Independent corrected memory campaigns retain their own GPU gates.


if __name__ == "__main__":
    main()
