"""Run a whole angular curve on one allocated GPU, preserving every failure."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("tp", "uvu", "radial", "so2", "ictp"))
    parser.add_argument("rows", type=int)
    args = parser.parse_args()
    result = ROOT / "results" / f"{args.family}-{args.rows}"
    result.mkdir(parents=True, exist_ok=True)
    attempts = []
    def run(stage, order, parity, **options):
        extra = f"-{options.get('role','all')}-{options.get('metric','both')}-r{options.get('repeat',0)}"
        receipt = result / f"{stage}-L{order}-{parity}{extra}.json"
        command = [sys.executable, "-S", "-u", str(ROOT / "benchmark.py"), stage,
                   args.family, str(order), parity, str(args.rows), str(receipt)]
        for key, value in options.items():
            command.extend(["--"+key, str(value)])
        tick = time.monotonic()
        with receipt.with_suffix(".log").open("w") as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           check=False, timeout=1200)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = 124
        if not receipt.exists():
            receipt.write_text(json.dumps(dict(stage=stage, family=args.family,
                order=order, parity=parity, rows=args.rows, status="timeout" if code == 124 else "process-failure",
                passed=False, returncode=code)))
        data = json.loads(receipt.read_text())
        attempts.append(dict(stage=stage, order=order, parity=parity, options={k:str(v) for k,v in options.items()},
                             returncode=code, wall_s=time.monotonic()-tick, status=data["status"]))
        (result / "attempts.json").write_text(json.dumps(attempts, indent=2))
        print(args.family, order, parity, stage, options, data["status"], data.get("accepted"), flush=True)
        return receipt, data
    parities = ("canonical", "full") if args.family in ("tp", "uvu", "radial") else ("natural",)
    for order in range(1, 9):
        for parity in parities:
            _, build = run("build", order, parity)
            if not build["passed"]:
                continue
            q, gate = run("qualify", order, parity)
            if not gate["passed"]:
                continue
            for repeat in range(5):
                run("time", order, parity, qualification=q, repeat=repeat)
            for metric in ("forward", "forward_vjp"):
                for role in gate["accepted_by_metric"][metric]:
                    run("memory", order, parity, qualification=q, role=role, metric=metric)


if __name__ == "__main__":
    main()
