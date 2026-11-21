"""Corrected fresh-process memory pairs with a new same-GPU FP32 qualification."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("tp", "uvu", "radial"))
    parser.add_argument("rows", type=int)
    args = parser.parse_args()
    result = ROOT / "results" / f"memory-corrected-{args.family}-{args.rows}"
    result.mkdir(parents=True, exist_ok=True)
    def run(stage, order, parity, **options):
        receipt = result / f"{stage}-L{order}-{parity}-{options.get('role','all')}-{options.get('metric','both')}.json"
        command = [sys.executable, "-S", "-u", str(ROOT / "benchmark.py"), stage,
                   args.family, str(order), parity, str(args.rows), str(receipt)]
        for key, value in options.items():
            command.extend(["--"+key, str(value)])
        with receipt.with_suffix(".log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1200, check=False)
        if not receipt.exists():
            receipt.write_text(json.dumps(dict(stage=stage, family=args.family, order=order,
                parity=parity, rows=args.rows, status="process-failure", passed=False)))
        data = json.loads(receipt.read_text())
        print(args.family, order, parity, stage, options, data["status"], flush=True)
        return receipt, data
    for order in range(1, 9):
        for parity in ("canonical", "full"):
            q, gate = run("qualify", order, parity)
            if not gate["passed"]:
                continue
            for metric in ("forward", "forward_vjp"):
                for role in gate["accepted_by_metric"][metric]:
                    run("memory", order, parity, qualification=q, role=role, metric=metric)


if __name__ == "__main__":
    main()
