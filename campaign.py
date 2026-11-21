"""Sequential, fresh-process qualification and five alternating paired rounds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("b1", "b8", "n512", "n4096"))
    args = parser.parse_args()
    output = root / ("results/campaign" + os.getenv("O2_RESULT_SUFFIX", ""))
    output.mkdir(parents=True, exist_ok=True)
    attempts = []
    def run(stage, role, case, boundary="fixed", repeat=0, **kwargs):
        receipt = output / f"{stage}-{role}-{case}-{boundary}-r{repeat}.json"
        command = [sys.executable] + (["-S"] if os.getenv("O2_CLEAN_RUNTIME") else []) + ["-u", str(root / "whole.py"), stage, role, case,
                   boundary, str(receipt), "--repeat", str(repeat)]
        for key, value in kwargs.items():
            command.extend(["--" + key, str(value)])
        tick = time.monotonic()
        with receipt.with_suffix(".log").open("w") as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=600, check=False)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = 124
        attempts.append(dict(stage=stage, role=role, case=case, boundary=boundary,
                             repeat=repeat, returncode=code, wall_s=time.monotonic()-tick))
        (output / f"attempts-{args.case or 'all'}.json").write_text(json.dumps(attempts, indent=2))
        if not receipt.exists():
            receipt.write_text(json.dumps(dict(stage=stage, role=role, case=case,
                boundary=boundary, repeat=repeat, status="timeout" if code == 124 else "process-failure",
                passed=False, returncode=code)))
        result = json.loads(receipt.read_text())
        print(stage, role, case, boundary, repeat, result["status"], flush=True)
        return receipt, result

    roles_to_test = os.getenv("O2_ROLES", "native,official,fork,local-hpx,eqx-current,cue-current").split(",")
    builds = {}
    preflight = Path(os.getenv("O2_PREFLIGHT_ROOT", str(root / "results/preflight")))
    prefix = os.getenv("O2_RUNTIME", "26")
    for role in roles_to_test:
        cached = preflight / f"{prefix}-{role}-build.json"
        builds[role] = cached.exists() and json.loads(cached.read_text()).get("passed", False)
    for case in ((args.case,) if args.case else ("b1", "b8", "n512", "n4096")):
        cross = {}
        if os.getenv("O2_CROSS_REFERENCE_ROOT"):
            cross["cross-reference"] = Path(os.environ["O2_CROSS_REFERENCE_ROOT"]) / f"ref-native-{case}-fixed-r0.pt"
        ref, result = run("ref", "native", case, **cross)
        if not result.get("passed"):
            continue
        qualifications = {}
        for role in roles_to_test:
            if role != "native" and not builds.get(role):
                continue
            q, result = run("qualify", role, case, reference=ref.with_suffix(".pt"), **cross)
            if result.get("passed"):
                qualifications[role] = q
        roles = list(qualifications)
        for repeat in range(5):
            order = roles if repeat % 2 == 0 else list(reversed(roles))
            for role in order:
                run("time-both", role, case, "both", repeat,
                    qualification=qualifications[role])
        for boundary in ("fixed", "e2e"):
            for role in roles:
                run("memory", role, case, boundary, qualification=qualifications[role])
        for role in roles:
            run("trace", role, case, qualification=qualifications[role])


if __name__ == "__main__":
    main()
