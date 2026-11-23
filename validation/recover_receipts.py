"""Bounded CPU-node receipt recovery; never submits GPU or whole-model jobs."""
import json
import os
from pathlib import Path
import subprocess
import time

assert os.environ.get("SLURM_JOB_ID")
ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / "provenance/approved-execution"
REMOTE = "/public_hw/home/phy_bjxiao/o2-benchmark/runs/o2-hp-approved-20261123-r1-"
SSH = ["ssh", "-p", "18022", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-i",
       "/public/home/sps-xia/bjxiao/.ssh/id_rsa_GBU", "-o", "StrictHostKeyChecking=yes",
       "-o", "UpdateHostKeys=no", "-o", "ConnectTimeout=15", "-o", "ConnectionAttempts=1"]
DEST = "phy_bjxiao@10.10.4.99"
jobs = (318804, 318805, 318806, 318807, 318808)
attempts = []
recovered = False
terminal = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"}
for attempt in range(8):
    try:
        probe = subprocess.run(SSH + [DEST, "source /etc/profile; sacct -n -X -P -j 318804,318805,318806,318807,318808 --format=JobIDRaw,State,ElapsedRaw"],
                               text=True, capture_output=True, timeout=25)
        rows = {int(p[0]): p[1].split()[0] for line in probe.stdout.splitlines()
                if len(p := line.split("|")) >= 3 and p[0].isdigit()}
        attempts.append(dict(attempt=attempt + 1, unix_time=time.time(), returncode=probe.returncode,
                             stdout=probe.stdout, stderr=probe.stderr, states=rows))
        if probe.returncode == 0 and all(rows.get(j) in terminal for j in jobs):
            for cohort in ("14e7d16", "d621827", "fddf444"):
                destination = ROOT / "receipts" / cohort / "results"
                destination.mkdir(parents=True, exist_ok=True)
                subprocess.run(["rsync", "-a", "--timeout=30", "-e", " ".join(SSH),
                                f"{DEST}:{REMOTE}{cohort}/results/", str(destination) + "/"], check=True, timeout=240)
            recovered = True
            break
    except (subprocess.SubprocessError, OSError) as exc:
        attempts.append(dict(attempt=attempt + 1, unix_time=time.time(), error=repr(exc)))
    (PROV / "RECOVERY-ATTEMPTS.json").write_text(json.dumps(attempts, indent=2))
    if attempt != 7:
        time.sleep(60)


def numerical_group(cohort, job, expected_count):
    directory = ROOT / "receipts" / cohort / "results" / str(job)
    paths = [p for p in directory.glob("*.json") if p.name != "SUMMARY.json"]
    cases = [json.loads(p.read_text()) for p in paths]
    passed = len(cases) == expected_count and all(c.get("passed") and not c.get("built_only")
                  and c.get("checks") and all(v["passed"] for v in c["checks"])
                  and c.get("tf32") is False and c.get("cuda_graph") is False for c in cases)
    return dict(passed=passed, cases=len(cases), job=job,
                failures=[str(p.relative_to(ROOT)) for p, c in zip(paths, cases) if not c.get("passed")])


record = dict(job=os.environ["SLURM_JOB_ID"], recovered=recovered, attempts=attempts,
              gpu_jobs_submitted=0, whole_model_jobs_submitted=0, g1_all_passed=False)
if recovered:
    record["uvu_full"] = {str(order): numerical_group("14e7d16", job, 4)
                           for order, job in ((6, 318787), (7, 318789), (8, 318767))}
    record["radial_small"] = numerical_group("fddf444", 318805, 32)
    record["radial_full_L8"] = numerical_group("fddf444", 318808, 4)
    fp64 = ROOT / "receipts/fddf444/results/318806/FP64-ORACLE.json"
    record["fp64_diagnostic_passed"] = json.loads(fp64.read_text())["passed"] if fp64.exists() else None
    record["still_required"] = ["radial E16384/full L4..7", "A100 fallback receipt", "whole-model equivalent full-parity and half-bond mapping"]
(PROV / "RECOVERED-G1.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2), flush=True)
if not recovered:
    raise SystemExit(2)
