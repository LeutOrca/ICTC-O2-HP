"""Read receipts only on a Slurm CPU compute node, preserving all failures."""
import json
import os
from collections import Counter
from pathlib import Path

assert os.environ.get("SLURM_JOB_ID")
root = Path(__file__).resolve().parents[1]
original = []
for path in sorted(root.glob("receipts/9cc0634/results/*/*-original.json")):
    record = json.loads(path.read_text())
    original.append(dict(path=str(path.relative_to(root)), stage=record["stage"],
                         execution_completed=record.get("execution_completed"),
                         original_cases=[dict(family=c["family"],
                             input_gate=c.get("original_fp32_input_gate"),
                             double_backward=c.get("double_backward"),
                             full_parameter_vjp=c.get("full_parameter_vjp"))
                             for c in record.get("cases", [])]))
summaries = []
failed = []
numeric_cases = 0
cohort_failures = {}
for path in sorted(root.glob("receipts/*/results/*/SUMMARY.json")):
    record = json.loads(path.read_text())
    summaries.append(dict(path=str(path.relative_to(root)), **record))
    if record.get("stage") == "build":
        continue
    for case_path in sorted(path.parent.glob("*.json")):
        if case_path.name == "SUMMARY.json":
            continue
        case = json.loads(case_path.read_text())
        numeric_cases += 1
        if not case.get("passed"):
            cohort = path.relative_to(root).parts[1]
            count = cohort_failures.setdefault(cohort, Counter())
            count.update(str(c.get("output_index", "unknown")) for c in case.get("checks", []) if not c["passed"])
            failed.append(dict(path=str(case_path.relative_to(root)), error=case.get("error"),
                               failed_checks=[c for c in case.get("checks", []) if not c["passed"]]))
report = dict(job=os.environ["SLURM_JOB_ID"], original=original, summaries=summaries,
              failed_output_roles_by_cohort={k: dict(v) for k, v in cohort_failures.items()},
              numerical_cases=numeric_cases, failed=failed,
              numerical_pass=numeric_cases > 0 and not failed,
              whole_model_pass=False)
target = root / "provenance/approved-execution/COLLECTED-AUDIT.json"
target.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
