"""Audit paired E/F, wall throughput and allocator peaks on a single GPU."""
import json
import os
from pathlib import Path
import statistics
import sys

target = Path(f"paired-{int(sys.argv[1])}")
receipts = {}
for file in (target / "receipts").glob("*.json"):
    row = json.loads(file.read_text())
    if row["job"] != sys.argv[1]:
        raise ValueError(f"Receipt from another job: {file}")
    key = tuple(row[field] for field in ("backend", "case", "boundary", "stage")) + (
        int(file.stem.rsplit("-", 1)[-1]),)
    if key in receipts:
        raise ValueError(f"Duplicate receipt: {file}")
    receipts[key] = row

validation = []
rows = []
gates = {
    "native-indexadd": "native",
    "hp-scalar": "native",
    "hp-local-o2": "hp-scalar",
    "hp-uvu": "hp-scalar",
    "eqx": "native",
}
for candidate, step_base in gates.items():
    for case in ("b8", "n512"):
        qkey = (candidate, case, "fixed", "qualify", 0)
        if qkey not in receipts:
            validation.append(dict(candidate=candidate, case=case, state="missing"))
            continue
        qualified = receipts[qkey]
        validation.append(dict(candidate=candidate, case=case,
                               state="pass" if qualified["passed"] else "fail",
                               numerics=qualified["numerics"]))
        if not qualified["passed"]:
            continue
        for base in sorted(set(("native", step_base))):
            for boundary in ("fixed", "e2e"):
                runs = []
                for repetition in range(1, 6):
                    keys = [(mode, case, boundary, stage, repetition)
                            for mode in (base, candidate) for stage in ("time", "memory")]
                    if any(key not in receipts for key in keys):
                        continue
                    time_base, memory_base, time_candidate, memory_candidate = (
                        receipts[key] for key in keys)
                    shape = [(r["atoms"], r["edges"], r["structures"], r["gpu"], r["torch"])
                             for r in (time_base, memory_base, time_candidate, memory_candidate)]
                    if len(set(shape)) != 1:
                        raise ValueError(f"Non-comparable pair: {keys}")
                    runs.append(dict(
                        round=repetition, basis_ms=time_base["median_ms"],
                        candidate_ms=time_candidate["median_ms"],
                        ratio=time_base["median_ms"] / time_candidate["median_ms"],
                        basis_atoms_s=time_base["atoms_s"], candidate_atoms_s=time_candidate["atoms_s"],
                        basis_structures_s=time_base["structures_s"],
                        candidate_structures_s=time_candidate["structures_s"],
                        basis_allocated=memory_base["peak_allocated"],
                        candidate_allocated=memory_candidate["peak_allocated"],
                        allocation_ratio=memory_candidate["peak_allocated"] / memory_base["peak_allocated"],
                        basis_reserved=memory_base["peak_reserved"],
                        candidate_reserved=memory_candidate["peak_reserved"],
                    ))
                medians = {key: statistics.median(r[key] for r in runs)
                           for key in runs[0] if key != "round"} if runs else {}
                rows.append(dict(base=base, candidate=candidate, case=case, boundary=boundary,
                                 complete=len(runs) == 5, runs=runs, medians=medians,
                                 target_met=len(runs) == 5 and medians["ratio"] >= 2
                                 and medians["allocation_ratio"] <= 0.90))

audit = dict(job=os.environ["SLURM_JOB_ID"], measurement_job=sys.argv[1],
             gpu=(target / "gpu.csv").read_text().strip() if (target / "gpu.csv").is_file() else None,
             validation=validation, comparisons=rows, count=len(receipts))
output = target / "AUDIT.json"
output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
print(output, len(receipts), len(rows), flush=True)
