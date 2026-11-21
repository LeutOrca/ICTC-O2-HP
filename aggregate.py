"""Audit receipts and produce qualified same-GPU paired tables on a CPU node."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    paths = sorted((ROOT / "measurements").rglob("*.json"))
    receipts, failures = [], []
    for path in paths:
        if path.name.endswith(".trace.json") or path.name.startswith("environment-"):
            continue
        try:
            row = json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(row, dict) or "stage" not in row:
            continue
        row = dict(row, receipt=str(path.relative_to(ROOT)))
        if row.get("stage") == "qualify" or not row.get("passed", False):
            failures.append({key: row.get(key, "") for key in
                             ("stage", "role", "case", "status", "passed", "torch", "error", "receipt")})
        receipts.append(row)
    timings = [r for r in receipts if r.get("stage") == "time" and r.get("passed")]
    qualifications = {hashlib.sha256((ROOT / row["receipt"]).read_bytes()).hexdigest(): row
                      for row in receipts if row.get("stage") == "qualify" and row.get("passed")}
    groups = {}
    for row in timings:
        key = row["torch"], row["case"], row["role"], row["boundary"]
        groups.setdefault(key, []).append(row)
    summary = []
    for (runtime, case, role, boundary), rows in sorted(groups.items()):
        if len(rows) != 5 or {r["repeat"] for r in rows} != set(range(5)):
            raise RuntimeError(f"Incomplete paired repeats: {(runtime, case, role, boundary)}")
        native = groups[runtime, case, "native", boundary]
        if len(native) != 5:
            raise RuntimeError("Missing native five-round reference")
        if len({r["gpu_uuid"] for r in rows + native}) != 1:
            raise RuntimeError("A purported same-card pair used different GPU UUIDs")
        if len({r["input_sha256"] for r in rows + native}) != 1:
            raise RuntimeError("Inputs differ inside a pair")
        if any(r["dtype"] != "float32" or r["tf32"] or r["amp"] or not r["cuda_graph"] for r in rows + native):
            raise RuntimeError("Precision/Graph contract mismatch")
        medians = [r["wall_ms"] for r in rows]
        reference = {r["repeat"]: r for r in native}
        ratios = [reference[r["repeat"]]["wall_ms"] / r["wall_ms"] for r in rows]
        atom_rates = [r["atoms_s"] for r in rows]
        one = rows[0]
        for row in rows:
            q = qualifications.get(row["qualification_sha256"])
            if q is None or (q["role"], q["case"], q["torch"], q["input_sha256"]) != (
                    role, case, runtime, row["input_sha256"]):
                raise RuntimeError("A timing lacks its exact accepted qualification receipt")
        memory = [r for r in receipts if r.get("stage") == "memory" and r.get("passed")
                  and (r.get("torch"), r.get("case"), r.get("role"), r.get("boundary"))
                  == (runtime, case, role, boundary)]
        native_memory = [r for r in receipts if r.get("stage") == "memory" and r.get("passed")
                         and (r.get("torch"), r.get("case"), r.get("role"), r.get("boundary"))
                         == (runtime, case, "native", boundary)]
        if len(memory) != 1 or len(native_memory) != 1:
            raise RuntimeError("Expected one separate fresh memory process per arm/boundary")
        if {r["gpu_uuid"] for r in memory + native_memory + rows} != {one["gpu_uuid"]}:
            raise RuntimeError("Memory pair used a different GPU")
        record = dict(runtime=runtime, role=role, case=case, boundary=boundary,
                      atoms=one["atoms"], edges=one["edges"], structures=one["structures"],
                      gpu_uuid=one["gpu_uuid"], input_sha256=one["input_sha256"],
                      checkpoint_sha256=one["checkpoint_sha256"], coverage=one["coverage"],
                      latency_ms=median(medians), latency_q25=float(np.quantile(medians, .25)),
                      latency_q75=float(np.quantile(medians, .75)), speedup=median(ratios),
                      speedup_q25=float(np.quantile(ratios, .25)), speedup_q75=float(np.quantile(ratios, .75)),
                      atoms_s=one["atoms"]*1000/median(medians),
                      atoms_s_q25=float(np.quantile(atom_rates, .25)),
                      atoms_s_q75=float(np.quantile(atom_rates, .75)),
                      structures_s=one["structures"]*1000/median(medians),
                      peak_allocated_mib=memory[0]["peak_allocated"]/2**20,
                      peak_reserved_mib=memory[0]["peak_reserved"]/2**20,
                      allocated_drop_pct=100*(1-memory[0]["peak_allocated"]/native_memory[0]["peak_allocated"]),
                      reserved_drop_pct=100*(1-memory[0]["peak_reserved"]/native_memory[0]["peak_reserved"]),
                      time_processes=5, memory_processes=1, qualified=True)
        record["e2e_2x"] = boundary == "e2e" and record["speedup"] >= 2
        record["allocated_10pct"] = record["allocated_drop_pct"] >= 10
        record["reserved_10pct"] = record["reserved_drop_pct"] >= 10
        summary.append(record)
    target = ROOT / "results/summary.csv"
    if not summary:
        raise RuntimeError("No completed qualified pairs")
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (ROOT / "results/status.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("stage", "role", "case", "status", "passed", "torch", "error", "receipt"))
        writer.writeheader()
        writer.writerows(failures)
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths if not path.name.endswith(".trace.json")}
    (ROOT / "results/RECEIPT-SHA256.json").write_text(json.dumps(hashes, indent=2))
    (ROOT / "SUMMARY.json").write_text(json.dumps(dict(summary=summary, qualification_and_failures=failures,
        audit=dict(same_gpu=True, matched_inputs=True, fp32=True, cuda_graph=True,
                   time_processes_per_arm=5, separate_memory_processes_per_arm=1)), indent=2))
    figure = ROOT.parents[1] / "figures/o2-hp-github-whole-20261121-r1/data"
    (figure / "summary.csv").write_bytes(target.read_bytes())
    (figure / "status.csv").write_bytes((ROOT / "results/status.csv").read_bytes())
    print("Audited", len(summary), "rows and", len(hashes), "receipts", flush=True)


if __name__ == "__main__":
    main()
