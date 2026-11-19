"""Independently recompute the final B1 gates from per-round H100 receipts."""
import json
from pathlib import Path
import statistics
import sys


job = sys.argv[1]
root = Path(f"paired-{job}")
gpu = (root / "gpu.csv").read_text().strip()
assert gpu.count("\n") == 0 and "NVIDIA H100 80GB" in gpu
receipts = root / "receipts"
candidate, previous = (
    sys.argv[2:4] if len(sys.argv) == 4
    else ("hp-uvu-compact-graph", "hp-uvu-species-graph")
)


def read(backend, boundary, stage, repeat):
    suffix = f"cuda-{backend}-b1-{boundary}-{stage}-{repeat}.json"
    path = receipts / suffix
    result = json.loads(path.read_text())
    assert result["job"] == job, path
    assert result["gpu"] in gpu and result["torch"] == "2.6.0+cu124", path
    assert (result["atoms"], result["edges"], result["structures"]) == (6, 272, 1), path
    assert not result["tf32"] and not result["amp"], path
    return result


comparisons = []
for reference in ("native", "native-graph", previous):
    first, second = [read(name, "fixed", "qualify", 0) for name in (reference, candidate)]
    assert first["passed"] and second["passed"]
    for boundary in ("fixed", "e2e"):
        time_a, time_b, mem_a, mem_b, reserve_a, reserve_b = ([] for _ in range(6))
        for r in range(1, 6):
            a = read(reference, boundary, "time", r)
            b = read(candidate, boundary, "time", r)
            assert len(a["wall_ms"]) == len(b["wall_ms"]) == 15
            time_a.append(statistics.median(a["wall_ms"]))
            time_b.append(statistics.median(b["wall_ms"]))
            for backend, live, reserved in [(reference, mem_a, reserve_a), (candidate, mem_b, reserve_b)]:
                measured = read(backend, boundary, "memory", r)
                live.append(measured["peak_allocated"])
                reserved.append(measured["peak_reserved"])
        assert max(mem_a) <= 1.02 * min(mem_a) and max(mem_b) <= 1.02 * min(mem_b)
        assert max(reserve_a) <= 1.02 * min(reserve_a) and max(reserve_b) <= 1.02 * min(reserve_b)
        ratio = statistics.median(a / b for a, b in zip(time_a, time_b))
        saving = 1 - statistics.median(b / a for a, b in zip(mem_a, mem_b))
        reserved_saving = 1 - statistics.median(b / a for a, b in zip(reserve_a, reserve_b))
        a, b = statistics.median(time_a), statistics.median(time_b)
        comparisons.append(dict(
            reference=reference, candidate=candidate, boundary=boundary, speed=ratio,
            allocated_saving=saving, reserved_saving=reserved_saving,
            basis_ms=a, candidate_ms=b, basis_atoms_s=6000/a, candidate_atoms_s=6000/b,
            basis_allocated=statistics.median(mem_a), candidate_allocated=statistics.median(mem_b),
            basis_reserved=statistics.median(reserve_a), candidate_reserved=statistics.median(reserve_b),
            gates_met=ratio>=2 and saving>=0.10,
            reserved_target_met=reserved_saving>=0.10,
            max_force_abs=second["numerics"][boundary]["forces"]["max_abs"],
            max_force_rel_l2=second["numerics"][boundary]["forces"]["relative_l2"],
        ))
assert len(comparisons) == 6
data = dict(job=job, gpu=gpu, fixed_composition=True, comparisons=comparisons)
Path("FINAL_GATE.json").write_text(json.dumps(data, indent=2) + "\n")
for row in comparisons:
    print(row, flush=True)
