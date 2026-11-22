"""Independent CPU audit of frozen inputs and full original FP32 gate coverage."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "o2-hp-angular-operators-20261121-r1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    with (BASE / "summary.csv").open() as stream:
        old = list(csv.DictReader(stream))
    expected = [(n, "eqx-o2-repair", "eqx-o2") for n in range(4, 9)] + [
                (n, "public-unfused-repair", "public-unfused") for n in range(4, 7)]
    accepted, pending, failures = [], [], []
    for order, role, baseline_role in expected:
        path = ROOT / "measurements/repair-a5cbc8c/numerical-repair" / f"qualify-L{order}-full-{role}.json"
        if not path.exists():
            pending.append(dict(order=order, role=role))
            continue
        data = json.loads(path.read_text())
        if role not in data.get("accepted_by_metric", {}).get("forward_vjp", []):
            failures.append(dict(order=order, role=role, receipt=str(path.relative_to(ROOT))))
            continue
        assert data["dtype"] == "float32" and data["torch"] == "2.11.0+cu128" and "H100" in data["gpu"]
        assert not any(data[key] for key in ("tf32", "amp", "cuda_graph"))
        references = [r for r in old if r["family"] == "radial" and r["rows"] == "16384" and
                      r["parity"] == "full" and r["order"] == str(order) and r["role"] == baseline_role]
        assert references, (order, baseline_role)
        for key in ("input_sha256", "canonical_weights_sha256"):
            assert data[key] == references[0][key], (order, role, key)
        assert set(data["numerics"]) == {f"{role}/{seed}/{scale}" for seed in (20261003, 20261004) for scale in (.1, 1.)}
        values = []
        for result in data["numerics"].values():
            assert result["passed"] and result["forward_passed"] and result["vjp_passed"] and result["full_workload"]
            assert result["reference_chunk_size"] == 4096 and result["check_window_size"] == 512
            assert [(r["start"], r["end"]) for r in result["chunks"]] == [(n, n+512) for n in range(0, 16384, 512)]
            for chunk in result["chunks"]:
                assert len(chunk["checks"]) == 4
                for check in chunk["checks"]:
                    assert check["passed"] and check["arithmetic"] == "float32"
                    values.append(check)
        accepted.append(dict(order=order, role=role, job=data["job"], gpu_uuid=data["gpu_uuid"],
            input_sha256=data["input_sha256"], canonical_weights_sha256=data["canonical_weights_sha256"],
            receipt=str(path.relative_to(ROOT)), qualification_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            checks=len(values), max_abs=max(c["max_abs"] for c in values),
            max_relative_l2=max(c["relative_l2"] for c in values)))
    report = dict(all_eight_passed=len(accepted) == 8 and not failures, accepted=accepted, pending=pending,
                  failures=failures, unchanged_baseline="6f641cf", original_fp32_gates=True,
                  timing_memory_not_measured=True)
    (ROOT / "provenance/REPAIR-GATE-AUDIT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if args.require_all and not report["all_eight_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
