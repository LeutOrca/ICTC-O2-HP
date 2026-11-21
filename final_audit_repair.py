"""Independent CPU delivery audit; numerical gates stay exclusively FP32."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = json.loads((root / "SUMMARY.json").read_text())
    assert summary["performance_complete"]
    baseline = root.parent / "hpx-custom-kernels-20261120-r1"
    baseline_files = ("matrix_kernels.py", "so2_layer.py", "frozen_so2.py", "v3_native.py",
                      "cuda_matrix.py", "csrc/matrix.cu")
    baseline_hashes = {name: digest(root / name) for name in baseline_files}
    assert all(value == digest(baseline / name) for name, value in baseline_hashes.items())
    measured = root / "cohorts/a977215"
    candidate_files = (*baseline_files, "fused_block.py", "so2_ordered.py", "scheduled_matrix.py", "fp32_gate.py")
    candidate_hashes = {name: digest(root / name) for name in candidate_files}
    assert all(value == digest(measured / name) for name, value in candidate_hashes.items())
    scope_trace = json.loads((measured / "results/SO2-OWN-TRACE.json").read_text())
    assert scope_trace["passed"] and len(scope_trace["layers"]) == 14
    assert not scope_trace["forbidden_library_matrix"]
    for path in (measured / "cache/triton").glob("*/*.ptx"):
        source = path.read_text()
        assert ".f64" not in source and ".tf32" not in source, path
    pairs = [row for row in summary["whole_network_pairs"] if row["implementation"] == "fused-block"]
    assert {row["case"] for row in pairs} == {"mptrj-b1", "mptrj-b8"}
    reports = []
    for row in pairs:
        directory = (root / row["path"]).parent
        for role_qualification in ("qualification-", "qualification-c591b16-"):
            qualification = json.loads((directory / f"{role_qualification}{row['case']}.json").read_text())
            assert qualification["passed"] and len(qualification["records"]) == 4
            assert qualification["metrics_dtype"] == "torch.float32"
            assert all(value["passed"] and value["metrics_dtype"] == "torch.float32"
                       for record in qualification["records"] for value in record["numerics"].values())
        traces = []
        for boundary in ("fixed", "e2e"):
            for role in ("native", "base", "own"):
                path = directory / f"v3-trace-{row['case']}-{boundary}-{role}-0.json"
                trace = json.loads(path.read_text())
                raw = path.with_suffix(".trace.json")
                assert trace["cuda_kernels"] and raw.exists() and raw.stat().st_size > 0
                traces.append(dict(path=str(raw.relative_to(root)), bytes=raw.stat().st_size, sha256=digest(raw)))
        qualification = json.loads((directory / f"qualification-{row['case']}.json").read_text())
        force_max = max(record["numerics"]["forces"]["relative_l2"] for record in qualification["records"])
        energy_max = max(record["numerics"]["energy"]["max_abs"] for record in qualification["records"])
        reports.append(dict(case=row["case"], source=row["source"], gpu_uuid=row["gpu_uuid"],
                            force_relative_l2_max=force_max, energy_max_abs=energy_max, traces=traces))
    checkpoint = root.parent / "o2-hpx-equiformerv3-oam-20261119-r2/checkpoint/omat24-mptrj-salex_gradient.pt"
    assert digest(checkpoint) == summary["checkpoint_sha256"]
    compiler = json.loads((root / "provenance/COMPILER_QA.json").read_text())
    snapshot = json.loads((root / "provenance/SNAPSHOT_QA.json").read_text())
    assert compiler["passed"] and snapshot["passed"]
    report = dict(passed=True, metrics_dtype="torch.float32", baseline="c591b16",
        checkpoint_sha256=summary["checkpoint_sha256"], candidate_kernel_source="a977215",
        baseline_files=baseline_hashes, candidate_files=candidate_hashes, numerics=reports,
        summary_sha256=digest(root / "SUMMARY.json"),
        notes="Numerical and artifact acceptance only; speed and memory goals remain separate in SUMMARY.json")
    (root / "provenance/DELIVERY_QA.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(passed=True, numerics=[{key: value for key, value in row.items() if key != "traces"}
                                               for row in reports])), flush=True)


if __name__ == "__main__":
    main()
