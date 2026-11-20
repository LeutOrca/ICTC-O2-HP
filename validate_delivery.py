"""Validate final receipt coverage, frozen weights and kernel source equivalence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main():
    summary = json.loads((ROOT / "SUMMARY.json").read_text())
    selected = summary["mace_h100_best_cases"]
    assert summary["mace_h100_three_case_target"]
    rows = []
    for case, item in selected.items():
        audit = item["audit"]
        directory = (ROOT / item["path"]).parent
        qualification = json.loads((directory / "qualification.json").read_text())
        assert qualification["passed"] and qualification["cuda_graph"]
        assert all(metric["passed"] for group in ("numerics", "capture", "native_capture")
                   for row in qualification[group].values() for metric in row.values())
        candidate = audit["candidate"]
        roles = ("native", "base", *(('final',) if "vs_final" in audit["boundaries"]["e2e"] else ()), candidate)
        count = 0
        for stage in ("time", "memory", "trace"):
            for boundary in ("fixed", "e2e"):
                for role in roles:
                    for repeat in range(1 if stage == "trace" else 5):
                        path = directory / f"mace-{stage}-{case}-{boundary}-{role}-{repeat}.json"
                        value = json.loads(path.read_text())
                        assert value["source"] == audit["source"]
                        assert value["gpu_uuid"] == audit["gpu_uuid"]
                        assert value["cuda_graph"] and not value["tf32"] and not value["amp"]
                        assert value["qualification_sha256"] == digest(directory / "qualification.json")
                        if stage == "trace":
                            assert path.with_suffix(".trace.json").exists()
                            if role == candidate:
                                assert {"_receiver", "_angular", "_radial", "_project"}.issubset(value["cuda_kernels"])
                        count += 1
        rows.append(dict(case=case, source=audit["source"], gpu_uuid=audit["gpu_uuid"],
                         receipts=count, e2e=audit["boundaries"]["e2e"]["vs_native"], passed=True))
    kernels = ("mace_scalar_tp.py", "mace_receiver_tp.py", "mace_species_projection.py",
               "mace_frozen_prefix.py", "native_reduce.py", "matrix_kernels.py")
    source_equivalence = {}
    for name in kernels:
        hashes = {cohort: digest(ROOT / "cohorts" / cohort / name)
                  for cohort in ("40af2f8", "fb3dc99", "64dbeb1", "20df315")}
        assert len(set(hashes.values())) == 1
        source_equivalence[name] = hashes
    checkpoint = ROOT.parent / "mace-ictc-o2-hp-gbu2-20261119-r1/frozen/models/pretrained/native.model"
    assert digest(checkpoint) == "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a"
    result = dict(passed=True, mace_three_cases=rows, kernel_source_equivalence=source_equivalence,
                  checkpoint_sha256=digest(checkpoint), training_qualified=False,
                  whole_v3_5x_claim=False, operator_6x_claim=False)
    (ROOT / "provenance/DELIVERY_QA.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(passed=True, mace_cases=len(rows), source_equivalence=True)), flush=True)


if __name__ == "__main__":
    main()
