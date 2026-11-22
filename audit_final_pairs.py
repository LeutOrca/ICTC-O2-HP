"""Independent CPU coverage and pairing audit of newly repaired operator points."""
from collections import Counter
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "o2-hp-angular-operators-20261121-r1"


def main():
    def rows(path):
        with path.open() as stream:
            return list(csv.DictReader(stream))
    original = rows(BASE / "summary.csv")
    points = rows(ROOT / "summary.csv")
    pairs = rows(ROOT / "speedups.csv")
    summary = json.loads((ROOT / "SUMMARY.json").read_text())
    roles = rows(ROOT / "role-coverage.csv")
    key = ("family", "rows", "order", "parity", "role", "metric")
    index = {tuple(r[k] for k in key): r for r in points}
    assert len(index) == len(points)
    for row in original:
        other = index[tuple(row[k] for k in key)]
        for field in ("input_sha256", "canonical_weights_sha256", "dtype", "equation", "rows", "channels", "hidden", "tf32", "amp", "cuda_graph"):
            assert row[field] == other[field], (key, field)
    repaired = [r for r in points if r["role"].endswith("repair")]
    wanted = {("uvu", str(n), "public-repair", m) for n in range(6, 9) for m in ("forward", "forward_vjp")}
    wanted |= {("radial", str(n), r, m) for n in range(4, 9) for r in (
        "public-repair", "public-unfused-repair", "eqx-o2-repair") for m in ("forward", "forward_vjp")}
    assert {(r["family"], r["order"], r["role"], r["metric"]) for r in repaired} == wanted
    for row in repaired:
        assert row["parity"] == "full" and int(row["repeats"]) == 5
        assert int(row["peak_allocated"]) > 0 and int(row["peak_reserved"]) > 0
        assert row["time_cohort"] == row["memory_cohort"]
        assert row["time_job"] == row["memory_job"]
        assert row["time_gpu_uuid"] == row["memory_gpu_uuid"]
        assert "H100" in row["gpu"] and row["torch"] == "2.11.0+cu128"
    repaired_pairs = [r for r in pairs if r["candidate"].endswith("repair")]
    assert len(repaired_pairs) == 26
    assert all(r["memory_status"] == "paired" for r in repaired_pairs)
    assert not summary["audit_errors"] and summary["memory_missing_points"] == 0
    assert summary["pending_role_metrics"] == 0
    # Original failed implementations remain failed; repaired roles are separate.
    assert len(points) == 489 and len(roles) == 532
    assert Counter(r["status"] for r in roles) == Counter({"measured": 489, "timeout": 15, "numerical-failure": 8, "unsupported": 20})
    protected = json.loads((BASE / "provenance/TAVOTTO-FINAL-EXPORTS.json").read_text())
    import hashlib
    for figure in protected["figures"]:
        for item in figure["files"]:
            assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    import tarfile
    gallery = ROOT.parents[1] / "figures/o2-hp-angular-operators-20261121-r1"
    original_files = 0
    with tarfile.open(BASE / "provenance/angular-figures-6f641cf.tar.gz") as packed:
        for member in packed.getmembers():
            if member.isfile() and Path(member.name).suffix.lower() in (".pdf", ".png", ".py", ".csv", ".ttf"):
                assert packed.extractfile(member).read() == (gallery / member.name).read_bytes(), member.name
                original_files += 1
    result = dict(passed=True, original_points_preserved=len(original), repaired_points=len(repaired),
                  repaired_same_gpu_pairs=len(repaired_pairs), all_original_input_weight_hashes_preserved=True,
                  all_new_points_five_repeats_and_same_gpu_job_allocated_reserved=True,
                  original_nine_exports_hash_unchanged=True, original_failed_roles_not_relabelled=True,
                  original_gallery_scientific_files_byte_identical=original_files,
                  role_status_counts=dict(Counter(r["status"] for r in roles)),
                  maximum_accepted_relative_l2=summary["max_accepted_relative_l2"])
    (ROOT / "provenance/FINAL-PAIRED-AUDIT.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
