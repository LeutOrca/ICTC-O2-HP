"""Audit collected terminal results and the frozen night payload on CPU Slurm."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    summary = json.loads((ROOT / "SUMMARY.json").read_text())
    data, roles, ratios = map(read_csv, ("summary.csv", "role-coverage.csv", "speedups.csv"))
    assert not summary["audit_errors"] and summary["pending_role_metrics"] == 0
    assert len(data) == summary["summary_points"]
    assert all(int(r["repeats"]) == 5 and r["dtype"] == "float32" for r in data)
    assert all(r[k] == "False" for r in data for k in ("tf32", "amp", "cuda_graph"))
    assert all(int(r["peak_allocated"]) > 0 and int(r["peak_reserved"]) > 0 for r in data)
    assert all(r["memory_status"] == "paired" for r in ratios)
    counts = dict(Counter(r["status"] for r in roles))
    assert sum(counts.values()) == summary["expected_role_metrics"]
    failures = [r for r in roles if r["status"] not in ("measured", "unsupported")]
    jobs = []
    angular_jobs = []
    wanted = {"317369", "317370", "317398", "317399", "317400", "317401", "317429", "317430", "317431", "317432"}
    for line in (ROOT / "provenance/project-gbu2-accounting-extended.tsv").read_text().splitlines():
        fields = line.split("|", 9)
        if len(fields) == 10 and "o2-hp-angular-operators-20261121-r1" in fields[4]:
            record = dict(job=fields[0], state=fields[1], seconds=int(fields[2]), node=fields[5])
            angular_jobs.append(record)
            if fields[0] in wanted:
                jobs.append(record)
    assert {r["job"] for r in jobs} == wanted
    assert all(r["state"].split()[0] in ("COMPLETED", "TIMEOUT", "FAILED", "CANCELLED", "OUT_OF_MEMORY") for r in angular_jobs)
    payload = ROOT / "night.tgz"
    assert sha(payload) == "97817444301772bcb83c7fdb01724db4e05d38abc82dfac26c5a3b7d21119c87"
    with tarfile.open(payload) as packed, tarfile.open(ROOT / "provenance/git-harness/d005f68.tar.gz") as git:
        originals = {m.name: git.extractfile(m).read() for m in git.getmembers() if m.isfile()}
        harness = []
        for name in ("models.py", "benchmark.py", "hpx_adapter.py", "cue_adapter.py", "runtime.sh",
                     "verify_sources.py", "fork_prelude.py", "fork_runner.py", "phase_campaign.py",
                     "night_fill.py", "night.sbatch", "CONTRACT.md"):
            raw = packed.extractfile(name).read()
            assert raw == originals[name], name
            harness.append(name)
    checked = []
    for path in sorted((ROOT / "measurements/overnight-final-d005f68/source-seals").glob("*.json")):
        seal = json.loads(path.read_text())
        for name, digest in seal["all_source_hashes"].items():
            assert sha(ROOT / "upstream" / name) == digest, name
        checked.append(dict(receipt=str(path.relative_to(ROOT)), files=len(seal["all_source_hashes"]), sha256=sha(path)))
    assert len(checked) == 4
    result = dict(passed=True, accepted_points=len(data), role_status_counts=counts,
                  terminal_failures=failures, jobs=jobs, all_angular_jobs_terminal=True,
                  all_accepted_points_have_five_repeats_and_memory=True,
                  all_ratios_have_same_gpu_memory_pairs=True,
                  max_accepted_relative_l2=summary["max_accepted_relative_l2"],
                  night_payload_sha256=sha(payload), night_harness_git_exact=harness,
                  night_gpu_upstream_source_seals=checked,
                  provenance_limit="Night harness verified from frozen transferred payload; fresh remote byte re-download unavailable at collection time.")
    (ROOT / "provenance/MORNING-AUDIT.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("night_gpu_upstream_source_seals", "night_harness_git_exact")}), flush=True)
    chosen = [r for r in ratios if (r["candidate"], r["baseline"]) in (("local-hpx", "public"), ("so2-triton", "so2-native"), ("public", "eqx-o2"))]
    groups = {}
    for r in chosen:
        key = "/".join(r[k] for k in ("family", "rows", "parity", "metric", "candidate", "baseline"))
        groups.setdefault(key, []).append(r)
    for key, values in sorted(groups.items()):
        print(key, "orders=" + ",".join(r["order"] for r in values),
              "speedup=" + f"{min(float(r['speedup']) for r in values):.4f}..{max(float(r['speedup']) for r in values):.4f}", flush=True)


if __name__ == "__main__":
    main()
