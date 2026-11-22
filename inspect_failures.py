"""Receipt-only diagnosis on a CPU compute node; never changes frozen receipts."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "o2-hp-angular-operators-20261121-r1"


def main():
    with (BASE / "role-coverage.csv").open() as stream:
        failures = [r for r in csv.DictReader(stream) if r["status"] == "numerical-failure"]
    output = []
    for target in failures:
        candidates = []
        for path in sorted((BASE / "measurements").glob("**/qualify-*.json")):
            data = json.loads(path.read_text())
            if any(str(data.get(k)) != target[k] for k in ("family", "rows", "order", "parity")):
                continue
            records = []
            for key, value in data.get("numerics", {}).items():
                if not key.startswith(target["role"] + "/"):
                    continue
                for chunk in value.get("chunks", []):
                    for index, check in enumerate(chunk["checks"]):
                        if not check["passed"]:
                            records.append(dict(seed_scale=key, start=chunk["start"], component=index, **check))
            if records:
                candidates.append(dict(receipt=str(path.relative_to(BASE)), job=data.get("job"),
                    reference_chunk=data.get("numerics", {}).get(next(iter(data.get("numerics", {})), ""), {}).get("reference_chunk_size"),
                    failed_checks=records))
        item = dict(target=target, failed_receipts=candidates)
        output.append(item)
        if candidates:
            preferred = [c for c in candidates if "overnight-final" in c["receipt"]] or candidates
            c = preferred[-1]
            checks = c["failed_checks"]
            print(json.dumps(dict(target=target, receipt=c["receipt"], job=c["job"],
                failed_windows=len(checks), components=sorted({r["component"] for r in checks}),
                max_abs=max(r["max_abs"] for r in checks),
                max_relative_l2=max(r["relative_l2"] for r in checks), first=checks[:2])), flush=True)
        else:
            item["classification"] = "no-failed-FP32-check-found; inspect forward-only eligibility and build timeouts"
            print(json.dumps(dict(target=target, classification=item["classification"])), flush=True)
    (ROOT / "provenance/FAILURE-RECEIPTS.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
