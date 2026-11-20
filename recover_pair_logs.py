"""Recover original per-arm JSON records from immutable scheduler stdout logs."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    results = args.destination / "results"
    results.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, line in enumerate(args.log.read_text().splitlines(), 1):
        if not line.startswith('{"stage":'):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("stage") not in ("time", "memory", "trace"):
            continue
        name = (f"mace-{record['stage']}-{record['case']}-{record['boundary']}-"
                f"{record['role']}-{record['repeat']}.json")
        path = results / name
        if path.exists():
            raise RuntimeError(f"Duplicate record in scheduler log: {name}")
        path.write_text(json.dumps(record, indent=2))
        rows.append(dict(line=number, record=name,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    provenance = dict(source_log=str(args.log.resolve()),
                      source_log_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(),
                      recovered_from_original_json_lines=True, records=rows)
    (args.destination / "LOG_RECOVERY.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(dict(recovered=len(rows), destination=str(args.destination))), flush=True)


if __name__ == "__main__":
    main()
