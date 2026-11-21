"""Summarize measured trace attribution on a scheduler CPU node."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("suite")
    args = parser.parse_args()
    rows = {}
    for boundary in ("fixed", "e2e"):
        rows[boundary] = {}
        for role in ("native", "base", "own"):
            path = args.directory / f"v3-trace-{args.suite}-{boundary}-{role}-0.json"
            report = json.loads(path.read_text())
            rows[boundary][role] = dict(source=report["source"], gpu_uuid=report["gpu_uuid"],
                top_device_operators=report["top_device_operators"][:12])
    output = dict(suite=args.suite, boundaries=rows,
                  note="Diagnostic profile attribution only; timing conclusions use the five paired fresh-process runs")
    (args.directory / "TRACE-SUMMARY.json").write_text(json.dumps(output, indent=2))
    for boundary, roles in rows.items():
        for role, row in roles.items():
            print(json.dumps(dict(boundary=boundary, role=role,
                                  top=row["top_device_operators"][:6])), flush=True)


if __name__ == "__main__":
    main()
