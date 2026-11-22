"""CPU aggregation of FP32 diagnostic receipts, never a numerical gate."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for path in sorted((ROOT / "measurements").glob("**/radial-isolation-*.json")):
    data = json.loads(path.read_text())
    print(path.relative_to(ROOT), data["job"], data["canonical_weights_sha256"], flush=True)
    for index, rows in enumerate(data["end_to_end_checks"]):
        print("component", index, "passed", all(r["passed"] for r in rows),
              "max_abs", max(r["max_abs"] for r in rows),
              "max_relative_l2", max(r["relative_l2"] for r in rows), flush=True)
    for name, rows in data["isolation"].items():
        if name == "strides":
            print(name, rows, flush=True)
            continue
        print(name, "passed", all(r["passed"] for r in rows),
              "max_abs", max(r["max_abs"] for r in rows),
              "max_relative_l2", max(r["relative_l2"] for r in rows), flush=True)
