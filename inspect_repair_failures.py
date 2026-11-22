"""Extract remaining FP32 failures from candidate qualification receipts on CPU."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
failures = []
for path in sorted((root / "measurements/repair-a5cbc8c/numerical-repair").glob("qualify-*.json")):
    data = json.loads(path.read_text())
    for key, value in data.get("numerics", {}).items():
        for chunk in value.get("chunks", []):
            for index, check in enumerate(chunk["checks"]):
                if not check["passed"]:
                    item = dict(order=data["order"], role=data["role"], seed_scale=key, window=chunk["start"],
                                reference_start=chunk["start"] // 4096 * 4096, component=index, **check)
                    failures.append(item)
                    print(json.dumps(item), flush=True)
(root / "provenance/REPAIR-REMAINING-FAILURES.json").write_text(json.dumps(failures, indent=2))
print("Remaining failed FP32 checks", len(failures), flush=True)
