"""Attribute actual CUDA kernel events, without double-counting CPU annotations."""
from collections import defaultdict
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
report = {}
for path in root.glob("v3*trace*own.trace.json"):
    events = json.loads(path.read_text())["traceEvents"]
    total = 0.
    kernels = defaultdict(lambda: dict(us=0., count=0))
    for event in events:
        if event.get("cat") != "kernel":
            continue
        value = kernels[event["name"]]
        value["us"] += event["dur"]
        value["count"] += 1
        total += event["dur"]
    rows = sorted((dict(name=name, **value, fraction=value["us"] / total)
                   for name, value in kernels.items()), key=lambda row: row["us"], reverse=True)
    report[path.name] = dict(kernel_us=total, top=rows[:15])
(root / "CUDA_KERNEL_AUDIT.json").write_text(json.dumps(report, indent=2))
for name, row in report.items():
    print(name, json.dumps(row["top"][:4]), flush=True)
