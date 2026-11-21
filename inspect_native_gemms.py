"""Map saved native forward/reverse matrix shapes to actual CUDA kernels on CPU."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    events = json.loads(args.trace.read_text())["traceEvents"]
    operators = {event.get("args", {}).get("External id"): event for event in events
                 if event.get("cat") == "cpu_op" and event.get("name") in ("aten::mm", "aten::bmm", "aten::addmm")}
    rows = {}
    for event in events:
        if event.get("cat") != "kernel":
            continue
        parent = operators.get(event.get("args", {}).get("External id"))
        if parent is None:
            continue
        shapes = parent["args"].get("Input Dims", parent["args"].get("Input Shapes"))
        key = json.dumps((parent["name"], shapes, event["name"]))
        row = rows.setdefault(key, dict(operation=parent["name"], shapes=shapes, kernel=event["name"], calls=0))
        row["calls"] += 1
    args.output.write_text(json.dumps(list(rows.values()), indent=2))
    for row in rows.values():
        shapes = row["shapes"]
        if row["operation"] == "aten::bmm" and shapes and len(shapes[0]) == 3 and shapes[0][1] == 2:
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
