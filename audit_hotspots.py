"""Summarize kernel attribution from saved traces on a Slurm CPU node."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(path):
    events = json.loads(path.read_text())["traceEvents"]
    totals = defaultdict(lambda: [0, 0.])
    for event in events:
        if event.get("cat") == "kernel" and event.get("ph") == "X":
            row = totals[event["name"]]
            row[0] += 1
            row[1] += event["dur"]
    elapsed = sum(row[1] for row in totals.values())
    rows = [dict(kernel=name, calls=count, duration_us=duration,
                 fraction=duration / elapsed if elapsed else 0.)
            for name, (count, duration) in sorted(totals.items(), key=lambda item: -item[1][1])]
    return dict(path=str(path), summed_kernel_us=elapsed, kernels=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    result = [summarize(path) for path in args.traces]
    args.output.write_text(json.dumps(result, indent=2))
    for item in result:
        print(json.dumps(dict(path=item["path"], summed_kernel_us=item["summed_kernel_us"],
                              top=item["kernels"][:12])), flush=True)


if __name__ == "__main__":
    main()
