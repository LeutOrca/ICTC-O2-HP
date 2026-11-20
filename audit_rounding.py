"""Summarize native matrix algorithms and own rounding on a CPU compute node."""
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
report = json.loads(path.read_text())
trace = json.loads(path.with_suffix(".trace.json").read_text())["traceEvents"]
operators = [event for event in trace if event.get("cat") == "cpu_op"
             and event["name"] in ("aten::mm", "aten::bmm")]
operators.sort(key=lambda event: event["ts"])
if len(operators) != len(report["records"]):
    raise RuntimeError((len(operators), len(report["records"])))
for row, event in zip(report["records"], operators):
    expected_shapes = [row["left_shape"], row["right_shape"]]
    if event["args"].get("Input Dims") != expected_shapes:
        raise RuntimeError(dict(index=row["index"], expected=expected_shapes,
                                actual=event["args"].get("Input Dims")))
    identity = event["args"]["External id"]
    row["native_kernels"] = sorted({kernel["name"] for kernel in trace
                                    if kernel.get("cat") == "kernel"
                                    and kernel.get("args", {}).get("External id") == identity})
    best = min(row["variants"], key=lambda backend: row["variants"][backend]["relative_l2"])
    row["closest_backend"] = best
    if row["variants"]["triton"]["relative_l2"] != 0:
        print(json.dumps(dict(index=row["index"], left=row["left_shape"], right=row["right_shape"],
                              kernels=row["native_kernels"], closest=best,
                              errors={name: value["relative_l2"]
                                      for name, value in row["variants"].items()})), flush=True)
path.with_name("MACE_ROUNDING_AUDIT.json").write_text(json.dumps(report, indent=2))
