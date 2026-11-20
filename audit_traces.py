"""Summarize H100 Chrome traces by launch region and CUDA kernel."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


def group_regions(events):
    result = defaultdict(list)
    for event in events:
        name = event.get("name", "")
        if event.get("ph") == "X" and (
            name.startswith("MACE::") or name.startswith("autograd::engine")
            or name in ("mace_native_neighbors_and_input_transfer", "mace_pinned_graph_input_transfer")
        ):
            result[event.get("tid")].append(
                (event["ts"], event["ts"] + event["dur"], name)
            )
    return result


def origin(event, regions):
    timestamp = event["ts"]
    matches = [
        (end - start, name) for start, end, name in regions[event.get("tid")]
        if start <= timestamp <= end
    ]
    return min(matches)[1] if matches else "Other"


def collect(path):
    events = json.loads(path.read_text())["traceEvents"]
    regions = group_regions(events)
    runtime = {
        event.get("args", {}).get("correlation"): event
        for event in events
        if event.get("cat") in ("cuda_runtime", "cuda_driver")
        and event.get("args", {}).get("correlation") is not None
    }
    op_by_id = {
        event.get("args", {}).get("External id"): event
        for event in events
        if event.get("cat") == "cpu_op"
        and event.get("args", {}).get("External id") is not None
    }
    kernels = Counter()
    phases = Counter()
    operators = Counter()
    neighbor_us = 0.0
    pinned_us = 0.0
    memory = Counter()
    peak = 0
    for event in events:
        name = event.get("name", "")
        args = event.get("args", {})
        if name == "mace_native_neighbors_and_input_transfer":
            neighbor_us += event.get("dur", 0)
        if name == "mace_pinned_graph_input_transfer":
            pinned_us += event.get("dur", 0)
        if event.get("cat") == "kernel":
            time = event["dur"]
            kernels[name] += time
            producer = runtime.get(args.get("correlation"))
            if producer is None:
                producer = op_by_id.get(args.get("External id"))
            if producer is not None:
                phases[origin(producer, regions)] += time
            op_event = op_by_id.get(args.get("External id"))
            if op_event is not None:
                operators[op_event["name"]] += time
        if name == "[memory]" and args.get("Device Type") == 1:
            size = args.get("Bytes", 0)
            if size > 0:
                memory[origin(event, regions)] += size
            peak = max(peak, args.get("Total Allocated", 0))
    return {
        "trace": str(path),
        "gpu_kernel_total_us": sum(kernels.values()),
        "top_kernels_us": kernels.most_common(25),
        "top_host_attributed_cuda_operators_us": operators.most_common(25),
        "top_launch_regions_us": phases.most_common(30),
        "native_neighbors_and_transfer_host_us": neighbor_us,
        "pinned_input_transfer_host_us": pinned_us,
        "allocation_bytes_by_region": memory.most_common(20),
        "trace_peak_allocated": peak,
    }


root = Path(sys.argv[1])
traces = [collect(path) for path in sorted((root / "traces").glob("*.json"))]
if not traces:
    raise ValueError(f"No traces in {root}")
(root / "TRACE_AUDIT.json").write_text(json.dumps(traces, indent=2) + "\n")
for trace in traces:
    print(trace["trace"], trace["gpu_kernel_total_us"], trace["native_neighbors_and_transfer_host_us"])
    print(trace["top_host_attributed_cuda_operators_us"][:5])
    print(trace["top_launch_regions_us"][:5])
