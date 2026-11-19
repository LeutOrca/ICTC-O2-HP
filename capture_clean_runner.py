"""Run one GPU process per stage to amortize imports on shared storage."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import traceback
import time

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=("qualify", "trace", "time", "memory"))
parser.add_argument("--trace-limit", type=int, default=1)
parser.add_argument("--qualification", type=Path)
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument("--wait-for", type=Path)
parser.add_argument("--signal", type=Path, required=True)
args = parser.parse_args()

if args.stage in ("time", "memory"):
    import conservative_runner as bench
else:
    import paired_runner as bench

BASE = ("native", "native-graph", "hp-uvu-graph", "hp-uvu-species-graph")
CASES = ("b1",)
if args.stage != "qualify" and args.qualification is None:
    parser.error("--qualification is required")

base_model = bench.model_for


def profiled(model):
    active = defaultdict(list)

    def before(label):
        def enter(module, inputs):
            region = torch.profiler.record_function(label)
            region.__enter__()
            active[label].append(region)
        return enter

    def after(label):
        def leave(module, inputs, output):
            active[label].pop().__exit__(None, None, None)
        return leave

    for name, module in model.named_modules():
        fields = name.split(".")
        if name and (len(fields) == 1
                     or fields[0] == "interactions" and len(fields) in (2, 3)
                     or fields[0] == "products" and len(fields) == 2):
            module.register_forward_pre_hook(before(f"MACE::{name}"))
            module.register_forward_hook(after(f"MACE::{name}"), always_call=True)
    return model


def model_for(backend, device):
    if backend == "hp-local-o2":
        from local_o2_compat import configure_local_o2
        model = configure_local_o2(base_model("native", device))
    else:
        model = base_model(backend, device)
    return profiled(model) if args.stage == "trace" else model


bench.model_for = model_for
from trace_optimizations import install as install_hotspots
from capture_conservative import install
install_hotspots(bench, profile=args.stage == "trace")
install(bench, validate_live_input=args.stage == "qualify")


def execute(backend, point, boundary="fixed", round_number=0):
    print(f"start {args.stage} {backend} {point} {boundary} {round_number}", flush=True)
    if args.stage == "memory" and torch.cuda.memory_allocated() > 1024 ** 2:
        raise RuntimeError("GPU allocations carried over from the preceding memory arm")
    try:
        bench.run(argparse.Namespace(
            device="cuda", case=point, backend=backend, stage=args.stage,
            boundary=boundary, warmup=6, repeats=15, round=round_number,
        ))
    except Exception as exc:
        traceback.print_exc()
        with (bench.ROOT / "errors.tsv").open("a") as log:
            log.write(f"{args.stage}\t{backend}\t{point}\t{boundary}\t{round_number}\t{type(exc).__name__}\n")
    finally:
        gc.collect()
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()


def accepted(root, backend, case):
    receipt = root / "receipts" / f"cuda-{backend}-{case}-fixed-qualify-0.json"
    return receipt.is_file() and json.loads(receipt.read_text())["passed"]


if args.wait_for:
    while not args.wait_for.is_file():
        if args.wait_for.with_suffix(".failed").exists():
            raise SystemExit(f"Previous stage failed: {args.wait_for.stem}")
        time.sleep(0.25)

if args.stage == "qualify":
    for case in CASES:
        for backend in BASE:
            if backend != "native" and not accepted(bench.ROOT, "native", case):
                continue
            execute(backend, case)
    if not any(accepted(bench.ROOT, "native", case) for case in CASES):
        raise SystemExit("No qualified native point")
elif args.stage == "trace":
    for backend in BASE:
        selected = [case for case in CASES
                    if accepted(args.qualification, backend, case)]
        for case in selected[:args.trace_limit]:
            for boundary in ("fixed", "e2e"):
                execute(backend, case, boundary)
    if not any((bench.ROOT / "traces").glob("native-*.json")):
        raise SystemExit("No native trace")
else:
    for repeat in range(1, args.rounds + 1):
        backends = BASE if repeat % 2 else BASE[::-1]
        for case in CASES:
            if not accepted(args.qualification, "native", case):
                continue
            if not any(accepted(args.qualification, backend, case) for backend in BASE[1:]):
                continue
            for boundary in ("fixed", "e2e"):
                for backend in backends:
                    if accepted(args.qualification, backend, case):
                        execute(backend, case, boundary, repeat)

args.signal.touch()
