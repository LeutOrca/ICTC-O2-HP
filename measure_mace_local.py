"""Fresh-process same-Graph native/base/prefix whole MACE measurements."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import torch

sys.path.insert(0, os.environ["HPX_MACE_ENTRY"])
import qualify_force as prior
from mace_frozen_prefix import install, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("time", "memory", "trace"))
    parser.add_argument("role", choices=("native", "base", "prefix"))
    parser.add_argument("case", choices=tuple(prior.bench.CASES))
    parser.add_argument("boundary", choices=("fixed", "e2e"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()
    qualification = json.loads(args.qualification.read_text())
    if not qualification["passed"] or not qualification.get("cuda_graph") or qualification["case"] != args.case:
        raise RuntimeError("The matching candidate whole-network E/F and Graph qualification is missing")
    prior.bench.configure()
    torch.use_deterministic_algorithms(True)
    model = prior.setup("native-graph" if args.role == "native" else "receiver-csr-partition-graph")
    if args.role == "prefix":
        install(model)
    geometry = prior.bench.load_geometry(prior.bench.CASES[args.case])
    fixed = prior.inputs_for(geometry, model)
    digest = hashlib.sha256()
    for name, value in sorted(fixed.items()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    record = dict(stage=args.stage, role=args.role, case=args.case, boundary=args.boundary,
                  repeat=args.repeat, source=Path("provenance/commit.txt").read_text().strip(),
                  base="78231f3482fc9999857cc29e1cf5c220e73aa435",
                  checkpoint_sha256="d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a",
                  qualification_sha256=hashlib.sha256(args.qualification.read_bytes()).hexdigest(),
                  input_sha256=digest.hexdigest(), atoms=len(fixed["positions"]),
                  edges=fixed["edge_index"].size(1), structures=len(geometry),
                  tf32=False, amp=False, cuda_graph=True, dtype="torch.float32",
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
    if args.stage == "memory":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        record["resident_allocated"] = torch.cuda.memory_allocated()
    tick = perf_counter()
    runner = prior.replay_for(model, fixed)
    torch.cuda.synchronize()
    record["capture_prepare_wall_ms"] = (perf_counter() - tick) * 1000

    def call():
        data = fixed if args.boundary == "fixed" else prior.inputs_for(geometry, model)
        return runner.call(data)

    if args.stage == "time":
        for _ in range(5):
            output = call()
            del output
        walls = []
        for _ in range(15):
            torch.cuda.synchronize()
            tick = perf_counter()
            output = call()
            torch.cuda.synchronize()
            walls.append((perf_counter() - tick) * 1000)
            del output
        record.update(wall_ms=median(walls), samples_ms=walls,
                      atoms_s=record["atoms"] * 1000 / median(walls),
                      structures_s=record["structures"] * 1000 / median(walls))
    elif args.stage == "memory":
        output = call()
        torch.cuda.synchronize()
        record.update(peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved(), includes_capture=True)
        del output
    else:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as trace:
            output = call()
            torch.cuda.synchronize()
        trace.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
        record["cuda_kernels"] = sorted({event.name for event in trace.events()
                                          if event.device_type == torch.autograd.DeviceType.CUDA})
        del output
    if args.role == "prefix":
        record["prefix_cache"] = receipt(model)
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
