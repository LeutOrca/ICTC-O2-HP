"""Fresh-process whole EquiformerV3 E/F timing, memory and kernel traces."""
import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import torch

sys.path.insert(0, os.environ["HPX_V3_ENTRY"])
import whole_network as native
from matrix_kernels import OwnMatrixKernels
from v3_native import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("time", "memory", "trace"))
    parser.add_argument("role", choices=("native", "own"))
    parser.add_argument("suite", choices=("mptrj-b1", "mptrj-b8"))
    parser.add_argument("boundary", choices=("fixed", "e2e"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--registry", action="store_true")
    args = parser.parse_args()
    qualification = json.loads(args.qualification.read_text())
    if (qualification["suite"] != args.suite or len(qualification["records"]) != 4
            or bool(qualification.get("registry", False)) != args.registry
            or not qualification["passed"] or not all(
                metric["passed"] for row in qualification["records"]
                for metric in row["numerics"].values())):
        raise RuntimeError("The inherited whole-network E/F qualification did not pass")
    native.strict(torch)
    torch.set_num_threads(4)
    net = native.Network("native", "cuda")
    net.freeze_ema()
    net.model.requires_grad_(False)
    raw = net.batch(net.manifest["suites"][args.suite])
    fixed, topology = net.topology(raw)
    installed = {}
    if args.role == "own":
        from optimizations import install as install_previous
        install_previous("gather-radial", net.model)
        installed = install(net.model, qualification.get("kernel", "triton"),
                            use_norm=bool(qualification["installed"]["norm"]),
                            split_complex=qualification.get("split_complex", False),
                            start_block=qualification.get("start_block", 0))
        if args.registry:
            from kernel_registry import install as register
            registered_calls = register()
    batch = fixed if args.boundary == "fixed" else raw
    mode = OwnMatrixKernels()

    def call():
        torch.manual_seed(native.SEEDS[0])
        with mode if (args.role == "own" and not args.registry
                      and not qualification.get("local_only", False)) else nullcontext():
            return net.forward(batch, args.boundary == "fixed")

    for _ in range(5):
        value = call()
        del value
    torch.cuda.synchronize()
    record = dict(stage=args.stage, role=args.role, suite=args.suite,
                  boundary=args.boundary, repeat=args.repeat, cuda_graph=False,
                  tf32=False, amp=False, dtype="torch.float32", seed=native.SEEDS[0],
                  registry=args.registry,
                  local_only=qualification.get("local_only", False),
                  checkpoint_sha256=native.CHECKPOINT_SHA,
                  qualification_sha256=native.sha(args.qualification),
                  qualification_path=str(args.qualification),
                  source=Path("provenance/commit.txt").read_text().strip(),
                  base="9787a3a6ca7fe67eeda3f21ec7c24d6634cbac03",
                  atoms=int(raw.natoms.sum()), structures=len(raw.natoms),
                  edges=fixed.edge_index.size(1), topology_sha256=topology,
                  input_positions_sha256=native.tensor_sha(raw.pos),
                  input_manifest_sha256=native.sha(native.PROV / "geometry.json"),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), installed=installed)
    if args.stage == "time":
        walls, events = [], []
        for _ in range(15):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            tick = perf_counter()
            start.record()
            value = call()
            end.record()
            end.synchronize()
            walls.append((perf_counter() - tick) * 1000)
            events.append(start.elapsed_time(end))
            del value
        record.update(wall_ms=median(walls), event_ms=median(events),
                      wall_samples_ms=walls, event_samples_ms=events,
                      atoms_s=record["atoms"] * 1000 / median(walls),
                      structures_s=record["structures"] * 1000 / median(walls))
    elif args.stage == "memory":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        record["resident_allocated"] = torch.cuda.memory_allocated()
        record["resident_reserved"] = torch.cuda.memory_reserved()
        value = call()
        torch.cuda.synchronize()
        record.update(peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved())
        del value
    else:
        with native.annotations(net), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True,
        ) as trace:
            value = call()
            torch.cuda.synchronize()
        trace.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
        kernels = sorted({event.name for event in trace.events()
                          if event.device_type == torch.autograd.DeviceType.CUDA})
        forbidden = [name for name in kernels if any(
            token in name.lower() for token in ("cublas", "gemm", "gemv", "cutlass"))]
        hot = sorted(trace.key_averages(), key=lambda event: event.self_device_time_total,
                     reverse=True)
        record.update(cuda_kernels=kernels, unexpected_library_matrix=forbidden,
                      matrix_calls=dict(registered_calls if args.role == "own" and args.registry else mode.calls),
                      top_device_operators=[dict(name=event.key, us=event.self_device_time_total,
                                                 count=event.count) for event in hot[:40]])
        if args.role == "own" and forbidden and not qualification.get("local_only", False):
            raise RuntimeError(dict(unexpected_library_matrix=forbidden))
        del value
    native.strict(torch)
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
