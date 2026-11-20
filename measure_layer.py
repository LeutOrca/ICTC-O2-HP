"""Fresh-process paired timing and memory for complete frozen SO2Linear."""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
from statistics import median

import torch

from frozen_so2 import FrozenWeightCache
from matrix_kernels import OwnMatrixKernels
from qualify_layer import official, tuple_output
from so2_layer import frozen_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("time", "memory", "trace"))
    parser.add_argument("role", choices=("native", "own"))
    parser.add_argument("mode", choices=("forward", "input_vjp"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--edges", type=int, default=160000)
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(20261120)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    module = official()(128, 128, 4, 2).cuda().requires_grad_(False)
    inputs = torch.randn(args.edges, 19, 128, device="cuda") * .1
    inputs.requires_grad_(args.mode == "input_vjp")
    cotangent = torch.randn_like(inputs)
    caches = tuple(FrozenWeightCache() for _ in range(2))

    def call():
        context = OwnMatrixKernels() if args.role == "own" else nullcontext()
        with context:
            value = (frozen_layer(inputs, module, caches, "triton")
                     if args.role == "own" else module(inputs))
            if args.mode == "input_vjp":
                return torch.autograd.grad(tuple_output(value), inputs, (cotangent,))[0]
            return value

    for _ in range(5):
        result = call()
        del result
    torch.cuda.synchronize()
    record = dict(role=args.role, mode=args.mode, repeat=args.repeat, edges=args.edges,
                  channels=128, lmax=4, mmax=2, cuda_graph=False, tf32=False, amp=False,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name())
    if args.stage == "time":
        events, walls = [], []
        from time import perf_counter
        for _ in range(15):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = perf_counter()
            start.record()
            result = call()
            end.record()
            end.synchronize()
            walls.append((perf_counter() - wall_start) * 1000)
            events.append(start.elapsed_time(end))
            del result
        record.update(event_ms=median(events), wall_ms=median(walls),
                      event_samples_ms=events, wall_samples_ms=walls)
    elif args.stage == "memory":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        resident = torch.cuda.memory_allocated()
        result = call()
        torch.cuda.synchronize()
        record.update(resident=resident, peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved())
        del result
    else:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as trace:
            result = call()
            torch.cuda.synchronize()
        trace.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
        kernels = sorted({event.name for event in trace.events()
                          if event.device_type == torch.autograd.DeviceType.CUDA})
        unexpected = [name for name in kernels
                      if any(word in name.lower() for word in ("cublas", "gemm", "cutlass"))]
        record.update(cuda_kernels=kernels, unexpected_library_gemm=unexpected)
        if args.role == "own" and unexpected:
            raise RuntimeError(record)
        del result
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
