"""Validate process-local own CUDA registration, gradients and live Graph input."""
import argparse
import json
from pathlib import Path

import torch

from kernel_registry import install
from qualify_matrix import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(20261120)
    torch.backends.cuda.matmul.allow_tf32 = False
    left = torch.randn(37, 65, device="cuda", requires_grad=True)
    right = torch.randn(65, 93, device="cuda", requires_grad=True)
    incoming = torch.randn(37, 93, device="cuda")
    expected = left.double().cpu() @ right.double().cpu()
    ref_grad = torch.autograd.grad(expected, (left, right), incoming.double().cpu())
    calls = install()
    actual = torch.mm(left, right)
    grad = torch.autograd.grad(actual, (left, right), incoming)
    records = [compare(actual, expected.to("cuda")),
               *[compare(a, b) for a, b in zip(grad, ref_grad)]]
    out = torch.empty_like(actual)
    torch.mm(left.detach(), right.detach(), out=out)
    records.append(compare(out, expected.to("cuda")))
    for _ in range(3):
        torch.mm(left, right)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = torch.mm(left, right)
    with torch.no_grad():
        left.add_(.01)
    graph.replay()
    torch.cuda.synchronize()
    expected_update = left.detach().double().cpu() @ right.detach().double().cpu()
    records.append(compare(captured, expected_update.to("cuda")))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as trace:
        torch.mm(left, right)
        torch.cuda.synchronize()
    kernels = sorted({event.name for event in trace.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA})
    forbidden = [name for name in kernels if any(
        token in name.lower() for token in ("gemm", "gemv", "cublas", "cutlass"))]
    receipt = dict(passed=all(row["passed"] for row in records) and not forbidden,
                   records=records, calls=dict(calls), kernels=kernels, forbidden=forbidden)
    args.output.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    if not receipt["passed"]:
        raise RuntimeError(receipt)


if __name__ == "__main__":
    main()
