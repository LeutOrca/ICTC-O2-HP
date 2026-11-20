"""GPU qualification for own FP32 Triton and CUDA matrix kernels."""
import argparse
import json
from pathlib import Path
from statistics import median

import torch

from matrix_kernels import OwnMatrixKernels, matrix


def compare(actual, expected):
    difference = actual.detach().double() - expected.detach().double()
    limit = 1e-5 + 1e-4 * expected.detach().double().abs()
    return dict(
        passed=bool(torch.isfinite(actual).all() and (difference.abs() <= limit).all()),
        max_abs=float(difference.abs().max()) if difference.numel() else 0.,
        relative_l2=float(difference.norm() / expected.double().norm().clamp_min(1e-30)),
    )


def time_ms(call):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(8):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        value = call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del value
    return median(samples)


def qualify(backend, output):
    torch.manual_seed(20261120)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if backend == "cuda":
        from cuda_matrix import prepare
        prepare()
    shapes = [(0, 7, 9), (11, 0, 13), (1, 128, 128),
              (37, 65, 93), (68, 128, 256), (272, 128, 512),
              (3472, 256, 128), (4096, 384, 256), (16384, 128, 128)]
    records = []
    for rows, inner, columns in shapes:
        for transposed in (False, True):
            a = torch.randn((inner, rows) if transposed else (rows, inner),
                            device="cuda", dtype=torch.float32) * .1
            if transposed:
                a = a.T
            b = torch.randn((columns, inner), device="cuda", dtype=torch.float32).T * .1
            bias = torch.randn(columns, device="cuda", dtype=torch.float32)
            expected = a.double().cpu() @ b.double().cpu() + bias.double().cpu()
            actual = matrix(a, b, bias=bias, backend=backend)
            result = compare(actual.cpu(), expected)
            row = dict(shape=(rows, inner, columns), transposed=transposed, **result)
            if rows and inner:
                row["own_event_ms"] = time_ms(lambda: matrix(a, b, bias=bias, backend=backend))
                row["native_event_ms"] = time_ms(lambda: torch.addmm(bias, a, b))
            records.append(row)
            output.write_text(json.dumps(dict(backend=backend, records=records), indent=2))
            if not result["passed"]:
                raise RuntimeError(row)
    a = torch.randn(3, 13, 17, device="cuda", requires_grad=True)
    b = torch.randn(3, 19, 17, device="cuda").transpose(1, 2).requires_grad_(True)
    reference = torch.bmm(a, b)
    cotangent = torch.randn_like(reference)
    ref_grad = torch.autograd.grad(reference, (a, b), cotangent)
    mode = OwnMatrixKernels(backend)
    with mode:
        actual = torch.bmm(a, b)
        gradients = torch.autograd.grad(actual, (a, b), cotangent)
    derivatives = [compare(actual, reference)] + [
        compare(got, ref) for got, ref in zip(gradients, ref_grad)
    ]
    records.append(dict(kind="batched_input_adjoint", results=derivatives, calls=dict(mode.calls)))
    if not all(item["passed"] for item in derivatives):
        raise RuntimeError(records[-1])
    invalid_bias = torch.full((13, 19), float("nan"), device="cuda")
    left, right = a.detach()[0], b.detach()[0]
    actual = matrix(left, right, bias=invalid_bias, beta=0, backend=backend)
    zero_beta = compare(actual, left @ right)
    if not zero_beta["passed"]:
        raise RuntimeError(zero_beta)
    # Capture is checked after every used signature has been compiled.
    with OwnMatrixKernels(backend):
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                torch.mm(left, right)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph, stream=stream):
            captured = torch.mm(left, right)
    left.add_(.01)
    graph.replay()
    capture = compare(captured, left @ right)
    if not capture["passed"]:
        raise RuntimeError(capture)
    with OwnMatrixKernels(backend):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA,
                                               torch.profiler.ProfilerActivity.CPU]) as profile:
            torch.mm(left, right)
            torch.cuda.synchronize()
    cuda_kernels = [event.name for event in profile.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA]
    if not cuda_kernels or any("cublas" in name.lower() or "gemm" in name.lower()
                               for name in cuda_kernels):
        raise RuntimeError(dict(unexpected_kernels=cuda_kernels))
    report = dict(backend=backend, device=torch.cuda.get_device_name(),
                  torch=torch.__version__, tf32=False, amp=False,
                  records=records, zero_beta=zero_beta, cuda_graph=capture,
                  candidate_cuda_kernels=cuda_kernels, passed=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("triton", "cuda"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    qualify(args.backend, args.output)
