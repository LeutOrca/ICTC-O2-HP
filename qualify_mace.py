"""Qualify own-matrix execution through the latest native MACE E/F entry."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

ENTRY = Path(os.environ["HPX_MACE_ENTRY"])
sys.path.insert(0, str(ENTRY))
import qualify_force as prior
from matrix_kernels import OwnMatrixKernels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=tuple(prior.bench.CASES))
    parser.add_argument("--kernel", choices=("triton", "cuda", "cuda2", "cuda4", "cuda8", "split2", "split4", "split8", "warp4", "warp4tree", "warp4adjacent"), default="triton")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior.bench.configure()
    torch.use_deterministic_algorithms(True)
    if args.kernel == "cuda":
        from cuda_matrix import prepare
        prepare()
    model = prior.setup("receiver-csr-partition-graph")
    geometry = prior.bench.load_geometry(prior.bench.CASES[args.case])
    fixed = prior.inputs_for(geometry, model)
    for _ in range(3):
        prior.graph_evaluate(model, fixed)
    reference = prior.graph_evaluate(model, fixed)
    reference_repeat = prior.graph_evaluate(model, fixed)
    reference_stability = prior.bench.errors(reference_repeat, reference)
    if not all(item["passed"] for item in reference_stability.values()):
        raise RuntimeError(dict(native_repeat_failed=reference_stability))
    mode = OwnMatrixKernels(args.kernel)
    with mode:
        candidate = prior.graph_evaluate(model, fixed)
    torch.cuda.synchronize()
    numeric = prior.bench.errors(candidate, reference)
    receipt = dict(case=args.case, kernel=args.kernel,
                   checkpoint_sha256="d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a",
                   base="78231f3482fc9999857cc29e1cf5c220e73aa435",
                   gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   atoms=len(fixed["positions"]), edges=fixed["edge_index"].size(1),
                   matrix_calls=dict(mode.calls), numerics=numeric,
                   native_self_repeat=reference_stability,
                   passed=all(item["passed"] for item in numeric.values()))
    args.output.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    if not receipt["passed"]:
        raise RuntimeError("Own matrix kernels failed the unchanged whole-network FP32 gate")
    with OwnMatrixKernels(args.kernel):
        replay = prior.replay_for(model, fixed)
        actual = replay.call(fixed)
    capture = prior.bench.errors(actual, reference)
    receipt["capture"] = capture
    receipt["passed"] = all(item["passed"] for item in capture.values())
    args.output.write_text(json.dumps(receipt, indent=2))
    if not receipt["passed"]:
        raise RuntimeError("Whole-network CUDA Graph failed FP32")
    # This trace covers the candidate E/F, including its position adjoint.
    with OwnMatrixKernels(args.kernel):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as trace:
            replay.call(fixed)
            torch.cuda.synchronize()
    names = sorted({event.name for event in trace.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA})
    unexpected = [name for name in names
                  if any(token in name.lower() for token in ("cublas", "gemm", "cutlass"))]
    receipt["candidate_cuda_kernels"] = names
    receipt["unexpected_matrix_kernels"] = unexpected
    receipt["passed"] = not unexpected and receipt["passed"] and bool(mode.calls)
    trace.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
    args.output.write_text(json.dumps(receipt, indent=2))
    if not receipt["passed"]:
        raise RuntimeError("Whole-network candidate trace still contains a library GEMM")


if __name__ == "__main__":
    main()
