"""Full measured-shape numerical qualification using kernels from the job cwd."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
import torch

from frozen_so2 import FrozenWeightCache
from matrix_kernels import OwnMatrixKernels
from qualify_layer import official
from so2_layer import frozen_layer


def fingerprint(tensor):
    digest = hashlib.sha256()
    for block in tensor.detach().split(4096):
        digest.update(block.contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def strict_compare(actual, reference):
    square_error, square_ref, maximum, violations = 0., 0., 0., 0
    for value, expected in zip(actual.detach().split(4096), reference.detach().split(4096)):
        value, expected = value.cpu().double(), expected.cpu().double()
        if not torch.isfinite(value).all():
            raise RuntimeError("Nonfinite candidate output")
        error = value - expected
        square_error += float(error.square().sum())
        square_ref += float(expected.square().sum())
        maximum = max(maximum, float(error.abs().max()))
        violations += int((error.abs() > 2e-5 + 2e-5 * expected.abs()).sum())
    relative = (square_error / max(square_ref, 1e-60)) ** .5
    return dict(max_abs=maximum, relative_l2=relative, violations=violations,
                passed=violations == 0 and relative <= 1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(20261120)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    module = official()(128, 128, 4, 2).cuda().requires_grad_(False)
    inputs = (torch.randn(160000, 19, 128, device="cuda") * .1).requires_grad_()
    cotangent = torch.randn_like(inputs)
    reference = module(inputs)
    expected, = torch.autograd.grad(reference, inputs, cotangent)
    caches = tuple(FrozenWeightCache() for _ in range(2))
    with OwnMatrixKernels():
        actual = frozen_layer(inputs, module, caches, "triton")
        gradient, = torch.autograd.grad(actual, inputs, cotangent)
    errors = dict(forward=strict_compare(actual, reference),
                  input_vjp=strict_compare(gradient, expected))
    weights = hashlib.sha256()
    for parameter in module.parameters():
        weights.update(parameter.detach().cpu().numpy().tobytes())
    receipt = dict(errors=errors, passed=all(row["passed"] for row in errors.values()),
                   source=Path("provenance/commit.txt").read_text().strip(),
                   driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   input_sha256=fingerprint(inputs), cotangent_sha256=fingerprint(cotangent),
                   weights_sha256=weights.hexdigest(), edges=160000, channels=128, lmax=4, mmax=2,
                   atol=2e-5, rtol=2e-5, relative_l2=1e-5, tf32=False, amp=False, cuda_graph=False,
                   gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
    args.output.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    if not receipt["passed"]:
        raise RuntimeError(receipt)


if __name__ == "__main__":
    main()
