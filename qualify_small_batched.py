"""Validate coalesced small batched FP32 GEMM against independent FP64."""
import argparse
import json
from pathlib import Path

import torch

from matrix_kernels import OwnMatrixKernels, matrix
from qualify_matrix import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(20261120)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    for batches, rows, inner, columns in ((0, 3, 3, 3), (1, 3, 3, 3), (1272, 3, 3, 3),
                                          (1272, 5, 5, 5), (3472, 9, 9, 9),
                                          (8704, 1, 16, 16), (37, 7, 5, 9), (37, 5, 0, 7)):
        left = torch.randn(batches, inner, rows, device="cuda").transpose(-2, -1).requires_grad_()
        right = torch.randn(batches, columns, inner, device="cuda").transpose(-2, -1).requires_grad_()
        expected = left.double().bmm(right.double())
        actual = matrix(left, right)
        numeric = compare(actual, expected)
        view = torch.empty(batches, rows, columns + 3, device="cuda")[:, :, 1:columns + 1]
        matrix(left, right, output=view)
        strided = compare(view, expected)
        incoming = torch.randn_like(actual)
        with OwnMatrixKernels():
            forward = torch.bmm(left, right)
            grad = torch.autograd.grad(forward, (left, right), incoming)
        ref = torch.autograd.grad(expected, (left, right), incoming.double())
        gradients = [compare(a, b) for a, b in zip(grad, ref)]
        row = dict(shape=[batches, rows, inner, columns], forward=numeric,
                   strided_output=strided, gradients=gradients)
        records.append(row)
        if not all(value["passed"] for value in (numeric, strided, *gradients)):
            raise RuntimeError(row)
    args.output.write_text(json.dumps(dict(records=records, passed=True), indent=2))
    print(json.dumps(dict(records=records, passed=True)), flush=True)


if __name__ == "__main__":
    main()
