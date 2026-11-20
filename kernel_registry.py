"""Process-local CUDA kernel replacement, without intercepting non-matrix ops."""
from collections import Counter

import torch

from matrix_kernels import matrix

_REGISTRATIONS = []


def install(backend="triton"):
    calls = Counter()
    library = torch.library.Library("aten", "IMPL", "CUDA")

    def product(left, right):
        calls["mm/bmm"] += 1
        return matrix(left, right, backend=backend)

    def product_out(left, right, *, out):
        calls["mm/bmm.out"] += 1
        return matrix(left, right, backend=backend, output=out)

    def affine(bias, left, right, *, beta=1, alpha=1):
        calls["addmm/baddbmm"] += 1
        return matrix(left, right, bias, alpha, beta, backend)

    def affine_out(bias, left, right, *, beta=1, alpha=1, out):
        calls["addmm/baddbmm.out"] += 1
        return matrix(left, right, bias, alpha, beta, backend, output=out)

    def vector(left, right):
        calls["mv"] += 1
        return matrix(left, right.unsqueeze(-1), backend=backend).squeeze(-1)

    def vector_out(left, right, *, out):
        calls["mv.out"] += 1
        matrix(left, right.unsqueeze(-1), backend=backend, output=out.unsqueeze(-1))
        return out

    def affine_vector(bias, left, right, *, beta=1, alpha=1):
        calls["addmv"] += 1
        return matrix(left, right.unsqueeze(-1), bias.reshape(-1, 1), alpha, beta,
                      backend).squeeze(-1)

    def dot(left, right):
        calls["dot/vdot"] += 1
        return matrix(left.reshape(1, -1), right.reshape(-1, 1), backend=backend).reshape(())

    def unsupported(*args, **kwargs):
        raise RuntimeError("An uncovered matrix overload requires an explicit own CUDA kernel")

    implementations = {
        "mm": product, "bmm": product, "mm.out": product_out, "bmm.out": product_out,
        "addmm": affine, "baddbmm": affine, "addmm.out": affine_out,
        "baddbmm.out": affine_out, "mv": vector, "mv.out": vector_out,
        "addmv": affine_vector, "addmv.out": unsupported,
        "dot": dot, "vdot": dot, "dot.out": unsupported, "vdot.out": unsupported,
        "addbmm": unsupported, "addbmm.out": unsupported, "_scaled_mm": unsupported,
    }
    for name, implementation in implementations.items():
        library.impl(name, implementation)
    _REGISTRATIONS.append(library)
    return calls
