"""Own IEEE FP32 matrix kernels for frozen-network inference."""
from collections import Counter

import torch
import triton
import triton.language as tl
from torch.utils._python_dispatch import TorchDispatchMode


@triton.jit
def _matrix(A, B, Bias, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            AS0: tl.constexpr, AS1: tl.constexpr, AS2: tl.constexpr,
            BS0: tl.constexpr, BS1: tl.constexpr, BS2: tl.constexpr,
            DS0: tl.constexpr, DS1: tl.constexpr, DS2: tl.constexpr,
            HAS_BIAS: tl.constexpr, ALPHA: tl.constexpr, BETA: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row_tiles = tl.cdiv(M, BM)
    batch = tl.program_id(0).to(tl.int64) // row_tiles
    row = (tl.program_id(0) % row_tiles) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    total = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + inner
        left = tl.load(A + batch * AS0 + row[:, None] * AS1 + k[None, :] * AS2,
                       (row[:, None] < M) & (k[None, :] < K), other=0.)
        right = tl.load(B + batch * BS0 + k[:, None] * BS1 + col[None, :] * BS2,
                        (k[:, None] < K) & (col[None, :] < N), other=0.)
        total = tl.dot(left, right, total, input_precision="ieee")
    result = total * ALPHA
    if HAS_BIAS and BETA != 0:
        bias = tl.load(Bias + batch * DS0 + row[:, None] * DS1 + col[None, :] * DS2,
                       (row[:, None] < M) & (col[None, :] < N), other=0.)
        result = result + BETA * bias
    tl.store(C + (batch * M + row[:, None]) * N + col[None, :], result,
             (row[:, None] < M) & (col[None, :] < N))


def _matrix_strides(value):
    if value.ndim == 2:
        return (0, *value.stride())
    return value.stride()


def matrix(left, right, bias=None, alpha=1., beta=1., backend="triton"):
    if (left.ndim not in (2, 3) or right.ndim != left.ndim
            or left.dtype != torch.float32 or right.dtype != torch.float32
            or not left.is_cuda or left.device != right.device
            or left.shape[-1] != right.shape[-2]):
        raise ValueError("Expected compatible CUDA FP32 matrices or batches")
    if left.ndim == 3 and left.shape[0] != right.shape[0]:
        raise ValueError("Batched matrix counts must match")
    shape = (*left.shape[:-1], right.shape[-1])
    rows, inner, columns = left.shape[-2], left.shape[-1], right.shape[-1]
    batches = left.shape[0] if left.ndim == 3 else 1
    output = left.new_empty(shape)
    if output.numel() == 0:
        return output
    if bias is not None:
        if bias.dtype != torch.float32 or bias.device != left.device:
            raise ValueError("Bias must use the same FP32 device")
        bias = bias.expand(shape)
    a_strides, b_strides = _matrix_strides(left), _matrix_strides(right)
    d_strides = (0, 0, 0) if bias is None else _matrix_strides(bias)
    if backend in ("cuda", "cuda2", "cuda4", "cuda8"):
        from cuda_matrix import compiled_matrix
        with torch.cuda.device(left.device):
            return compiled_matrix(left, right, bias, alpha, beta, groups=int(backend[4:] or 1))
    if backend != "triton":
        raise ValueError("Unknown matrix kernel")
    block_rows, block_columns = (64, 128) if rows >= 512 and columns >= 128 else (32, 64)
    with torch.cuda.device(left.device):
        _matrix[(triton.cdiv(rows, block_rows) * batches, triton.cdiv(columns, block_columns))](
            left, right, bias if bias is not None else output, output,
            rows, columns, inner, *a_strides, *b_strides, *d_strides,
            bias is not None, alpha, beta, block_rows, block_columns, 32, num_warps=4,
        )
    return output


class OwnMatrixKernels(TorchDispatchMode):
    """Route ATen matrix products, including input adjoints, to own kernels."""
    def __init__(self, backend="triton"):
        super().__init__()
        self.backend = backend
        self.calls = Counter()

    def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = operation._schema.name
        if name in ("aten::mv", "aten::addmv", "aten::dot", "aten::vdot"):
            self.calls[name] += 1
            if name == "aten::mv":
                left, vector = args
                return matrix(left, vector.unsqueeze(-1), backend=self.backend).squeeze(-1)
            if name == "aten::addmv":
                bias, left, vector = args
                return matrix(left, vector.unsqueeze(-1), bias.reshape(-1, 1),
                              alpha=kwargs.get("alpha", 1), beta=kwargs.get("beta", 1),
                              backend=self.backend).squeeze(-1)
            left, right = args
            return matrix(left.reshape(1, -1), right.reshape(-1, 1),
                          backend=self.backend).reshape(())
        if name in ("aten::mm", "aten::bmm", "aten::addmm", "aten::baddbmm"):
            if operation._schema.overload_name not in ("", "default"):
                raise RuntimeError(f"Unsupported matrix overload {operation}")
            self.calls[name] += 1
            if name in ("aten::mm", "aten::bmm"):
                return matrix(*args, backend=self.backend)
            bias, left, right = args
            return matrix(left, right, bias=bias, alpha=kwargs.get("alpha", 1),
                          beta=kwargs.get("beta", 1), backend=self.backend)
        if name in ("aten::_scaled_mm", "aten::addbmm"):
            raise RuntimeError(f"Matrix operation requires an explicit own kernel: {name}")
        return operation(*args, **kwargs)
