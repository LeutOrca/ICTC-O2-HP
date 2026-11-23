"""Own IEEE FP32 matrices and column reductions with int64 addressing."""
import torch
import triton
import triton.language as tl


@triton.jit
def _mm(A, B, Bias, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        A0: tl.constexpr, A1: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr,
        BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    col = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK).to(tl.int64)
    total = tl.full((BM, BN), 0., tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block.to(tl.int64) * BK + inner
        left = tl.load(A + row[:, None] * A0 + k[None, :] * A1,
                       (row[:, None] < M) & (k[None, :] < K), 0.)
        right = tl.load(B + k[:, None] * B0 + col[None, :] * B1,
                        (k[:, None] < K) & (col[None, :] < N), 0.)
        total = tl.dot(left, right, total, input_precision="ieee")
    if BIAS:
        total += tl.load(Bias + col[None, :], col[None, :] < N, 0.)
    tl.store(C + row[:, None] * N + col[None, :], total,
             (row[:, None] < M) & (col[None, :] < N))


@triton.jit
def _sum_part(A, Parts, M: tl.constexpr, N: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    col = tl.program_id(1).to(tl.int64) * COLS + tl.arange(0, COLS)
    value = tl.load(A + row[:, None] * N + col[None, :],
                    (row[:, None] < M) & (col[None, :] < N), 0.)
    tl.store(Parts + tl.program_id(0).to(tl.int64) * N + col,
             tl.sum(value, 0), col < N)


@triton.jit
def _sum_join(Parts, Out, N: tl.constexpr, COUNT: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    row = tl.arange(0, R).to(tl.int64)
    col = tl.program_id(0).to(tl.int64) * C + tl.arange(0, C)
    value = tl.load(Parts + row[:, None] * N + col[None, :],
                    (row[:, None] < COUNT) & (col[None, :] < N), 0.)
    tl.store(Out + col, tl.sum(value, 0), col < N)


@triton.jit
def _bias_rows(Bias, Out, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    value = tl.load(Bias + i % N, i < M * N, 0.)
    tl.store(Out + i, value, i < M * N)


def mm(left, right, bias=None):
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("Incompatible matrix shapes")
    if not left.is_cuda or left.dtype != torch.float32 or right.device != left.device or right.dtype != left.dtype:
        raise TypeError("Own matrix requires matching CUDA FP32 tensors")
    m, k, n = left.shape[0], left.shape[1], right.shape[1]
    if bias is not None and (bias.shape != (n,) or bias.device != left.device or bias.dtype != left.dtype):
        raise ValueError("Invalid column bias")
    out = left.new_empty((m, n))
    if m and n:
        _mm[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
            left, right, out if bias is None else bias, out, m, n, k,
            *left.stride(), *right.stride(), bias is not None, 16, 32, 32,
            num_warps=4, enable_fp_fusion=False)
    return out


def sum_rows(value):
    if not value.is_contiguous() or value.ndim != 2:
        raise ValueError("Column reduction requires a contiguous matrix")
    m, n = value.shape
    if not m:
        return value.new_zeros(n)
    count = triton.cdiv(m, 128)
    parts = value.new_empty((count, n))
    out = value.new_empty(n)
    _sum_part[(count, triton.cdiv(n, 64))](value, parts, m, n, 128, 64, num_warps=4)
    _sum_join[(triton.cdiv(n, 64),)](parts, out, n, count, triton.next_power_of_2(count), 64, num_warps=4)
    return out


def bias_rows(bias, rows):
    out = bias.new_empty((rows, len(bias)))
    if out.numel():
        _bias_rows[(triton.cdiv(out.numel(), 256),)](bias, out, rows, len(bias), 256, num_warps=4)
    return out
