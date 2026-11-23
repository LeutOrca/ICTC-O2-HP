"""Own FP32 split-K GEMM with explicitly ordered partials and joins.

This preserves each scalar FMA chain. It does not use a higher precision
accumulator, Tensor Cores, atomics, or a vendor matrix routine.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _partials(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              A0: tl.constexpr, A1: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr,
              GROUPS: tl.constexpr, STRIPE: tl.constexpr, CONTIGUOUS: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              CHUNK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    col = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    group = tl.program_id(2).to(tl.int64)
    inner = tl.arange(0, BK).to(tl.int64)
    total = tl.full((BM, BN), 0., tl.float32)
    span: tl.constexpr = K // GROUPS
    if CHUNK:
        for block in range(tl.cdiv(K, CHUNK)):
            item = inner
            k = block * CHUNK + (item // STRIPE) * GROUPS * STRIPE + group * STRIPE + item % STRIPE
            left = tl.load(A + row[:, None] * A0 + k[None, :] * A1,
                           (row[:, None] < M) & (item[None, :] < CHUNK // GROUPS) & (k[None, :] < K), other=0.)
            right = tl.load(B + k[:, None] * B0 + col[None, :] * B1,
                            (item[:, None] < CHUNK // GROUPS) & (k[:, None] < K) & (col[None, :] < N), other=0.)
            subtotal = tl.dot(left, right, input_precision="ieee")
            # Keep the block's rounding boundary. Triton otherwise rewrites
            # dot(A, B, 0) + total into dot(A, B, total), changing FP32 results.
            subtotal = tl.inline_asm_elementwise(
                "mov.b32 $0, $1;", constraints="=f,f", args=[subtotal],
                dtype=tl.float32, is_pure=True, pack=1)
            total = total + subtotal
    else:
        for block in range(tl.cdiv(span, BK)):
            item = block * BK + inner
            if CONTIGUOUS:
                k = group * span + item
            else:
                k = (item // STRIPE) * GROUPS * STRIPE + group * STRIPE + item % STRIPE
            left = tl.load(A + row[:, None] * A0 + k[None, :] * A1,
                           (row[:, None] < M) & (item[None, :] < span), other=0.)
            right = tl.load(B + k[:, None] * B0 + col[None, :] * B1,
                            (item[:, None] < span) & (col[None, :] < N), other=0.)
            total = tl.dot(left, right, total, input_precision="ieee")
    tl.store(C + group * M * N + row[:, None] * N + col[None, :], total,
             (row[:, None] < M) & (col[None, :] < N))


@triton.jit
def _join(Parts, Bias, C, M: tl.constexpr, N: tl.constexpr,
          C0: tl.constexpr, C1: tl.constexpr, D0: tl.constexpr, D1: tl.constexpr,
          HAS_BIAS: tl.constexpr, GROUPS: tl.constexpr, JOIN: tl.constexpr,
          LEVELS: tl.constexpr, TILE: tl.constexpr):
    item = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    values = ()
    for group in tl.static_range(GROUPS):
        values += (tl.load(Parts + group * M * N + item, item < M * N, other=0.),)
    if JOIN == "l":
        total = values[0]
        for group in tl.static_range(1, GROUPS):
            total = total + values[group]
    else:
        for level in tl.static_range(LEVELS):
            reduced = ()
            for group in tl.static_range(GROUPS >> (level + 1)):
                if JOIN == "a":
                    reduced += (values[2 * group] + values[2 * group + 1],)
                else:
                    reduced += (values[group] + values[group + (GROUPS >> (level + 1))],)
            values = reduced
        total = values[0]
    row, col = item // N, item % N
    if HAS_BIAS:
        total = total + tl.load(Bias + row * D0 + col * D1, row < M, other=0.)
    tl.store(C + row * C0 + col * C1, total, row < M)


def scheduled(left, right, configuration, output=None, bias=None):
    configuration, _, chunk = configuration.partition("@")
    chunk = int(chunk or 0)
    join = configuration[-1]
    groups, stripe = map(int, configuration[:-1].split("x"))
    count = abs(groups)
    assert count > 1 and count & (count - 1) == 0 and join in "lab"
    assert left.ndim == right.ndim == 2 and left.dtype == right.dtype == torch.float32
    m, k, n = left.shape[0], left.shape[1], right.shape[1]
    assert right.shape[0] == k and k % (count * stripe) == 0
    assert not chunk or (groups > 0 and chunk % count == 0 and chunk // count <= 32)
    if output is None:
        output = left.new_empty((m, n))
    if not output.numel():
        return output
    partials = left.new_empty((count, m, n))
    _partials[(triton.cdiv(m, 32), triton.cdiv(n, 64), count)](
        left, right, partials, m, n, k, *left.stride(), *right.stride(),
        count, stripe, groups < 0, 32, 64, 32, chunk, num_warps=4, enable_fp_fusion=False)
    if bias is not None:
        bias = bias.expand(m, n)
    _join[(triton.cdiv(m * n, 256),)](
        partials, output if bias is None else bias, output, m, n,
        *output.stride(), *((0, 0) if bias is None else bias.stride()),
        bias is not None, count, join, count.bit_length() - 1, 256,
        num_warps=4, enable_fp_fusion=False)
    return output
