"""Own fused 16-lane FP32 GEMM preserving native 256-element FMA resets."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_weight(W, Packed, K: tl.constexpr, N: tl.constexpr,
                 WS0: tl.constexpr, WS1: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    value = tl.load(W + col[None, :] * WS0 + row[:, None] * WS1,
                    (row[:, None] < K) & (col[None, :] < N), other=0.)
    tl.store(Packed + row[:, None] * N + col[None, :], value,
             (row[:, None] < K) & (col[None, :] < N))


@triton.jit
def _fused(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            A0: tl.constexpr, A1: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    lane = tl.arange(0, 16)
    total = tl.full((16, BM, BN), 0., tl.float32)
    for start in range(tl.cdiv(K, 256)):
        subtotal = tl.full((16, BM, BN), 0., tl.float32)
        for step in tl.static_range(16):
            k = start * 256 + lane + step * 16
            left = tl.load(A + row[None, :] * A0 + k[:, None] * A1,
                           (row[None, :] < M) & (k[:, None] < K), other=0.)
            right = tl.load(B + k[:, None] * N + col[None, :],
                            (k[:, None] < K) & (col[None, :] < N), other=0.)
            subtotal = tl.fma(left[:, :, None], right[:, None, :], subtotal)
        subtotal = tl.inline_asm_elementwise(
            "mov.b32 $0, $1;", constraints="=f,f", args=[subtotal],
            dtype=tl.float32, is_pure=True, pack=1)
        total = total + subtotal
    result = tl.full((BM, BN), 0., tl.float32)
    for group in tl.static_range(16):
        index = tl.full((1, BM, BN), group, tl.int32)
        value = tl.gather(total, index, 0).reshape((BM, BN))
        result = result + value
    tl.store(C + row[:, None] * N + col[None, :], result,
             (row[:, None] < M) & (col[None, :] < N))


def fused(left, linear, compact=False):
    weight = linear.weight
    assert not weight.requires_grad and left.dtype == weight.dtype == torch.float32
    assert left.is_cuda and weight.device == left.device and left.shape[1] == weight.shape[1]
    n, k = weight.shape
    signature = (weight.data_ptr(), weight._version, weight.dtype, weight.device, tuple(weight.shape), tuple(weight.stride()))
    retain = not compact or left.shape[0] >= 4096
    if not retain and hasattr(linear, "_hpx_fused_block_weight"):
        delattr(linear, "_hpx_fused_block_weight")
    cache = getattr(linear, "_hpx_fused_block_weight", None) if retain else None
    if cache is None or cache[0] != signature:
        packed = weight.new_empty((k, n))
        _pack_weight[(triton.cdiv(k, 32), triton.cdiv(n, 32))](
            weight, packed, k, n, *weight.stride(), 32)
        if retain:
            linear._hpx_fused_block_weight = (signature, packed)
    else:
        packed = cache[1]
    output = left.new_empty((left.shape[0], n))
    _fused[(triton.cdiv(left.shape[0], 8), triton.cdiv(n, 32))](
        left, packed, output, left.shape[0], n, k, *left.stride(),
        8, 32, num_warps=4, enable_fp_fusion=False)
    return output
