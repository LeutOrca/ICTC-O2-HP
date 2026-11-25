"""FP32 IEEE tiled channel contraction with differentiable Triton adjoints.

No Tensor Core/TF32/half inputs, atomics, detached weight caches, or native
double-backward recomputation. FP32 segment boundaries and their final sum
preserve the measured A100 channel/adjoint arithmetic order.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_rn(left, right):
    # Keep a sum of completed segments separate from the dot-product FMAs.
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[left, right],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _dot_piece(A, B, rows, cols, q, offset, acc,
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, SPAN: tl.constexpr,
               SAM: tl.constexpr, SAK: tl.constexpr, SBK: tl.constexpr, SBN: tl.constexpr):
    k = q + offset
    a = tl.load(A + rows[:, None] * SAM + k[None, :] * SAK,
                (rows[:, None] < M) & (q[None, :] < SPAN) & (k[None, :] < K), 0)
    b = tl.load(B + k[:, None] * SBK + cols[None, :] * SBN,
                (q[:, None] < SPAN) & (k[:, None] < K) & (cols[None, :] < N), 0)
    return tl.dot(a, b, acc, input_precision="ieee")


@triton.jit
def _uvw_mm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            SAM: tl.constexpr, SAK: tl.constexpr, SBK: tl.constexpr, SBN: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PARTS: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN).to(tl.int64)
    kk = tl.arange(0, BK).to(tl.int64)
    span = tl.cdiv(K, PARTS)
    c0 = tl.full((BM, BN), 0, tl.float32)
    if PARTS >= 2:
        c1 = tl.full((BM, BN), 0, tl.float32)
    if PARTS >= 3:
        c2 = tl.full((BM, BN), 0, tl.float32)
    if PARTS >= 4:
        c3 = tl.full((BM, BN), 0, tl.float32)
    if PARTS == 6:
        c4 = tl.full((BM, BN), 0, tl.float32)
        c5 = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(span, BK)):
        q = tl.cast(start, tl.int64) * BK + kk
        c0 = _dot_piece(A, B, rows, cols, q, 0, c0, M, N, K, span, SAM, SAK, SBK, SBN)
        if PARTS >= 2:
            c1 = _dot_piece(A, B, rows, cols, q, span, c1, M, N, K, span, SAM, SAK, SBK, SBN)
        if PARTS >= 3:
            c2 = _dot_piece(A, B, rows, cols, q, 2 * span, c2, M, N, K, span, SAM, SAK, SBK, SBN)
        if PARTS >= 4:
            c3 = _dot_piece(A, B, rows, cols, q, 3 * span, c3, M, N, K, span, SAM, SAK, SBK, SBN)
        if PARTS == 6:
            c4 = _dot_piece(A, B, rows, cols, q, 4 * span, c4, M, N, K, span, SAM, SAK, SBK, SBN)
            c5 = _dot_piece(A, B, rows, cols, q, 5 * span, c5, M, N, K, span, SAM, SAK, SBK, SBN)
    if PARTS >= 2:
        c0 = _add_rn(c0, c1)
    if PARTS >= 3:
        c0 = _add_rn(c0, c2)
    if PARTS >= 4:
        c0 = _add_rn(c0, c3)
    if PARTS == 6:
        c0 = _add_rn(c0, c4)
        c0 = _add_rn(c0, c5)
    tl.store(C + rows[:, None] * N + cols[None, :], c0,
             (rows[:, None] < M) & (cols[None, :] < N))


class _Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right):
        ctx.save_for_backward(left, right)
        m, k = left.shape
        n = right.shape[1]
        out = left.new_empty((m, n))
        parts = 1
        if left.is_contiguous():
            if right.stride() == (1, k):
                if (k, n) == (1024, 256):
                    parts = 2
                elif (k, n) == (768, 192):
                    parts = 4
            elif right.is_contiguous():
                if (k, n) == (512, 128):
                    parts = 6
                elif (k, n) == (384, 96):
                    parts = 3
        if m and n:
            _uvw_mm[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
                left, right, out, m, n, k, *left.stride(), *right.stride(),
                32, 32, 16 if parts > 1 else 32, parts, num_warps=4,
            )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        left, right = ctx.saved_tensors
        dl = _Matmul.apply(grad_output, right.transpose(0, 1)) if ctx.needs_input_grad[0] else None
        dr = _Matmul.apply(left.transpose(0, 1), grad_output) if ctx.needs_input_grad[1] else None
        return dl, dr


def channel_matmul(left, right):
    """Strided 2D channel contraction; FP32 CUDA uses IEEE Triton at all orders."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("Expected compatible two-dimensional channel matrices")
    if left.dtype != right.dtype or left.device != right.device:
        raise ValueError("Channel matrices must have the same dtype/device")
    if left.is_cuda and left.dtype == torch.float32:
        return _Matmul.apply(left, right)
    return left @ right


def canonical_linear(features, weight):
    """U=1, fixed scalar 0e x m -> m UVW specialization, preserving original K.

    features: [edges, angular_components, V]; weight: [W,V]. Reshape may stage
    non-flattenable layouts; that cost belongs to the measured call.
    """
    shape = features.shape
    if len(shape) != 3 or weight.ndim != 2 or shape[-1] != weight.shape[1]:
        raise ValueError("Expected [edges,components,V] and original [W,V] K")
    return channel_matmul(features.reshape(-1, shape[-1]), weight.T).reshape(
        shape[0], shape[1], weight.shape[0]
    )
