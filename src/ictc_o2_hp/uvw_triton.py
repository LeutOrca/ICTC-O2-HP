"""FP32 IEEE tiled channel contraction with differentiable Triton adjoints.

No Tensor Core/TF32/half inputs, atomics, detached weight caches, or native
double-backward recomputation. All derivative matrix products use this kernel.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _uvw_mm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            SAM: tl.constexpr, SAK: tl.constexpr, SBK: tl.constexpr, SBN: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        a = tl.load(A + rows[:, None] * SAM + k[None, :] * SAK,
                    (rows[:, None] < M) & (k[None, :] < K), 0)
        b = tl.load(B + k[:, None] * SBK + cols[None, :] * SBN,
                    (k[:, None] < K) & (cols[None, :] < N), 0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(C + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


class _Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right):
        ctx.save_for_backward(left, right)
        m, k = left.shape
        n = right.shape[1]
        out = left.new_empty((m, n))
        if m and n:
            _uvw_mm[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
                left, right, out, m, n, k, *left.stride(), *right.stride(),
                32, 32, 32, num_warps=4,
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
