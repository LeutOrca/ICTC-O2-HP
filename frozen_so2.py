"""Frozen complex SO(2) channel mixing with own FP32 matrix kernels."""
import weakref

import torch
import triton
import triton.language as tl

from matrix_kernels import matrix


@triton.jit
def _pack_input(X, Packed, E, K: tl.constexpr, S0: tl.constexpr,
                S1: tl.constexpr, S2: tl.constexpr, TILE: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    edge = index // (2 * K)
    part = (index // K) % 2
    channel = index % K
    valid = edge < E
    value = tl.load(X + edge * S0 + part * S1 + channel * S2, valid, other=0.)
    tl.store(Packed + index, value, valid)


@triton.jit
def _pack_weight(W, Packed, K: tl.constexpr, O: tl.constexpr,
                 S0: tl.constexpr, S1: tl.constexpr, TILE: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    row = index // (2 * K)
    col = index % (2 * K)
    imaginary_block = (row // O) != (col // K)
    source_row = row % O + imaginary_block.to(tl.int64) * O
    value = tl.load(W + source_row * S0 + (col % K) * S1, row < 2 * O, other=0.)
    negate = (row < O) & (col >= K)
    tl.store(Packed + index, tl.where(negate, -value, value), row < 2 * O)


class FrozenWeightCache:
    """Invalidate packed immutable weights after version, storage or dtype changes."""
    def __init__(self):
        self.key = None
        self.packed = None
        self.reference = None

    def prepare(self, weight):
        if weight.requires_grad:
            raise ValueError("Frozen SO(2) inference requires immutable weights")
        key = (id(weight), weight._version, weight.data_ptr(), weight.storage_offset(),
               weight.shape, weight.stride(), weight.device, weight.dtype)
        if key != self.key or self.reference is None or self.reference() is not weight:
            width = weight.shape[1]
            outputs = weight.shape[0] // 2
            packed = weight.new_empty((2 * outputs, 2 * width))
            with torch.cuda.device(weight.device):
                _pack_weight[(triton.cdiv(packed.numel(), 256),)](
                    weight, packed, width, outputs, *weight.stride(), 256,
                )
            self.key, self.packed = key, packed
            self.reference = weakref.ref(weight)
        return self.packed


class _FrozenSO2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, backend):
        edges, _, width = inputs.shape
        packed_input = inputs.new_empty((edges, 2 * width))
        with torch.cuda.device(inputs.device):
            if edges:
                _pack_input[(triton.cdiv(packed_input.numel(), 256),)](
                    inputs, packed_input, edges, width, *inputs.stride(), 256,
                )
        output = matrix(packed_input, weight.T, backend=backend)
        ctx.save_for_backward(weight)
        ctx.input_shape = inputs.shape
        ctx.backend = backend
        return output.reshape(edges, 2, weight.shape[0] // 2)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, derivative):
        weight, = ctx.saved_tensors
        packed = derivative.reshape(derivative.shape[0], weight.shape[0])
        gradient = matrix(packed, weight, backend=ctx.backend)
        return gradient.reshape(ctx.input_shape), None, None


def frozen_so2(inputs, weight, cache, backend="triton"):
    if (inputs.ndim != 3 or inputs.shape[1] != 2 or weight.ndim != 2
            or weight.shape[0] % 2 or inputs.shape[2] != weight.shape[1]
            or not inputs.is_cuda or inputs.dtype != torch.float32
            or weight.dtype != torch.float32 or inputs.device != weight.device):
        raise ValueError("Expected CUDA FP32 SO(2) inputs [E,2,K] and weights [2O,K]")
    return _FrozenSO2.apply(inputs, cache.prepare(weight), backend)
