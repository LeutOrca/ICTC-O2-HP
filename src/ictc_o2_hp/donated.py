"""Inference with explicit caller donation of the left tensor."""
import torch
import triton
import triton.language as tl

from .triton_o2 import _preloaded_path, _read_canonical, _write_raw
from .packed_large import cooperative


@triton.jit
def _consume_copy(left, right, weight, edge, base, mask, COPY: tl.constexpr,
                  WIDTH: tl.constexpr, C: tl.constexpr, KEYS: tl.constexpr,
                  PATHS: tl.constexpr, FULL: tl.constexpr, BLOCK: tl.constexpr):
    channel = base + COPY * C
    first0, first1, second0, second1 = (), (), (), ()
    for k in tl.static_range(len(KEYS.value)):
        index = channel % KEYS.value[k][1]
        a, b = _read_canonical(left, edge, index, mask, k, WIDTH, C, KEYS, FULL)
        x, y = _read_canonical(right, edge, index, mask, k, WIDTH, C, KEYS, FULL)
        first0 += (a,)
        first1 += (b,)
        second0 += (x,)
        second1 += (y,)
    for k in tl.static_range(len(KEYS.value)):
        if COPY == 0 or KEYS.value[k][3]:
            a = tl.full((BLOCK,), 0., tl.float32)
            b = tl.full((BLOCK,), 0., tl.float32)
            for p in tl.static_range(len(PATHS.value[k])):
                x, y = _preloaded_path(first0, first1, second0, second1, weight,
                                       channel, mask, PATHS, k, p)
                a += x
                b += y
            _write_raw(left, edge, channel, a, b, mask, k, WIDTH, C, KEYS, FULL)


@triton.jit(do_not_specialize=['edges'])
def _consume_left(left, right, weight, edges, WIDTH: tl.constexpr, C: tl.constexpr,
                  KEYS: tl.constexpr, PATHS: tl.constexpr, FULL: tl.constexpr,
                  BLOCK: tl.constexpr):
    # Each lane owns both parity copies and all frequencies of one (edge, base).
    # The upper copy reads shared scalars but writes only upper frequencies.
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    edge, base = idx // C, idx % C
    mask = edge < edges
    if FULL:
        _consume_copy(left, right, weight, edge, base, mask, 1,
                      WIDTH, C, KEYS, PATHS, FULL, BLOCK)
    _consume_copy(left, right, weight, edge, base, mask, 0,
                  WIDTH, C, KEYS, PATHS, FULL, BLOCK)


class DonatedLeftInference:
    """Overwrite the uniquely owned left input, return it, and preserve the right.

    The caller owns the left storage and must discard its previous contents
    and all external aliases. CUDA FP32 native flat inference under no_grad.
    """
    def __init__(self, module):
        if any(k[4] != 0 for k in module._triton_keys):
            raise ValueError('Native flat storage is required')
        self.module = module
        self.keys = module._triton_keys
        self.paths = tuple(tuple(p for p in module._triton_paths if p[2] == k)
                           for k in range(len(self.keys)))
        self.full = module.parity == 'full'
        self.packed_family = self.full and module.channels == 256 and module.max_frequency == 6

    def __call__(self, left, right):
        weight = self.module.weight
        if torch.is_grad_enabled() or left.requires_grad:
            raise ValueError('Donate only an inference tensor under no_grad')
        if left.ndim != 2 or left.shape != right.shape or left.shape[1] != self.module.representation_dimension:
            raise ValueError('Expected matching native flat tensors')
        for t in (left, right, weight):
            if t.device != weight.device or not t.is_cuda or t.dtype != torch.float32 or not t.is_contiguous():
                raise ValueError('Expected contiguous FP32 tensors on the module CUDA device')
        if not left.numel():
            return left
        storage = left.untyped_storage().data_ptr()
        if storage in (right.untyped_storage().data_ptr(), weight.untyped_storage().data_ptr()):
            raise ValueError('Donated storage must be distinct from right and weight')
        kernel = _consume_left
        if (self.packed_family and left.data_ptr() % 8 == 0 and right.data_ptr() % 8 == 0
                and torch.cuda.get_device_capability(left.device) == (8, 0)):
            kernel = cooperative
        # The module holds its parameters live; honour the caller's stream.
        with torch.cuda.device(left.device):
            self.last_kernel = kernel[(triton.cdiv(left.shape[0] * self.module.channels, 128),)](
                left, right, weight, left.shape[0], left.shape[1], self.module.channels,
                self.keys, self.paths, self.full, 128, num_warps=4)
        torch.autograd.graph.increment_version(left)
        return left
