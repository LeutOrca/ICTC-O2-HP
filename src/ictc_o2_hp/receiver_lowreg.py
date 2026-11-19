"""Fused receiver CSR schedules for diagonal O(2) uuu inference."""

import torch
import triton
import triton.language as tl

from .receiver_csr import ReceiverCSRGraphInference
from .triton_o2 import _preloaded_path, _read_canonical, _write_raw


@triton.jit
def _read_source(left, node, channel, mask, KEY: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
                 KEYS: tl.constexpr, FULL: tl.constexpr, CACHE: tl.constexpr):
    # GraphInference requires a native flat layout.
    start: tl.constexpr = KEYS.value[KEY][0]
    dim: tl.constexpr = KEYS.value[KEY][2]
    offset = node.to(tl.int64) * WIDTH + start + channel * dim
    a = tl.load(left + offset, mask, 0., cache_modifier=CACHE)
    if dim == 1:
        return a, a * 0.
    b = tl.load(left + offset + 1, mask, 0., cache_modifier=CACHE)
    if FULL and KEYS.value[KEY][3]:
        odd = channel >= C
        return tl.where(odd, -b, a), tl.where(odd, a, b)
    return a, b


@triton.jit
def _edges(sources, edge_ids, scale, start, end, EDGE_TILE: tl.constexpr):
    i = start.to(tl.int64) + tl.arange(0, EDGE_TILE)
    live = i < end
    source = tl.load(sources + i, live, 0)
    edge = tl.load(edge_ids + i, live, 0)
    factor = tl.load(scale + edge, live, 0.)
    return source, factor, live


@triton.jit
def _direct_tile(left, scale, weight, sources, edge_ids, start, end, channel, right0, right1, valid,
                 WIDTH: tl.constexpr, C: tl.constexpr, KEYS: tl.constexpr, PATHS: tl.constexpr,
                 FULL: tl.constexpr, KEY: tl.constexpr, EDGE_TILE: tl.constexpr, TILE: tl.constexpr,
                 CACHE: tl.constexpr):
    source, factor, live = _edges(sources, edge_ids, scale, start, end, EDGE_TILE)
    left_mask = live[:, None] & valid[None, :]
    l0, l1, r0, r1 = (), (), (), ()
    for k in tl.static_range(len(KEYS.value)):
        ch = channel % KEYS.value[k][1]
        a, b = _read_source(left, source[:, None], ch[None, :], left_mask, k, WIDTH, C, KEYS, FULL, CACHE)
        l0 += (a,)
        l1 += (b,)
        r0 += (right0[k][None, :] * factor[:, None],)
        r1 += (right1[k][None, :] * factor[:, None],)
    x = tl.full((EDGE_TILE, TILE), 0., tl.float32)
    y = tl.full((EDGE_TILE, TILE), 0., tl.float32)
    for p in tl.static_range(len(PATHS.value[KEY])):
        a, b = _preloaded_path(l0, l1, r0, r1, weight, channel[None, :], valid[None, :], PATHS, KEY, p)
        x += a
        y += b
    return tl.where(live[:, None], x, 0.), tl.where(live[:, None], y, 0.)


@triton.jit
def _one_output(left, receiver, scale, weight, sources, edge_ids, output, node, start, end, channel,
                WIDTH: tl.constexpr, C: tl.constexpr, KEYS: tl.constexpr, PATHS: tl.constexpr,
                FULL: tl.constexpr, KEY: tl.constexpr, EDGE_TILE: tl.constexpr, TILE: tl.constexpr,
                STATIC_STEPS: tl.constexpr, CACHE: tl.constexpr):
    valid = channel < KEYS.value[KEY][1]
    right0, right1 = (), ()
    for k in tl.static_range(len(KEYS.value)):
        ch = channel % KEYS.value[k][1]
        receiver0, receiver1 = _read_canonical(receiver, node, ch, valid, k, WIDTH, C, KEYS, FULL)
        right0 += (receiver0,)
        right1 += (receiver1,)
    x = tl.full((EDGE_TILE, TILE), 0., tl.float32)
    y = tl.full((EDGE_TILE, TILE), 0., tl.float32)
    if STATIC_STEPS:
        for i in tl.static_range(STATIC_STEPS):
            a, b = _direct_tile(left, scale, weight, sources, edge_ids, start + i * EDGE_TILE, end,
                                channel, right0, right1, valid, WIDTH, C, KEYS, PATHS, FULL, KEY,
                                EDGE_TILE, TILE, CACHE)
            x += a
            y += b
    else:
        for i in range(start, end, EDGE_TILE):
            a, b = _direct_tile(left, scale, weight, sources, edge_ids, i, end, channel, right0,
                                right1, valid, WIDTH, C, KEYS, PATHS, FULL, KEY, EDGE_TILE, TILE, CACHE)
            x += a
            y += b
    _write_raw(output, node, channel, tl.sum(x, 0), tl.sum(y, 0), valid, KEY, WIDTH, C, KEYS, FULL)


@triton.jit
def _csr_by_output(left, receiver, scale, weight, sources, edge_ids, row_ptr, output,
                   WIDTH: tl.constexpr, C: tl.constexpr, KEYS: tl.constexpr, PATHS: tl.constexpr,
                   FULL: tl.constexpr, KEY_TILES: tl.constexpr, TILES_PER_NODE: tl.constexpr,
                   EDGE_TILE: tl.constexpr, TILE: tl.constexpr, STATIC_STEPS: tl.constexpr,
                   CACHE: tl.constexpr):
    node = tl.program_id(0) // TILES_PER_NODE
    slot = tl.program_id(0) % TILES_PER_NODE
    begin, end = tl.load(row_ptr + node), tl.load(row_ptr + node + 1)
    first = 0
    for key in tl.static_range(len(KEYS.value)):
        stop = first + KEY_TILES.value[key]
        if slot >= first and slot < stop:
            ch = (slot - first) * TILE + tl.arange(0, TILE)
            _one_output(left, receiver, scale, weight, sources, edge_ids, output, node, begin, end, ch,
                        WIDTH, C, KEYS, PATHS, FULL, key, EDGE_TILE, TILE, STATIC_STEPS, CACHE)
        first = stop


@triton.jit
def _factored_tile(left, src, factor, live, channel, valid, WIDTH: tl.constexpr, C: tl.constexpr,
                   KEYS: tl.constexpr, FULL: tl.constexpr, KEY: tl.constexpr, CACHE: tl.constexpr):
    a, b = _read_source(left, src[:, None], channel[None, :], live[:, None] & valid[None, :],
                        KEY, WIDTH, C, KEYS, FULL, CACHE)
    return a * factor[:, None], b * factor[:, None]


@triton.jit
def _csr_bilinear(left, receiver, scale, weight, sources, edge_ids, row_ptr, output,
                  WIDTH: tl.constexpr, C: tl.constexpr, KEYS: tl.constexpr, PATHS: tl.constexpr,
                  FULL: tl.constexpr, MULTIPLICITY: tl.constexpr, EDGE_TILE: tl.constexpr,
                  TILE: tl.constexpr, ONE_TILE: tl.constexpr, CACHE: tl.constexpr):
    blocks: tl.constexpr = (MULTIPLICITY + TILE - 1) // TILE
    node = tl.program_id(0) // blocks
    channel = (tl.program_id(0) % blocks) * TILE + tl.arange(0, TILE)
    start, end = tl.load(row_ptr + node), tl.load(row_ptr + node + 1)
    valid = channel < MULTIPLICITY
    l0, l1, r0, r1 = (), (), (), ()
    if ONE_TILE:
        source, factor, live = _edges(sources, edge_ids, scale, start, end, EDGE_TILE)
    for key in tl.static_range(len(KEYS.value)):
        ch = channel % KEYS.value[key][1]
        if ONE_TILE:
            a, b = _factored_tile(left, source, factor, live, ch, valid, WIDTH, C, KEYS, FULL, key, CACHE)
        else:
            a = tl.full((EDGE_TILE, TILE), 0., tl.float32)
            b = tl.full((EDGE_TILE, TILE), 0., tl.float32)
            for i in range(start, end, EDGE_TILE):
                source, factor, live = _edges(sources, edge_ids, scale, i, end, EDGE_TILE)
                x, y = _factored_tile(left, source, factor, live, ch, valid, WIDTH, C, KEYS, FULL, key, CACHE)
                a += x
                b += y
        l0 += (tl.sum(a, 0),)
        l1 += (tl.sum(b, 0),)
        c, d = _read_canonical(receiver, node, ch, valid, key, WIDTH, C, KEYS, FULL)
        r0 += (c,)
        r1 += (d,)
    for key in tl.static_range(len(KEYS.value)):
        live_ch = channel < KEYS.value[key][1]
        a = tl.full((TILE,), 0., tl.float32)
        b = tl.full((TILE,), 0., tl.float32)
        for p in tl.static_range(len(PATHS.value[key])):
            x, y = _preloaded_path(l0, l1, r0, r1, weight, channel, live_ch, PATHS, key, p)
            a += x
            b += y
        _write_raw(output, node, channel, tl.where(start < end, a, 0.), tl.where(start < end, b, 0.),
                   live_ch, key, WIDTH, C, KEYS, FULL)


class LowRegisterReceiverCSRInference(ReceiverCSRGraphInference):
    """Fuse gather, scaled right TP and reduction while scheduling output keys separately."""

    def __init__(self, module, graph, *, channel_tile=16, edge_tile=16, warps=4,
                 unroll_segments=False, cache_modifier="", bilinear=False):
        super().__init__(module, graph, channel_tile=channel_tile, edge_tile=edge_tile)
        if warps not in (1, 2, 4, 8) or cache_modifier not in ("", ".cg", ".ca"):
            raise ValueError("Unsupported Triton launch options")
        self.warps = warps
        if bilinear:
            # For scalar-scaled right features and shared path weights only.
            # FP32 rounding can differ after moving the edge scalar onto left.
            self.kernel = _csr_bilinear
            self.constants = (*self.constants, cache_modifier)
        else:
            self.kernel = _csr_by_output
            key_tiles = tuple(triton.cdiv(k[1], channel_tile) for k in self.keys)
            blocks = sum(key_tiles)
            self.grid = (graph.nodes * blocks, 1, 1)
            segments = triton.cdiv(graph.max_degree, edge_tile)
            static_steps = segments if (unroll_segments and segments <= 4) or segments == 1 else 0
            self.constants = (
                self.width, self.channels, self.keys, self.paths, self.full,
                key_tiles, blocks, edge_tile, channel_tile, static_steps, cache_modifier,
            )

    def __call__(self, left, receiver, scale):
        weight = self._check_inputs(left, receiver, scale)
        graph = self.graph
        if not graph.edges:
            return left.new_zeros((graph.nodes, self.width))
        output = left.new_empty((graph.nodes, self.width))
        aligned = tuple(t.data_ptr() % 16 == 0 for t in (left, receiver, scale, weight))
        args = (left, receiver, scale, weight, graph.sources, graph.edge_ids, graph.row_ptr,
                output, *self.constants)
        with torch.cuda.device(left.device):
            runner = self.runners.get(aligned)
            if runner is None:
                self.last_kernel = self.kernel[self.grid](*args, num_warps=self.warps)
                self.runners[aligned] = self.last_kernel[self.grid]
            else:
                runner(*args)
        return output
