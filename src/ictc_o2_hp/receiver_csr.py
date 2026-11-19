"""Receiver-owned FP32 reduction for scaled-right uuu graph inference."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .scaled_graph import _ScaledReceiverGraphInference
from .triton_o2 import _preloaded_path, _read_canonical, _write_raw


@dataclass(frozen=True)
class PreparedReceiverCSR:
    nodes: int
    edges: int
    sources: torch.Tensor
    row_ptr: torch.Tensor
    edge_ids: torch.Tensor
    max_degree: int


def prepare_receiver_csr(graph):
    """Group incoming edges without changing the caller's edge-aligned inputs."""
    index_dtype = torch.int32 if max(graph.nodes, graph.edges) <= 2**31 - 1 else torch.int64
    order = torch.argsort(graph.destinations, stable=True)
    degree = torch.bincount(graph.destinations, minlength=graph.nodes)
    row_ptr = torch.cat((degree.new_zeros(1), degree.cumsum(0))).to(index_dtype)
    return PreparedReceiverCSR(
        graph.nodes, graph.edges, graph.sources.index_select(0, order).to(index_dtype),
        row_ptr, order.to(index_dtype), graph.max_degree,
    )


@triton.jit
def _receiver_csr_tile(left, scale, weight, sources, edge_ids, start, stop, channel,
                       right0, right1, accum0, accum1,
                       WIDTH: tl.constexpr, C: tl.constexpr, KEYS: tl.constexpr,
                       PATHS: tl.constexpr, FULL: tl.constexpr, MULTIPLICITY: tl.constexpr,
                       EDGE_TILE: tl.constexpr, CHANNEL_TILE: tl.constexpr):
    position = start.to(tl.int64) + tl.arange(0, EDGE_TILE)
    active = position < stop
    src = tl.load(sources + position, active, 0)
    edge = tl.load(edge_ids + position, active, 0)
    factor = tl.load(scale + edge, active, 0)
    left_mask = active[:, None] & (channel[None, :] < MULTIPLICITY)
    first0, first1, second0, second1 = (), (), (), ()
    for key in tl.static_range(len(KEYS.value)):
        key_channel = channel % KEYS.value[key][1]
        a, b = _read_canonical(left, src[:, None], key_channel[None, :], left_mask,
                               key, WIDTH, C, KEYS, FULL)
        first0 += (a,)
        first1 += (b,)
        second0 += (right0[key][None, :] * factor[:, None],)
        second1 += (right1[key][None, :] * factor[:, None],)

    next0, next1 = (), ()
    for key in tl.static_range(len(KEYS.value)):
        value0 = tl.full((EDGE_TILE, CHANNEL_TILE), 0., tl.float32)
        value1 = tl.full((EDGE_TILE, CHANNEL_TILE), 0., tl.float32)
        for path in tl.static_range(len(PATHS.value[key])):
            a, b = _preloaded_path(first0, first1, second0, second1, weight, channel[None, :],
                                   channel[None, :] < KEYS.value[key][1], PATHS, key, path)
            value0 += a
            value1 += b
        next0 += (accum0[key] + tl.where(active[:, None], value0, 0.),)
        next1 += (accum1[key] + tl.where(active[:, None], value1, 0.),)
    return next0, next1


@triton.jit
def _receiver_csr(left, receiver, scale, weight, sources, edge_ids, row_ptr, output,
                  WIDTH: tl.constexpr, C: tl.constexpr, KEYS: tl.constexpr,
                  PATHS: tl.constexpr, FULL: tl.constexpr, MULTIPLICITY: tl.constexpr,
                  EDGE_TILE: tl.constexpr, CHANNEL_TILE: tl.constexpr, SINGLE_TILE: tl.constexpr):
    channel_blocks: tl.constexpr = (MULTIPLICITY + CHANNEL_TILE - 1) // CHANNEL_TILE
    node = tl.program_id(0) // channel_blocks
    channel = (tl.program_id(0) % channel_blocks) * CHANNEL_TILE + tl.arange(0, CHANNEL_TILE)
    begin = tl.load(row_ptr + node)
    end = tl.load(row_ptr + node + 1)
    zero = tl.full((EDGE_TILE, CHANNEL_TILE), 0., tl.float32)
    right0, right1, accum0, accum1 = (), (), (), ()
    for key in tl.static_range(len(KEYS.value)):
        a, b = _read_canonical(receiver, node, channel % KEYS.value[key][1],
                               channel < MULTIPLICITY, key, WIDTH, C, KEYS, FULL)
        right0 += (a,)
        right1 += (b,)
        accum0 += (zero,)
        accum1 += (zero,)
    if SINGLE_TILE:
        accum0, accum1 = _receiver_csr_tile(
            left, scale, weight, sources, edge_ids, begin, end, channel, right0, right1,
            accum0, accum1, WIDTH, C, KEYS, PATHS, FULL, MULTIPLICITY, EDGE_TILE, CHANNEL_TILE,
        )
    else:
        for start in range(begin, end, EDGE_TILE):
            accum0, accum1 = _receiver_csr_tile(
                left, scale, weight, sources, edge_ids, start, end, channel, right0, right1,
                accum0, accum1, WIDTH, C, KEYS, PATHS, FULL, MULTIPLICITY, EDGE_TILE, CHANNEL_TILE,
            )
    for key in tl.static_range(len(KEYS.value)):
        _write_raw(output, node, channel, tl.sum(accum0[key], 0), tl.sum(accum1[key], 0),
                   channel < KEYS.value[key][1], key, WIDTH, C, KEYS, FULL)


class ReceiverCSRGraphInference(_ScaledReceiverGraphInference):
    """Own each receiver/channel tile in one CTA and store its local neighbor sum."""

    def __init__(self, module, graph, *, channel_tile=8, edge_tile=64):
        super().__init__(module, graph, channel_tile=channel_tile)
        if edge_tile not in (16, 32, 64):
            raise ValueError("edge_tile must be 16, 32 or 64")
        self.constants = (*self.constants[:-1], edge_tile, channel_tile, graph.max_degree <= edge_tile)
        # Adjacent CTAs visit channel tiles of the same receiver before advancing.
        self.grid = (graph.nodes * triton.cdiv(self.multiplicity, channel_tile), 1, 1)

    def __call__(self, left, receiver, scale):
        weight = self._check_inputs(left, receiver, scale)
        graph = self.graph
        if not graph.edges:
            return left.new_zeros((graph.nodes, self.width))
        output = left.new_empty((graph.nodes, self.width))
        aligned = tuple(tensor.data_ptr() % 16 == 0 for tensor in (left, receiver, scale, weight))
        args = (left, receiver, scale, weight, graph.sources, graph.edge_ids, graph.row_ptr,
                output, *self.constants)
        with torch.cuda.device(left.device):
            runner = self.runners.get(aligned)
            if runner is None:
                self.last_kernel = _receiver_csr[self.grid](*args, num_warps=4)
                self.runners[aligned] = self.last_kernel[self.grid]
            else:
                runner(*args)
        return output
