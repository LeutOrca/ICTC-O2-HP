"""Reduce a graph TP whose right operand is a receiver feature times an edge scalar."""

import torch
import triton
import triton.language as tl

from .graph import GraphInference, _add_raw, _edge_product
from .triton_o2 import _preloaded_path, _read_canonical


@triton.jit(do_not_specialize=["E"])
def _receiver_rows(receiver, target, scale, output, E, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    edge = offset // WIDTH
    active = edge < E
    node = tl.load(target + edge, active, 0)
    factor = tl.load(scale + edge, active, 0)
    value = tl.load(receiver + node * WIDTH + offset % WIDTH, active, 0)
    tl.store(output + offset, value * factor, active)


def _materialize_scaled_receivers(receiver, destinations, scale):
    """Materialize exactly one range with a coalesced gather-and-scale kernel."""
    edges, width = destinations.numel(), receiver.size(1)
    output = receiver.new_empty((edges, width))
    if edges:
        block = 256
        _receiver_rows[(triton.cdiv(edges * width, block),)](
            receiver, destinations, scale, output, edges, width, block, num_warps=4
        )
    return output


class _ScaledReceiverGraphInference(GraphInference):
    def _check_inputs(self, left, receiver, scale):
        if torch.is_grad_enabled():
            raise RuntimeError("Use torch.no_grad for graph inference.")
        node_shape = (self.graph.nodes, self.width)
        if left.shape != node_shape or receiver.shape != node_shape or scale.shape != (self.graph.edges,):
            raise ValueError("Expected two node feature matrices and one scalar per directed edge.")
        weight = self.module.weight
        for operand in (left, receiver, scale, weight):
            if operand.device != self.graph.sources.device or operand.dtype != torch.float32 or not operand.is_contiguous():
                raise ValueError("Contiguous CUDA FP32 operands on the graph device are required.")
        return weight


class ChunkedScaledReceiverGraphInference(_ScaledReceiverGraphInference):
    """Generate a tile of scaled receiver features and reduce it into one node output."""

    def __init__(self, module, graph, *, chunk_edges=16384):
        super().__init__(module, graph)
        if type(chunk_edges) is not int or chunk_edges <= 0:
            raise ValueError("chunk_edges must be a positive integer.")
        self.chunk_edges = chunk_edges

    def __call__(self, left, receiver, scale):
        weight = self._check_inputs(left, receiver, scale)
        output = left.new_zeros((self.graph.nodes, self.width))
        with torch.cuda.device(left.device):
            for start in range(0, self.graph.edges, self.chunk_edges):
                stop = min(start + self.chunk_edges, self.graph.edges)
                sources = self.graph.sources[start:stop]
                targets = self.graph.destinations[start:stop]
                values = _materialize_scaled_receivers(receiver, targets, scale[start:stop])
                grid = (triton.cdiv((stop - start) * self.multiplicity, self.block),)
                self.last_kernel = _edge_product[grid](
                    left, values, weight, sources, targets, output, stop - start, *self.constants, num_warps=4
                )
                del values
        return output


@triton.jit(do_not_specialize=["E"])
def _scaled_edge_product(left, receiver, scale, weight, sources, targets, output, E,
                         WIDTH: tl.constexpr, C: tl.constexpr, KEYS: tl.constexpr, PATHS: tl.constexpr,
                         FULL: tl.constexpr, MULTIPLICITY: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    edge, channel = offset // MULTIPLICITY, offset % MULTIPLICITY
    active = edge < E
    source = tl.load(sources + edge, active, 0)
    target = tl.load(targets + edge, active, 0)
    factor = tl.load(scale + edge, active, 0)
    first0, first1, second0, second1 = (), (), (), ()
    for key in tl.static_range(len(KEYS.value)):
        key_channel = channel % KEYS.value[key][1]
        a0, a1 = _read_canonical(left, source, key_channel, active, key, WIDTH, C, KEYS, FULL)
        b0, b1 = _read_canonical(receiver, target, key_channel, active, key, WIDTH, C, KEYS, FULL)
        first0 += (a0,)
        first1 += (a1,)
        second0 += (b0 * factor,)
        second1 += (b1 * factor,)
    for key in tl.static_range(len(KEYS.value)):
        mask = active & (channel < KEYS.value[key][1])
        value0 = tl.full((BLOCK,), 0., tl.float32)
        value1 = tl.full((BLOCK,), 0., tl.float32)
        for p in tl.static_range(len(PATHS.value[key])):
            a, b = _preloaded_path(first0, first1, second0, second1, weight, channel, mask, PATHS, key, p)
            value0 += a
            value1 += b
        _add_raw(output, target, channel, value0, value1, mask, key, WIDTH, C, KEYS, FULL)


class FusedScaledReceiverGraphInference(_ScaledReceiverGraphInference):
    """Generate the right operand in registers and accumulate each edge into its receiver."""

    def __call__(self, left, receiver, scale):
        weight = self._check_inputs(left, receiver, scale)
        output = left.new_zeros((self.graph.nodes, self.width))
        if not self.graph.edges:
            return output
        aligned = tuple(tensor.data_ptr() % 16 == 0 for tensor in (left, receiver, scale, weight))
        args = (left, receiver, scale, weight, self.graph.sources, self.graph.destinations,
                output, self.graph.edges, *self.constants)
        with torch.cuda.device(left.device):
            runner = self.runners.get(aligned)
            if runner is None:
                self.last_kernel = _scaled_edge_product[self.grid](*args, num_warps=4)
                self.runners[aligned] = self.last_kernel[self.grid]
            else:
                runner(*args)
        return output
