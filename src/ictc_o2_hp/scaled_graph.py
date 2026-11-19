"""Reduce a graph TP whose right operand is a receiver feature times an edge scalar."""

import torch
import triton
import triton.language as tl

from .graph import GraphInference, _edge_product


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
