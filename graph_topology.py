"""Graph-safe distance replay of a validated, fixed official V3 topology."""
from types import MethodType

import torch


def install(net, fixed):
    count = fixed.edge_index.size(1)
    # The original neighbor builder supplied this topology and already excluded
    # zero-distance edges. Cache only integer topology metadata, never geometry.
    system = torch.repeat_interleave(torch.arange(len(fixed.natoms), device=fixed.pos.device),
                                     fixed.neighbors, output_size=count)
    indices = torch.arange(count, device=fixed.pos.device)

    def generate_graph(model, data, *args, **kwargs):
        if model.otf_graph:
            raise RuntimeError("The fixed-topology replay requires frozen edge lists")
        row, col = data.edge_index
        vectors = data.pos[row] - data.pos[col]
        cells = data.cell.index_select(0, system)
        offsets = data.cell_offsets.float().view(-1, 1, 3).bmm(cells.float()).view(-1, 3)
        vectors += offsets
        distances = vectors.norm(dim=-1)
        torch._assert_async(torch.all(distances > 0), "Frozen topology acquired a zero-length edge")
        return (data.edge_index[:, indices], distances[indices], vectors[indices],
                data.cell_offsets, offsets[indices], data.neighbors)

    net.model.generate_graph = MethodType(generate_graph, net.model)
