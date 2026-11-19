"""Two independent whole-network changes selected from the B1 H100 trace."""
from contextlib import nullcontext
import math

import numpy as np
import torch


class SpeciesProjection(torch.nn.Module):
    """Apply one-hot species-dependent UVW channel mixing by species group."""

    def __init__(self, reference):
        super().__init__()
        channels, scalar = reference.irreps_in2[0]
        if (len(reference.irreps_in2) != 1 or scalar.l != 0 or scalar.p != 1
                or not reference.internal_weights or not reference.shared_weights):
            raise ValueError("The species leg must contain shared even-scalar weights")
        self.reference = reference
        self.channels = channels
        self.groups = None
        left = reference.irreps_in1.slices()
        right = reference.irreps_out.slices()
        offset, paths = 0, []
        for inst in reference.instructions:
            ni, ri = reference.irreps_in1[inst.i_in1]
            no, ro = reference.irreps_out[inst.i_out]
            if (inst.connection_mode != "uvw" or not inst.has_weight or ri != ro
                    or inst.i_in2 != 0 or inst.path_shape != (ni, channels, no)):
                raise ValueError("Unsupported first-layer selector path")
            count = ni * channels * no
            paths.append((inst.i_out, left[inst.i_in1], right[inst.i_out], ri.dim,
                          ni, no, offset, offset + count, inst.path_weight))
            offset += count
        self.paths = tuple(sorted(paths))
        if ([p[0] for p in self.paths] != list(range(len(right)))
                or offset != reference.weight.numel()):
            raise ValueError("The selector must cover each original output once")

    def forward(self, features, node_attrs):
        if self.groups is None:
            return self.reference(features, node_attrs)
        groups, n = self.groups, features.shape[0]
        homogeneous = len(groups) == 1
        blocks = []
        for _, source, _, dim, ni, no, lo, hi, scale in self.paths:
            x = features[:, source].reshape(n, ni, dim).transpose(1, 2)
            weights = self.reference.weight[lo:hi].view(ni, self.channels, no)
            if homogeneous:
                selected = x @ weights[:, groups[0][0], :]
            else:
                selected = x.new_zeros((n, dim, no))
                for z, indices in groups:
                    value = x.index_select(0, indices) @ weights[:, z, :]
                    selected.index_copy_(0, indices, value)
            output = (scale * (selected / math.sqrt(dim))).transpose(1, 2)
            blocks.append(output.reshape(n, no * dim))
        return torch.cat(blocks, dim=-1)


def install(bench, profile=False):
    model_for, native_graph, evaluate = bench.model_for, bench.native_graph, bench.evaluate

    def setup(backend, device):
        if backend not in ("hp-uvu-species", "hp-uvu-packed"):
            return model_for(backend, device)
        model = model_for("hp-uvu", device)
        model._packed_input_transfer = backend == "hp-uvu-packed"
        if backend == "hp-uvu-species":
            interaction = model.interactions[0]
            interaction.skip_tp = SpeciesProjection(interaction.skip_tp)
        return model

    def prepare(geometry, model, device):
        if torch.device(device).type == "cuda" and getattr(model, "_packed_input_transfer", False):
            region = torch.profiler.record_function("MACE::packed_host_to_device") if profile else nullcontext()
            with region:
                data = native_graph(geometry, model, "cpu")
                for dtype in (torch.float32, torch.int64):
                    batch = {key: value for key, value in data.items() if value.dtype == dtype}
                    pack = torch.cat([value.flatten() for value in batch.values()]).to(device)
                    slices = pack.split([value.numel() for value in batch.values()])
                    for (key, value), part in zip(batch.items(), slices):
                        data[key] = part.reshape_as(value)
        else:
            data = native_graph(geometry, model, device)
        if isinstance(model.interactions[0].skip_tp, SpeciesProjection):
            species = np.concatenate([row[1] for row in geometry])
            indices = np.searchsorted(model._benchmark_elements, species)
            elements = np.unique(indices)
            data["_selector_groups"] = tuple(
                (int(z), None if len(elements) == 1 else
                 torch.from_numpy(np.flatnonzero(indices == z)).to(device))
                for z in elements
            )
        return data

    def evaluate_grouped(model, data):
        selector = model.interactions[0].skip_tp
        if not isinstance(selector, SpeciesProjection):
            return evaluate(model, data)
        selector.groups = data["_selector_groups"]
        try:
            network_input = {key: value for key, value in data.items() if key != "_selector_groups"}
            return evaluate(model, network_input)
        finally:
            selector.groups = None

    bench.model_for = setup
    bench.native_graph = prepare
    bench.evaluate = evaluate_grouped
