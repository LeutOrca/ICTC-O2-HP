"""Keep only exact selector coefficients used by a frozen composition."""
import torch

from trace_optimizations import SpeciesProjection, install as install_species


class PackedSpeciesProjection(SpeciesProjection):
    def __init__(self, reference):
        super().__init__(reference)
        self.active_elements = None

    def bind(self, elements):
        reference = self.reference
        packed = []
        paths = []
        offset = 0
        keep = torch.tensor(elements, device=reference.weight.device, dtype=torch.long)
        for number, source, target, dim, ni, no, start, end, scale in self.paths:
            values = reference.weight.detach()[start:end].view(ni, self.channels, no)
            compact = values.index_select(1, keep).reshape(-1)
            packed.append(compact)
            paths.append((number, source, target, dim, ni, no, offset,
                          offset + compact.numel(), scale))
            offset += compact.numel()
        active = torch.nn.Module()
        active.register_buffer("weight", torch.cat(packed))
        self.reference = active
        self.paths = tuple(paths)
        self.channels = len(elements)
        self.active_elements = elements
        print(f"species selector uses {len(elements)} elements and {offset} coefficients", flush=True)

    def forward(self, features, node_attrs):
        if self.groups is None:
            raise ValueError("A species group schedule is required")
        actual = self.groups
        elements = tuple(z for z, _ in actual)
        if self.active_elements is None:
            self.bind(elements)
        if elements != self.active_elements:
            raise ValueError("Changed elements require a new packed selector")
        self.groups = tuple((index, indices) for index, (_, indices) in enumerate(actual))
        try:
            return super().forward(features, node_attrs)
        finally:
            self.groups = actual


def install(bench, profile=False):
    install_species(bench, profile=profile)
    base_model = bench.model_for

    def model_for(backend, device):
        if backend == "hp-uvu-compact":
            model = base_model("hp-uvu-species", device)
            block = model.interactions[0]
            block.skip_tp = PackedSpeciesProjection(block.skip_tp.reference)
            return model
        return base_model(backend, device)

    bench.model_for = model_for
