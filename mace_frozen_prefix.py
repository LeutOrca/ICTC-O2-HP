"""Eliminate position-independent MACE prefix projections for frozen inference."""
import torch


class FrozenPrefix(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.reference = reference
        self.cached = None
        self.signature = None
        self.hits = 0
        self.misses = 0

    def forward(self, *inputs):
        if any(value.requires_grad for value in inputs) or any(
                parameter.requires_grad for parameter in self.reference.parameters()):
            raise RuntimeError("Frozen prefix projections require constant attributes and weights")
        signature = tuple((value.data_ptr(), value._version, tuple(value.shape), tuple(value.stride()),
                           value.dtype, value.device) for value in inputs)
        signature += tuple((parameter.data_ptr(), parameter._version)
                           for parameter in self.reference.parameters())
        if self.cached is None or signature != self.signature:
            self.cached = self.reference(*inputs).detach()
            self.signature = signature
            self.misses += 1
        else:
            self.hits += 1
        return self.cached


def install(model):
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen prefix elimination supports frozen inference only")
    model.node_embedding = FrozenPrefix(model.node_embedding)
    first = model.interactions[0]
    first.linear_up = FrozenPrefix(first.linear_up)
    names = ["node_embedding", "interactions.0.linear_up"]
    if first.__class__.__name__ == "RealAgnosticResidualInteractionBlock":
        first.skip_tp = FrozenPrefix(first.skip_tp)
        names.append("interactions.0.skip_tp")
    return tuple(names)


def receipt(model):
    return {name: dict(hits=module.hits, misses=module.misses,
                       bytes=module.cached.numel() * module.cached.element_size()
                       if module.cached is not None else 0)
            for name, module in model.named_modules() if isinstance(module, FrozenPrefix)}
