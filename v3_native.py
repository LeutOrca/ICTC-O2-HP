"""Install own frozen SO(2) mixing and merge norm into official V3 modules."""
from types import MethodType

from frozen_so2 import FrozenWeightCache, frozen_so2
from merge_norm import merge_norm


def install(model, backend="triton", use_norm=True):
    model.requires_grad_(False)
    installed = dict(so2=[], norm=[])
    for name, module in model.named_modules():
        if module.__class__.__name__ == "SO2MLinear":
            cache = FrozenWeightCache()

            def forward(layer, inputs, concat_outputs=True, _cache=cache):
                value = frozen_so2(inputs, layer.fc.weight, _cache, backend)
                return value if concat_outputs else (value[:, :1], value[:, 1:])

            module.forward = MethodType(forward, module)
            installed["so2"].append(name)
        if use_norm and module.__class__.__name__ == "EquivariantMergeLayerNorm":
            module.forward = MethodType(lambda layer, value: merge_norm(value, layer), module)
            installed["norm"].append(name)
    if not installed["so2"] or use_norm and not installed["norm"]:
        raise RuntimeError(f"Expected frozen V3 SO2 and norm modules, received {installed}")
    return installed
