"""Install own frozen SO(2) mixing and merge norm into official V3 modules."""
from types import MethodType

from frozen_so2 import FrozenWeightCache
from merge_norm import merge_norm
from so2_layer import frozen_layer


def install(model, backend="triton", use_norm=True, split_complex=False):
    model.requires_grad_(False)
    installed = dict(so2=[], norm=[])
    for name, module in model.named_modules():
        if module.__class__.__name__ == "SO2Linear":
            module._own_split_complex = split_complex
            caches = tuple(FrozenWeightCache() for _ in module.so2_m_linear)

            def forward(layer, inputs, _caches=caches):
                return frozen_layer(inputs, layer, _caches, backend)

            module.forward = MethodType(forward, module)
            installed["so2"].append(name)
        if use_norm and module.__class__.__name__ == "EquivariantMergeLayerNorm":
            module.forward = MethodType(lambda layer, value: merge_norm(value, layer), module)
            installed["norm"].append(name)
    if not installed["so2"] or use_norm and not installed["norm"]:
        raise RuntimeError(f"Expected frozen V3 SO2 and norm modules, received {installed}")
    return installed
