"""Frozen HPX 5f278ef, independent output and FP32 input-only first adjoints."""
import importlib.util
import os
from pathlib import Path
import sys

import torch
from torch.autograd.function import once_differentiable

PACKAGE = None


def package():
    global PACKAGE
    if PACKAGE is None:
        folder = Path(__file__).resolve().parent / "upstream/local-hpx/ictc_o2_hp"
        settings = {"O2_HPX_FORWARD": "preload", "O2_HPX_ADJOINT": "replay", "O2_OPT_VARIANT": "angular_frozen_input_vjp"}
        previous = {key: os.environ.get(key) for key in settings}
        os.environ.update(settings)
        try:
            spec = importlib.util.spec_from_file_location("angular_hpx", folder / "__init__.py", submodule_search_locations=[str(folder)])
            PACKAGE = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = PACKAGE
            spec.loader.exec_module(PACKAGE)
            PACKAGE.TensorProduct
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return PACKAGE


class InputAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, module):
        kernel = sys.modules["angular_hpx.triton_o2"]
        ctx.module = module
        ctx.save_for_backward(left, right)
        return kernel._o2_forward_op(left, right, module.weight, module._weight_path_info,
            module._weight_path_coefficients, module._triton_metadata_id)

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        left, right = ctx.saved_tensors
        module = ctx.module
        kernel = sys.modules["angular_hpx.triton_o2"]
        first, second, _ = kernel._o2_backward_op(left, right, module.weight, cotangent,
            module._weight_path_info, module._weight_path_coefficients,
            module._triton_metadata_id, 3)
        return first, second, None


def frozen(native):
    module = package().TensorProduct(native.channels, native.max_frequency,
        parity=native.parity).cuda().float().requires_grad_(False)
    if module.paths != native.paths or module.weight.shape != native.weight.shape:
        raise RuntimeError("HPX paths/weights differ from the frozen public equation")
    with torch.no_grad():
        module.weight.copy_(native.weight)
    return lambda left, right: InputAdjoint.apply(left, right, module)
