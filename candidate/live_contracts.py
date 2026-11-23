"""Conservative implementation-domain checks for explicit live raw-kernel APIs."""
import torch

from .contracts import TP_BUFFERS
from .reduction_policy import _has_hooks


def tensor_types(*values):
    if any(type(value) not in (torch.Tensor, torch.nn.Parameter) for value in values):
        raise RuntimeError("Raw live entry does not support tensor subclasses; use native module semantics")


def original_plan(tp):
    from ictc_o2_hp.uvu_o2 import UvuTensorProduct
    if (type(tp) is not UvuTensorProduct or
            getattr(getattr(tp, "forward", None), "__func__", None) is not UvuTensorProduct.forward or
            _has_hooks(tp)):
        raise RuntimeError("Raw live entry requires the unmodified TP implementation without module hooks")
    if (tp.normalization != "component" or tp.path_normalization != "none" or
            tp.m_max_in2 != tp.m_max):
        raise ValueError("Raw live normalization/angular-input domain has not been qualified")
    tensor_types(*(getattr(tp, name) for name in TP_BUFFERS))


def original_linear(linear):
    if (type(linear) is not torch.nn.Linear or
            getattr(getattr(linear, "forward", None), "__func__", None) is not torch.nn.Linear.forward or
            _has_hooks(linear)):
        raise RuntimeError("Raw live entry requires the unmodified Linear implementation without module hooks")
    if linear.bias is None:
        raise ValueError("Raw live affine domain requires the original bias")
    tensor_types(linear.weight, linear.bias)
    if not linear.weight.is_contiguous() or not linear.bias.is_contiguous():
        raise ValueError("Raw live noncontiguous affine parameters require separate qualification")
