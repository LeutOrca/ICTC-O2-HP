"""Host-metadata contracts for the frozen CUDA FP32 V=1 candidate.

No tensor values are inspected or cast. Layout copies preserve logical values
and are performed by the public wrappers, outside custom autograd functions.
"""
import torch


TP_BUFFERS = ("_path_info", "_coefficients", "_first_order", "_second_order")


def tensor_contract(name, value, *, device=None, dtype=torch.float32, ndim=2):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.layout != torch.strided or value.ndim != ndim:
        raise ValueError(f"{name} must be a dense {ndim}D tensor")
    if value.is_neg() or value.is_conj():
        raise ValueError(f"{name} must materialize lazy negative/conjugate views before entry")
    if value.dtype != dtype or value.device.type != "cuda":
        raise TypeError(f"{name} must use CUDA {dtype}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} is on a different device")


def check_tp(tp, device):
    from ictc_o2_hp.uvu_o2 import UvuTensorProduct
    if not isinstance(tp, UvuTensorProduct):
        raise TypeError("Candidate requires the frozen UvuTensorProduct plan type")
    if tp.channels_in2 != 1 or tp.shared_weights or tp.internal_weights:
        raise ValueError("Candidate requires V=1 external unshared weights")
    if tp.representation_dimension != tp.input1_dimension:
        raise ValueError("Candidate requires identical first-input/output widths")
    if tuple(tp.entries_out) != tuple(tp.entries_in1):
        raise ValueError("Candidate requires the same first-input/output entries")
    if tp.weight_numel != len(tp.paths) * tp.channels:
        raise ValueError("TP weight count does not match its V=1 paths")
    records = len(tp.paths) + len(tp.entries_out) + len(tp.entries_in1) + len(tp.entries_in2)
    shapes = ((records, 7), (len(tp.paths), 2, 2, 2), (len(tp.paths),), (len(tp.paths),))
    dtypes = (torch.int32, torch.float32, torch.int32, torch.int32)
    for name, shape, dtype in zip(TP_BUFFERS, shapes, dtypes):
        value = getattr(tp, name)
        tensor_contract("tp." + name, value, device=device, dtype=dtype, ndim=len(shape))
        if tuple(value.shape) != shape or not value.is_contiguous():
            raise ValueError(f"tp.{name} has an invalid shape or layout")
        if value.requires_grad:
            raise ValueError("Candidate requires frozen TP plan buffers")


def check_features(tp, x, y):
    tensor_contract("left", x)
    tensor_contract("angular", y, device=x.device)
    check_tp(tp, x.device)
    if x.shape[1] != tp.input1_dimension:
        raise ValueError("left width does not match the TP")
    if tuple(y.shape) != (len(x), tp.input2_dimension):
        raise ValueError("angular batch or width does not match left/TP")


def check_linear(linear, hidden, *, rows=None, device=None, outputs=None):
    if not isinstance(linear, torch.nn.Linear):
        raise TypeError("Radial output must be an nn.Linear")
    tensor_contract("radial_hidden", hidden, device=device)
    if hidden.shape[1] != linear.in_features or (rows is not None and len(hidden) != rows):
        raise ValueError("radial_hidden has an invalid batch or hidden width")
    tensor_contract("linear.weight", linear.weight, device=hidden.device)
    if tuple(linear.weight.shape) != (linear.out_features, linear.in_features):
        raise ValueError("Linear weight shape does not match its declared widths")
    if outputs is not None and linear.out_features != outputs:
        raise ValueError("Linear output width does not match TP weights")
    if linear.bias is not None:
        tensor_contract("linear.bias", linear.bias, device=hidden.device, ndim=1)
        if tuple(linear.bias.shape) != (linear.out_features,):
            raise ValueError("Linear bias has an invalid shape")
    if any(value.requires_grad for value in linear.parameters()):
        raise ValueError("Candidate supports frozen affine parameters only")


def prepare_uvu(tp, x, y, w):
    check_features(tp, x, y)
    tensor_contract("external_weight", w, device=x.device)
    if tuple(w.shape) != (len(x), tp.weight_numel):
        raise ValueError("External weights have an invalid batch or width")
    return x.contiguous(), y.contiguous(), w.contiguous()


def prepare_radial(tp, linear, x, y, hidden):
    check_features(tp, x, y)
    check_linear(linear, hidden, rows=len(x), device=x.device, outputs=tp.weight_numel)
    return x.contiguous(), y.contiguous(), hidden.contiguous()


def prepare_cotangent(value, *, rows, width, device):
    tensor_contract("output_cotangent", value, device=device)
    if tuple(value.shape) != (rows, width):
        raise ValueError("Output cotangent shape does not match the saved forward")
    return value.contiguous()
