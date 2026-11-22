"""Conservative, metadata-only coverage and native fallback.

The table records the old benchmark's structural coverage, not a numerical
guarantee for arbitrary FP32 values. This safety candidate is still untested.
No online tuning, tensor-value reads, dtype changes or CUDA synchronization.
"""
from dataclasses import asdict, dataclass
import os

import torch
import triton


PROFILE = "h100-torch211-triton360-full-c64-h64-e16384-v1"
BLOCK_ROWS = 4096
# Exact (angular order, weight width) pairs; do not extrapolate a K threshold.
COVERAGE = {
    4: (15616, "16x1b@128"),
    5: (23296, "16x1b@128"),
    6: (32512, "16x1b@128"),
    7: (43264, "16x1b@128"),
    8: (55552, "-8x1b"),
}


@dataclass(frozen=True)
class Decision:
    route: str
    mode: str
    configuration: str | None
    reasons: tuple
    execution: str
    matmul_rows: int | None = None
    profile: str = PROFILE
    candidate_validated: bool = False

    def metadata(self):
        return asdict(self)


def policy_mode(request):
    if request in (None, "shape-v2", "auto"):
        return "auto"
    if request in ("native", "required"):
        return request
    raise ValueError("Use auto/native/required (shape-v2 aliases auto); raw sum overrides are not a guarded policy")


def require_fp32_policy():
    # Reject unsafe precision instead of changing caller-wide settings or
    # falling back to a native operation that could silently use TF32/AMP.
    if torch.is_autocast_enabled():
        raise RuntimeError("Candidate and native fallback require CUDA autocast off")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("Candidate and native fallback require TF32 off")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("Set float32 matmul precision to highest before using this candidate")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") == "1":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE=1 violates the frozen FP32 contract")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise RuntimeError("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 violates the frozen FP32 contract")


def _runtime_reasons(sample):
    reasons = []
    if str(torch.__version__) != "2.11.0+cu128" or torch.version.cuda != "12.8":
        reasons.append("torch/CUDA runtime outside frozen coverage")
    if triton.__version__ != "3.6.0":
        reasons.append("Triton version outside frozen coverage")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":16:8":
        reasons.append("native reference workspace policy outside frozen coverage")
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and getattr(compiler, "is_compiling", lambda: False)():
        reasons.append("compiler tracing outside eager benchmark coverage")
    if torch.is_inference_mode_enabled():
        reasons.append("inference_mode outside the guarded-autograd benchmark")
    if reasons:
        return reasons
    # Match HPX's eager-only boundary before any raw storage/device access.
    from torch.fx.experimental.proxy_tensor import get_proxy_mode
    from torch._subclasses.fake_tensor import is_fake
    if torch._C._are_functorch_transforms_active():
        reasons.append("functorch transforms require native autograd")
    if get_proxy_mode() is not None or is_fake(sample):
        reasons.append("proxy/fake tracing requires native autograd")
    if reasons:
        return reasons
    properties = torch.cuda.get_device_properties(sample.device)
    if (properties.major, properties.minor) != (9, 0) or properties.name != "NVIDIA H100 80GB HBM3":
        reasons.append("device outside exact H100 coverage")
    with torch.cuda.device(sample.device):
        if torch.cuda.is_current_stream_capturing():
            reasons.append("CUDA Graph capture outside benchmark coverage")
    return reasons


def _tp_reasons(tp, rows):
    from ictc_o2_hp.uvu_o2 import UvuTensorProduct
    reasons = []
    if type(tp) is not UvuTensorProduct or getattr(tp.forward, "__func__", None) is not UvuTensorProduct.forward:
        reasons.append("custom TP forward requires native module semantics")
    if _has_hooks(tp):
        reasons.append("TP/module hooks require native module semantics")
    coverage = COVERAGE.get(tp.m_max)
    if tp.channels != 64 or tp.channels_in2 != 1 or rows != 16384:
        reasons.append("requires C64/V1/E16384")
    if tp.parity != "full" or tp.parity_in2 != "full" or tp.m_max_in2 != tp.m_max:
        reasons.append("parity or angular-input cutoff outside full-parity coverage")
    if tp.normalization != "component" or tp.path_normalization != "none":
        reasons.append("normalization outside frozen equation")
    if coverage is None or tp.weight_numel != coverage[0]:
        reasons.append("angular order/weight width not in the exact coverage table")
    if tp.input1_dimension != 64 * (4 * tp.m_max + 2) or tp.input2_dimension != 4 * tp.m_max + 2:
        reasons.append("representation widths outside frozen coverage")
    return reasons


def _has_hooks(module):
    local = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
    global_names = ("_global_forward_hooks", "_global_forward_pre_hooks",
                    "_global_backward_hooks", "_global_backward_pre_hooks")
    return getattr(module, "_compiled_call_impl", None) is not None or any(
        getattr(module, name, {}) for name in local) or any(
        getattr(torch.nn.modules.module, name, {}) for name in global_names)


def assert_reduction_context(sample):
    """An active custom backward must reject context drift, never switch route."""
    require_fp32_policy()
    reasons = _runtime_reasons(sample)
    if reasons:
        raise RuntimeError("Reduction context changed after forward: " + "; ".join(reasons))


def _finish(mode, execution, configuration, reasons, matmul_rows=None):
    if mode == "native":
        return Decision("native", mode, None, ("native policy explicitly requested",), execution)
    if reasons:
        if mode == "required":
            raise RuntimeError("Own reduction unavailable: " + "; ".join(reasons))
        return Decision("native", mode, None, tuple(reasons), execution)
    return Decision("ordered", mode, configuration, (), execution, matmul_rows)


def choose_radial(tp, linear, hidden, request="auto", *, execution="unfused", features=()):
    mode = policy_mode(request)
    require_fp32_policy()
    if execution not in ("unfused", "streamed"):
        raise ValueError("Unknown radial execution boundary")
    if mode == "native":
        return _finish(mode, execution, None, [])
    reasons = ["TP context required"] if tp is None else _tp_reasons(tp, len(hidden))
    if hidden.shape[1] != 64 or linear.in_features != 64 or linear.bias is None:
        reasons.append("requires the frozen H64 biased affine")
    if not hidden.is_contiguous() or not linear.weight.is_contiguous() or (
            linear.bias is not None and not linear.bias.is_contiguous()):
        reasons.append("affine strides outside frozen coverage")
    if (type(linear) is not torch.nn.Linear
            or getattr(linear.forward, "__func__", None) is not torch.nn.Linear.forward
            or _has_hooks(linear)):
        reasons.append("custom Linear/forward-backward hooks require native module semantics")
    values = [hidden, linear.weight, linear.bias, *features]
    if not all(value.is_contiguous() for value in features):
        reasons.append("feature strides outside canonicalized coverage")
    if tp is not None:
        values += [tp._path_info, tp._coefficients, tp._first_order, tp._second_order]
    if any(value is not None and value.is_inference() for value in values):
        reasons.append("guarded adjoints require tensor version counters")
    if any(value is not None and type(value) not in (torch.Tensor, torch.nn.Parameter) for value in values):
        reasons.append("tensor subclasses require native dispatch semantics")
    # Reject cheap structural misses before touching device properties.
    if not reasons:
        reasons.extend(_runtime_reasons(hidden))
    configuration = None if tp is None or tp.m_max not in COVERAGE else COVERAGE[tp.m_max][1]
    rows = BLOCK_ROWS if execution == "streamed" else len(hidden)
    return _finish(mode, execution, configuration, reasons, rows)


def choose_uvu(tp, x, y, w, request="auto"):
    mode = policy_mode(request)
    require_fp32_policy()
    if mode == "native":
        return _finish(mode, "uvu", None, [])
    reasons = _tp_reasons(tp, len(x))
    if tp.m_max not in (6, 7, 8):
        reasons.append("standalone bounded UVU coverage is L6..8")
    values = [x, y, w, tp._path_info, tp._coefficients, tp._first_order, tp._second_order]
    if any(value.is_inference() for value in values):
        reasons.append("guarded adjoints require tensor version counters")
    if any(type(value) not in (torch.Tensor, torch.nn.Parameter) for value in values):
        reasons.append("tensor subclasses require native dispatch semantics")
    if not all(value.is_contiguous() for value in (x, y, w)):
        reasons.append("UVU strides outside frozen coverage")
    if not reasons:
        reasons.extend(_runtime_reasons(x))
    return _finish(mode, "uvu", None, reasons)
