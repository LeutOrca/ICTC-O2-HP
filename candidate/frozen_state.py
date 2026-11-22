"""Forward-owned tensor/plan references with host-only mutation guards."""
from dataclasses import dataclass, replace

from .contracts import TP_BUFFERS


TP_METADATA = (
    "channels", "channels_in2", "m_max", "m_max_in2", "parity", "parity_in2",
    "normalization", "path_normalization", "shared_weights", "internal_weights",
    "weight_numel", "input1_dimension", "input2_dimension", "representation_dimension",
    "_kernel_metadata_id", "entries_in1", "entries_in2", "entries_out", "paths",
)


def tensor_stamp(value):
    if value is None:
        return None
    if value.is_inference():
        raise ValueError("Guarded adjoints require tensors with version counters")
    return (id(value), value.data_ptr(), value.untyped_storage().data_ptr(),
            value.storage_offset(), tuple(value.shape), tuple(value.stride()),
            value.dtype, value.device, value.requires_grad, value._version)


def tp_metadata(tp):
    # The registered plan uses immutable tuples/NamedTuples, not GPU values.
    return tuple(tuple(tuple(item) for item in getattr(tp, name)) if name in (
        "entries_in1", "entries_in2", "entries_out", "paths") else getattr(tp, name)
        for name in TP_METADATA)


@dataclass(frozen=True)
class TPPlan:
    channels: int
    channels_in2: int
    shared_weights: bool
    internal_weights: bool
    weight_numel: int
    input1_dimension: int
    input2_dimension: int
    representation_dimension: int
    entries_in1: tuple
    entries_in2: tuple
    entries_out: tuple
    paths: tuple
    _path_info: object
    _coefficients: object
    _first_order: object
    _second_order: object

    @classmethod
    def capture(cls, tp):
        return cls(tp.channels, tp.channels_in2, tp.shared_weights, tp.internal_weights,
                   tp.weight_numel, tp.input1_dimension, tp.input2_dimension,
                   tp.representation_dimension, tuple(tp.entries_in1), tuple(tp.entries_in2),
                   tuple(tp.entries_out), tuple(tp.paths),
                   *(getattr(tp, name) for name in TP_BUFFERS))

    @property
    def buffers(self):
        return tuple(getattr(self, name) for name in TP_BUFFERS)

    def with_buffers(self, values):
        if len(values) != len(TP_BUFFERS):
            raise RuntimeError("Saved TP buffer count changed")
        return replace(self, **dict(zip(TP_BUFFERS, values)))


class ForwardState:
    """Reject ordinary parameter/buffer mutations and use forward-owned values.

    No clones or value hashes are taken. Autograd's saved-tensor checks cover
    in-place updates; identity/storage/metadata stamps also catch replacement.
    This does not make concurrent raw storage writes safe.
    """
    def __init__(self, tp, linear=None):
        self.tp = tp
        self.metadata = tp_metadata(tp)
        self.plan = TPPlan.capture(tp)
        self.buffer_stamps = tuple(tensor_stamp(value) for value in self.plan.buffers)
        self.linear = linear
        self.weight = None if linear is None else linear.weight
        self.bias = None if linear is None else linear.bias
        self.linear_widths = None if linear is None else (linear.in_features, linear.out_features)
        self.parameter_stamps = (tensor_stamp(self.weight), tensor_stamp(self.bias))

    def assert_current(self):
        if tp_metadata(self.tp) != self.metadata:
            raise RuntimeError("TP plan metadata changed after the forward state was captured")
        for name, before in zip(TP_BUFFERS, self.buffer_stamps):
            if tensor_stamp(getattr(self.tp, name)) != before:
                raise RuntimeError(f"TP buffer {name} changed between forward and backward")
        if self.linear is not None:
            if (self.linear.in_features, self.linear.out_features) != self.linear_widths:
                raise RuntimeError("Affine dimensions changed between forward and backward")
            now = (tensor_stamp(self.linear.weight), tensor_stamp(self.linear.bias))
            if now != self.parameter_stamps:
                raise RuntimeError("Affine weight/bias changed between forward and backward")
