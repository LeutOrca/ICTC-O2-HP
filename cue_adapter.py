"""Original audited O2 descriptor, using FP32 math and native layouts."""
from importlib.metadata import version
import torch


class CueO2(torch.nn.Module):
    def __init__(self, native):
        super().__init__()
        import cuequivariance as cue
        import cuequivariance_torch as cuet
        descriptor = cue.SegmentedTensorProduct.from_subscripts("u,iu,ju,ku+ijk")
        indices = {}
        for block in native.blocks:
            for copy in range(block.parity_copies):
                index = descriptor.add_segment(1, (block.irrep_dimension, native.channels))
                descriptor.add_segment(2, (block.irrep_dimension, native.channels))
                descriptor.add_segment(3, (block.irrep_dimension, native.channels))
                indices[block.key, copy] = index
        for a, b, out in native.paths:
            coefficient = native.o2.o2_cg_tensor(a, b, out, dtype=torch.float32)
            coefficient *= native.block_by_key[out].irrep_dimension**.5 / coefficient.norm()
            for copy in range(native.block_by_key[out].parity_copies):
                w = descriptor.add_segment(0, (native.channels,))
                descriptor.add_path(w, indices[a, copy if a[0] == "freq" else 0],
                    indices[b, copy if b[0] == "freq" else 0], indices[out, copy], c=coefficient.numpy())
        if descriptor.operands[0].size != native.weight.numel():
            raise ValueError("Descriptor changed the path weights")
        self.weight = torch.nn.Parameter(native.weight.detach().clone(), requires_grad=False)
        if version("cuequivariance-torch") != "0.13.0":
            raise RuntimeError("This contender is pinned to cuEq0.13.0")
        self.operation = cuet.SegmentedPolynomial(cue.SegmentedPolynomial.eval_last_operand(descriptor),
                                                method="uniform_1d", math_dtype=torch.float32)
        if self.operation.method != "uniform_1d":
            raise RuntimeError("Unexpected cuEq fallback")

    def forward(self, left, right):
        return self.operation([self.weight.unsqueeze(0), left, right])[0]
