"""Exact descriptor for the existing O2 paths; no representation inference."""
import os
from importlib.metadata import version
import torch
from torch import nn

class CueO2(nn.Module):
    def __init__(self,native):
        super().__init__()
        import cuequivariance as cue
        import cuequivariance_torch as cuet
        descriptor=cue.SegmentedTensorProduct.from_subscripts('u,iu,ju,ku+ijk')
        indices={}
        for block in native.blocks:
            for copy in range(block.parity_copies):
                index=descriptor.add_segment(1,(block.irrep_dimension,native.channels))
                descriptor.add_segment(2,(block.irrep_dimension,native.channels))
                descriptor.add_segment(3,(block.irrep_dimension,native.channels))
                indices[block.key,copy]=index
        for k1,k2,k3 in native.paths:
            coefficient=native.o2.o2_cg_tensor(k1,k2,k3,dtype=torch.float64)
            coefficient=coefficient*(native.block_by_key[k3].irrep_dimension**.5/coefficient.norm())
            for copy in range(native.block_by_key[k3].parity_copies):
                w=descriptor.add_segment(0,(native.channels,))
                descriptor.add_path(w,indices[k1,copy if k1[0]=='freq' else 0],
                    indices[k2,copy if k2[0]=='freq' else 0],indices[k3,copy],c=coefficient.numpy())
        assert descriptor.operands[0].size==native.weight.numel()
        self.weight=nn.Parameter(native.weight.detach().clone())
        self.polynomial=tuple(map(int,version('cuequivariance-torch').split('.')[:2])) >= (0,13)
        if self.polynomial:
            method=os.environ.get('CUE_METHOD','uniform_1d')
            self.operation=cuet.SegmentedPolynomial(cue.SegmentedPolynomial.eval_last_operand(descriptor),
                                                   method=method,math_dtype=torch.float32)
            if self.operation.method!=method:raise RuntimeError('Unexpected cuEq fallback')
        else:
            self.operation=cuet.TensorProduct(descriptor,device=torch.device('cuda'),
                                             math_dtype=torch.float32,use_fallback=False)
    def forward(self,left,right):
        if self.polynomial:return self.operation([self.weight.unsqueeze(0),left,right])[0]
        return self.operation(self.weight.unsqueeze(0),left,right)
