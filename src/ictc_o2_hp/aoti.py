"""Experimental bounded dynamic-E deployment for the native flat FP32 product.

Build on a GPU compute node. The returned callable reads the existing module's
canonical weight on every call, so state_dict and parameter ownership stay with
that module. This interface supports inference under torch.no_grad only.
"""
import hashlib
import json
from pathlib import Path
import torch
from . import triton_o2 as K


def _equation(module):
    return hashlib.sha256(repr((module._triton_keys,module._triton_paths)).encode()).hexdigest()


class _Functional(torch.nn.Module):
    def __init__(self,module):
        super().__init__()
        self.register_buffer('info',module._weight_path_info,persistent=False)
        self.register_buffer('coefficients',module._weight_path_coefficients,persistent=False)
        self.mid=module._triton_metadata_id
    def forward(self,left,right,weight):
        return K._o2_forward_op(left,right,weight,self.info,self.coefficients,self.mid)


def compile_inference(module,path,*,max_edges=131072):
    """Compile one shape family, keeping E dynamic and weights as Tensor inputs."""
    if not K._HPX_PRELOAD:raise ValueError('Enable the preloaded dynamic-E source before import')
    if max_edges<2:raise ValueError('max_edges must be >=2')
    weight=module.weight
    if weight.device.type!='cuda' or weight.dtype!=torch.float32:raise ValueError('CUDA FP32 required')
    if weight.device.index!=0 or torch.cuda.current_device()!=0:raise ValueError('Select the target GPU with CUDA_VISIBLE_DEVICES')
    if module._triton_keys[0][4]!=0:raise ValueError('This prototype supports native flat storage')
    path=Path(path)
    example=torch.zeros(2,module.representation_dimension,device=weight.device)
    edges=torch.export.Dim('edges',min=1,max=max_edges)
    with torch.no_grad():
        exported=torch.export.export(_Functional(module),(example,example.clone(),weight.detach()),
            dynamic_shapes=({0:edges},{0:edges},None),strict=True)
    from torch._inductor import aoti_compile_and_package
    aoti_compile_and_package(exported,package_path=str(path),inductor_configs={
        'triton.cudagraphs':False,'aot_inductor.package_cpp_only':False})
    metadata=dict(equation_sha256=_equation(module),width=module.representation_dimension,
        weight_count=weight.numel(),max_edges=max_edges,torch=torch.__version__,
        capability=list(torch.cuda.get_device_capability()),
        package_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),runtime_weights=True)
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


class FlatInference:
    """Two-input callable sharing the original module and its live weights."""
    def __init__(self,module,path):
        weight=module.weight
        if weight.device!=torch.device('cuda',0) or weight.dtype!=torch.float32 or torch.cuda.current_device()!=0:
            raise ValueError('Load with CUDA FP32 weights on the selected cuda:0 device')
        path=Path(path)
        metadata=json.loads(path.with_suffix('.json').read_text())
        if metadata['equation_sha256']!=_equation(module):raise ValueError('Artifact equation differs')
        if metadata['torch']!=torch.__version__:raise ValueError('Artifact Torch ABI differs')
        if metadata['capability']!=list(torch.cuda.get_device_capability()):raise ValueError('Artifact GPU target differs')
        if metadata['package_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('Artifact hash differs')
        try:self.loader=torch._C._aoti.AOTIModelPackageLoader(str(path),'model',False,1,0)
        except TypeError:self.loader=torch._C._aoti.AOTIModelPackageLoader(str(path),'model',False)
        self.module=module
        self.width=metadata['width']
        self.max_edges=metadata['max_edges']
        self.weight_count=metadata['weight_count']
        self.device=module.weight.device
    def __call__(self,left,right):
        weight=self.module.weight
        if torch.is_grad_enabled():raise RuntimeError('Use torch.no_grad for this inference-only artifact')
        if left.ndim!=2 or left.shape!=right.shape or left.shape[1]!=self.width:raise ValueError('Input shape differs')
        if not 1<=left.shape[0]<=self.max_edges:raise ValueError('E is outside the exported range')
        if not left.is_contiguous() or not right.is_contiguous() or not weight.is_contiguous():raise ValueError('Contiguous tensors required')
        if left.device!=self.device or right.device!=self.device or weight.device!=self.device:raise ValueError('Device differs')
        if left.dtype!=torch.float32 or right.dtype!=torch.float32 or weight.dtype!=torch.float32:raise ValueError('FP32 tensors required')
        if weight.ndim!=1 or weight.numel()!=self.weight_count:raise ValueError('Weight shape differs')
        return self.loader.boxed_run([left,right,weight])[0]
