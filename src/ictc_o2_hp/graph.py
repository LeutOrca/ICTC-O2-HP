"""Exact O(2) edge-local products consumed directly by node accumulation."""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from .triton_o2 import _read_canonical,_preloaded_path

@dataclass(frozen=True)
class PreparedGraph:
    nodes:int
    edges:int
    sources:torch.Tensor
    destinations:torch.Tensor
    max_degree:int

def prepare_graph(sources,destinations,nodes):
    if nodes<1 or int(nodes)!=nodes:
        raise ValueError('nodes must be a positive integer')
    if sources.ndim!=1 or sources.shape!=destinations.shape:
        raise ValueError('one source/destination per directed edge required')
    if sources.dtype!=torch.int64 or destinations.dtype!=torch.int64:
        raise ValueError('int64 graph indices required')
    if sources.device.type!='cuda' or destinations.device!=sources.device:
        raise ValueError('CUDA graph indices on one device required')
    if sources.numel() and (min(int(sources.min()),int(destinations.min()))<0 or
                            max(int(sources.max()),int(destinations.max()))>=nodes):
        raise ValueError('graph index outside node range')
    degree=torch.bincount(destinations,minlength=nodes)
    return PreparedGraph(int(nodes),sources.numel(),sources.clone(memory_format=torch.contiguous_format),
                         destinations.clone(memory_format=torch.contiguous_format),int(degree.max()))

@triton.jit
def _add_raw(output,node,channel,a,b,mask,KEY:tl.constexpr,WIDTH:tl.constexpr,
             C:tl.constexpr,KEYS:tl.constexpr,FULL:tl.constexpr):
    start:tl.constexpr=KEYS.value[KEY][0]
    dim:tl.constexpr=KEYS.value[KEY][2]
    frequency:tl.constexpr=KEYS.value[KEY][3]
    offset=node.to(tl.int64)*WIDTH+start+channel*dim
    if dim==1:
        tl.atomic_add(output+offset,a,mask,sem='relaxed')
    else:
        if FULL and frequency:
            odd=channel>=C
            a,b=tl.where(odd,b,a),tl.where(odd,-a,b)
        tl.atomic_add(output+offset,a,mask,sem='relaxed')
        tl.atomic_add(output+offset+1,b,mask,sem='relaxed')

@triton.jit(do_not_specialize=['E'])
def _edge_product(left,right,weight,sources,destinations,output,E,
                  WIDTH:tl.constexpr,C:tl.constexpr,KEYS:tl.constexpr,PATHS:tl.constexpr,
                  FULL:tl.constexpr,MULTIPLICITY:tl.constexpr,BLOCK:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    position,channel=offsets//MULTIPLICITY,offsets%MULTIPLICITY
    mask=position<E
    source=tl.load(sources+position,mask,0)
    edge=position
    node=tl.load(destinations+position,mask,0)
    first0,first1,second0,second1=(),(),(),()
    for key in tl.static_range(len(KEYS.value)):
        a0,a1=_read_canonical(left,source,channel%KEYS.value[key][1],mask,key,WIDTH,C,KEYS,FULL)
        b0,b1=_read_canonical(right,edge,channel%KEYS.value[key][1],mask,key,WIDTH,C,KEYS,FULL)
        first0+=(a0,);first1+=(a1,);second0+=(b0,);second1+=(b1,)
    for key in tl.static_range(len(KEYS.value)):
        active=mask&(channel<KEYS.value[key][1])
        value0=tl.full((BLOCK,),0.,tl.float32)
        value1=tl.full((BLOCK,),0.,tl.float32)
        for p in tl.static_range(len(PATHS.value[key])):
            a,b=_preloaded_path(first0,first1,second0,second1,weight,channel,active,PATHS,key,p)
            value0+=a;value1+=b
        _add_raw(output,node,channel,value0,value1,active,key,WIDTH,C,KEYS,FULL)

class GraphInference:
    def __init__(self,module,graph,*,channel_tile=8):
        if channel_tile not in (4,8,16,32):
            raise ValueError('channel_tile must be4,8,16,32')
        if module.weight.device!=graph.sources.device or any(k[4]!=0 for k in module._triton_keys):
            raise ValueError('native flat module and graph on one device required')
        self.module,self.graph=module,graph
        self.width,self.channels=module.representation_dimension,module.channels
        self.keys=module._triton_keys
        self.paths=tuple(tuple(p for p in module._triton_paths if p[2]==key) for key in range(len(self.keys)))
        self.full=module.parity=='full'
        self.block=channel_tile*16
        self.multiplicity = max(k[1] for k in self.keys)
        self.grid = (triton.cdiv(graph.edges * self.multiplicity, self.block), 1, 1)
        self.constants = (self.width, self.channels, self.keys, self.paths, self.full, self.multiplicity, self.block)
        self.runners = {}
    def __call__(self,left,right):
        if torch.is_grad_enabled():raise RuntimeError('Use torch.no_grad')
        if left.shape!=(self.graph.nodes,self.width) or right.shape!=(self.graph.edges,self.width):
            raise ValueError('node/edge feature shape differs')
        weight=self.module.weight
        for value in (left,right,weight):
            if value.device!=self.graph.sources.device or value.dtype!=torch.float32 or not value.is_contiguous():
                raise ValueError('contiguous CUDA FP32 tensors required')
        output=left.new_zeros((self.graph.nodes,self.width))
        if self.graph.edges == 0:
            return output
        # A prepared graph fixes sizes/strides. The Triton pointer specialization
        # additionally depends on alignment, including live replacement weights.
        aligned = tuple(t.data_ptr() % 16 == 0 for t in (left, right, weight))
        args = (left, right, weight, self.graph.sources, self.graph.destinations,
                output, self.graph.edges, *self.constants)
        with torch.cuda.device(left.device):
            runner = self.runners.get(aligned)
            if runner is None:
                self.last_kernel = _edge_product[self.grid](*args, num_warps=4)
                self.runners[aligned] = self.last_kernel[self.grid]
            else:
                runner(*args)
        return output
