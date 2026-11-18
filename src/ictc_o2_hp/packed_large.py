"""Paired loads and stores for aligned native-flat FP32 components."""
import triton
import triton.language as tl
from .triton_o2 import _preloaded_path

@triton.jit
def _read_canonical(ptr,edge,ch,valid,K:tl.constexpr,WIDTH:tl.constexpr,C:tl.constexpr,
                    KEYS:tl.constexpr,FULL:tl.constexpr):
    dim:tl.constexpr=KEYS.value[K][2]
    offset=edge.to(tl.int64)*WIDTH+KEYS.value[K][0]+ch*dim
    if dim==1:
        a=tl.load(ptr+offset,valid,0.)
        return a,a*0.
    else:
        word=tl.load((ptr+offset).to(tl.pointer_type(tl.uint64)),valid,0)
        a=word.to(tl.uint32).to(tl.float32,bitcast=True)
        b=(word>>32).to(tl.uint32).to(tl.float32,bitcast=True)
        if FULL:
            return tl.where(ch>=C,-b,a),tl.where(ch>=C,a,b)
        else:
            return a,b

@triton.jit
def _write_raw(ptr,edge,ch,a,b,valid,K:tl.constexpr,WIDTH:tl.constexpr,C:tl.constexpr,
               KEYS:tl.constexpr,FULL:tl.constexpr):
    dim:tl.constexpr=KEYS.value[K][2]
    offset=edge.to(tl.int64)*WIDTH+KEYS.value[K][0]+ch*dim
    if dim==1:
        tl.store(ptr+offset,a,valid)
    else:
        if FULL:
            a,b=tl.where(ch>=C,b,a),tl.where(ch>=C,-a,b)
        word=a.to(tl.uint32,bitcast=True).to(tl.uint64)|(b.to(tl.uint32,bitcast=True).to(tl.uint64)<<32)
        tl.store((ptr+offset).to(tl.pointer_type(tl.uint64)),word,valid)
@triton.jit(do_not_specialize=["edges"])
def cooperative(left,right,weight,edges, WIDTH:tl.constexpr,C:tl.constexpr,
                KEYS:tl.constexpr,PATHS:tl.constexpr,FULL:tl.constexpr,TILE:tl.constexpr):
    copies:tl.constexpr=1+FULL
    groups:tl.constexpr=triton.cdiv(C,TILE)
    pid=tl.program_id(0).to(tl.int64)
    lane=tl.arange(0,TILE*copies)
    edge=pid//groups
    base=pid%groups*TILE+lane%TILE
    channel=base+lane//TILE*C
    valid=(edge<edges)&(base<C)
    x0,x1,y0,y1=(),(),(),()
    for k in tl.static_range(len(KEYS.value)):
        ch=channel%KEYS.value[k][1]
        a,b=_read_canonical(left,edge,ch,valid,k,WIDTH,C,KEYS,FULL)
        u,v=_read_canonical(right,edge,ch,valid,k,WIDTH,C,KEYS,FULL)
        x0+=(a,);x1+=(b,);y0+=(u,);y1+=(v,)
    values0,values1=(),()
    for k in tl.static_range(len(KEYS.value)):
        active=valid&(channel<KEYS.value[k][1])
        a=tl.full((TILE*copies,),0.,tl.float32)
        b=tl.full((TILE*copies,),0.,tl.float32)
        for p in tl.static_range(len(PATHS.value[k])):
            u,v=_preloaded_path(x0,x1,y0,y1,weight,channel,active,PATHS,k,p)
            a+=u;b+=v
        values0+=(a,);values1+=(b,)
    if FULL:
        tl.debug_barrier()
    for k in tl.static_range(len(KEYS.value)):
        _write_raw(left,edge,channel,values0[k],values1[k],valid&(channel<KEYS.value[k][1]),k,WIDTH,C,KEYS,FULL)
