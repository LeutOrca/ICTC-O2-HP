"""FP32 bounded angular bilinear contraction with recursively differentiable VJPs."""
import torch
import triton
import triton.language as tl

@triton.jit
def _add(a,b):
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[a,b],dtype=tl.float32,is_pure=True,pack=1)

@triton.jit
def _piece(K,A,B,rows,cols,start,acc,
           OA:tl.constexpr,LA:tl.constexpr,LB:tl.constexpr,H:tl.constexpr,N:tl.constexpr,
           K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,
           A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,
           B0:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,LANE:tl.constexpr):
    q=start+LANE*8+tl.arange(0,16).to(tl.int64)
    valid=(tl.arange(0,16)<8)&(q<LA*LB)
    ia=q//LB;ib=q%LB
    coeff=tl.load(K+rows[:,None]*K0+ia[None,:]*K1+ib[None,:]*K2,(rows[:,None]<OA)&valid[None,:],0)
    n=cols//H;h=cols%H
    aa=tl.load(A+n[None,:]*A0+ia[:,None]*A1+h[None,:]*A2,valid[:,None]&(n[None,:]<N),0)
    bb=tl.load(B+n[None,:]*B0+ib[:,None]*B1+h[None,:]*B2,valid[:,None]&(n[None,:]<N),0)
    outer=tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[aa,bb],dtype=tl.float32,is_pure=True,pack=1)
    return tl.dot(coeff,outer,acc,input_precision='ieee')

@triton.jit
def _bilinear(K,A,B,Y,OA:tl.constexpr,LA:tl.constexpr,LB:tl.constexpr,H:tl.constexpr,N:tl.constexpr,
              K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,
              B0:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    rows=tl.program_id(0).to(tl.int64)*BM+tl.arange(0,BM).to(tl.int64)
    cols=tl.program_id(1).to(tl.int64)*BN+tl.arange(0,BN).to(tl.int64)
    c0=tl.full((BM,BN),0,tl.float32);c1=tl.full((BM,BN),0,tl.float32)
    c2=tl.full((BM,BN),0,tl.float32);c3=tl.full((BM,BN),0,tl.float32)
    for block in range(tl.cdiv(LA*LB,32)):
        start=tl.cast(block,tl.int64)*32
        c0=_piece(K,A,B,rows,cols,start,c0,OA,LA,LB,H,N,K0,K1,K2,A0,A1,A2,B0,B1,B2,0)
        c1=_piece(K,A,B,rows,cols,start,c1,OA,LA,LB,H,N,K0,K1,K2,A0,A1,A2,B0,B1,B2,1)
        c2=_piece(K,A,B,rows,cols,start,c2,OA,LA,LB,H,N,K0,K1,K2,A0,A1,A2,B0,B1,B2,2)
        c3=_piece(K,A,B,rows,cols,start,c3,OA,LA,LB,H,N,K0,K1,K2,A0,A1,A2,B0,B1,B2,3)
    value=_add(_add(c0,c1),_add(c2,c3))
    n=cols//H;h=cols%H
    tl.store(Y+(n[None,:]*OA+rows[:,None])*H+h[None,:],value,(rows[:,None]<OA)&(n[None,:]<N))

class Bilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,k,a,b):
        if k.requires_grad:raise ValueError('Angular coefficients are the original fixed buffer; active model weights are not frozen')
        if k.ndim!=3 or a.ndim!=3 or b.ndim!=3 or a.shape[0]!=b.shape[0] or a.shape[2]!=b.shape[2] or k.shape[1:]!=(a.shape[1],b.shape[1]):raise ValueError('Incompatible angular bilinear shapes')
        if not all(v.is_cuda and v.dtype==torch.float32 and v.device==a.device for v in (k,a,b)):raise TypeError('CUDA FP32 operands required')
        ctx.save_for_backward(k,a,b)
        n,_,h=a.shape;oa,la,lb=k.shape
        y=a.new_empty((n,oa,h))
        if n and h:
            _bilinear[(triton.cdiv(oa,32),triton.cdiv(n*h,32))](k,a,b,y,oa,la,lb,h,n,*k.stride(),*a.stride(),*b.stride(),32,32,num_warps=4)
        return y
    @staticmethod
    def backward(ctx,g):
        k,a,b=ctx.saved_tensors
        da=Bilinear.apply(k.permute(1,0,2),g,b) if ctx.needs_input_grad[1] else None
        db=Bilinear.apply(k.permute(2,0,1),g,a) if ctx.needs_input_grad[2] else None
        return None,da,db

def product(k,a,b,mode='direct'):
    if mode!='direct':raise ValueError('Only the original direct canonical bilinear is supported')
    return Bilinear.apply(k,a,b)
