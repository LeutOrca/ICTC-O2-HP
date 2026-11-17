# Bounded dynamic-E AOT inference, 2026-11-17

This increment adds compile_inference and FlatInference for native flat,
contiguous, two-dimensional CUDA FP32 inputs. Enable O2_HPX_FORWARD=preload
before importing the library. The original TensorProduct owns the canonical
weight, and the deployment callable reads that weight on every invocation.
Checkpoints and parameter ownership stay with the original module.

```python
import os
os.environ['O2_HPX_FORWARD'] = 'preload'
import torch
from ictc_o2_hp import TensorProduct, compile_inference, FlatInference
native = TensorProduct(32, 1, parity='canonical').cuda()
compile_inference(native, 'tp.pt2', max_edges=131072)
deployed = FlatInference(native, 'tp.pt2')
with torch.no_grad():
    output = deployed(left, right)
```

Build and verification must run on scheduler-managed compute nodes in this
workspace. Building needs a host C++ compiler and complete CUDA headers/libraries.
The study uses an isolated verified CUDA12.8 toolchain. A package is specific to
its Torch ABI and GPU target. The loader checks those fields, the equation and
package hashes, input shapes, dtype, device, contiguity, weight size and the
exported positive-E range. Select the target with CUDA_VISIBLE_DEVICES; its
process-local index is0. Inference uses torch.no_grad. The ordinary operator
remains available for training, other layouts and unsupported deployment inputs.

The preloaded forward now accepts E as a runtime kernel argument. torch.export
keeps E symbolic and weights as runtime Tensor inputs. aoti_compile_and_package
emits the native wrapper.so and cubin with package_cpp_only=False and CUDA graphs
disabled. Direct use of the packaged C++ runner avoids repeated Python pytree
handling. Its thread-safe runner remains enabled. No parameter is baked into
the package, and no precision or coefficient change is introduced.

A100 confirmation:24 screen cases, five independent same-GPU allocations,
30 warmups/30 samples per backend process, counterbalanced order. The fastest
of original native and accepted preload is chosen separately at each point.
Geometric mean improvements are1.264490x CUDA-event and1.210778x synchronized
wall versus that prior. Versus original native the figures are1.799713x and
1.696322x. No point regresses by more than3%. All backends retain two operands;
unused cotangents are excluded consistently. The full raw paired summary is
hpx-aoti-paired-screen-20261117.json.

Numerical gates:110842 verifies every actual screen output against frozen native
and independent FP64 equations, both seeds/scales, rotations/reflections and live
weight changes.110860 checks E1/37/257/513 in all12 representation families.
110857 passes all first/mixed second derivatives and20 energy/force-loss AdamW
steps including parameters and optimizer moments.110834 preserves reference-only
imports without the optional GPU runtime. Pair array110862, summary CPU110868.

A source-independent direct C++ load is separately verified on H100/Torch2.11
by312133 across multiple E and weight changes. Its single diagnostic run reports
28.842ms artifact loading and13.221s script-start-to-first-output; these are not
five-run cold-start claims. The package contains its wrapper.so and native cubin.

This confirms an incremental deployment improvement. The full240-point >=2x
objective against the fastest current same-equation external implementation
remains OPEN. This forward-only artifact does not qualify a compiled VJP,
complete MLIP force inference, or full-model training.
