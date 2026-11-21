export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 EQX_USE_TF32=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:16:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=9.0
remote_runs=/public_hw/home/phy_bjxiao/o2-benchmark/runs
export HPX_MACE_ENTRY="$remote_runs/mace-ictc-o2-hp-gbu2-20261120-r1-78231f3"
export O2_MACE_SOURCE="$HPX_MACE_ENTRY/frozen"
export O2_HPX_SOURCE="$PWD/c591b16"
export PYTHONPATH="$HPX_MACE_ENTRY/compatibility"
export CUDA_HOME="$remote_runs/o2-hpx-opt-20261117-r1/toolchain/cuda_host_v3"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export TRITON_CACHE_DIR="$PWD/cache/triton" TORCH_EXTENSIONS_DIR="$PWD/cache/extensions"
export XDG_CACHE_HOME="$PWD/cache/xdg"
py=/public_hw/home/phy_bjxiao/RNA-Au-database/envs/mace_gpu/bin/python
export PATH="${py%/python}:$PATH"
export LD_LIBRARY_PATH="${py%/bin/python}/lib/python3.11/site-packages/nvidia/cuda_cupti/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p results/preflight logs "$XDG_CACHE_HOME/torch/kernels"
