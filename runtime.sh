export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 EQX_USE_TF32=0 EQX_USE_CUDA_GRAPH=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:16:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=9.0
runtime_root=/public_hw/home/phy_bjxiao/o2-benchmark/environments/torch211-ac7e624
prior=/public_hw/home/phy_bjxiao/o2-benchmark/runs/o2-hp-github-whole-20261121-r1-ad08623
export PYTHONPATH="$runtime_root:$prior/deps-211"
export PATH="$prior/build-tools/ninja-1.13.0.data/scripts:$PATH"
export CUDA_HOME=/public_hw/home/phy_bjxiao/o2-benchmark/runs/o2-hpx-opt-20261117-r1/toolchain/cuda_host_v3
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$runtime_root/nvidia/cuda_cupti/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="$PWD/cache/extensions" TRITON_CACHE_DIR="$PWD/cache/triton"
export XDG_CACHE_HOME="$PWD/cache/xdg"
py=/public_hw/home/phy_bjxiao/RNA-Au-database/envs/mace_gpu/bin/python
mkdir -p logs results/preflight "$XDG_CACHE_HOME/torch/kernels"
