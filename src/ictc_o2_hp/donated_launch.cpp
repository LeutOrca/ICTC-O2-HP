#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <dlfcn.h>

struct Driver {
  void* library;
  decltype(&cuModuleLoad) load;
  decltype(&cuModuleGetFunction) function;
  decltype(&cuLaunchKernel) launch;
  decltype(&cuModuleUnload) unload;
  decltype(&cuGetErrorString) error_string;
  Driver() {
    library=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
    TORCH_CHECK(library,"Cannot load CUDA driver");
    load=reinterpret_cast<decltype(load)>(dlsym(library,"cuModuleLoad"));
    function=reinterpret_cast<decltype(function)>(dlsym(library,"cuModuleGetFunction"));
    launch=reinterpret_cast<decltype(launch)>(dlsym(library,"cuLaunchKernel"));
    unload=reinterpret_cast<decltype(unload)>(dlsym(library,"cuModuleUnload"));
    error_string=reinterpret_cast<decltype(error_string)>(dlsym(library,"cuGetErrorString"));
    TORCH_CHECK(load&&function&&launch&&unload&&error_string,"Missing CUDA driver symbols");
  }
  void check(CUresult status) {
    if(status!=CUDA_SUCCESS) {
      const char* message=nullptr;
      error_string(status,&message);
      TORCH_CHECK(false,"Native O2 launch: ",message?message:"CUDA driver error");
    }
  }
};
static Driver& driver() {static Driver value;return value;}

struct DonatedKernel {
  CUmodule module{};
  CUfunction function{};
  int width, channels, weight_count, block, warps, shared, device;
  DonatedKernel(pybind11::bytes cubin, std::string name, int width_, int channels_,
                int weight_count_, int block_, int warps_, int shared_, int device_)
      : width(width_), channels(channels_), weight_count(weight_count_), block(block_),
        warps(warps_), shared(shared_), device(device_) {
    c10::cuda::CUDAGuard guard(device);
    using LoadData = decltype(&cuModuleLoadData);
    auto load_data = reinterpret_cast<LoadData>(dlsym(driver().library, "cuModuleLoadData"));
    TORCH_CHECK(load_data, "CUDA module loader is unavailable");
    std::string image = cubin;
    driver().check(load_data(&module, image.data()));
    driver().check(driver().function(&function, module, name.c_str()));
  }
  ~DonatedKernel() { if (module) driver().unload(module); }
  at::Tensor run(const at::Tensor& left, const at::Tensor& right, const at::Tensor& weight) {
    TORCH_CHECK(!at::GradMode::is_enabled() && !left.requires_grad(), "Donated inference requires no_grad");
    TORCH_CHECK(left.is_cuda() && left.get_device()==device && right.device()==left.device() &&
                weight.device()==left.device(), "CUDA device differs");
    TORCH_CHECK(left.scalar_type()==at::kFloat && right.scalar_type()==at::kFloat &&
                weight.scalar_type()==at::kFloat, "FP32 required");
    TORCH_CHECK(left.dim()==2 && right.sizes()==left.sizes() && left.size(1)==width &&
                weight.dim()==1 && weight.numel()==weight_count, "Invalid input/weight shape");
    TORCH_CHECK(left.is_contiguous() && right.is_contiguous() && weight.is_contiguous(),
                "Contiguous tensors required");
    if (left.numel()==0) return left;
    TORCH_CHECK(!left.is_alias_of(right) && !left.is_alias_of(weight), "Donated input aliases another operand");
    TORCH_CHECK(left.size(0)<=INT32_MAX && left.size(0)*channels <= int64_t(INT32_MAX)*block,
                "Edge count exceeds launch range");
    c10::cuda::CUDAGuard guard(device);
    void* a=left.data_ptr(); void* b=right.data_ptr(); void* w=weight.data_ptr();
    TORCH_CHECK(!(reinterpret_cast<uintptr_t>(a)%16 || reinterpret_cast<uintptr_t>(b)%16 ||
                  reinterpret_cast<uintptr_t>(w)%16), "Prepared launch requires 16-byte alignment");
    int32_t edges=static_cast<int32_t>(left.size(0));
    void* scratch=nullptr;
    void* args[]={&a,&b,&w,&edges,&scratch};
    auto stream=reinterpret_cast<CUstream>(c10::cuda::getCurrentCUDAStream(device).stream());
    unsigned grid=(int64_t(edges)*channels+block-1)/block;
    driver().check(driver().launch(function,grid,1,1,warps*32,1,1,shared,stream,args,nullptr));
    if (!left.is_inference()) left.unsafeGetTensorImpl()->bump_version();
    return left;
  }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  pybind11::class_<DonatedKernel>(m,"DonatedKernel")
    .def(pybind11::init<pybind11::bytes,std::string,int,int,int,int,int,int,int>())
    .def("__call__",&DonatedKernel::run);
}
