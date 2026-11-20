#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

__global__ void ieee_matrix(
    const float* left, const float* right, const float* bias, float* output,
    int64_t rows, int64_t columns, int64_t inner,
    int64_t as0, int64_t as1, int64_t as2,
    int64_t bs0, int64_t bs1, int64_t bs2,
    int64_t ds0, int64_t ds1, int64_t ds2,
    float alpha, float beta) {
    __shared__ float left_tile[64][16];
    __shared__ float right_tile[16][64];
    const int thread = threadIdx.y * 16 + threadIdx.x;
    const int64_t row_base = int64_t(blockIdx.x) * 64;
    const int64_t column_base = int64_t(blockIdx.y) * 64;
    const int64_t batch = blockIdx.z;
    float sums[4][4] = {};
    for (int64_t start = 0; start < inner; start += 16) {
        #pragma unroll
        for (int item = thread; item < 1024; item += 256) {
            const int ar = item / 16;
            const int ak = item % 16;
            const int bk = item / 64;
            const int bc = item % 64;
            left_tile[ar][ak] = row_base + ar < rows && start + ak < inner
                ? left[batch * as0 + (row_base + ar) * as1 + (start + ak) * as2] : 0.f;
            right_tile[bk][bc] = start + bk < inner && column_base + bc < columns
                ? right[batch * bs0 + (start + bk) * bs1 + (column_base + bc) * bs2] : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float a = left_tile[threadIdx.y + 16 * i][k];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sums[i][j] = __fmaf_rn(a, right_tile[k][threadIdx.x + 16 * j], sums[i][j]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int64_t row = row_base + threadIdx.y + 16 * i;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t column = column_base + threadIdx.x + 16 * j;
            if (row < rows && column < columns) {
                float value = __fmul_rn(alpha, sums[i][j]);
                if (bias != nullptr && beta != 0.f) {
                    value = __fadd_rn(value, __fmul_rn(beta, bias[batch * ds0 + row * ds1 + column * ds2]));
                }
                output[(batch * rows + row) * columns + column] = value;
            }
        }
    }
}

torch::Tensor matrix(torch::Tensor a, torch::Tensor b,
                     c10::optional<torch::Tensor> bias, double alpha, double beta) {
    c10::cuda::CUDAGuard guard(a.device());
    const int64_t rows = a.size(-2);
    const int64_t inner = a.size(-1);
    const int64_t columns = b.size(-1);
    const int64_t batches = a.dim() == 3 ? a.size(0) : 1;
    auto sizes = a.sizes().vec();
    sizes.back() = columns;
    auto output = torch::empty(sizes, a.options());
    if (output.numel() == 0) return output;
    const float* d = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    const int64_t ds0 = bias.has_value() && a.dim() == 3 ? bias->stride(0) : 0;
    const int64_t ds1 = bias.has_value() ? bias->stride(-2) : 0;
    const int64_t ds2 = bias.has_value() ? bias->stride(-1) : 0;
    ieee_matrix<<<dim3((rows + 63) / 64, (columns + 63) / 64, batches), dim3(16, 16),
                  0, c10::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), d, output.data_ptr<float>(),
        rows, columns, inner,
        a.dim() == 3 ? a.stride(0) : 0, a.stride(-2), a.stride(-1),
        b.dim() == 3 ? b.stride(0) : 0, b.stride(-2), b.stride(-1),
        ds0, ds1, ds2, alpha, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("matrix", &matrix);
}
