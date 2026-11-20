using int64_t = long long;
#ifndef K_GROUPS
#define K_GROUPS 1
#endif

extern "C" __global__ void ieee_matrix(
    const float* left, const float* right, const float* bias, float* output,
    int64_t rows, int64_t columns, int64_t inner,
    int64_t as0, int64_t as1, int64_t as2,
    int64_t bs0, int64_t bs1, int64_t bs2,
    int64_t ds0, int64_t ds1, int64_t ds2,
    float alpha, float beta) {
    __shared__ float left_tile[64][16];
    __shared__ float right_tile[16][64];
    const int thread = threadIdx.y * 16 + threadIdx.x;
    const int64_t row_tiles = (rows + 63) / 64;
    const int64_t row_base = (int64_t(blockIdx.x) % row_tiles) * 64;
    const int64_t column_base = int64_t(blockIdx.y) * 64;
    const int64_t batch = int64_t(blockIdx.x) / row_tiles;
    float sums[4][4][K_GROUPS] = {};
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
                    const int group = k % K_GROUPS;
                    sums[i][j][group] = __fmaf_rn(
                        a, right_tile[k][threadIdx.x + 16 * j], sums[i][j][group]);
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
                #pragma unroll
                for (int stride = K_GROUPS / 2; stride > 0; stride /= 2) {
                    #pragma unroll
                    for (int group = 0; group < stride; ++group) {
                        sums[i][j][group] = __fadd_rn(sums[i][j][group], sums[i][j][group + stride]);
                    }
                }
                float value = __fmul_rn(alpha, sums[i][j][0]);
                if (bias != nullptr && beta != 0.f) {
                    value = __fadd_rn(value, __fmul_rn(beta, bias[batch * ds0 + row * ds1 + column * ds2]));
                }
                output[(batch * rows + row) * columns + column] = value;
            }
        }
    }
}

