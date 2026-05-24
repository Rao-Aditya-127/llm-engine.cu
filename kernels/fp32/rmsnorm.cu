#include "kernels.cuh"
#include "config.h"

// One block normalizes the whole hidden vector.
// Step 1: each thread sums squares of its strided slice.
// Step 2: tree reduction in shared memory gives sum-of-squares.
// Step 3: each thread writes its outputs with the shared scale factor.
__global__ void rmsnorm_kernel(float* out, const float* x, const float* w,
                               int n) {
    extern __shared__ float red[];
    int tid = threadIdx.x;

    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) local += x[i] * x[i];
    red[tid] = local;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(red[0] / n + qwen2::RMS_NORM_EPS);

    for (int i = tid; i < n; i += blockDim.x)
        out[i] = x[i] * scale * w[i];
}

void rmsnorm_cuda(float* out, const float* x, const float* w, int n) {
    const int threads = 256;  // power of two — required by the reduction
    rmsnorm_kernel<<<1, threads, threads * sizeof(float)>>>(out, x, w, n);
}
