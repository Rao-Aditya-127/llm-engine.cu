#include "kernels.cuh"
#include "config.h"

// Same shape as the FP32 RMSNorm: one block does the whole 896-dim vector.
// Inputs/outputs are __half on the GPU; the sum-of-squares accumulates in
// FP32 (kept in shared memory) for numerical stability.
__global__ void rmsnorm_fp16_kernel(__half* out, const __half* x,
                                    const __half* w, int n) {
    extern __shared__ float red[];
    int tid = threadIdx.x;

    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float xf = __half2float(x[i]);
        local += xf * xf;
    }
    red[tid] = local;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(red[0] / n + qwen2::RMS_NORM_EPS);

    for (int i = tid; i < n; i += blockDim.x) {
        float xf = __half2float(x[i]);
        float wf = __half2float(w[i]);
        out[i] = __float2half(xf * scale * wf);
    }
}

void rmsnorm_fp16_cuda(__half* out, const __half* x, const __half* w, int n) {
    const int threads = 256;
    rmsnorm_fp16_kernel<<<1, threads, threads * sizeof(float)>>>(out, x, w, n);
}
