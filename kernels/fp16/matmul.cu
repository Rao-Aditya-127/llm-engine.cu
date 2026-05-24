#include "kernels.cuh"

// Warp-per-row GEMV, FP16 weights/activations, FP32 accumulator.
// The 32 lanes stride across the input dimension (coalesced FP16 reads) and
// each accumulates in FP32. A warp-shuffle reduction sums the partials.
// FP32 accumulation is the standard trick for keeping FP16 inference accurate.
__global__ void matmul_fp16_kernel(__half* y, const __half* W, const __half* x,
                                   const __half* bias, int n_out, int n_in) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_out) return;

    const __half* w = W + (size_t)warp * n_in;
    float acc = 0.0f;
    for (int i = lane; i < n_in; i += 32)
        acc += __half2float(w[i]) * __half2float(x[i]);

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0) {
        float b = bias ? __half2float(bias[warp]) : 0.0f;
        y[warp] = __float2half(acc + b);
    }
}

// LM-head variant: same math, but writes FP32 logits straight to host-bound
// memory. No bias for the LM head.
__global__ void matmul_fp16_to_fp32_kernel(float* y, const __half* W,
                                            const __half* x,
                                            int n_out, int n_in) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_out) return;

    const __half* w = W + (size_t)warp * n_in;
    float acc = 0.0f;
    for (int i = lane; i < n_in; i += 32)
        acc += __half2float(w[i]) * __half2float(x[i]);

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0) y[warp] = acc;
}

void matmul_fp16_cuda(__half* y, const __half* W, const __half* x,
                      const __half* bias, int n_out, int n_in) {
    const int threads = 256;
    int blocks = (n_out * 32 + threads - 1) / threads;
    matmul_fp16_kernel<<<blocks, threads>>>(y, W, x, bias, n_out, n_in);
}

void matmul_fp16_to_fp32_cuda(float* y, const __half* W, const __half* x,
                              int n_out, int n_in) {
    const int threads = 256;
    int blocks = (n_out * 32 + threads - 1) / threads;
    matmul_fp16_to_fp32_kernel<<<blocks, threads>>>(y, W, x, n_out, n_in);
}
