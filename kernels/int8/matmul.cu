#include "kernels.cuh"

// Fused dequant + matmul, W8A16.
//
// One warp per output row (same skeleton as the FP16 matmul). The 32 lanes
// stride across the input dimension, each reading int8 weights, casting to
// FP32, and multiplying by the FP16 activation. A warp-shuffle sums the
// partials in FP32.
//
// The per-row FP16 scale is constant within a warp, so it factors *out* of the
// inner loop: we multiply the final accumulator by `scale` exactly once.
// This is what makes W8A16 cheap — one extra FMA per output, not per element.
__global__ void matmul_int8_kernel(
    __half* y, const int8_t* W, const __half* scales, const __half* x,
    const __half* bias, int n_out, int n_in) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_out) return;

    const int8_t* w = W + (size_t)warp * n_in;
    float acc = 0.0f;
    for (int i = lane; i < n_in; i += 32)
        acc += (float)w[i] * __half2float(x[i]);

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0) {
        float s = __half2float(scales[warp]);
        float b = bias ? __half2float(bias[warp]) : 0.0f;
        y[warp] = __float2half(acc * s + b);
    }
}

// LM-head variant: same kernel structure, FP32 output, no bias.
__global__ void matmul_int8_to_fp32_kernel(
    float* y, const int8_t* W, const __half* scales, const __half* x,
    int n_out, int n_in) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_out) return;

    const int8_t* w = W + (size_t)warp * n_in;
    float acc = 0.0f;
    for (int i = lane; i < n_in; i += 32)
        acc += (float)w[i] * __half2float(x[i]);

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0) y[warp] = acc * __half2float(scales[warp]);
}

void matmul_int8_cuda(__half* y, const int8_t* W, const __half* scales,
                      const __half* x, const __half* bias,
                      int n_out, int n_in) {
    const int threads = 256;
    int blocks = (n_out * 32 + threads - 1) / threads;
    matmul_int8_kernel<<<blocks, threads>>>(y, W, scales, x, bias, n_out, n_in);
}

void matmul_int8_to_fp32_cuda(float* y, const int8_t* W, const __half* scales,
                              const __half* x, int n_out, int n_in) {
    const int threads = 256;
    int blocks = (n_out * 32 + threads - 1) / threads;
    matmul_int8_to_fp32_kernel<<<blocks, threads>>>(y, W, scales, x, n_out, n_in);
}
