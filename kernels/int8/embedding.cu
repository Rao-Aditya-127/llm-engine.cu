#include "kernels.cuh"

// Read one row of an INT8 [n_out, n_in] matrix, dequantize to FP16.
// Used by the GPU runner for the embedding lookup — replaces the simple
// row memcpy that the FP16 runner uses, because here the embedding is
// stored INT8 + a per-row scale.
__global__ void embed_dequant_int8_kernel(__half* out, const int8_t* W,
                                          const __half* scales,
                                          int row, int n_in) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_in) return;
    float s = __half2float(scales[row]);
    int8_t w = W[(size_t)row * n_in + i];
    out[i] = __float2half((float)w * s);
}

void embed_dequant_int8_cuda(__half* out, const int8_t* W, const __half* scales,
                             int row, int n_in) {
    const int threads = 256;
    int blocks = (n_in + threads - 1) / threads;
    embed_dequant_int8_kernel<<<blocks, threads>>>(out, W, scales, row, n_in);
}
