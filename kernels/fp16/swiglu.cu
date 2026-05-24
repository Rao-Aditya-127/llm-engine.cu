#include "kernels.cuh"

// Fused silu × multiply, FP32 math.
__global__ void swiglu_fp16_kernel(__half* gate, const __half* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    float silu_g = g / (1.0f + expf(-g));
    gate[i] = __float2half(silu_g * u);
}

void swiglu_fp16_cuda(__half* gate, const __half* up, int n) {
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    swiglu_fp16_kernel<<<blocks, threads>>>(gate, up, n);
}
