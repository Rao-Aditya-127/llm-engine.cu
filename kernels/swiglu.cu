#include "kernels.cuh"

// Fused SwiGLU activation: gate[i] = silu(gate[i]) * up[i], in place.
// "Fused" = the silu and the multiply happen in one kernel, so the intermediate
// silu(gate) is never written back to global memory — it stays in a register.
__global__ void swiglu_kernel(float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    gate[i] = (g / (1.0f + expf(-g))) * up[i];  // silu(g) * up
}

void swiglu_cuda(float* gate, const float* up, int n) {
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads>>>(gate, up, n);
}
