#include "kernels.cuh"

// Single-token decode => this is really a matrix-VECTOR product (M = 1).
//
// One WARP (32 threads) computes one output row:
//   - the 32 lanes stride across the input dimension, each accumulating a
//     partial dot product. Lanes read CONSECUTIVE weights, so global-memory
//     loads are coalesced — this is the bandwidth-friendly layout.
//   - a warp-shuffle reduction sums the 32 partials with no shared memory.
//
// Next optimization (Phase 2 stretch): float4 vectorized loads of W and x.
// Profiling in Phase 3 should confirm these weight GEMMs are the bottleneck.
__global__ void matmul_kernel(float* y, const float* W, const float* x,
                              const float* bias, int n_out, int n_in) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_out) return;

    const float* w = W + (size_t)warp * n_in;
    float acc = 0.0f;
    for (int i = lane; i < n_in; i += 32)
        acc += w[i] * x[i];

    // warp reduction — every lane is active so the full mask is valid
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0)
        y[warp] = acc + (bias ? bias[warp] : 0.0f);
}

void matmul_cuda(float* y, const float* W, const float* x, const float* bias,
                 int n_out, int n_in) {
    const int threads = 256;            // 8 warps per block
    int blocks = (n_out * 32 + threads - 1) / threads;
    matmul_kernel<<<blocks, threads>>>(y, W, x, bias, n_out, n_in);
}
