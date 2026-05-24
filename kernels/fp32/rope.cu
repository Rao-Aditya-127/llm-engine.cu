#include "kernels.cuh"
#include "config.h"

// Element-wise: one thread per (head, pair) index.
// Each thread rotates one (i, i+half) pair inside one head — the rotate-half
// form, identical to the CPU version in infer_cpu.cpp.
__global__ void rope_kernel(float* vec, int n_heads, int head_dim, int pos) {
    int half = head_dim >> 1;
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_heads * half) return;

    int h = idx / half;
    int i = idx % half;

    float inv_freq = powf(qwen2::ROPE_THETA, -2.0f * i / head_dim);
    float ang = pos * inv_freq;
    float c = cosf(ang), s = sinf(ang);

    float* v = vec + h * head_dim;
    float x0 = v[i], x1 = v[i + half];
    v[i]        = x0 * c - x1 * s;
    v[i + half] = x1 * c + x0 * s;
}

void rope_cuda(float* vec, int n_heads, int head_dim, int pos) {
    int total = n_heads * (head_dim >> 1);
    const int threads = 128;
    int blocks = (total + threads - 1) / threads;
    rope_kernel<<<blocks, threads>>>(vec, n_heads, head_dim, pos);
}
