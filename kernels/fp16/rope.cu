#include "kernels.cuh"
#include "config.h"

// FP16 storage, FP32 math: read pair as half, rotate in FP32, write back as half.
__global__ void rope_fp16_kernel(__half* vec, int n_heads, int head_dim,
                                 int pos) {
    int half_d = head_dim >> 1;
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_heads * half_d) return;

    int h = idx / half_d;
    int i = idx % half_d;

    float inv_freq = powf(qwen2::ROPE_THETA, -2.0f * i / head_dim);
    float ang = pos * inv_freq;
    float c = cosf(ang), s = sinf(ang);

    __half* v = vec + h * head_dim;
    float x0 = __half2float(v[i]);
    float x1 = __half2float(v[i + half_d]);
    v[i]          = __float2half(x0 * c - x1 * s);
    v[i + half_d] = __float2half(x1 * c + x0 * s);
}

void rope_fp16_cuda(__half* vec, int n_heads, int head_dim, int pos) {
    int total = n_heads * (head_dim >> 1);
    const int threads = 128;
    int blocks = (total + threads - 1) / threads;
    rope_fp16_kernel<<<blocks, threads>>>(vec, n_heads, head_dim, pos);
}

// ---------------------------------------------------------------------------
// Batched RoPE: applies position t to row t of vec[seq_len × n_heads × head_dim].
// Each thread handles one (seq_pos, head, pair) triple.
// ---------------------------------------------------------------------------
__global__ void rope_batched_fp16_kernel(__half* vec, int n_heads, int head_dim,
                                         int seq_len) {
    int half_d = head_dim >> 1;
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * n_heads * half_d) return;

    int t = idx / (n_heads * half_d);        // sequence position
    int h = (idx / half_d) % n_heads;        // head index
    int i = idx % half_d;                    // pair index within head

    float inv_freq = powf(qwen2::ROPE_THETA, -2.0f * i / head_dim);
    float ang = t * inv_freq;
    float c = cosf(ang), s = sinf(ang);

    __half* v = vec + (size_t)t * n_heads * head_dim + h * head_dim;
    float x0 = __half2float(v[i]);
    float x1 = __half2float(v[i + half_d]);
    v[i]          = __float2half(x0 * c - x1 * s);
    v[i + half_d] = __float2half(x1 * c + x0 * s);
}

void rope_batched_fp16_cuda(__half* vec, int n_heads, int head_dim, int seq_len) {
    int total = seq_len * n_heads * (head_dim >> 1);
    const int threads = 128;
    int blocks = (total + threads - 1) / threads;
    rope_batched_fp16_kernel<<<blocks, threads>>>(vec, n_heads, head_dim, seq_len);
}

// ---------------------------------------------------------------------------
// Decode-batch RoPE: row r is a different sequence at position positions[r].
// Each thread handles one (row, head, pair) triple; angle uses positions[r].
// ---------------------------------------------------------------------------
__global__ void rope_decode_batched_fp16_kernel(__half* vec, const int* positions,
                                                int n_heads, int head_dim,
                                                int batch) {
    int half_d = head_dim >> 1;
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * n_heads * half_d) return;

    int r = idx / (n_heads * half_d);        // row = sequence in the batch
    int h = (idx / half_d) % n_heads;        // head index
    int i = idx % half_d;                    // pair index within head

    float inv_freq = powf(qwen2::ROPE_THETA, -2.0f * i / head_dim);
    float ang = positions[r] * inv_freq;
    float c = cosf(ang), s = sinf(ang);

    __half* v = vec + (size_t)r * n_heads * head_dim + h * head_dim;
    float x0 = __half2float(v[i]);
    float x1 = __half2float(v[i + half_d]);
    v[i]          = __float2half(x0 * c - x1 * s);
    v[i + half_d] = __float2half(x1 * c + x0 * s);
}

void rope_decode_batched_fp16_cuda(__half* vec, const int* positions,
                                   int n_heads, int head_dim, int batch) {
    int total = batch * n_heads * (head_dim >> 1);
    const int threads = 128;
    int blocks = (total + threads - 1) / threads;
    rope_decode_batched_fp16_kernel<<<blocks, threads>>>(
        vec, positions, n_heads, head_dim, batch);
}
