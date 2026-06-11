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

// ---------------------------------------------------------------------------
// Batched GEMM: Y[seq_len × n_out] = X[seq_len × n_in] × W[n_out × n_in]^T
// One warp handles one output element (s, o). Layout mirrors the GEMV above.
// ---------------------------------------------------------------------------
__global__ void matmul_batched_fp16_kernel(__half* Y, const __half* W,
                                           const __half* X, const __half* bias,
                                           int seq_len, int n_out, int n_in) {
    int total_warps = seq_len * n_out;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= total_warps) return;

    int s = warp / n_out;   // sequence position
    int o = warp % n_out;   // output row index

    const __half* w  = W + (size_t)o * n_in;
    const __half* xi = X + (size_t)s * n_in;

    float acc = 0.0f;
    for (int i = lane; i < n_in; i += 32)
        acc += __half2float(w[i]) * __half2float(xi[i]);

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0) {
        float b = bias ? __half2float(bias[o]) : 0.0f;
        Y[(size_t)s * n_out + o] = __float2half(acc + b);
    }
}

// ---------------------------------------------------------------------------
// Tiled GEMM with weight reuse (fast path for batch <= TILE_MAX_BATCH).
//
// One warp per OUTPUT ROW o (not per (s,o)). The warp streams the weight row
// W[o] once from HBM and reuses each weight element across the whole batch.
//
// Batch is a TEMPLATE parameter so `acc[B]` is a compile-time-sized register
// array and the batch loop fully unrolls — without this, a runtime-indexed
// `acc[batch]` spills to local memory (DRAM) and every FMA hits DRAM, which is
// what made the first attempt slower than the per-(s,o) kernel.
//
// Loads are vectorized: each lane pulls a float4 (= 8 halves) of weights once
// and reuses it across all B activations. n_in is a multiple of 8 for every
// matmul in this model (H=896, I=4864, KV=128, QD=896, vocab=151936); the host
// wrapper falls back to the loop kernel otherwise.
// ---------------------------------------------------------------------------
#define TILE_MAX_BATCH 16

// Reinterpret one float4 (16 bytes) as 8 consecutive halves.
__device__ __forceinline__ void f4_to_h8(const float4& v, __half h[8]) {
    const __half2* p = reinterpret_cast<const __half2*>(&v);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        h[2 * j]     = __low2half(p[j]);
        h[2 * j + 1] = __high2half(p[j]);
    }
}

template <int B>
__global__ void matmul_tiled_fp16_kernel(__half* Y, const __half* W,
                                         const __half* X, const __half* bias,
                                         int batch, int n_out, int n_in) {
    int o    = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // output row
    int lane = threadIdx.x & 31;
    if (o >= n_out) return;

    const float4* w4    = reinterpret_cast<const float4*>(W + (size_t)o * n_in);
    int           n_vec = n_in >> 3;   // n_in / 8

    float acc[B];
    #pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;

    // Each weight float4 is loaded once and reused across all B sequences.
    for (int i = lane; i < n_vec; i += 32) {
        __half hw[8];
        f4_to_h8(w4[i], hw);
        #pragma unroll
        for (int b = 0; b < B; ++b) {
            const float4* x4 = reinterpret_cast<const float4*>(X + (size_t)b * n_in);
            __half hx[8];
            f4_to_h8(x4[i], hx);   // X is tiny -> L2-resident, cheap re-read
            #pragma unroll
            for (int k = 0; k < 8; ++k)
                acc[b] += __half2float(hw[k]) * __half2float(hx[k]);
        }
    }

    #pragma unroll
    for (int b = 0; b < B; ++b) {
        float a = acc[b];
        for (int off = 16; off > 0; off >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0 && b < batch) {   // padding rows (b >= batch) discarded
            float bv = bias ? __half2float(bias[o]) : 0.0f;
            Y[(size_t)b * n_out + o] = __float2half(a + bv);
        }
    }
}

// FP32-store / no-bias variant for the LM head (largest matmul, n_out = vocab).
template <int B>
__global__ void matmul_tiled_fp16_to_fp32_kernel(float* Y, const __half* W,
                                                 const __half* X,
                                                 int batch, int n_out, int n_in) {
    int o    = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (o >= n_out) return;

    const float4* w4    = reinterpret_cast<const float4*>(W + (size_t)o * n_in);
    int           n_vec = n_in >> 3;

    float acc[B];
    #pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;

    for (int i = lane; i < n_vec; i += 32) {
        __half hw[8];
        f4_to_h8(w4[i], hw);
        #pragma unroll
        for (int b = 0; b < B; ++b) {
            const float4* x4 = reinterpret_cast<const float4*>(X + (size_t)b * n_in);
            __half hx[8];
            f4_to_h8(x4[i], hx);
            #pragma unroll
            for (int k = 0; k < 8; ++k)
                acc[b] += __half2float(hw[k]) * __half2float(hx[k]);
        }
    }

    #pragma unroll
    for (int b = 0; b < B; ++b) {
        float a = acc[b];
        for (int off = 16; off > 0; off >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0 && b < batch) Y[(size_t)b * n_out + o] = a;
    }
}

// Round the real batch up to the next supported template instantiation.
static inline int tile_pad_batch(int b) {
    if (b <= 1) return 1;
    if (b <= 2) return 2;
    if (b <= 4) return 4;
    if (b <= 8) return 8;
    return 16;
}

void matmul_batched_fp16_cuda(__half* Y, const __half* W, const __half* X,
                              const __half* bias,
                              int seq_len, int n_out, int n_in) {
    const int threads = 256;
    if (seq_len <= TILE_MAX_BATCH && (n_in & 7) == 0) {
        int blocks = ((size_t)n_out * 32 + threads - 1) / threads;
        #define LAUNCH_TILED(B) \
            matmul_tiled_fp16_kernel<B><<<blocks, threads>>>( \
                Y, W, X, bias, seq_len, n_out, n_in)
        switch (tile_pad_batch(seq_len)) {
            case 1:  LAUNCH_TILED(1);  break;
            case 2:  LAUNCH_TILED(2);  break;
            case 4:  LAUNCH_TILED(4);  break;
            case 8:  LAUNCH_TILED(8);  break;
            default: LAUNCH_TILED(16); break;
        }
        #undef LAUNCH_TILED
    } else {
        // Fallback for large batches (long-prompt prefill): one warp per (s,o).
        int total_warps = seq_len * n_out;
        int blocks = ((size_t)total_warps * 32 + threads - 1) / threads;
        matmul_batched_fp16_kernel<<<blocks, threads>>>(Y, W, X, bias,
                                                        seq_len, n_out, n_in);
    }
}

// ---------------------------------------------------------------------------
// Batched LM head: Y[batch × n_out] FP32 = X[batch × n_in] × W[n_out × n_in]^T.
// One warp per output element (b, o); FP32 store, no bias.
// ---------------------------------------------------------------------------
__global__ void matmul_batched_fp16_to_fp32_kernel(float* Y, const __half* W,
                                                   const __half* X,
                                                   int batch, int n_out, int n_in) {
    int total_warps = batch * n_out;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= total_warps) return;

    int b = warp / n_out;
    int o = warp % n_out;

    const __half* w  = W + (size_t)o * n_in;
    const __half* xb = X + (size_t)b * n_in;

    float acc = 0.0f;
    for (int i = lane; i < n_in; i += 32)
        acc += __half2float(w[i]) * __half2float(xb[i]);

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0) Y[(size_t)b * n_out + o] = acc;
}

void matmul_batched_fp16_to_fp32_cuda(float* Y, const __half* W, const __half* X,
                                      int batch, int n_out, int n_in) {
    const int threads = 256;
    if (batch <= TILE_MAX_BATCH && (n_in & 7) == 0) {
        int blocks = ((size_t)n_out * 32 + threads - 1) / threads;
        #define LAUNCH_TILED_F32(B) \
            matmul_tiled_fp16_to_fp32_kernel<B><<<blocks, threads>>>( \
                Y, W, X, batch, n_out, n_in)
        switch (tile_pad_batch(batch)) {
            case 1:  LAUNCH_TILED_F32(1);  break;
            case 2:  LAUNCH_TILED_F32(2);  break;
            case 4:  LAUNCH_TILED_F32(4);  break;
            case 8:  LAUNCH_TILED_F32(8);  break;
            default: LAUNCH_TILED_F32(16); break;
        }
        #undef LAUNCH_TILED_F32
    } else {
        int total_warps = batch * n_out;
        int blocks = ((size_t)total_warps * 32 + threads - 1) / threads;
        matmul_batched_fp16_to_fp32_kernel<<<blocks, threads>>>(
            Y, W, X, batch, n_out, n_in);
    }
}
