#include "kernels.cuh"

// Largest batch the tiled (weight-reuse) path supports. Matches the server's
// 16 KV-cache slots. Batches larger than this (e.g. long-prompt prefill) fall
// back to the per-(s,o) loop kernel below.
#define MAX_BATCH 16

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
// Tiled GEMM with weight reuse (the fast path for batch <= MAX_BATCH).
//
// One warp per OUTPUT ROW o (not per (s,o)). The warp streams the weight row
// W[o] once from HBM and reuses each weight element across the whole batch,
// holding B FP32 accumulators per lane. Weight HBM traffic is now independent
// of batch size — the GEMV becomes a real GEMM. Activations X[b] are tiny
// (B*n_in halves) and stay hot in L2, so re-reading them per row is cheap.
// ---------------------------------------------------------------------------
__global__ void matmul_tiled_fp16_kernel(__half* Y, const __half* W,
                                         const __half* X, const __half* bias,
                                         int batch, int n_out, int n_in) {
    int o    = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // output row
    int lane = threadIdx.x & 31;
    if (o >= n_out) return;

    const __half* w = W + (size_t)o * n_in;

    float acc[MAX_BATCH];
    for (int b = 0; b < batch; ++b) acc[b] = 0.0f;

    // Each weight element is loaded once and FMA'd across all B sequences.
    for (int i = lane; i < n_in; i += 32) {
        float wv = __half2float(w[i]);
        for (int b = 0; b < batch; ++b)
            acc[b] += wv * __half2float(X[(size_t)b * n_in + i]);
    }

    // Reduce each sequence's partial sum across the warp.
    for (int b = 0; b < batch; ++b) {
        float a = acc[b];
        for (int off = 16; off > 0; off >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0) {
            float bv = bias ? __half2float(bias[o]) : 0.0f;
            Y[(size_t)b * n_out + o] = __float2half(a + bv);
        }
    }
}

void matmul_batched_fp16_cuda(__half* Y, const __half* W, const __half* X,
                              const __half* bias,
                              int seq_len, int n_out, int n_in) {
    const int threads = 256;
    if (seq_len <= MAX_BATCH) {
        // Fast path: one warp per output row, weight read once, reused across batch.
        int blocks = ((size_t)n_out * 32 + threads - 1) / threads;
        matmul_tiled_fp16_kernel<<<blocks, threads>>>(Y, W, X, bias,
                                                      seq_len, n_out, n_in);
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

// Tiled FP32-store variant (LM head) — same weight-reuse idea, float output.
// This is the largest matmul in the model (n_out = vocab = 151936), so it
// benefits most from reading each weight once per batch.
__global__ void matmul_tiled_fp16_to_fp32_kernel(float* Y, const __half* W,
                                                 const __half* X,
                                                 int batch, int n_out, int n_in) {
    int o    = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (o >= n_out) return;

    const __half* w = W + (size_t)o * n_in;

    float acc[MAX_BATCH];
    for (int b = 0; b < batch; ++b) acc[b] = 0.0f;

    for (int i = lane; i < n_in; i += 32) {
        float wv = __half2float(w[i]);
        for (int b = 0; b < batch; ++b)
            acc[b] += wv * __half2float(X[(size_t)b * n_in + i]);
    }

    for (int b = 0; b < batch; ++b) {
        float a = acc[b];
        for (int off = 16; off > 0; off >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0) Y[(size_t)b * n_out + o] = a;
    }
}

void matmul_batched_fp16_to_fp32_cuda(float* Y, const __half* W, const __half* X,
                                      int batch, int n_out, int n_in) {
    const int threads = 256;
    if (batch <= MAX_BATCH) {
        int blocks = ((size_t)n_out * 32 + threads - 1) / threads;
        matmul_tiled_fp16_to_fp32_kernel<<<blocks, threads>>>(
            Y, W, X, batch, n_out, n_in);
    } else {
        int total_warps = batch * n_out;
        int blocks = ((size_t)total_warps * 32 + threads - 1) / threads;
        matmul_batched_fp16_to_fp32_kernel<<<blocks, threads>>>(
            Y, W, X, batch, n_out, n_in);
    }
}
