#include "kernels.cuh"

// One block per query head. Scores and softmax sum live in shared memory as
// FP32 — softmax is numerically delicate in FP16, so we keep it in float
// even though Q/K/V/O are __half.
__global__ void attention_fp16_kernel(__half* out, const __half* q,
                                       const __half* kbase,
                                       const __half* vbase,
                                       int pos, int n_heads, int n_kv_heads,
                                       int head_dim) {
    int h        = blockIdx.x;
    int tid      = threadIdx.x;
    int nthreads = blockDim.x;

    int group   = n_heads / n_kv_heads;
    int kvh     = h / group;
    int kv_dim  = n_kv_heads * head_dim;
    int seqlen  = pos + 1;
    float scale = rsqrtf((float)head_dim);

    extern __shared__ float smem[];
    float* scores = smem;
    float* red    = smem + seqlen;

    const __half* qh = q + h * head_dim;

    // 1. scores[t] = (q . K[t]) * scale, in FP32
    for (int t = tid; t < seqlen; t += nthreads) {
        const __half* kt = kbase + (size_t)t * kv_dim + kvh * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += __half2float(qh[d]) * __half2float(kt[d]);
        scores[t] = dot * scale;
    }
    __syncthreads();

    // 2. block max
    float m = -1e30f;
    for (int t = tid; t < seqlen; t += nthreads) m = fmaxf(m, scores[t]);
    red[tid] = m;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float maxv = red[0];
    __syncthreads();

    // 3. exp in place + block sum
    float sum = 0.0f;
    for (int t = tid; t < seqlen; t += nthreads) {
        float e = expf(scores[t] - maxv);
        scores[t] = e;
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float total = red[0];
    __syncthreads();

    // 4. weighted sum of V, write as FP16
    __half* outh = out + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float acc = 0.0f;
        for (int t = 0; t < seqlen; ++t) {
            const __half* vt = vbase + (size_t)t * kv_dim + kvh * head_dim;
            acc += scores[t] * __half2float(vt[d]);
        }
        outh[d] = __float2half(acc / total);
    }
}

void attention_fp16_cuda(__half* out, const __half* q, const __half* kbase,
                         const __half* vbase, int pos, int n_heads,
                         int n_kv_heads, int head_dim) {
    const int threads = 128;
    int seqlen = pos + 1;
    size_t shared = ((size_t)seqlen + threads) * sizeof(float);
    attention_fp16_kernel<<<n_heads, threads, shared>>>(
        out, q, kbase, vbase, pos, n_heads, n_kv_heads, head_dim);
}

// ---------------------------------------------------------------------------
// Causal prefill attention.
// Grid: dim3(seq_len, n_heads) — one block per (query position, query head).
// Block t processes Q[t] attending causally to K/V rows 0..t.
// Shared memory layout: scores[0..seq_len-1]  |  reduction[0..nthreads-1]
// Both regions are allocated at max (seq_len + nthreads) floats per block.
// ---------------------------------------------------------------------------
__global__ void attention_prefill_fp16_kernel(
        __half* out, const __half* q,
        const __half* kbase, const __half* vbase,
        int seq_len, int n_heads, int n_kv_heads, int head_dim) {

    int t   = blockIdx.x;   // query position; attends to 0..t (causal)
    int h   = blockIdx.y;   // query head
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    int group   = n_heads / n_kv_heads;
    int kvh     = h / group;
    int kv_dim  = n_kv_heads * head_dim;
    int seqlen  = t + 1;
    float scale = rsqrtf((float)head_dim);

    extern __shared__ float smem[];
    float* scores = smem;
    float* red    = smem + seq_len;   // nthreads floats for block reduction

    const __half* qh = q + (size_t)t * n_heads * head_dim + h * head_dim;

    // 1. dot products Q[t] · K[tp] for tp = 0..t
    for (int tp = tid; tp < seqlen; tp += nthreads) {
        const __half* kt = kbase + (size_t)tp * kv_dim + kvh * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += __half2float(qh[d]) * __half2float(kt[d]);
        scores[tp] = dot * scale;
    }
    __syncthreads();

    // 2. block max for numerically stable softmax
    float m = -1e30f;
    for (int tp = tid; tp < seqlen; tp += nthreads) m = fmaxf(m, scores[tp]);
    red[tid] = m;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float maxv = red[0];
    __syncthreads();

    // 3. exp in-place + block sum
    float sum = 0.0f;
    for (int tp = tid; tp < seqlen; tp += nthreads) {
        float e = expf(scores[tp] - maxv);
        scores[tp] = e;
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float total = red[0];
    __syncthreads();

    // 4. weighted sum of V, write FP16 output
    __half* outh = out + (size_t)t * n_heads * head_dim + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float acc = 0.0f;
        for (int tp = 0; tp < seqlen; ++tp) {
            const __half* vt = vbase + (size_t)tp * kv_dim + kvh * head_dim;
            acc += scores[tp] * __half2float(vt[d]);
        }
        outh[d] = __float2half(acc / total);
    }
}

void attention_prefill_fp16_cuda(__half* out, const __half* q,
                                 const __half* kbase, const __half* vbase,
                                 int seq_len, int n_heads, int n_kv_heads,
                                 int head_dim) {
    const int threads = 128;
    dim3 grid(seq_len, n_heads);
    size_t shared = ((size_t)seq_len + threads) * sizeof(float);
    attention_prefill_fp16_kernel<<<grid, threads, shared>>>(
        out, q, kbase, vbase, seq_len, n_heads, n_kv_heads, head_dim);
}

// ---------------------------------------------------------------------------
// Decode-batch attention.
// Grid: dim3(batch, n_heads) — block (b,h) is sequence b's query head h.
// Sequence b lives in KV-cache slot slots[b] and attends over rows
// 0..positions[b]. q is [batch × n_heads × head_dim] (one new token per row).
// Shared memory: scores[0..seqlen-1] | reduction[0..nthreads-1], sized for the
// batch-wide max_seqlen.
// ---------------------------------------------------------------------------
__global__ void attention_decode_batched_fp16_kernel(
        __half* out, const __half* q,
        const __half* kcache, const __half* vcache,
        const int* positions, const int* slots,
        int batch, int n_heads, int n_kv_heads, int head_dim,
        int layer, int num_layers, int cap, int max_seqlen) {

    int b   = blockIdx.x;   // sequence in the batch
    int h   = blockIdx.y;   // query head
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    int group   = n_heads / n_kv_heads;
    int kvh     = h / group;
    int kv_dim  = n_kv_heads * head_dim;
    int seqlen  = positions[b] + 1;
    float scale = rsqrtf((float)head_dim);

    // Base of this (slot, layer) KV cache region.
    size_t base = ((size_t)slots[b] * num_layers + layer) * cap * kv_dim;
    const __half* kbase = kcache + base;
    const __half* vbase = vcache + base;

    extern __shared__ float smem[];
    float* scores = smem;
    float* red    = smem + max_seqlen;   // nthreads floats for block reduction

    const __half* qh = q + (size_t)b * n_heads * head_dim + h * head_dim;

    // 1. dot products Q · K[tp] for tp = 0..positions[b]
    for (int tp = tid; tp < seqlen; tp += nthreads) {
        const __half* kt = kbase + (size_t)tp * kv_dim + kvh * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += __half2float(qh[d]) * __half2float(kt[d]);
        scores[tp] = dot * scale;
    }
    __syncthreads();

    // 2. block max
    float m = -1e30f;
    for (int tp = tid; tp < seqlen; tp += nthreads) m = fmaxf(m, scores[tp]);
    red[tid] = m;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float maxv = red[0];
    __syncthreads();

    // 3. exp in-place + block sum
    float sum = 0.0f;
    for (int tp = tid; tp < seqlen; tp += nthreads) {
        float e = expf(scores[tp] - maxv);
        scores[tp] = e;
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float total = red[0];
    __syncthreads();

    // 4. weighted sum of V, write FP16 output
    __half* outh = out + (size_t)b * n_heads * head_dim + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float acc = 0.0f;
        for (int tp = 0; tp < seqlen; ++tp) {
            const __half* vt = vbase + (size_t)tp * kv_dim + kvh * head_dim;
            acc += scores[tp] * __half2float(vt[d]);
        }
        outh[d] = __float2half(acc / total);
    }
}

void attention_decode_batched_fp16_cuda(__half* out, const __half* q,
                                        const __half* kcache, const __half* vcache,
                                        const int* positions, const int* slots,
                                        int batch, int n_heads, int n_kv_heads,
                                        int head_dim, int layer, int num_layers,
                                        int cap, int max_seqlen) {
    const int threads = 128;
    dim3 grid(batch, n_heads);
    size_t shared = ((size_t)max_seqlen + threads) * sizeof(float);
    attention_decode_batched_fp16_kernel<<<grid, threads, shared>>>(
        out, q, kcache, vcache, positions, slots,
        batch, n_heads, n_kv_heads, head_dim, layer, num_layers, cap, max_seqlen);
}
