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
