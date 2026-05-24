#include "kernels.cuh"

// Causal GQA attention for ONE query token at position `pos`.
//
// One block per query head. Shared memory holds the score vector
// (one float per past position) plus a small scratch array for reductions.
// GQA: query head h reads key/value head h / (n_heads / n_kv_heads).
//
// Naive-but-correct: scores -> softmax -> weighted sum of V, with proper
// block reductions for the max and the sum. A fully fused single-pass
// (flash-attention style) softmax is the Phase 2 stretch optimization.
__global__ void attention_kernel(float* out, const float* q,
                                  const float* kbase, const float* vbase,
                                  int pos, int n_heads, int n_kv_heads,
                                  int head_dim) {
    int h        = blockIdx.x;          // query head
    int tid      = threadIdx.x;
    int nthreads = blockDim.x;          // power of two

    int group   = n_heads / n_kv_heads;
    int kvh     = h / group;            // shared key/value head
    int kv_dim  = n_kv_heads * head_dim;
    int seqlen  = pos + 1;
    float scale = rsqrtf((float)head_dim);

    extern __shared__ float smem[];
    float* scores = smem;               // [seqlen]
    float* red    = smem + seqlen;      // [nthreads] reduction scratch

    const float* qh = q + h * head_dim;

    // 1. scores[t] = (q . K[t]) * scale
    for (int t = tid; t < seqlen; t += nthreads) {
        const float* kt = kbase + (size_t)t * kv_dim + kvh * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kt[d];
        scores[t] = dot * scale;
    }
    __syncthreads();

    // 2. block max (for numerically stable softmax)
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

    // 4. out[d] = sum_t softmax[t] * V[t][d]
    float* outh = out + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float acc = 0.0f;
        for (int t = 0; t < seqlen; ++t) {
            const float* vt = vbase + (size_t)t * kv_dim + kvh * head_dim;
            acc += scores[t] * vt[d];
        }
        outh[d] = acc / total;
    }
}

void attention_cuda(float* out, const float* q, const float* kbase,
                    const float* vbase, int pos, int n_heads, int n_kv_heads,
                    int head_dim) {
    const int threads = 128;            // power of two — required by reductions
    int seqlen = pos + 1;
    size_t shared = ((size_t)seqlen + threads) * sizeof(float);
    attention_kernel<<<n_heads, threads, shared>>>(
        out, q, kbase, vbase, pos, n_heads, n_kv_heads, head_dim);
}
