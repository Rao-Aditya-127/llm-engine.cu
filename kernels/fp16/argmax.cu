#include "kernels.cuh"
#include <math_constants.h>

// Greedy argmax over each row of logits[batch × vocab]. One block per row.
// Keeps (max value, lowest index on ties) so the result is bit-identical to the
// host greedy sampler in sampler.cpp (which takes the first strict maximum).
//
// This replaces a per-step DtoH copy of batch×vocab FP32 logits (≈9.7 MB at
// batch=16) + a host argmax over millions of floats with one kernel and a
// batch-int copyback (≈64 bytes).
__global__ void argmax_rows_fp32_kernel(int* out, const float* logits,
                                        int vocab) {
    int row      = blockIdx.x;
    int tid      = threadIdx.x;
    int nthreads = blockDim.x;
    const float* L = logits + (size_t)row * vocab;

    float best_val = -CUDART_INF_F;
    int   best_idx = 0;
    // Strided scan; indices increase within a thread so the first (lowest-index)
    // maximum is naturally retained.
    for (int i = tid; i < vocab; i += nthreads) {
        float v = L[i];
        if (v > best_val) { best_val = v; best_idx = i; }
    }

    extern __shared__ char smem[];
    float* sval = reinterpret_cast<float*>(smem);
    int*   sidx = reinterpret_cast<int*>(sval + nthreads);
    sval[tid] = best_val;
    sidx[tid] = best_idx;
    __syncthreads();

    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float vo = sval[tid + s];
            int   io = sidx[tid + s];
            // Prefer larger value; on ties prefer the lower index.
            if (vo > sval[tid] || (vo == sval[tid] && io < sidx[tid])) {
                sval[tid] = vo;
                sidx[tid] = io;
            }
        }
        __syncthreads();
    }

    if (tid == 0) out[row] = sidx[0];
}

void argmax_rows_fp32_cuda(int* out, const float* logits, int batch, int vocab) {
    const int threads = 256;
    size_t shared = (size_t)threads * (sizeof(float) + sizeof(int));
    argmax_rows_fp32_kernel<<<batch, threads, shared>>>(out, logits, vocab);
}
