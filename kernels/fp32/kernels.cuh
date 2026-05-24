#pragma once
#include "common.cuh"

// Host-side launch wrappers. Each runs on FP32 device buffers.
// These are the Phase 2 ports of the CPU ops in src/infer_cpu.cpp.

// out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
void rmsnorm_cuda(float* out, const float* x, const float* weight, int n);

// Rotary embedding, "rotate-half" form, applied in place per head.
void rope_cuda(float* vec, int n_heads, int head_dim, int pos);

// gate[i] = silu(gate[i]) * up[i]   (fused — no intermediate buffer)
void swiglu_cuda(float* gate, const float* up, int n);

// y = W @ x (+ bias).  W is row-major [n_out, n_in].
void matmul_cuda(float* y, const float* W, const float* x, const float* bias,
                 int n_out, int n_in);

// Causal GQA attention for a single query token at sequence position `pos`.
// kbase/vbase point at one layer's cache: [KV_CACHE_CAP, n_kv_heads*head_dim].
void attention_cuda(float* out, const float* q, const float* kbase,
                    const float* vbase, int pos, int n_heads, int n_kv_heads,
                    int head_dim);
