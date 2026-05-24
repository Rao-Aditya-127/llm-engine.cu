#pragma once
#include <cuda_fp16.h>
#include "common.cuh"

// Phase 3 FP16 launch wrappers.
// Storage: weights, activations, KV cache are __half.
// Math:    accumulators (matmul, softmax, RMSNorm sum-of-squares) are FP32.
// Output:  matmul writes __half except the LM head, which writes FP32 directly
//          for the sampler.

void rmsnorm_fp16_cuda(__half* out, const __half* x, const __half* weight,
                       int n);

void rope_fp16_cuda(__half* vec, int n_heads, int head_dim, int pos);

void swiglu_fp16_cuda(__half* gate, const __half* up, int n);

// Standard FP16 matmul: __half y = W @ x (+ bias).
void matmul_fp16_cuda(__half* y, const __half* W, const __half* x,
                      const __half* bias, int n_out, int n_in);

// Variant used for the LM head — writes FP32 logits, no bias.
void matmul_fp16_to_fp32_cuda(float* y, const __half* W, const __half* x,
                              int n_out, int n_in);

void attention_fp16_cuda(__half* out, const __half* q, const __half* kbase,
                         const __half* vbase, int pos, int n_heads,
                         int n_kv_heads, int head_dim);
