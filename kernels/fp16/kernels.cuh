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

// ---------------------------------------------------------------------------
// Batched variants — used by the prefill (GEMM) path.
// All take seq_len as the number of tokens being processed simultaneously.
// ---------------------------------------------------------------------------

// RMSNorm applied independently to each of seq_len rows of x[seq_len × n].
void rmsnorm_batched_fp16_cuda(__half* out, const __half* x, const __half* w,
                               int n, int seq_len);

// RoPE applied to vec[seq_len × n_heads × head_dim] with position = row index.
void rope_batched_fp16_cuda(__half* vec, int n_heads, int head_dim, int seq_len);

// GEMM: Y[seq_len × n_out] = X[seq_len × n_in] × W[n_out × n_in]^T  (+bias).
void matmul_batched_fp16_cuda(__half* Y, const __half* W, const __half* X,
                              const __half* bias,
                              int seq_len, int n_out, int n_in);

// Causal self-attention for a full prompt of seq_len tokens.
// q[seq_len × Q_dim], kbase/vbase are the layer's KV-cache rows 0..seq_len-1.
void attention_prefill_fp16_cuda(__half* out, const __half* q,
                                 const __half* kbase, const __half* vbase,
                                 int seq_len, int n_heads, int n_kv_heads,
                                 int head_dim);
