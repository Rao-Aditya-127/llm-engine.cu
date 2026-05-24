#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include "common.cuh"

// Phase 4 — W8A16 launch wrappers.
//
// Weights: INT8 + per-output-row FP16 scale.  Dequant happens inside the
// matmul, with no intermediate FP16 weight tensor ever materialized.
// Activations and KV cache: FP16 (same as Phase 3).
// Matmul accumulators: FP32 (numerical stability).

// y = (W_int8 * scale_per_row) @ x_fp16 (+ bias_fp16), written as fp16.
void matmul_int8_cuda(__half* y, const int8_t* W, const __half* scales,
                      const __half* x, const __half* bias,
                      int n_out, int n_in);

// LM head variant — writes FP32 logits straight to the sampler buffer.
void matmul_int8_to_fp32_cuda(float* y, const int8_t* W, const __half* scales,
                              const __half* x, int n_out, int n_in);

// Dequantize one row of the INT8 embedding matrix into an FP16 vector.
// Used in place of the embedding-lookup memcpy when weights are INT8.
void embed_dequant_int8_cuda(__half* out, const int8_t* W, const __half* scales,
                             int row, int n_in);
