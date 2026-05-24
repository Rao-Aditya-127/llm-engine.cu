#pragma once
#include <vector>
#include "model.h"

// GPU FP32 forward pass for Qwen2-0.5B. Same public interface as CpuRunner,
// so main.cpp can swap between them. All device pointers are plain float* —
// no CUDA types leak into this header, so it stays a normal C++ header.
class GpuRunner {
public:
    explicit GpuRunner(const Model& model);
    ~GpuRunner();

    GpuRunner(const GpuRunner&) = delete;
    GpuRunner& operator=(const GpuRunner&) = delete;

    // Run one token at sequence position `pos`.
    // Returns a host pointer to the logits buffer [vocab_size].
    const float* forward(int token_id, int pos);

    int vocab_size() const { return vocab_; }

private:
    TinyllmHeader header_{};
    int vocab_ = 0;

    // All weights live in one device blob; these point into it.
    float* d_weights_ = nullptr;
    const float* d_embed_ = nullptr;       // also the tied LM head
    const float* d_final_norm_ = nullptr;
    std::vector<LayerWeights> d_layers_;   // device pointers per layer

    // Device KV cache: [num_layers][KV_CACHE_CAP][kv_dim], one K and one V.
    float* d_kcache_ = nullptr;
    float* d_vcache_ = nullptr;

    // Per-step device scratch.
    float* d_x_ = nullptr;
    float* d_xn_ = nullptr;
    float* d_q_ = nullptr;
    float* d_attn_ = nullptr;
    float* d_gate_ = nullptr;
    float* d_up_ = nullptr;
    float* d_logits_ = nullptr;

    std::vector<float> logits_;            // host copy returned to caller
};
