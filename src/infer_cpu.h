#pragma once
#include <vector>
#include "model.h"

// Single-threaded (or OpenMP) FP32 forward pass for Qwen2-0.5B.
// Process one token at a time; the KV cache grows as positions advance.
class CpuRunner {
public:
    explicit CpuRunner(const Model& model);

    // Run one token at sequence position `pos`.
    // Returns a pointer to the logits buffer [vocab_size] (owned by this object).
    const float* forward(int token_id, int pos);

    int vocab_size() const { return model_.header.vocab_size; }

private:
    const Model& model_;

    // KV cache: [num_layers][KV_CACHE_CAP][kv_dim], one for K and one for V.
    std::vector<float> kcache_;
    std::vector<float> vcache_;

    // Per-step scratch buffers.
    std::vector<float> x_, xn_, q_, k_, v_, attn_, scores_, gate_, up_, logits_;
};
