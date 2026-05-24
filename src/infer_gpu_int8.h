#pragma once
#include <cstdint>
#include <vector>
#include "model.h"

// Phase 4 — W8A16 GPU forward pass.
// Weights: INT8 + per-output-row FP16 scales (the tied embedding too).
// Activations + KV cache: FP16 (reuses the Phase 3 kernels for everything
// except the matmul and embedding lookup).
// Logits: FP32 (for the sampler).
class GpuRunnerInt8 {
public:
    explicit GpuRunnerInt8(const Model& model);
    ~GpuRunnerInt8();

    GpuRunnerInt8(const GpuRunnerInt8&) = delete;
    GpuRunnerInt8& operator=(const GpuRunnerInt8&) = delete;

    const float* forward(int token_id, int pos);
    int vocab_size() const { return vocab_; }

private:
    TinyllmHeader header_{};
    int vocab_ = 0;

    // One device blob holds all weight bytes (heterogeneous: int8 + fp16).
    uint8_t* d_weights_ = nullptr;

    // Rebased device pointers into d_weights_.
    const int8_t*   d_embed_int8_   = nullptr;   // also the tied LM head
    const uint16_t* d_embed_scales_ = nullptr;
    const uint16_t* d_final_norm_   = nullptr;
    std::vector<LayerWeightsInt8> d_layers_;

    // FP16 KV cache + per-step scratch — same as the FP16 runner.
    uint16_t* d_kcache_ = nullptr;
    uint16_t* d_vcache_ = nullptr;
    uint16_t* d_x_ = nullptr;
    uint16_t* d_xn_ = nullptr;
    uint16_t* d_q_ = nullptr;
    uint16_t* d_attn_ = nullptr;
    uint16_t* d_gate_ = nullptr;
    uint16_t* d_up_ = nullptr;

    // Logits stay FP32 (LM head writes FP32 directly).
    float* d_logits_ = nullptr;
    std::vector<float> logits_;
};
