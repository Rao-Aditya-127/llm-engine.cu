#pragma once
#include <cstdint>
#include <vector>
#include "model.h"

// FP16 GPU forward pass for Qwen2-0.5B. Same public interface as the FP32
// runners. Device pointers are uint16_t* in this header (plain C++) — the
// CUDA implementation in infer_fp16.cu reinterprets them as __half*.
class GpuRunnerFP16 {
public:
    explicit GpuRunnerFP16(const Model& model);
    ~GpuRunnerFP16();

    GpuRunnerFP16(const GpuRunnerFP16&) = delete;
    GpuRunnerFP16& operator=(const GpuRunnerFP16&) = delete;

    // Run one token at sequence position `pos`.
    // Returns a host pointer to FP32 logits [vocab_size] (for the sampler).
    const float* forward(int token_id, int pos);

    int vocab_size() const { return vocab_; }

private:
    TinyllmHeader header_{};
    int vocab_ = 0;

    // All FP16 weights live in one device blob; the pointers below index it.
    uint16_t* d_weights_h_ = nullptr;
    const uint16_t* d_embed_h_ = nullptr;        // also the tied LM head
    const uint16_t* d_final_norm_h_ = nullptr;
    std::vector<LayerWeightsHalf> d_layers_h_;

    // KV cache and per-step scratch — all FP16.
    uint16_t* d_kcache_ = nullptr;
    uint16_t* d_vcache_ = nullptr;
    uint16_t* d_x_ = nullptr;
    uint16_t* d_xn_ = nullptr;
    uint16_t* d_q_ = nullptr;
    uint16_t* d_attn_ = nullptr;
    uint16_t* d_gate_ = nullptr;
    uint16_t* d_up_ = nullptr;

    // Logits stay FP32 (the LM-head matmul writes float directly).
    float* d_logits_ = nullptr;
    std::vector<float> logits_;
};
