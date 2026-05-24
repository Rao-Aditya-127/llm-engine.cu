#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "config.h"

// Pointers into the weight buffer for one transformer layer.
// All matmul weights are row-major [out_features, in_features].
struct LayerWeights {
    const float* input_layernorm;   // [hidden]
    const float* q_proj_w;          // [q_dim, hidden]
    const float* q_proj_b;          // [q_dim]
    const float* k_proj_w;          // [kv_dim, hidden]
    const float* k_proj_b;          // [kv_dim]
    const float* v_proj_w;          // [kv_dim, hidden]
    const float* v_proj_b;          // [kv_dim]
    const float* o_proj_w;          // [hidden, q_dim]
    const float* post_attn_layernorm; // [hidden]
    const float* gate_proj_w;       // [inter, hidden]
    const float* up_proj_w;         // [inter, hidden]
    const float* down_proj_w;       // [hidden, inter]
};

// FP16 variant for Phase 3. Pointers are uint16_t* (plain C++) so this header
// stays usable by the g++ CPU build; the GPU code reinterpret_casts to __half*.
struct LayerWeightsHalf {
    const uint16_t* input_layernorm;
    const uint16_t* q_proj_w;
    const uint16_t* q_proj_b;
    const uint16_t* k_proj_w;
    const uint16_t* k_proj_b;
    const uint16_t* v_proj_w;
    const uint16_t* v_proj_b;
    const uint16_t* o_proj_w;
    const uint16_t* post_attn_layernorm;
    const uint16_t* gate_proj_w;
    const uint16_t* up_proj_w;
    const uint16_t* down_proj_w;
};

// The whole model: one big float buffer + pointers into it.
class Model {
public:
    explicit Model(const std::string& path);

    TinyllmHeader header;

    // FP32 fields — populated only when header.dtype == 0.
    const float* embed_tokens = nullptr;  // [vocab, hidden], also tied lm_head
    const float* final_norm   = nullptr;  // [hidden]
    std::vector<LayerWeights> layers;

    // FP16 fields — populated only when header.dtype == 1.
    const uint16_t* embed_tokens_h = nullptr;
    const uint16_t* final_norm_h   = nullptr;
    std::vector<LayerWeightsHalf> layers_h;

    // Raw access to the contiguous weight blob — used by GPU runners to upload
    // everything at once, then rebase host pointers to device offsets.
    const float*    base()      const { return buffer_.data(); }
    size_t          num_floats() const { return buffer_.size(); }
    const uint16_t* base_h()    const { return buffer_h_.data(); }
    size_t          num_halves() const { return buffer_h_.size(); }

private:
    std::vector<float>    buffer_;    // fp32 data (dtype == 0)
    std::vector<uint16_t> buffer_h_;  // fp16 data (dtype == 1)
};
