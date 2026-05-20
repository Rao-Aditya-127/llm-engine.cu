#pragma once
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

// The whole model: one big float buffer + pointers into it.
class Model {
public:
    explicit Model(const std::string& path);

    TinyllmHeader header;
    const float* embed_tokens = nullptr;  // [vocab, hidden], also tied lm_head
    const float* final_norm   = nullptr;  // [hidden]
    std::vector<LayerWeights> layers;

private:
    std::vector<float> buffer_;  // owns all weight data
};
