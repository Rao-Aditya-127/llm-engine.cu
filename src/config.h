#pragma once
#include <cstdint>

// Qwen2-0.5B architecture constants. These are fixed for this model.
namespace qwen2 {

constexpr int   HIDDEN_SIZE        = 896;
constexpr int   INTERMEDIATE_SIZE  = 4864;
constexpr int   NUM_LAYERS         = 24;
constexpr int   NUM_HEADS          = 14;
constexpr int   NUM_KV_HEADS       = 2;
constexpr int   HEAD_DIM           = 64;        // HIDDEN_SIZE / NUM_HEADS
constexpr int   Q_DIM              = NUM_HEADS    * HEAD_DIM;   // 896
constexpr int   KV_DIM             = NUM_KV_HEADS * HEAD_DIM;   // 128
constexpr int   GQA_GROUP          = NUM_HEADS / NUM_KV_HEADS;  // 7
constexpr int   VOCAB_SIZE         = 151936;
constexpr float RMS_NORM_EPS       = 1e-6f;
constexpr float ROPE_THETA         = 1000000.0f;
constexpr int   KV_CACHE_CAP       = 4096;       // capped context for this engine

// q/k/v projections have bias; o_proj and the MLP do not.
// lm_head is tied to embed_tokens (weights stored once).

} // namespace qwen2

// tinyllm.bin file format.
// Layout: [Header][tensor data ...] all little-endian.
// Header magic identifies the file; dtype: 0=fp32, 1=fp16, 2=int8.
constexpr uint32_t TINYLLM_MAGIC   = 0x4D4C4E54; // "TNLM"
constexpr uint32_t TINYLLM_VERSION = 1;

struct TinyllmHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t dtype;            // 0=fp32, 1=fp16, 2=int8
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
    float    rms_norm_eps;
    float    rope_theta;
};

// Runtime sampling configuration.
struct RunConfig {
    int   max_new_tokens = 64;
    float temperature    = 0.0f;   // 0 => greedy
    float top_p          = 1.0f;
    unsigned long long seed = 1234ULL;
};
