#include "model.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

Model::Model(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);

    if (std::fread(&header, sizeof(header), 1, f) != 1)
        throw std::runtime_error("cannot read header");
    if (header.magic != TINYLLM_MAGIC)
        throw std::runtime_error("bad magic — not a tinyllm.bin file");

    const int H  = header.hidden_size;
    const int I  = header.intermediate_size;
    const int QD = header.num_heads * header.head_dim;
    const int KV = header.num_kv_heads * header.head_dim;
    const int V  = header.vocab_size;

    long pos = std::ftell(f);
    std::fseek(f, 0, SEEK_END);
    long end = std::ftell(f);
    std::fseek(f, pos, SEEK_SET);

    if (header.dtype == 0) {
        // ---- FP32 path ----
        size_t n_floats = (end - pos) / sizeof(float);
        buffer_.resize(n_floats);
        if (std::fread(buffer_.data(), sizeof(float), n_floats, f) != n_floats)
            throw std::runtime_error("cannot read weights");
        std::fclose(f);

        const float* p = buffer_.data();
        auto take = [&](size_t n) { const float* r = p; p += n; return r; };

        embed_tokens = take((size_t)V * H);
        layers.resize(header.num_layers);
        for (auto& L : layers) {
            L.input_layernorm    = take(H);
            L.q_proj_w           = take((size_t)QD * H);
            L.q_proj_b           = take(QD);
            L.k_proj_w           = take((size_t)KV * H);
            L.k_proj_b           = take(KV);
            L.v_proj_w           = take((size_t)KV * H);
            L.v_proj_b           = take(KV);
            L.o_proj_w           = take((size_t)H * QD);
            L.post_attn_layernorm = take(H);
            L.gate_proj_w        = take((size_t)I * H);
            L.up_proj_w          = take((size_t)I * H);
            L.down_proj_w        = take((size_t)H * I);
        }
        final_norm = take(H);

        if ((size_t)(p - buffer_.data()) != n_floats)
            throw std::runtime_error("weight buffer size mismatch");

    } else if (header.dtype == 1) {
        // ---- FP16 path (uint16_t bits; GPU side reinterprets as __half) ----
        size_t n_halves = (end - pos) / sizeof(uint16_t);
        buffer_h_.resize(n_halves);
        if (std::fread(buffer_h_.data(), sizeof(uint16_t), n_halves, f) != n_halves)
            throw std::runtime_error("cannot read fp16 weights");
        std::fclose(f);

        const uint16_t* p = buffer_h_.data();
        auto take = [&](size_t n) { const uint16_t* r = p; p += n; return r; };

        embed_tokens_h = take((size_t)V * H);
        layers_h.resize(header.num_layers);
        for (auto& L : layers_h) {
            L.input_layernorm    = take(H);
            L.q_proj_w           = take((size_t)QD * H);
            L.q_proj_b           = take(QD);
            L.k_proj_w           = take((size_t)KV * H);
            L.k_proj_b           = take(KV);
            L.v_proj_w           = take((size_t)KV * H);
            L.v_proj_b           = take(KV);
            L.o_proj_w           = take((size_t)H * QD);
            L.post_attn_layernorm = take(H);
            L.gate_proj_w        = take((size_t)I * H);
            L.up_proj_w          = take((size_t)I * H);
            L.down_proj_w        = take((size_t)H * I);
        }
        final_norm_h = take(H);

        if ((size_t)(p - buffer_h_.data()) != n_halves)
            throw std::runtime_error("fp16 weight buffer size mismatch");

    } else {
        std::fclose(f);
        throw std::runtime_error("unsupported dtype in tinyllm.bin header");
    }
}
