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
    if (header.dtype != 0)
        throw std::runtime_error("Phase 1 expects an fp32 file");

    // Read the rest of the file (all weights) into one buffer.
    long pos = std::ftell(f);
    std::fseek(f, 0, SEEK_END);
    long end = std::ftell(f);
    std::fseek(f, pos, SEEK_SET);
    size_t n_floats = (end - pos) / sizeof(float);
    buffer_.resize(n_floats);
    if (std::fread(buffer_.data(), sizeof(float), n_floats, f) != n_floats)
        throw std::runtime_error("cannot read weights");
    std::fclose(f);

    const int H  = header.hidden_size;
    const int I  = header.intermediate_size;
    const int QD = header.num_heads * header.head_dim;
    const int KV = header.num_kv_heads * header.head_dim;
    const int V  = header.vocab_size;

    // Walk the buffer in the exact order convert.py wrote it.
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
}
