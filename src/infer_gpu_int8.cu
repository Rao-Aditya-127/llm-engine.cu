#include "infer_gpu_int8.h"
#include "fp16/kernels.cuh"     // reuse rmsnorm / rope / swiglu / attention
#include "int8/kernels.cuh"     // int8 matmul + embedding dequant
#include "config.h"
#include <stdexcept>

using namespace qwen2;

// Reinterpret helpers (header keeps int8/uint16 plain so g++ can read it).
static inline __half*       asHalf(uint16_t* p)       { return reinterpret_cast<__half*>(p); }
static inline const __half* asHalf(const uint16_t* p) { return reinterpret_cast<const __half*>(p); }

// Residual add (FP16). Same as the FP16 runner — keeping it local to avoid
// cross-file linkage.
__global__ void residual_add_kernel_int8(__half* x, const __half* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __hadd(x[i], y[i]);
}
static void residual_add(uint16_t* x, const uint16_t* y, int n) {
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    residual_add_kernel_int8<<<blocks, threads>>>(asHalf(x), asHalf(y), n);
}

// ---------------------------------------------------------------------------

GpuRunnerInt8::GpuRunnerInt8(const Model& model) {
    if (model.header.dtype != 2)
        throw std::runtime_error("GpuRunnerInt8 expects an int8 .bin (dtype=2)");

    header_ = model.header;
    vocab_  = header_.vocab_size;

    const int H  = header_.hidden_size;
    const int I  = header_.intermediate_size;
    const int QD = header_.num_heads * header_.head_dim;
    const int KV = header_.num_kv_heads * header_.head_dim;

    // Upload the heterogeneous weight blob in one shot.
    size_t nbytes = model.num_bytes();
    CUDA_CHECK(cudaMalloc(&d_weights_, nbytes));
    CUDA_CHECK(cudaMemcpy(d_weights_, model.base_q(), nbytes,
                          cudaMemcpyHostToDevice));

    // Rebase host pointers to device offsets, preserving int8 vs fp16 typing.
    const uint8_t* hbase = model.base_q();
    auto dev_i8 = [&](const int8_t* h) {
        return reinterpret_cast<const int8_t*>(
            d_weights_ + (reinterpret_cast<const uint8_t*>(h) - hbase));
    };
    auto dev_h = [&](const uint16_t* h) {
        return reinterpret_cast<const uint16_t*>(
            d_weights_ + (reinterpret_cast<const uint8_t*>(h) - hbase));
    };

    d_embed_int8_   = dev_i8(model.embed_tokens_int8);
    d_embed_scales_ = dev_h(model.embed_tokens_int8_scales);
    d_final_norm_   = dev_h(model.final_norm_int8);

    d_layers_.resize(header_.num_layers);
    for (int l = 0; l < (int)header_.num_layers; ++l) {
        const LayerWeightsInt8& s = model.layers_int8[l];
        LayerWeightsInt8& d = d_layers_[l];
        d.input_layernorm     = dev_h(s.input_layernorm);
        d.q_proj_w            = dev_i8(s.q_proj_w);
        d.q_proj_scales       = dev_h(s.q_proj_scales);
        d.q_proj_b            = dev_h(s.q_proj_b);
        d.k_proj_w            = dev_i8(s.k_proj_w);
        d.k_proj_scales       = dev_h(s.k_proj_scales);
        d.k_proj_b            = dev_h(s.k_proj_b);
        d.v_proj_w            = dev_i8(s.v_proj_w);
        d.v_proj_scales       = dev_h(s.v_proj_scales);
        d.v_proj_b            = dev_h(s.v_proj_b);
        d.o_proj_w            = dev_i8(s.o_proj_w);
        d.o_proj_scales       = dev_h(s.o_proj_scales);
        d.post_attn_layernorm = dev_h(s.post_attn_layernorm);
        d.gate_proj_w         = dev_i8(s.gate_proj_w);
        d.gate_proj_scales    = dev_h(s.gate_proj_scales);
        d.up_proj_w           = dev_i8(s.up_proj_w);
        d.up_proj_scales      = dev_h(s.up_proj_scales);
        d.down_proj_w         = dev_i8(s.down_proj_w);
        d.down_proj_scales    = dev_h(s.down_proj_scales);
    }

    size_t cache = (size_t)header_.num_layers * KV_CACHE_CAP * KV;
    CUDA_CHECK(cudaMalloc(&d_kcache_, cache * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_vcache_, cache * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_x_,      H  * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_xn_,     H  * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_q_,      QD * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_attn_,   QD * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_gate_,   I  * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_up_,     I  * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_logits_, (size_t)vocab_ * sizeof(float)));

    logits_.resize(vocab_);
}

GpuRunnerInt8::~GpuRunnerInt8() {
    cudaFree(d_weights_);
    cudaFree(d_kcache_); cudaFree(d_vcache_);
    cudaFree(d_x_); cudaFree(d_xn_); cudaFree(d_q_); cudaFree(d_attn_);
    cudaFree(d_gate_); cudaFree(d_up_); cudaFree(d_logits_);
}

const float* GpuRunnerInt8::forward(int token_id, int pos) {
    const int H  = header_.hidden_size;
    const int I  = header_.intermediate_size;
    const int NH = header_.num_heads;
    const int NKV = header_.num_kv_heads;
    const int HD = header_.head_dim;
    const int QD = NH * HD;
    const int KV = NKV * HD;

    // 1. embedding lookup — dequant one INT8 row into FP16
    embed_dequant_int8_cuda(asHalf(d_x_), d_embed_int8_, asHalf(d_embed_scales_),
                            token_id, H);

    for (int l = 0; l < (int)header_.num_layers; ++l) {
        const LayerWeightsInt8& L = d_layers_[l];
        uint16_t* kbase = d_kcache_ + (size_t)l * KV_CACHE_CAP * KV;
        uint16_t* vbase = d_vcache_ + (size_t)l * KV_CACHE_CAP * KV;
        uint16_t* k_dst = kbase + (size_t)pos * KV;
        uint16_t* v_dst = vbase + (size_t)pos * KV;

        // attention block
        rmsnorm_fp16_cuda(asHalf(d_xn_), asHalf(d_x_),
                          asHalf(L.input_layernorm), H);
        matmul_int8_cuda(asHalf(d_q_), L.q_proj_w, asHalf(L.q_proj_scales),
                         asHalf(d_xn_), asHalf(L.q_proj_b), QD, H);
        matmul_int8_cuda(asHalf(k_dst), L.k_proj_w, asHalf(L.k_proj_scales),
                         asHalf(d_xn_), asHalf(L.k_proj_b), KV, H);
        matmul_int8_cuda(asHalf(v_dst), L.v_proj_w, asHalf(L.v_proj_scales),
                         asHalf(d_xn_), asHalf(L.v_proj_b), KV, H);
        rope_fp16_cuda(asHalf(d_q_),  NH,  HD, pos);
        rope_fp16_cuda(asHalf(k_dst), NKV, HD, pos);
        attention_fp16_cuda(asHalf(d_attn_), asHalf(d_q_),
                            asHalf(kbase), asHalf(vbase),
                            pos, NH, NKV, HD);
        matmul_int8_cuda(asHalf(d_xn_), L.o_proj_w, asHalf(L.o_proj_scales),
                         asHalf(d_attn_), nullptr, H, QD);
        residual_add(d_x_, d_xn_, H);

        // SwiGLU FFN
        rmsnorm_fp16_cuda(asHalf(d_xn_), asHalf(d_x_),
                          asHalf(L.post_attn_layernorm), H);
        matmul_int8_cuda(asHalf(d_gate_), L.gate_proj_w,
                         asHalf(L.gate_proj_scales),
                         asHalf(d_xn_), nullptr, I, H);
        matmul_int8_cuda(asHalf(d_up_),   L.up_proj_w,
                         asHalf(L.up_proj_scales),
                         asHalf(d_xn_), nullptr, I, H);
        swiglu_fp16_cuda(asHalf(d_gate_), asHalf(d_up_), I);
        matmul_int8_cuda(asHalf(d_xn_), L.down_proj_w,
                         asHalf(L.down_proj_scales),
                         asHalf(d_gate_), nullptr, H, I);
        residual_add(d_x_, d_xn_, H);
    }

    // final norm + LM head (tied INT8 embedding) -> FP32 logits
    rmsnorm_fp16_cuda(asHalf(d_xn_), asHalf(d_x_), asHalf(d_final_norm_), H);
    matmul_int8_to_fp32_cuda(d_logits_, d_embed_int8_,
                             asHalf(d_embed_scales_), asHalf(d_xn_), vocab_, H);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(logits_.data(), d_logits_,
                          (size_t)vocab_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return logits_.data();
}
