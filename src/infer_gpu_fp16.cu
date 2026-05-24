#include "infer_gpu_fp16.h"
#include "fp16/kernels.cuh"
#include "config.h"
#include <stdexcept>

using namespace qwen2;

// Helpers to bridge the plain-C++ header (uint16_t*) and the CUDA side (__half*).
static inline __half*       asHalf(uint16_t* p)       { return reinterpret_cast<__half*>(p); }
static inline const __half* asHalf(const uint16_t* p) { return reinterpret_cast<const __half*>(p); }

// Residual add in FP16. __hadd is the native FP16 add (sm_53+, T4 = sm_75).
__global__ void residual_add_fp16_kernel(__half* x, const __half* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __hadd(x[i], y[i]);
}
static void residual_add(uint16_t* x, const uint16_t* y, int n) {
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    residual_add_fp16_kernel<<<blocks, threads>>>(asHalf(x), asHalf(y), n);
}

// ---------------------------------------------------------------------------

GpuRunnerFP16::GpuRunnerFP16(const Model& model) {
    if (model.header.dtype != 1)
        throw std::runtime_error("GpuRunnerFP16 expects an fp16 .bin (dtype=1)");

    header_ = model.header;
    vocab_  = header_.vocab_size;

    const int H  = header_.hidden_size;
    const int I  = header_.intermediate_size;
    const int QD = header_.num_heads * header_.head_dim;
    const int KV = header_.num_kv_heads * header_.head_dim;

    // Upload the entire FP16 weight blob in one copy.
    size_t nh = model.num_halves();
    CUDA_CHECK(cudaMalloc(&d_weights_h_, nh * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_weights_h_, model.base_h(), nh * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));

    // Rebase host pointers to device offsets.
    const uint16_t* hbase = model.base_h();
    auto dev = [&](const uint16_t* h) { return d_weights_h_ + (h - hbase); };

    d_embed_h_      = dev(model.embed_tokens_h);
    d_final_norm_h_ = dev(model.final_norm_h);

    d_layers_h_.resize(header_.num_layers);
    for (int l = 0; l < (int)header_.num_layers; ++l) {
        const LayerWeightsHalf& s = model.layers_h[l];
        LayerWeightsHalf& d = d_layers_h_[l];
        d.input_layernorm    = dev(s.input_layernorm);
        d.q_proj_w           = dev(s.q_proj_w);
        d.q_proj_b           = dev(s.q_proj_b);
        d.k_proj_w           = dev(s.k_proj_w);
        d.k_proj_b           = dev(s.k_proj_b);
        d.v_proj_w           = dev(s.v_proj_w);
        d.v_proj_b           = dev(s.v_proj_b);
        d.o_proj_w           = dev(s.o_proj_w);
        d.post_attn_layernorm = dev(s.post_attn_layernorm);
        d.gate_proj_w        = dev(s.gate_proj_w);
        d.up_proj_w          = dev(s.up_proj_w);
        d.down_proj_w        = dev(s.down_proj_w);
    }

    size_t cache = (size_t)header_.num_layers * KV_CACHE_CAP * KV;
    CUDA_CHECK(cudaMalloc(&d_kcache_, cache * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_vcache_, cache * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_x_,      H * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_xn_,     H * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_q_,      QD * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_attn_,   QD * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_gate_,   I * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_up_,     I * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_logits_, (size_t)vocab_ * sizeof(float)));

    logits_.resize(vocab_);
}

GpuRunnerFP16::~GpuRunnerFP16() {
    cudaFree(d_weights_h_);
    cudaFree(d_kcache_); cudaFree(d_vcache_);
    cudaFree(d_x_); cudaFree(d_xn_); cudaFree(d_q_); cudaFree(d_attn_);
    cudaFree(d_gate_); cudaFree(d_up_); cudaFree(d_logits_);
}

const float* GpuRunnerFP16::forward(int token_id, int pos) {
    const int H  = header_.hidden_size;
    const int I  = header_.intermediate_size;
    const int NH = header_.num_heads;
    const int NKV = header_.num_kv_heads;
    const int HD = header_.head_dim;
    const int QD = NH * HD;
    const int KV = NKV * HD;

    // 1. token embedding lookup (DtoD copy of one FP16 row)
    CUDA_CHECK(cudaMemcpy(d_x_, d_embed_h_ + (size_t)token_id * H,
                          H * sizeof(uint16_t), cudaMemcpyDeviceToDevice));

    for (int l = 0; l < (int)header_.num_layers; ++l) {
        const LayerWeightsHalf& L = d_layers_h_[l];
        uint16_t* kbase = d_kcache_ + (size_t)l * KV_CACHE_CAP * KV;
        uint16_t* vbase = d_vcache_ + (size_t)l * KV_CACHE_CAP * KV;
        uint16_t* k_dst = kbase + (size_t)pos * KV;
        uint16_t* v_dst = vbase + (size_t)pos * KV;

        // attention block — K/V are written straight into the cache
        rmsnorm_fp16_cuda(asHalf(d_xn_), asHalf(d_x_),
                          asHalf(L.input_layernorm), H);
        matmul_fp16_cuda(asHalf(d_q_), asHalf(L.q_proj_w),
                         asHalf(d_xn_), asHalf(L.q_proj_b), QD, H);
        matmul_fp16_cuda(asHalf(k_dst), asHalf(L.k_proj_w),
                         asHalf(d_xn_), asHalf(L.k_proj_b), KV, H);
        matmul_fp16_cuda(asHalf(v_dst), asHalf(L.v_proj_w),
                         asHalf(d_xn_), asHalf(L.v_proj_b), KV, H);
        rope_fp16_cuda(asHalf(d_q_),  NH,  HD, pos);
        rope_fp16_cuda(asHalf(k_dst), NKV, HD, pos);
        attention_fp16_cuda(asHalf(d_attn_), asHalf(d_q_),
                            asHalf(kbase), asHalf(vbase),
                            pos, NH, NKV, HD);
        matmul_fp16_cuda(asHalf(d_xn_), asHalf(L.o_proj_w),
                         asHalf(d_attn_), nullptr, H, QD);
        residual_add(d_x_, d_xn_, H);

        // SwiGLU FFN
        rmsnorm_fp16_cuda(asHalf(d_xn_), asHalf(d_x_),
                          asHalf(L.post_attn_layernorm), H);
        matmul_fp16_cuda(asHalf(d_gate_), asHalf(L.gate_proj_w),
                         asHalf(d_xn_), nullptr, I, H);
        matmul_fp16_cuda(asHalf(d_up_),   asHalf(L.up_proj_w),
                         asHalf(d_xn_), nullptr, I, H);
        swiglu_fp16_cuda(asHalf(d_gate_), asHalf(d_up_), I);
        matmul_fp16_cuda(asHalf(d_xn_), asHalf(L.down_proj_w),
                         asHalf(d_gate_), nullptr, H, I);
        residual_add(d_x_, d_xn_, H);
    }

    // final norm + LM head (tied embedding). LM head writes FP32 logits
    // directly so the host sampler doesn't need a conversion step.
    rmsnorm_fp16_cuda(asHalf(d_xn_), asHalf(d_x_), asHalf(d_final_norm_h_), H);
    matmul_fp16_to_fp32_cuda(d_logits_, asHalf(d_embed_h_), asHalf(d_xn_),
                             vocab_, H);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(logits_.data(), d_logits_,
                          (size_t)vocab_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return logits_.data();
}
