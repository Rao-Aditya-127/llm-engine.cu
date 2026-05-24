#include "infer_gpu_fp32.h"
#include "fp32/kernels.cuh"
#include "config.h"
#include <stdexcept>

using namespace qwen2;

// Residual add: x += y. Glue, not a "real" kernel, so it lives here.
__global__ void residual_add_kernel(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
static void residual_add(float* x, const float* y, int n) {
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads>>>(x, y, n);
}

// ---------------------------------------------------------------------------

GpuRunner::GpuRunner(const Model& model) {
    if (model.header.dtype != 0)
        throw std::runtime_error("GpuRunner expects an fp32 .bin (dtype=0)");
    header_ = model.header;
    vocab_  = header_.vocab_size;

    const int H  = header_.hidden_size;
    const int I  = header_.intermediate_size;
    const int QD = header_.num_heads * header_.head_dim;
    const int KV = header_.num_kv_heads * header_.head_dim;

    // Upload the entire weight blob to the device in one copy.
    size_t nf = model.num_floats();
    CUDA_CHECK(cudaMalloc(&d_weights_, nf * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_weights_, model.base(), nf * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Rebase every host weight pointer to the matching device offset.
    const float* hbase = model.base();
    auto dev = [&](const float* h) { return d_weights_ + (h - hbase); };

    d_embed_      = dev(model.embed_tokens);
    d_final_norm_ = dev(model.final_norm);

    d_layers_.resize(header_.num_layers);
    for (int l = 0; l < (int)header_.num_layers; ++l) {
        const LayerWeights& s = model.layers[l];
        LayerWeights& d = d_layers_[l];
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
    CUDA_CHECK(cudaMalloc(&d_kcache_, cache * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vcache_, cache * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_,      H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xn_,     H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_,      QD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_,   QD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gate_,   I * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up_,     I * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits_, (size_t)vocab_ * sizeof(float)));

    logits_.resize(vocab_);
}

GpuRunner::~GpuRunner() {
    cudaFree(d_weights_);
    cudaFree(d_kcache_); cudaFree(d_vcache_);
    cudaFree(d_x_); cudaFree(d_xn_); cudaFree(d_q_); cudaFree(d_attn_);
    cudaFree(d_gate_); cudaFree(d_up_); cudaFree(d_logits_);
}

const float* GpuRunner::forward(int token_id, int pos) {
    const int H  = header_.hidden_size;
    const int I  = header_.intermediate_size;
    const int NH = header_.num_heads;
    const int NKV = header_.num_kv_heads;
    const int HD = header_.head_dim;
    const int QD = NH * HD;
    const int KV = NKV * HD;

    // 1. token embedding lookup (device-to-device copy of one row)
    CUDA_CHECK(cudaMemcpy(d_x_, d_embed_ + (size_t)token_id * H,
                          H * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int l = 0; l < (int)header_.num_layers; ++l) {
        const LayerWeights& L = d_layers_[l];
        float* kbase = d_kcache_ + (size_t)l * KV_CACHE_CAP * KV;
        float* vbase = d_vcache_ + (size_t)l * KV_CACHE_CAP * KV;
        float* k_dst = kbase + (size_t)pos * KV;   // this token's K slot
        float* v_dst = vbase + (size_t)pos * KV;   // this token's V slot

        // attention block — K/V are written straight into the cache
        rmsnorm_cuda(d_xn_, d_x_, L.input_layernorm, H);
        matmul_cuda(d_q_, L.q_proj_w, d_xn_, L.q_proj_b, QD, H);
        matmul_cuda(k_dst, L.k_proj_w, d_xn_, L.k_proj_b, KV, H);
        matmul_cuda(v_dst, L.v_proj_w, d_xn_, L.v_proj_b, KV, H);
        rope_cuda(d_q_, NH, HD, pos);
        rope_cuda(k_dst, NKV, HD, pos);
        attention_cuda(d_attn_, d_q_, kbase, vbase, pos, NH, NKV, HD);
        matmul_cuda(d_xn_, L.o_proj_w, d_attn_, nullptr, H, QD);
        residual_add(d_x_, d_xn_, H);

        // SwiGLU FFN
        rmsnorm_cuda(d_xn_, d_x_, L.post_attn_layernorm, H);
        matmul_cuda(d_gate_, L.gate_proj_w, d_xn_, nullptr, I, H);
        matmul_cuda(d_up_,   L.up_proj_w,   d_xn_, nullptr, I, H);
        swiglu_cuda(d_gate_, d_up_, I);
        matmul_cuda(d_xn_, L.down_proj_w, d_gate_, nullptr, H, I);
        residual_add(d_x_, d_xn_, H);
    }

    // final norm + LM head (tied embedding)
    rmsnorm_cuda(d_xn_, d_x_, d_final_norm_, H);
    matmul_cuda(d_logits_, d_embed_, d_xn_, nullptr, vocab_, H);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(logits_.data(), d_logits_, (size_t)vocab_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return logits_.data();
}
