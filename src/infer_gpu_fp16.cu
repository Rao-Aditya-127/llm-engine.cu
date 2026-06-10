#include "infer_gpu_fp16.h"
#include "fp16/kernels.cuh"
#include "config.h"
#include "sampler.h"
#include <algorithm>
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

// Embedding gather: d_out[t × H] = table[d_ids[t] × H]  for t = 0..seq_len-1.
// One block per token, threads stride over H.
__global__ void embed_gather_fp16_kernel(__half* out, const __half* table,
                                         const int* ids, int H) {
    int t = blockIdx.x;
    const __half* row = table + (size_t)ids[t] * H;
    __half*       dst = out   + (size_t)t      * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        dst[i] = row[i];
}

// ---------------------------------------------------------------------------

GpuRunnerFP16::GpuRunnerFP16(const Model& model, int max_slots) {
    if (model.header.dtype != 1)
        throw std::runtime_error("GpuRunnerFP16 expects an fp16 .bin (dtype=1)");

    header_ = model.header;
    vocab_  = header_.vocab_size;
    max_slots_ = max_slots < 1 ? 1 : max_slots;
    slot_cfg_.resize(max_slots_);

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

    // KV cache gains a leading slot dimension: [max_slots × layers × cap × KV].
    size_t cache = (size_t)max_slots_ * header_.num_layers * KV_CACHE_CAP * KV;
    CUDA_CHECK(cudaMalloc(&d_kcache_, cache * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_vcache_, cache * sizeof(uint16_t)));
    // Activation buffers are over-allocated to KV_CACHE_CAP rows so the same
    // pointers work for both the single-token decode path (row 0) and the
    // batched prefill path (rows 0..seq_len-1).
    CUDA_CHECK(cudaMalloc(&d_x_,    (size_t)KV_CACHE_CAP * H  * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_xn_,   (size_t)KV_CACHE_CAP * H  * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_q_,    (size_t)KV_CACHE_CAP * QD * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_attn_, (size_t)KV_CACHE_CAP * QD * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_gate_, (size_t)KV_CACHE_CAP * I  * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_up_,   (size_t)KV_CACHE_CAP * I  * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_logits_, (size_t)max_slots_ * vocab_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ids_,    (size_t)KV_CACHE_CAP * sizeof(int)));

    // Decode-batch scratch (sized to max_slots).
    CUDA_CHECK(cudaMalloc(&d_ktmp_, (size_t)max_slots_ * KV * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_vtmp_, (size_t)max_slots_ * KV * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_pos_,  (size_t)max_slots_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_slot_, (size_t)max_slots_ * sizeof(int)));

    logits_.resize((size_t)max_slots_ * vocab_);
}

GpuRunnerFP16::~GpuRunnerFP16() {
    cudaFree(d_weights_h_);
    cudaFree(d_kcache_); cudaFree(d_vcache_);
    cudaFree(d_x_); cudaFree(d_xn_); cudaFree(d_q_); cudaFree(d_attn_);
    cudaFree(d_gate_); cudaFree(d_up_); cudaFree(d_logits_); cudaFree(d_ids_);
    cudaFree(d_ktmp_); cudaFree(d_vtmp_); cudaFree(d_pos_); cudaFree(d_slot_);
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

// ---------------------------------------------------------------------------
// Batched prefill: processes all seq_len prompt tokens in a single GEMM pass
// per layer rather than seq_len separate GEMV calls. Writes K/V into the given
// KV-cache slot's rows 0..seq_len-1. Returns host logits for the last token.
// ---------------------------------------------------------------------------
const float* GpuRunnerFP16::prefill(const int* ids, int seq_len) {
    return run_prefill(ids, seq_len, 0);
}

const float* GpuRunnerFP16::run_prefill(const int* ids, int seq_len, int slot) {
    const int H   = header_.hidden_size;
    const int I   = header_.intermediate_size;
    const int NH  = header_.num_heads;
    const int NKV = header_.num_kv_heads;
    const int HD  = header_.head_dim;
    const int QD  = NH  * HD;
    const int KV  = NKV * HD;
    const int NL  = header_.num_layers;

    // Copy token indices to device for the gather kernel.
    CUDA_CHECK(cudaMemcpy(d_ids_, ids, (size_t)seq_len * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Gather embeddings: d_x_[t × H] = embed[ids[t]]  for t = 0..seq_len-1
    embed_gather_fp16_kernel<<<seq_len, 256>>>(asHalf(d_x_), asHalf(d_embed_h_),
                                               d_ids_, H);

    for (int l = 0; l < NL; ++l) {
        const LayerWeightsHalf& L = d_layers_h_[l];
        // K/V written directly into slot `slot`, rows 0..seq_len-1.
        size_t kv_base = (((size_t)slot * NL + l) * KV_CACHE_CAP) * KV;
        uint16_t* kbase = d_kcache_ + kv_base;
        uint16_t* vbase = d_vcache_ + kv_base;

        // ---- attention block ----
        rmsnorm_batched_fp16_cuda(asHalf(d_xn_), asHalf(d_x_),
                                  asHalf(L.input_layernorm), H, seq_len);

        matmul_batched_fp16_cuda(asHalf(d_q_),    asHalf(L.q_proj_w),
                                 asHalf(d_xn_),   asHalf(L.q_proj_b),
                                 seq_len, QD, H);
        matmul_batched_fp16_cuda(asHalf(kbase),   asHalf(L.k_proj_w),
                                 asHalf(d_xn_),   asHalf(L.k_proj_b),
                                 seq_len, KV, H);
        matmul_batched_fp16_cuda(asHalf(vbase),   asHalf(L.v_proj_w),
                                 asHalf(d_xn_),   asHalf(L.v_proj_b),
                                 seq_len, KV, H);

        rope_batched_fp16_cuda(asHalf(d_q_),  NH,  HD, seq_len);
        rope_batched_fp16_cuda(asHalf(kbase), NKV, HD, seq_len);

        attention_prefill_fp16_cuda(asHalf(d_attn_), asHalf(d_q_),
                                    asHalf(kbase), asHalf(vbase),
                                    seq_len, NH, NKV, HD);

        matmul_batched_fp16_cuda(asHalf(d_xn_), asHalf(L.o_proj_w),
                                 asHalf(d_attn_), nullptr,
                                 seq_len, H, QD);
        residual_add(d_x_, d_xn_, H * seq_len);

        // ---- SwiGLU FFN ----
        rmsnorm_batched_fp16_cuda(asHalf(d_xn_), asHalf(d_x_),
                                  asHalf(L.post_attn_layernorm), H, seq_len);

        matmul_batched_fp16_cuda(asHalf(d_gate_), asHalf(L.gate_proj_w),
                                 asHalf(d_xn_), nullptr,
                                 seq_len, I, H);
        matmul_batched_fp16_cuda(asHalf(d_up_),   asHalf(L.up_proj_w),
                                 asHalf(d_xn_), nullptr,
                                 seq_len, I, H);
        // Element-wise SwiGLU: reuse existing kernel, just cover seq_len × I elements.
        swiglu_fp16_cuda(asHalf(d_gate_), asHalf(d_up_), I * seq_len);

        matmul_batched_fp16_cuda(asHalf(d_xn_), asHalf(L.down_proj_w),
                                 asHalf(d_gate_), nullptr,
                                 seq_len, H, I);
        residual_add(d_x_, d_xn_, H * seq_len);
    }

    // Final norm + LM head on the LAST token's hidden state only.
    const uint16_t* last_x = d_x_ + (size_t)(seq_len - 1) * H;
    rmsnorm_fp16_cuda(asHalf(d_xn_), asHalf(last_x), asHalf(d_final_norm_h_), H);
    matmul_fp16_to_fp32_cuda(d_logits_, asHalf(d_embed_h_), asHalf(d_xn_),
                             vocab_, H);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(logits_.data(), d_logits_,
                          (size_t)vocab_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return logits_.data();
}

// ---------------------------------------------------------------------------
// Continuous-batch: prefill a prompt into a slot, sample the first token.
// ---------------------------------------------------------------------------
int GpuRunnerFP16::prefill_slot(const int* ids, int seq_len, int slot,
                                float temperature, float top_p,
                                unsigned long long seed) {
    SlotSampling& sc = slot_cfg_[slot];
    sc.temperature = temperature;
    sc.top_p       = top_p;
    sc.rng         = seed;

    const float* logits = run_prefill(ids, seq_len, slot);

    RunConfig cfg;
    cfg.temperature = sc.temperature;
    cfg.top_p       = sc.top_p;
    return sample(logits, vocab_, cfg, sc.rng);
}

// ---------------------------------------------------------------------------
// Continuous-batch: one decode step over `batch` sequences. Each row is a
// different sequence at positions[r] in KV-cache slot slots[r]. Reuses the
// batched GEMM machinery; attention and RoPE read per-row position/slot arrays.
// ---------------------------------------------------------------------------
std::vector<int> GpuRunnerFP16::decode_batch(const int* tokens,
                                             const int* positions,
                                             const int* slots, int batch) {
    const int H   = header_.hidden_size;
    const int I   = header_.intermediate_size;
    const int NH  = header_.num_heads;
    const int NKV = header_.num_kv_heads;
    const int HD  = header_.head_dim;
    const int QD  = NH  * HD;
    const int KV  = NKV * HD;
    const int NL  = header_.num_layers;

    // Upload per-row tokens / positions / slots.
    CUDA_CHECK(cudaMemcpy(d_ids_, tokens, (size_t)batch * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_, positions, (size_t)batch * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slot_, slots, (size_t)batch * sizeof(int),
                          cudaMemcpyHostToDevice));

    int max_seqlen = 0;
    for (int b = 0; b < batch; ++b)
        max_seqlen = std::max(max_seqlen, positions[b] + 1);

    // Gather embeddings: d_x_[b × H] = embed[tokens[b]].
    embed_gather_fp16_kernel<<<batch, 256>>>(asHalf(d_x_), asHalf(d_embed_h_),
                                             d_ids_, H);

    for (int l = 0; l < NL; ++l) {
        const LayerWeightsHalf& L = d_layers_h_[l];

        // ---- attention block ----
        rmsnorm_batched_fp16_cuda(asHalf(d_xn_), asHalf(d_x_),
                                  asHalf(L.input_layernorm), H, batch);

        // Q into d_q_; K/V into temp buffers (then scattered to cache slots).
        matmul_batched_fp16_cuda(asHalf(d_q_),    asHalf(L.q_proj_w),
                                 asHalf(d_xn_),   asHalf(L.q_proj_b),
                                 batch, QD, H);
        matmul_batched_fp16_cuda(asHalf(d_ktmp_), asHalf(L.k_proj_w),
                                 asHalf(d_xn_),   asHalf(L.k_proj_b),
                                 batch, KV, H);
        matmul_batched_fp16_cuda(asHalf(d_vtmp_), asHalf(L.v_proj_w),
                                 asHalf(d_xn_),   asHalf(L.v_proj_b),
                                 batch, KV, H);

        // RoPE Q and K with per-row positions.
        rope_decode_batched_fp16_cuda(asHalf(d_q_),    d_pos_, NH,  HD, batch);
        rope_decode_batched_fp16_cuda(asHalf(d_ktmp_), d_pos_, NKV, HD, batch);

        // Scatter K/V into their per-slot, per-position cache cells.
        kv_scatter_fp16_cuda(asHalf(d_kcache_), asHalf(d_vcache_),
                             asHalf(d_ktmp_), asHalf(d_vtmp_),
                             d_pos_, d_slot_, batch, KV, l, NL, KV_CACHE_CAP);

        attention_decode_batched_fp16_cuda(asHalf(d_attn_), asHalf(d_q_),
                                           asHalf(d_kcache_), asHalf(d_vcache_),
                                           d_pos_, d_slot_, batch, NH, NKV, HD,
                                           l, NL, KV_CACHE_CAP, max_seqlen);

        matmul_batched_fp16_cuda(asHalf(d_xn_), asHalf(L.o_proj_w),
                                 asHalf(d_attn_), nullptr, batch, H, QD);
        residual_add(d_x_, d_xn_, H * batch);

        // ---- SwiGLU FFN ----
        rmsnorm_batched_fp16_cuda(asHalf(d_xn_), asHalf(d_x_),
                                  asHalf(L.post_attn_layernorm), H, batch);
        matmul_batched_fp16_cuda(asHalf(d_gate_), asHalf(L.gate_proj_w),
                                 asHalf(d_xn_), nullptr, batch, I, H);
        matmul_batched_fp16_cuda(asHalf(d_up_),   asHalf(L.up_proj_w),
                                 asHalf(d_xn_), nullptr, batch, I, H);
        swiglu_fp16_cuda(asHalf(d_gate_), asHalf(d_up_), I * batch);
        matmul_batched_fp16_cuda(asHalf(d_xn_), asHalf(L.down_proj_w),
                                 asHalf(d_gate_), nullptr, batch, H, I);
        residual_add(d_x_, d_xn_, H * batch);
    }

    // Final norm (per row) + batched LM head → [batch × vocab] FP32 logits.
    rmsnorm_batched_fp16_cuda(asHalf(d_xn_), asHalf(d_x_),
                              asHalf(d_final_norm_h_), H, batch);
    matmul_batched_fp16_to_fp32_cuda(d_logits_, asHalf(d_embed_h_),
                                     asHalf(d_xn_), batch, vocab_, H);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(logits_.data(), d_logits_,
                          (size_t)batch * vocab_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Sample one token per row using that slot's stored sampling config.
    std::vector<int> out(batch);
    for (int b = 0; b < batch; ++b) {
        SlotSampling& sc = slot_cfg_[slots[b]];
        RunConfig cfg;
        cfg.temperature = sc.temperature;
        cfg.top_p       = sc.top_p;
        out[b] = sample(logits_.data() + (size_t)b * vocab_, vocab_, cfg, sc.rng);
    }
    return out;
}
