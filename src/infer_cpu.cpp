#include "infer_cpu.h"
#include <cmath>
#include <cstring>

using namespace qwen2;

// -------- primitive ops --------------------------------------------------

// y = W @ x (+ bias).  W is row-major [n_out, n_in].
// The OpenMP pragma parallelizes over output rows — this is the single line
// that takes us from the naive baseline to the multi-threaded one.
static void matmul(float* y, const float* W, const float* x, const float* bias,
                   int n_out, int n_in) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < n_out; ++o) {
        const float* w = W + (size_t)o * n_in;
        float acc = bias ? bias[o] : 0.0f;
        for (int i = 0; i < n_in; ++i) acc += w[i] * x[i];
        y[o] = acc;
    }
}

// RMSNorm: normalize by root-mean-square, then per-channel scale.
static void rmsnorm(float* out, const float* x, const float* weight, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    float scale = 1.0f / std::sqrt(ss / n + RMS_NORM_EPS);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * weight[i];
}

// RoPE — rotary position embedding, "rotate-half" form (Llama/Qwen2 style).
// Applied independently to each head's head_dim-sized slice.
static void rope(float* vec, int n_heads, int head_dim, int pos) {
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; ++h) {
        float* v = vec + h * head_dim;
        for (int i = 0; i < half; ++i) {
            float inv_freq = std::pow(ROPE_THETA, -2.0f * i / head_dim);
            float ang = pos * inv_freq;
            float c = std::cos(ang), s = std::sin(ang);
            float x0 = v[i], x1 = v[i + half];
            v[i]        = x0 * c - x1 * s;
            v[i + half] = x1 * c + x0 * s;
        }
    }
}

static inline float silu(float z) { return z / (1.0f + std::exp(-z)); }

// -------- runner ---------------------------------------------------------

CpuRunner::CpuRunner(const Model& model) : model_(model) {
    const auto& h = model_.header;
    int H = h.hidden_size, I = h.intermediate_size;
    int QD = h.num_heads * h.head_dim, KV = h.num_kv_heads * h.head_dim;

    size_t cache = (size_t)h.num_layers * KV_CACHE_CAP * KV;
    kcache_.assign(cache, 0.0f);
    vcache_.assign(cache, 0.0f);

    x_.resize(H); xn_.resize(H);
    q_.resize(QD); k_.resize(KV); v_.resize(KV);
    attn_.resize(QD);
    scores_.resize(KV_CACHE_CAP);
    gate_.resize(I); up_.resize(I);
    logits_.resize(h.vocab_size);
}

const float* CpuRunner::forward(int token_id, int pos) {
    const auto& h = model_.header;
    const int H  = h.hidden_size;
    const int I  = h.intermediate_size;
    const int NH = h.num_heads;
    const int NKV = h.num_kv_heads;
    const int HD = h.head_dim;
    const int QD = NH * HD;
    const int KV = NKV * HD;
    const int group = NH / NKV;
    const float attn_scale = 1.0f / std::sqrt((float)HD);

    // 1. token embedding lookup
    std::memcpy(x_.data(), model_.embed_tokens + (size_t)token_id * H,
                H * sizeof(float));

    for (int l = 0; l < h.num_layers; ++l) {
        const LayerWeights& L = model_.layers[l];

        // 2. attention block
        rmsnorm(xn_.data(), x_.data(), L.input_layernorm, H);
        matmul(q_.data(), L.q_proj_w, xn_.data(), L.q_proj_b, QD, H);
        matmul(k_.data(), L.k_proj_w, xn_.data(), L.k_proj_b, KV, H);
        matmul(v_.data(), L.v_proj_w, xn_.data(), L.v_proj_b, KV, H);

        rope(q_.data(), NH, HD, pos);
        rope(k_.data(), NKV, HD, pos);

        // store K,V for this position into the layer's cache
        float* kc = kcache_.data() + ((size_t)l * KV_CACHE_CAP + pos) * KV;
        float* vc = vcache_.data() + ((size_t)l * KV_CACHE_CAP + pos) * KV;
        std::memcpy(kc, k_.data(), KV * sizeof(float));
        std::memcpy(vc, v_.data(), KV * sizeof(float));

        // causal attention, per query head (GQA: heads share KV heads)
        const float* kbase = kcache_.data() + (size_t)l * KV_CACHE_CAP * KV;
        const float* vbase = vcache_.data() + (size_t)l * KV_CACHE_CAP * KV;
        #pragma omp parallel for schedule(static)
        for (int hd = 0; hd < NH; ++hd) {
            int kvh = hd / group;
            const float* qh = q_.data() + hd * HD;
            float local_scores[KV_CACHE_CAP];

            float maxs = -1e30f;
            for (int t = 0; t <= pos; ++t) {
                const float* kt = kbase + (size_t)t * KV + kvh * HD;
                float dot = 0.0f;
                for (int d = 0; d < HD; ++d) dot += qh[d] * kt[d];
                dot *= attn_scale;
                local_scores[t] = dot;
                if (dot > maxs) maxs = dot;
            }
            float sum = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                local_scores[t] = std::exp(local_scores[t] - maxs);
                sum += local_scores[t];
            }
            float* out = attn_.data() + hd * HD;
            for (int d = 0; d < HD; ++d) out[d] = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                const float* vt = vbase + (size_t)t * KV + kvh * HD;
                float w = local_scores[t] / sum;
                for (int d = 0; d < HD; ++d) out[d] += w * vt[d];
            }
        }

        // output projection (no bias) + residual add
        matmul(xn_.data(), L.o_proj_w, attn_.data(), nullptr, H, QD);
        for (int i = 0; i < H; ++i) x_[i] += xn_[i];

        // 3. SwiGLU FFN
        rmsnorm(xn_.data(), x_.data(), L.post_attn_layernorm, H);
        matmul(gate_.data(), L.gate_proj_w, xn_.data(), nullptr, I, H);
        matmul(up_.data(),   L.up_proj_w,   xn_.data(), nullptr, I, H);
        for (int i = 0; i < I; ++i) gate_[i] = silu(gate_[i]) * up_[i];
        matmul(xn_.data(), L.down_proj_w, gate_.data(), nullptr, H, I);
        for (int i = 0; i < H; ++i) x_[i] += xn_[i];
    }

    // 4. final norm + LM head (tied to the embedding matrix)
    rmsnorm(xn_.data(), x_.data(), model_.final_norm, H);
    matmul(logits_.data(), model_.embed_tokens, xn_.data(), nullptr,
           h.vocab_size, H);
    return logits_.data();
}
