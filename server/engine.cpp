#include "engine.h"

#include "config.h"
#include "infer_gpu_fp16.h"
#include "model.h"
#include "sampler.h"

// ---------------------------------------------------------------------------
// Impl — owns Model + GpuRunnerFP16.  All CUDA types are confined here.
// ---------------------------------------------------------------------------
// Number of concurrent KV-cache slots the server can keep in flight.
static constexpr int kMaxSlots = 16;

struct LLMEngine::Impl {
    Model         model;
    GpuRunnerFP16 runner;

    explicit Impl(const std::string& path)
        : model(path), runner(model, kMaxSlots) {}
};

// ---------------------------------------------------------------------------
// LLMEngine
// ---------------------------------------------------------------------------
LLMEngine::LLMEngine(const std::string& path)
    : impl_(std::make_unique<Impl>(path)) {}

LLMEngine::~LLMEngine() = default;

std::vector<int> LLMEngine::generate_ids(
        const std::vector<int>& prompt_ids,
        int                     max_tokens,
        float                   temperature,
        float                   top_p,
        unsigned long long      seed) {

    // Prefill: single batched GEMM pass over all prompt tokens.
    const float* logits = impl_->runner.prefill(
        prompt_ids.data(), static_cast<int>(prompt_ids.size()));

    RunConfig cfg;
    cfg.temperature    = temperature;
    cfg.top_p          = top_p;
    cfg.max_new_tokens = max_tokens;
    unsigned long long rng = seed;

    std::vector<int> out;
    out.reserve(max_tokens);
    int pos = static_cast<int>(prompt_ids.size());

    for (int step = 0; step < max_tokens; ++step) {
        int tok = sample(logits, impl_->runner.vocab_size(), cfg, rng);
        out.push_back(tok);
        if (tok == qwen2::EOS_TOKEN_ID || tok == qwen2::IM_END_TOKEN_ID) break;
        logits = impl_->runner.forward(tok, pos++);
    }
    return out;
}

void LLMEngine::generate_ids_streaming(
        const std::vector<int>&         prompt_ids,
        int                             max_tokens,
        const std::function<void(int)>& on_token,
        float                           temperature,
        float                           top_p,
        unsigned long long              seed) {

    const float* logits = impl_->runner.prefill(
        prompt_ids.data(), static_cast<int>(prompt_ids.size()));

    RunConfig cfg;
    cfg.temperature    = temperature;
    cfg.top_p          = top_p;
    cfg.max_new_tokens = max_tokens;
    unsigned long long rng = seed;

    int pos = static_cast<int>(prompt_ids.size());

    for (int step = 0; step < max_tokens; ++step) {
        int tok = sample(logits, impl_->runner.vocab_size(), cfg, rng);
        if (tok == qwen2::EOS_TOKEN_ID || tok == qwen2::IM_END_TOKEN_ID) break;
        on_token(tok);
        logits = impl_->runner.forward(tok, pos++);
    }
}

int LLMEngine::prefill_slot(const std::vector<int>& prompt_ids, int slot,
                            float temperature, float top_p,
                            unsigned long long seed) {
    return impl_->runner.prefill_slot(
        prompt_ids.data(), static_cast<int>(prompt_ids.size()),
        slot, temperature, top_p, seed);
}

std::vector<int> LLMEngine::decode_batch(const std::vector<int>& tokens,
                                         const std::vector<int>& positions,
                                         const std::vector<int>& slots) {
    return impl_->runner.decode_batch(
        tokens.data(), positions.data(), slots.data(),
        static_cast<int>(tokens.size()));
}

int LLMEngine::max_slots() const {
    return impl_->runner.max_slots();
}

int LLMEngine::vocab_size() const {
    return impl_->runner.vocab_size();
}
