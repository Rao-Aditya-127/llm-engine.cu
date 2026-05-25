#include "engine.h"

#include "config.h"
#include "infer_gpu_fp16.h"
#include "model.h"
#include "sampler.h"

// ---------------------------------------------------------------------------
// Impl — owns Model + GpuRunnerFP16.  All CUDA types are confined here.
// ---------------------------------------------------------------------------
struct LLMEngine::Impl {
    Model         model;
    GpuRunnerFP16 runner;

    explicit Impl(const std::string& path) : model(path), runner(model) {}
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

    // Prefill: feed every prompt token to populate the KV cache.
    const float* logits = nullptr;
    for (int i = 0; i < static_cast<int>(prompt_ids.size()); ++i)
        logits = impl_->runner.forward(prompt_ids[i], i);

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

    const float* logits = nullptr;
    for (int i = 0; i < static_cast<int>(prompt_ids.size()); ++i)
        logits = impl_->runner.forward(prompt_ids[i], i);

    RunConfig cfg;
    cfg.temperature    = temperature;
    cfg.top_p          = top_p;
    cfg.max_new_tokens = max_tokens;
    unsigned long long rng = seed;

    int pos = static_cast<int>(prompt_ids.size());

    for (int step = 0; step < max_tokens; ++step) {
        int tok = sample(logits, impl_->runner.vocab_size(), cfg, rng);
        on_token(tok);                              // fire immediately
        logits = impl_->runner.forward(tok, pos++);
    }
}

int LLMEngine::vocab_size() const {
    return impl_->runner.vocab_size();
}
