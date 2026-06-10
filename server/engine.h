#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

// LLMEngine — a clean C++ interface over GpuRunnerFP16.
//
// PIMPL keeps all CUDA types out of this header so bindings.cpp can be
// compiled without nvcc.  engine.cpp (compiled by nvcc) holds the Impl
// struct that directly owns Model and GpuRunnerFP16.
class LLMEngine {
public:
    explicit LLMEngine(const std::string& model_path);
    ~LLMEngine();

    // Blocking: prefill + full decode loop. Returns every generated token ID.
    std::vector<int> generate_ids(const std::vector<int>& prompt_ids,
                                  int                    max_tokens,
                                  float                  temperature = 0.0f,
                                  float                  top_p       = 1.0f,
                                  unsigned long long     seed        = 1234ULL);

    // Streaming: calls on_token(token_id) immediately after each decode step.
    void generate_ids_streaming(const std::vector<int>&          prompt_ids,
                                int                              max_tokens,
                                const std::function<void(int)>&  on_token,
                                float                            temperature = 0.0f,
                                float                            top_p       = 1.0f,
                                unsigned long long               seed        = 1234ULL);

    // --- continuous-batch API (used by the server scheduler) ---

    // Prefill a prompt into KV-cache slot `slot`; returns the first sampled token.
    int prefill_slot(const std::vector<int>& prompt_ids, int slot,
                     float temperature = 0.0f, float top_p = 1.0f,
                     unsigned long long seed = 1234ULL);

    // One decode step over a batch of active sequences. Returns the next token
    // id for each. tokens/positions/slots are parallel arrays of equal length.
    std::vector<int> decode_batch(const std::vector<int>& tokens,
                                  const std::vector<int>& positions,
                                  const std::vector<int>& slots);

    int max_slots() const;
    int vocab_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
