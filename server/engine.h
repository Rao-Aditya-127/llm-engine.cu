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

    int vocab_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
