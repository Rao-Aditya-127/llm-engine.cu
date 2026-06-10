#pragma once
#include <cstdint>
#include <vector>
#include "model.h"
#include "config.h"

// FP16 GPU forward pass for Qwen2-0.5B. Same public interface as the FP32
// runners. Device pointers are uint16_t* in this header (plain C++) — the
// CUDA implementation in infer_fp16.cu reinterprets them as __half*.
//
// The runner can host up to `max_slots` independent KV-cache slots so the
// continuous-batch server can keep many sequences in flight. The CLI uses the
// default (1 slot); single-sequence forward()/prefill() always use slot 0.
class GpuRunnerFP16 {
public:
    explicit GpuRunnerFP16(const Model& model, int max_slots = 1);
    ~GpuRunnerFP16();

    GpuRunnerFP16(const GpuRunnerFP16&) = delete;
    GpuRunnerFP16& operator=(const GpuRunnerFP16&) = delete;

    // Run one token at sequence position `pos` (decode path, GEMV, slot 0).
    // Returns a host pointer to FP32 logits [vocab_size] (for the sampler).
    const float* forward(int token_id, int pos);

    // Run all prompt tokens in one batched GEMM pass (prefill path, slot 0).
    // Writes K/V for positions 0..seq_len-1 into the cache.
    // Returns host FP32 logits for the last token — feed directly into sample().
    const float* prefill(const int* ids, int seq_len);

    // --- continuous-batch API ---

    // Prefill a prompt into KV-cache slot `slot`, store that slot's sampling
    // config, and return the first sampled token id.
    int prefill_slot(const int* ids, int seq_len, int slot,
                     float temperature, float top_p, unsigned long long seed);

    // One decode step over `batch` sequences. tokens/positions/slots are host
    // arrays of length batch. Samples per-row with each slot's stored config
    // and returns the next token id for each sequence.
    std::vector<int> decode_batch(const int* tokens, const int* positions,
                                  const int* slots, int batch);

    int vocab_size() const { return vocab_; }
    int max_slots()  const { return max_slots_; }

private:
    TinyllmHeader header_{};
    int vocab_ = 0;
    int max_slots_ = 1;

    // Per-slot sampling state for the batched decode path.
    struct SlotSampling {
        float temperature = 0.0f;
        float top_p = 1.0f;
        unsigned long long rng = 1234ULL;
    };
    std::vector<SlotSampling> slot_cfg_;

    // All FP16 weights live in one device blob; the pointers below index it.
    uint16_t* d_weights_h_ = nullptr;
    const uint16_t* d_embed_h_ = nullptr;        // also the tied LM head
    const uint16_t* d_final_norm_h_ = nullptr;
    std::vector<LayerWeightsHalf> d_layers_h_;

    // KV cache and per-step scratch — all FP16.
    // The KV cache has a leading slot dimension:
    //   [max_slots × num_layers × KV_CACHE_CAP × KV_DIM].
    // Activation buffers are sized at KV_CACHE_CAP × dim so that both the
    // single-token decode path (uses row 0 only) and the batched prefill path
    // (uses rows 0..seq_len-1) share the same allocations. A decode batch of
    // up to max_slots rows fits comfortably (max_slots << KV_CACHE_CAP).
    uint16_t* d_kcache_ = nullptr;
    uint16_t* d_vcache_ = nullptr;
    uint16_t* d_x_    = nullptr;   // [KV_CACHE_CAP × H]
    uint16_t* d_xn_   = nullptr;   // [KV_CACHE_CAP × H]
    uint16_t* d_q_    = nullptr;   // [KV_CACHE_CAP × QD]
    uint16_t* d_attn_ = nullptr;   // [KV_CACHE_CAP × QD]
    uint16_t* d_gate_ = nullptr;   // [KV_CACHE_CAP × I]
    uint16_t* d_up_   = nullptr;   // [KV_CACHE_CAP × I]
    int*      d_ids_  = nullptr;   // [KV_CACHE_CAP] — embed gather / batch tokens

    // Decode-batch scratch.
    uint16_t* d_ktmp_ = nullptr;   // [max_slots × KV_DIM] — K before scatter
    uint16_t* d_vtmp_ = nullptr;   // [max_slots × KV_DIM] — V before scatter
    int*      d_pos_  = nullptr;   // [max_slots] — per-row positions
    int*      d_slot_ = nullptr;   // [max_slots] — per-row KV-cache slots

    // Logits stay FP32 (the LM-head matmul writes float directly).
    // Sized [max_slots × vocab] so the batched LM head can write all rows.
    float* d_logits_ = nullptr;
    std::vector<float> logits_;

    // Shared body for prefill() and prefill_slot(): runs the batched-GEMM
    // prompt pass into the given slot and returns host logits for the last token.
    const float* run_prefill(const int* ids, int seq_len, int slot);
};
