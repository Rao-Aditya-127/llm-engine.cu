#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <vector>

// xorshift64 — tiny deterministic RNG so runs are reproducible from a seed.
static double next_rand(unsigned long long& s) {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return ((s * 0x2545F4914F6CDD1DULL) >> 11) / (double)(1ULL << 53);
}

int sample(const float* logits, int vocab_size, const RunConfig& cfg,
           unsigned long long& rng_state) {
    // Greedy: just the argmax.
    if (cfg.temperature <= 0.0f) {
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < vocab_size; ++i)
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }

    // Temperature-scaled softmax.
    std::vector<float> probs(vocab_size);
    float maxl = logits[0];
    for (int i = 1; i < vocab_size; ++i) maxl = std::max(maxl, logits[i]);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        float e = std::exp((logits[i] - maxl) / cfg.temperature);
        probs[i] = e;
        sum += e;
    }
    for (float& p : probs) p /= sum;

    // Sort indices by probability, descending.
    std::vector<int> idx(vocab_size);
    for (int i = 0; i < vocab_size; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return probs[a] > probs[b]; });

    // Keep the smallest set whose cumulative probability >= top_p.
    float cum = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; ++i) {
        cum += probs[idx[i]];
        if (cum >= cfg.top_p) { cutoff = i + 1; break; }
    }

    // Sample within the nucleus.
    double r = next_rand(rng_state) * cum;
    double acc = 0.0;
    for (int i = 0; i < cutoff; ++i) {
        acc += probs[idx[i]];
        if (r <= acc) return idx[i];
    }
    return idx[cutoff - 1];
}
