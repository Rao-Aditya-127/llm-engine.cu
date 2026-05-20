#pragma once
#include "config.h"

// Picks the next token id from a logit vector [vocab_size].
// temperature == 0  -> greedy (argmax).
// otherwise         -> temperature + top-p (nucleus) sampling.
int sample(const float* logits, int vocab_size, const RunConfig& cfg,
           unsigned long long& rng_state);
