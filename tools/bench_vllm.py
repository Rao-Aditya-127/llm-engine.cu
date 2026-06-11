"""vLLM throughput vs batch size — the production baseline for our batching table.

Mirrors tools/bench_batch.py as closely as vLLM allows:
  - same model (TINYLLM_MODEL, default Qwen2-0.5B-Instruct), FP16
  - greedy decoding (temperature=0)
  - fixed output length with ignore_eos=True so every sequence in the batch
    generates exactly MAX_TOKENS — no early stop skewing the batch
  - a warm-up generate() before timing so CUDA-graph capture / torch.compile
    cost is excluded, matching our prefill-excluded measurement

Note: vLLM's reported time is end-to-end (prefill + decode). With a short prompt
and MAX_TOKENS=200 the decode phase dominates, so the number is directly
comparable to our decode throughput in practice.

Install (if needed):  pip install vllm
Run:                   python tools/bench_vllm.py
"""
import os
import time

from vllm import LLM, SamplingParams

MODEL_ID   = os.environ.get("TINYLLM_MODEL", "Qwen/Qwen2-0.5B-Instruct")
PROMPT     = "Tell me about machine learning."
MAX_TOKENS = 200
BATCHES    = (1, 2, 4, 8, 16)

# 0.5B is tiny — cap memory so vLLM doesn't try to grab 90% of the GPU and
# pre-allocate a huge KV cache. Raise this if you hit "No available memory for
# the cache blocks"; lower it if startup OOMs.
llm = LLM(model=MODEL_ID, dtype="float16",
          max_num_seqs=max(BATCHES), enforce_eager=False,
          gpu_memory_utilization=0.5)
tok = llm.get_tokenizer()

text = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    tokenize=False, add_generation_prompt=True)

sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)

print(f"{'batch':>6} {'tok/s (total)':>14} {'tok/s/seq':>11}")
for B in BATCHES:
    prompts = [text] * B

    # Warm-up (graph capture / cache) — not timed.
    llm.generate(prompts, sp, use_tqdm=False)

    t0 = time.perf_counter()
    out = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0

    total = sum(len(o.outputs[0].token_ids) for o in out)
    print(f"{B:>6} {total / dt:>14.1f} {total / dt / B:>11.1f}")
