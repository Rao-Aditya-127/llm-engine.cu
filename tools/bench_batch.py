"""Aggregate decode throughput vs batch size.

Drives the engine's batched decode directly (no HTTP) so the measurement is the
GPU work alone. Each weight matrix is read from HBM once per step regardless of
batch size, so aggregate tok/s should rise steeply with the batch — that is the
whole point of continuous batching.

Run from the repo root after `make server`:
    python tools/bench_batch.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

from transformers import AutoTokenizer
from llm_engine import LLMEngine

# Override with: export TINYLLM_MODEL=Qwen/Qwen2-1.5B-Instruct
MODEL_ID   = os.environ.get("TINYLLM_MODEL", "Qwen/Qwen2-0.5B-Instruct")
MODEL_PATH = "tinyllm_fp16.bin"
PROMPT     = "Tell me about machine learning."
STEPS      = 100

tok = AutoTokenizer.from_pretrained(MODEL_ID)
eng = LLMEngine(MODEL_PATH)

text = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    tokenize=False, add_generation_prompt=True)
ids = tok.encode(text, add_special_tokens=False)

print(f"{'batch':>6} {'tok/s (total)':>14} {'tok/s/seq':>11}")
for B in (1, 2, 4, 8, 16):
    if B > eng.max_slots():
        break
    slots = list(range(B))
    last  = [eng.prefill_slot(ids, s, 0.0, 1.0, 1234) for s in slots]
    pos   = [len(ids)] * B

    t0 = time.perf_counter()
    for _ in range(STEPS):
        last = eng.decode_batch(last, pos, slots)
        pos  = [p + 1 for p in pos]
    dt = time.perf_counter() - t0

    total = B * STEPS
    print(f"{B:>6} {total / dt:>14.1f} {STEPS / dt:>11.1f}")
