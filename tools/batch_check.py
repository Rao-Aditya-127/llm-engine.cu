"""Verify the continuous-batch decode path against single-sequence decode.

Two checks:
  1. All rows of a batch (same prompt) are token-for-token identical to each
     other — confirms slot indexing, per-row RoPE, KV scatter, and decode
     attention are correct.
  2. The batched output matches the single-sequence generate_ids() reference —
     confirms decode_batch is numerically equivalent to the GEMV decode path.

Run from the repo root after `make server`:
    python tools/batch_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

from transformers import AutoTokenizer
from llm_engine import LLMEngine

MODEL_ID   = "Qwen/Qwen2-1.5B-Instruct"
MODEL_PATH = "tinyllm_fp16.bin"
PROMPT     = "What is the capital of France?"
STEPS      = 20
BATCH      = 3

tok = AutoTokenizer.from_pretrained(MODEL_ID)
eng = LLMEngine(MODEL_PATH)

text = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    tokenize=False, add_generation_prompt=True)
ids = tok.encode(text, add_special_tokens=False)

# 1. Single-sequence reference (greedy).
ref = eng.generate_ids(ids, STEPS, 0.0, 1.0)

# 2. Batched: same prompt in BATCH slots, decoded together (greedy).
slots = list(range(BATCH))
last  = [eng.prefill_slot(ids, s, 0.0, 1.0, 1234) for s in slots]
pos   = [len(ids)] * BATCH
out   = [[t] for t in last]
for _ in range(STEPS - 1):
    nxt = eng.decode_batch(last, pos, slots)
    for b in range(BATCH):
        out[b].append(nxt[b])
        last[b] = nxt[b]
        pos[b] += 1

print("reference (single-seq):", ref)
for b in range(BATCH):
    print(f"batch row {b}:          ", out[b])

# Report.
rows_match = all(out[b] == out[0] for b in range(BATCH))
n = min(len(ref), len(out[0]))
ref_match = out[0][:n] == ref[:n]

print()
print("rows identical to each other :", "PASS" if rows_match else "FAIL")
print("batched matches single-seq   :", "PASS" if ref_match else "FAIL")
print("decoded text                 :",
      tok.decode(out[0], skip_special_tokens=True))
