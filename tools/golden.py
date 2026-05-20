#!/usr/bin/env python3
"""Dump a golden reference from the real HF Qwen2-0.5B model.

Every later phase greedy-decodes the same prompt and diffs against this.

Outputs:
  benchmarks/golden.txt        human-readable: prompt, ids, greedy continuation
  benchmarks/golden_logits.bin float32 logits of the FIRST forward step [vocab]
"""
import os
import struct
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2-0.5B"
PROMPT = "The capital of France is"
N_NEW = 32

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    prompt_ids = tok.encode(PROMPT)
    ids = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.no_grad():
        first_logits = model(ids).logits[0, -1].to(torch.float32)

    with open(os.path.join(OUT_DIR, "golden_logits.bin"), "wb") as f:
        f.write(first_logits.numpy().tobytes())

    # greedy decode
    gen = list(prompt_ids)
    cur = ids
    with torch.no_grad():
        for _ in range(N_NEW):
            logits = model(cur).logits[0, -1]
            nxt = int(torch.argmax(logits))
            gen.append(nxt)
            cur = torch.tensor([gen], dtype=torch.long)

    cont_ids = gen[len(prompt_ids):]
    top5 = torch.topk(first_logits, 5)

    with open(os.path.join(OUT_DIR, "golden.txt"), "w", encoding="utf-8") as f:
        f.write(f"prompt: {PROMPT}\n")
        f.write(f"prompt_ids: {' '.join(map(str, prompt_ids))}\n")
        f.write(f"first_step_argmax: {int(torch.argmax(first_logits))}\n")
        f.write("first_step_top5:\n")
        for v, i in zip(top5.values.tolist(), top5.indices.tolist()):
            f.write(f"  {i}\t{v:.5f}\n")
        f.write(f"greedy_ids: {' '.join(map(str, cont_ids))}\n")
        f.write(f"greedy_text: {tok.decode(cont_ids)}\n")

    print("wrote benchmarks/golden.txt and golden_logits.bin", file=sys.stderr)
    print(f"continuation: {tok.decode(cont_ids)}", file=sys.stderr)


if __name__ == "__main__":
    main()
