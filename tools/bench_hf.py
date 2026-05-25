#!/usr/bin/env python3
"""HuggingFace `transformers` baseline benchmark for Qwen2-0.5B.

Measures decode tok/s using the same prompt and methodology as our engine:
  prompt = "The capital of France is"  (5 tokens)
  max_new = 32 greedy tokens
  timing = the 32-token generation loop only (prefill excluded), matching
           main.cpp which times only the generation loop in `forward()`.

Warms up first so cuDNN/kernels are loaded before timing.
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2-0.5B"
PROMPT_IDS = [785, 6722, 315, 9625, 374]   # "The capital of France is"


def run_once(model, tok, max_new, prompt_ids_t):
    """One generation: prefill (untimed), then time the decode loop only."""
    with torch.no_grad():
        # ----- prefill (untimed) -----
        out = model(prompt_ids_t, use_cache=True)
        past = out.past_key_values
        last = out.logits[:, -1, :]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # ----- timed decode loop -----
        generated = []
        for _ in range(max_new):
            nxt = last.argmax(dim=-1, keepdim=True)        # greedy
            generated.append(nxt.item())
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            last = out.logits[:, -1, :]

        torch.cuda.synchronize()
        t1 = time.perf_counter()

    return generated, t1 - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs",   type=int, default=5)
    args = ap.parse_args()

    torch_dtype = {"fp32": torch.float32,
                   "fp16": torch.float16,
                   "bf16": torch.bfloat16}[args.dtype]

    print(f"Loading {MODEL_ID} in {args.dtype} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch_dtype).cuda()
    model.eval()

    prompt_ids_t = torch.tensor([PROMPT_IDS], dtype=torch.long, device="cuda")

    print(f"warming up ({args.warmup} runs) ...")
    for _ in range(args.warmup):
        run_once(model, tok, args.max_new, prompt_ids_t)

    print(f"timing ({args.runs} runs, {args.max_new} new tokens each):")
    rates = []
    for r in range(args.runs):
        gen, secs = run_once(model, tok, args.max_new, prompt_ids_t)
        rate = len(gen) / secs
        rates.append(rate)
        print(f"  run {r+1}: {len(gen)} tokens in {secs*1000:7.1f} ms  =>  {rate:7.2f} tok/s")

    rates.sort()
    median = rates[len(rates) // 2]
    print(f"\nmedian: {median:.2f} tok/s  (dtype={args.dtype}, model={MODEL_ID})")
    print(f"continuation: {tok.decode(gen)}")


if __name__ == "__main__":
    main()
