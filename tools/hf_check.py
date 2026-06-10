"""Compare our engine's greedy output against HuggingFace transformers.

Usage:
    python tools/hf_check.py
    python tools/hf_check.py --prompt "Explain gravity in one sentence"
    python tools/hf_check.py --max-new 30
"""
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Override with: export TINYLLM_MODEL=Qwen/Qwen2-1.5B-Instruct
MODEL_ID = os.environ.get("TINYLLM_MODEL", "Qwen/Qwen2-0.5B-Instruct")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt",   default="What is the capital of France?")
    ap.add_argument("--max-new",  type=int, default=20)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="cuda")

    # Render the chat template to text first, then encode separately.
    # apply_chat_template with tokenize=True can return a tokenizers.Encoding
    # object on fast tokenizers — splitting the steps avoids that.
    text = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([ids]).to("cuda")

    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=args.max_new,
            do_sample=False, temperature=1.0)

    new_ids = out[0][input_ids.shape[1]:]
    print("Prompt token IDs :", ids)
    print("Output token IDs :", new_ids.tolist())
    print("Output text      :", tok.decode(new_ids, skip_special_tokens=True))
    print()
    print("--- run our engine with ---")
    print(f"./build/tinyllm_gpu_fp16 tinyllm_fp16.bin "
          f"--ids \"{' '.join(map(str, ids))}\" --max-new {args.max_new}")

if __name__ == "__main__":
    main()
