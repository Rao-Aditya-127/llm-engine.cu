#!/usr/bin/env python3
"""Export Qwen2-0.5B weights to tinyllm.bin.

Phase 0 supports FP32 (--dtype fp32) and Phase 3 adds FP16 (--dtype fp16).
INT8 export is added in Phase 4.

File layout (little-endian):

  [Header]                    48 bytes, see struct fmt below
  embed_tokens.weight         [vocab, hidden]   (also serves as tied lm_head)
  for each of 24 layers, in order:
    input_layernorm.weight        [hidden]
    self_attn.q_proj.weight       [q_dim, hidden]
    self_attn.q_proj.bias         [q_dim]
    self_attn.k_proj.weight       [kv_dim, hidden]
    self_attn.k_proj.bias         [kv_dim]
    self_attn.v_proj.weight       [kv_dim, hidden]
    self_attn.v_proj.bias         [kv_dim]
    self_attn.o_proj.weight       [hidden, q_dim]
    post_attention_layernorm.weight [hidden]
    mlp.gate_proj.weight          [intermediate, hidden]
    mlp.up_proj.weight            [intermediate, hidden]
    mlp.down_proj.weight          [hidden, intermediate]
  model.norm.weight             [hidden]

All matmul weights keep HF's [out_features, in_features] row-major layout.
"""
import argparse
import struct
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2-0.5B"
MAGIC = 0x4D4C4E54
VERSION = 1
HEADER_FMT = "<10I2f"  # magic,version,dtype,hidden,inter,layers,heads,kv_heads,head_dim,vocab, eps,theta

DTYPE_FP32 = 0
DTYPE_FP16 = 1


def write_tensor(f, t, dtype):
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy()
    if dtype == "fp16":
        arr = arr.astype(np.float16)
    f.write(arr.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tinyllm.bin")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"],
                    help="weight precision; fp16 halves file size and GPU bandwidth")
    args = ap.parse_args()

    print(f"loading {MODEL_ID} ...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    cfg = model.config
    sd = model.state_dict()

    hidden = cfg.hidden_size
    inter = cfg.intermediate_size
    nlayers = cfg.num_hidden_layers
    nheads = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", hidden // nheads)
    vocab = cfg.vocab_size
    eps = cfg.rms_norm_eps
    theta = float(getattr(cfg, "rope_theta", 1000000.0))

    print(f"hidden={hidden} inter={inter} layers={nlayers} heads={nheads} "
          f"kv={nkv} head_dim={head_dim} vocab={vocab} eps={eps} theta={theta}",
          file=sys.stderr)

    dtype_id = DTYPE_FP16 if args.dtype == "fp16" else DTYPE_FP32
    with open(args.out, "wb") as f:
        f.write(struct.pack(HEADER_FMT, MAGIC, VERSION, dtype_id, hidden, inter,
                            nlayers, nheads, nkv, head_dim, vocab, eps, theta))

        write_tensor(f, sd["model.embed_tokens.weight"], args.dtype)

        for i in range(nlayers):
            p = f"model.layers.{i}."
            write_tensor(f, sd[p + "input_layernorm.weight"], args.dtype)
            write_tensor(f, sd[p + "self_attn.q_proj.weight"], args.dtype)
            write_tensor(f, sd[p + "self_attn.q_proj.bias"], args.dtype)
            write_tensor(f, sd[p + "self_attn.k_proj.weight"], args.dtype)
            write_tensor(f, sd[p + "self_attn.k_proj.bias"], args.dtype)
            write_tensor(f, sd[p + "self_attn.v_proj.weight"], args.dtype)
            write_tensor(f, sd[p + "self_attn.v_proj.bias"], args.dtype)
            write_tensor(f, sd[p + "self_attn.o_proj.weight"], args.dtype)
            write_tensor(f, sd[p + "post_attention_layernorm.weight"], args.dtype)
            write_tensor(f, sd[p + "mlp.gate_proj.weight"], args.dtype)
            write_tensor(f, sd[p + "mlp.up_proj.weight"], args.dtype)
            write_tensor(f, sd[p + "mlp.down_proj.weight"], args.dtype)

        write_tensor(f, sd["model.norm.weight"], args.dtype)

    print(f"wrote {args.out} ({args.dtype})", file=sys.stderr)


if __name__ == "__main__":
    main()
