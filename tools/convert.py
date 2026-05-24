#!/usr/bin/env python3
"""Export Qwen2-0.5B weights to tinyllm.bin.

Supports three precisions:
  --dtype fp32   Phase 0 / 1 / 2
  --dtype fp16   Phase 3
  --dtype int8   Phase 4: W8A16 — INT8 weights, per-output-row FP16 scales.
                 Matmul weights are quantized; biases / norms stay FP16.

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
DTYPE_INT8 = 2


def write_tensor(f, t, dtype):
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy()
    if dtype == "fp16":
        arr = arr.astype(np.float16)
    f.write(arr.tobytes())


def write_fp16(f, t):
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy().astype(np.float16)
    f.write(arr.tobytes())


def write_int8_with_scales(f, t):
    """Per-output-row symmetric INT8 quantization.

    For matmul weight W of shape [n_out, n_in], compute one scale per row:
        scale[i] = max(|W[i, :]|) / 127
        W_q[i, j] = round(W[i, j] / scale[i])    (clipped to [-127, 127])
    Dequantization is W[i, j] ~= W_q[i, j] * scale[i].

    File layout: int8 bytes (n_out * n_in), then fp16 scales (n_out).
    """
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy()
    abs_max = np.maximum(np.abs(arr).max(axis=1), 1e-8)
    scales = abs_max / 127.0
    q = np.clip(np.round(arr / scales[:, None]), -127, 127).astype(np.int8)
    f.write(q.tobytes())
    f.write(scales.astype(np.float16).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tinyllm.bin")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "int8"],
                    help="weight precision; fp16 halves file size, int8 halves it again")
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

    dtype_id = {"fp32": DTYPE_FP32, "fp16": DTYPE_FP16, "int8": DTYPE_INT8}[args.dtype]
    with open(args.out, "wb") as f:
        f.write(struct.pack(HEADER_FMT, MAGIC, VERSION, dtype_id, hidden, inter,
                            nlayers, nheads, nkv, head_dim, vocab, eps, theta))

        if args.dtype == "int8":
            # W8A16: matmul weights -> int8 + per-row fp16 scales.
            # Biases and norm weights stay fp16 (small + numerically sensitive).
            write_int8_with_scales(f, sd["model.embed_tokens.weight"])
            for i in range(nlayers):
                p = f"model.layers.{i}."
                write_fp16(f,              sd[p + "input_layernorm.weight"])
                write_int8_with_scales(f,  sd[p + "self_attn.q_proj.weight"])
                write_fp16(f,              sd[p + "self_attn.q_proj.bias"])
                write_int8_with_scales(f,  sd[p + "self_attn.k_proj.weight"])
                write_fp16(f,              sd[p + "self_attn.k_proj.bias"])
                write_int8_with_scales(f,  sd[p + "self_attn.v_proj.weight"])
                write_fp16(f,              sd[p + "self_attn.v_proj.bias"])
                write_int8_with_scales(f,  sd[p + "self_attn.o_proj.weight"])
                write_fp16(f,              sd[p + "post_attention_layernorm.weight"])
                write_int8_with_scales(f,  sd[p + "mlp.gate_proj.weight"])
                write_int8_with_scales(f,  sd[p + "mlp.up_proj.weight"])
                write_int8_with_scales(f,  sd[p + "mlp.down_proj.weight"])
            write_fp16(f, sd["model.norm.weight"])
        else:
            # fp32 / fp16: every tensor written with the same dtype.
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
