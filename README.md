# llm-engine.cu

**A from-scratch CUDA/C++ inference engine for Qwen2 — hand-written kernels,
continuous batching, and an HTTP serving layer. No cuBLAS, no cuDNN, no PyTorch in
the hot path.**

`llm-engine.cu` runs Qwen2-family models (0.5B / 1.5B) end-to-end on a single NVIDIA
GPU. Every operation in the forward pass — GEMMs, grouped-query attention, RMSNorm,
RoPE, SwiGLU, and quantized matmul — is a hand-written CUDA kernel. A continuous-batching
scheduler and a FastAPI server sit on top, so the same engine that runs from the command
line also serves concurrent HTTP requests with token streaming.

The engine is built around the fact that LLM decoding is **memory-bandwidth-bound**:
performance comes from keeping the GPU's memory system saturated and reading each weight
as few times as possible. On an NVIDIA L4 it matches vLLM's single-stream latency and
stays within ~1.2× of vLLM's tensor-core throughput through batch 8 — with no external
GEMM library.

---

## Highlights

- **Pure hand-written CUDA/C++** — no cuBLAS, CUTLASS, cuDNN, or framework runtime in the
  inference path. Tokenization is the only Python dependency.
- **Three precision backends** — FP32, FP16 (with FP32 accumulation), and INT8 (W8A16),
  selected at build time.
- **Continuous batching** — a request scheduler with per-sequence KV-cache slots decodes
  many sequences in a single batched GPU step, admitting and evicting sequences each step.
- **Weight-reuse tiled GEMM** — a templated batched matmul that streams each weight from
  HBM once and reuses it across the batch, turning decode-time GEMVs into true GEMMs.
- **HTTP serving** — FastAPI server with blocking and Server-Sent-Events streaming
  endpoints, a minimal web UI, and `pybind11` bindings into the C++ engine.
- **Correctness-verified** — outputs checked token-exact against HuggingFace Transformers.
- **Model-configurable** — `TINYLLM_MODEL` selects any Qwen2 checkpoint; weights are
  exported to a compact single-file binary format.

---

## Performance

### vs. vLLM — continuous batching (NVIDIA L4, Qwen2-0.5B-Instruct, FP16)

Throughput as concurrency scales. Both systems measured identically: greedy decoding,
200 fixed output tokens, warm-up excluded.

| Batch | llm-engine.cu (tok/s) | per-seq | vLLM (tok/s) | per-seq |
|------:|----------------------:|--------:|-------------:|--------:|
| 1     | 195                   | 195     | 202          | 202     |
| 2     | 385                   | 193     | 407          | 204     |
| 4     | 753                   | 188     | 813          | 203     |
| 8     | 1,312                 | 164     | 1,600        | 200     |
| 16    | 1,732                 | 108     | 3,081        | 193     |

`llm-engine.cu` is neck-and-neck with vLLM through batch 4 and within 1.2× at batch 8 —
a from-scratch engine keeping pace with a production tensor-core stack across most of the
curve. The remaining gap at batch 16 is the large GEMMs, which vLLM runs on tensor cores;
see [Performance engineering](#performance-engineering).

### Precision backends (NVIDIA T4, single-sequence decode)

Single-token decode throughput against HuggingFace Transformers as the baseline, same
prompt and hardware.

| Backend              | tok/s  | vs. HF FP16 |
|----------------------|-------:|------------:|
| HF Transformers FP16 | 28.88  | 1.00×       |
| GPU FP32             | 112.43 | 3.89×       |
| GPU FP16             | 184.66 | **6.39×**   |
| GPU INT8 (W8A16)     | 263.04 | **9.11×**   |
| CPU + OpenMP         | 5.13   | 0.18×       |
| CPU naive            | 1.60   | 0.06×       |

Output is verified token-exact against HuggingFace through FP16. The 6–9× advantage over
HF comes from eliminating Python dispatch overhead, not from faster kernels — see
[Performance engineering](#performance-engineering).

---

## Architecture

```
llm-engine.cu/
├── src/
│   ├── config.h               # Qwen2 architecture constants + binary header format
│   ├── model.{h,cpp}          # Weight loading from the .bin blob (FP32 / FP16 / INT8)
│   ├── sampler.{h,cpp}        # Greedy and top-p (nucleus) sampling
│   ├── main.cpp               # CLI entry point
│   ├── infer_cpu.{h,cpp}      # CPU reference forward pass (+ OpenMP)
│   ├── infer_gpu_fp32.{h,cu}  # FP32 GPU runner
│   ├── infer_gpu_fp16.{h,cu}  # FP16 GPU runner — prefill, continuous-batch decode
│   └── infer_gpu_int8.{h,cu}  # INT8 W8A16 GPU runner
│
├── kernels/                   # Hand-written CUDA kernels (fp32 / fp16 / int8 variants)
│   ├── */matmul.cu            # Warp-per-row GEMV, templated weight-reuse tiled GEMM
│   ├── */attention.cu         # Causal GQA attention (decode + prefill)
│   ├── */rmsnorm.cu rope.cu swiglu.cu
│   ├── fp16/kv_scatter.cu     # Scatter K/V into per-slot, per-position cache cells
│   └── fp16/argmax.cu         # On-GPU greedy argmax over the logits
│
├── server/
│   ├── engine.{h,cpp}         # C++ engine facade (PIMPL — keeps CUDA out of bindings)
│   ├── bindings.cpp           # pybind11 module
│   ├── server.py              # FastAPI app + continuous-batch scheduler
│   └── index.html             # Minimal streaming web UI
│
├── tools/
│   ├── convert.py             # Export a Qwen2 checkpoint → tinyllm.bin
│   ├── tokenizer.py           # Text ↔ token IDs
│   ├── hf_check.py            # Cross-check engine output against HuggingFace
│   ├── batch_check.py         # Verify batched decode == single-sequence decode
│   ├── bench_batch.py         # Throughput vs. batch size
│   └── bench_vllm.py          # vLLM baseline at matching settings
│
├── Makefile
└── details.md                 # Design deep-dive
```

**Design notes.** The C++ engine is exposed to Python through a PIMPL facade so the
binding layer compiles without `nvcc`; CUDA types stay confined to the `.cu` translation
units. The KV cache carries a leading slot dimension (`[slots × layers × cap × kv_dim]`)
so the scheduler can keep each in-flight sequence isolated. FP16 storage with FP32
accumulation is used throughout the matmul and softmax paths for numerical stability.

---

## Quickstart

### Requirements

- CUDA Toolkit 11.0+ with `nvcc`, and an NVIDIA GPU. The Makefile defaults to
  `-arch=sm_75` (T4); set it to your architecture (e.g. `sm_89` for L4/RTX 4090,
  `sm_80` for A100).
- A 64-bit C++ compiler for the CPU backend.
- Python 3.8+ with `torch`, `transformers`, `numpy` for weight export and tokenization;
  add `fastapi`, `uvicorn`, `pybind11` for the server.

### Setup

```bash
pip install torch transformers numpy fastapi uvicorn pybind11

# Select the model (default: Qwen/Qwen2-0.5B-Instruct)
export TINYLLM_MODEL=Qwen/Qwen2-0.5B-Instruct

# Export FP16 weights (~1 GB). Use --dtype fp32 / int8 for the other backends.
python tools/convert.py --out tinyllm_fp16.bin --dtype fp16
```

### Build

```bash
make gpu_fp16     # FP16 GPU binary
make gpu_int8     # INT8 W8A16 GPU binary
make gpu          # FP32 GPU binary
make              # CPU binaries (naive + OpenMP)
make server       # pybind11 shared library for the FastAPI server
```

### Run — CLI

```bash
python tools/tokenizer.py encode "The capital of France is"
#   → 785 6722 315 9625 374

./build/tinyllm_gpu_fp16 tinyllm_fp16.bin --ids "785 6722 315 9625 374" --max-new 32
#   prints generated token IDs to stdout, tok/s to stderr

python tools/tokenizer.py decode 12095 13 1084 374 279 7772 3283
#   → " Paris. It is the largest city..."
```

| Flag | Default | Description |
|------|---------|-------------|
| `--ids "..."` | required | Space-separated prompt token IDs |
| `--max-new N` | 32 | Tokens to generate |
| `--temp T` | 0.0 | Sampling temperature (0 = greedy) |
| `--top-p P` | 0.9 | Nucleus sampling threshold |
| `--seed S` | 42 | RNG seed |
| `--dump-logits PATH` | — | Write first-step logits for numeric verification |

### Run — server

```bash
make server
uvicorn server.server:app --host 0.0.0.0 --port 8000
```

Open `http://<host>:8000` for the web UI, or call the API directly:

```bash
# Blocking
curl -s -X POST localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain gravity in one sentence","max_tokens":64}'

# Streaming (Server-Sent Events)
curl -N -X POST localhost:8000/generate/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain gravity in one sentence","max_tokens":64}'
```

The server runs a single engine-owning worker thread behind a continuous-batch scheduler:
concurrent requests are admitted into free KV-cache slots, decoded together each step, and
evicted on EOS or length limit.

---

## Performance engineering

LLM decode reads the entire weight set from HBM for every token, so the relevant metric is
bytes moved, not FLOPs. Three design decisions follow directly from that.

**Decode is memory-bandwidth-bound.** At FP16 the model is ~990 MB; at the T4's 320 GB/s
that is a ~3 ms/token floor, capping single-stream throughput near ~320 tok/s regardless
of kernel cleverness. INT8 (W8A16) halves the bytes per weight and lifts decode to 263
tok/s — 82% of that ceiling. A roofline check (`ncu`) puts the large FFN GEMMs at **90.5%
of peak DRAM bandwidth**, confirming the matmuls are bandwidth-limited rather than
compute-limited.

**Batching only pays off with weight reuse.** Adding sequences to a batch is free in
bandwidth terms *only if* each weight is read once and reused across the batch. The first
batched matmul read the weights once per sequence, so per-sequence throughput collapsed as
the batch grew. The fix is a tiled kernel that streams each weight row from HBM once and
accumulates across the batch from registers — but the per-sequence accumulators must be a
**compile-time-sized** array (`template <int B>`); a runtime-indexed `acc[batch]` spills to
local memory (DRAM) and turns every multiply-add into a DRAM round-trip. Templating the
batch and using `float4` vectorized loads took batch-16 throughput from 492 to 1,554 tok/s.

**Keep the decode loop on the GPU.** Greedy decoding originally copied the full
`[batch × vocab]` logits to the host each step (~9.7 MB at batch 16) for a CPU argmax. Doing
the argmax on the GPU and returning `batch` integers (~64 bytes) lifted batch-16 throughput
to 1,732 tok/s. A built-in profiler (`TINYLLM_PROFILE=1`) then attributes the decode step to
**~80% matmul, ~13% norm/RoPE, ~7% attention, ~0.5% sampling** — the large GEMMs dominate
and are at the CUDA-core bandwidth ceiling.

**Why this beats HuggingFace by 6–9× at batch 1.** HF Transformers calls cuBLAS — fast
kernels — but dispatches 120+ separate Python-level ops per decode step; at batch 1 each
kernel runs in tens of microseconds and Python spends a comparable amount re-entering the
dispatcher. This engine runs one C++ `forward()` per token with zero Python in the loop, so
the win is on the host cost path, not raw kernel speed. (This is the same overhead
`torch.compile` and CUDA graphs exist to remove.)

---

## Limitations & roadmap

- **Tensor cores.** The remaining gap to vLLM at high batch is the large GEMMs, which vLLM
  runs on FP16 tensor cores (a 16×16×16 multiply per instruction with hardware weight reuse).
  A `wmma`/`mma` GEMM is the path to closing it; the current kernels are CUDA-core only.
- **INT8 on small models.** The W8A16 kernel is correct (first-token argmax matches HF), but
  per-row symmetric quantization accumulates error through the 24-layer stack on sub-1B
  models and the sequence diverges after a few tokens. Calibrated schemes (GPTQ/AWQ) sit on
  the same W8A16 skeleton and fix this; continuous batching currently targets the FP16 path.
- **Long context.** Attention is a standard causal/GQA kernel; FlashAttention-style tiling
  would matter only at long sequence lengths, which are out of scope for the current targets.

---

## Further reading

[details.md](details.md) — a design deep-dive covering the binary weight format, the KV-cache
layout, the quantization scheme, and the kernel-by-kernel rationale.
