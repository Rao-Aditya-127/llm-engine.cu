# llm-engine.cu

A single-GPU, raw C++/CUDA inference engine for **Qwen2-0.5B** — no cuBLAS, no
CUTLASS. Every kernel is written by hand.

The goal is to understand *why* LLM inference is slow (memory bandwidth) and what
actually fixes it (FP16, INT8, op fusion) — by building each optimization from
scratch and measuring the result.

See [details.md](details.md) for an in-depth, phase-by-phase explanation of every
design decision and concept.

---

## Results

Single-token decode throughput. Prompt: `"The capital of France is"`, 32 greedy
tokens. All rows measured on the same cloud GPU VM (lightning.ai T4) for an
apples-to-apples comparison.

| Engine               | Hardware | tok/s  | vs HF FP16 |
|----------------------|----------|--------|------------|
| CPU naive (1 thread) | VM CPU   | 1.60   | 0.06×      |
| CPU + OpenMP         | VM CPU   | 5.13   | 0.18×      |
| GPU FP32             | VM GPU   | 112.43 | 3.89×      |
| GPU FP16             | VM GPU   | 184.66 | **6.39×**  |
| GPU INT8 (W8A16)     | VM GPU   | 263.04 | **9.11×**  |
| HF transformers FP16 | VM GPU   | 28.88  | 1.00×      |

Output verified token-exact against HuggingFace `transformers` for all phases
through FP16. INT8 matches on the first token; subsequent tokens diverge due to
known error accumulation on sub-1B models (see [details.md](details.md), Phase 4).

---

## Repository structure

```
tinyllm/
│
├── Makefile                    # Primary build — see "Building" below
│
├── src/
│   ├── config.h                # Qwen2-0.5B architecture constants + binary header format
│   ├── model.h / model.cpp     # Weight loading from .bin (FP32 / FP16 / INT8)
│   ├── sampler.h / sampler.cpp # Greedy and top-p sampling
│   ├── main.cpp                # CLI entry point — times the generation loop
│   │
│   ├── infer_cpu.h / .cpp      # Phase 1: full FP32 CPU forward pass + KV cache
│   ├── infer_gpu_fp32.h / .cu  # Phase 2: FP32 GPU forward pass (GpuRunner)
│   ├── infer_gpu_fp16.h / .cu  # Phase 3: FP16 GPU forward pass (GpuRunnerFP16)
│   └── infer_gpu_int8.h / .cu  # Phase 4: INT8 W8A16 GPU forward pass (GpuRunnerInt8)
│
├── kernels/
│   ├── common.cuh              # Shared CUDA_CHECK macro
│   ├── fp32/                   # Phase 2 FP32 kernels
│   │   ├── kernels.cuh         # Host-side launch declarations
│   │   ├── rmsnorm.cu
│   │   ├── rope.cu
│   │   ├── swiglu.cu
│   │   ├── matmul.cu           # Warp-per-row GEMV with warp-shuffle reduction
│   │   └── attention.cu        # Causal attention + GQA (14Q / 2KV heads)
│   ├── fp16/                   # Phase 3 FP16 kernels (FP16 storage, FP32 accumulators)
│   │   ├── kernels.cuh
│   │   ├── rmsnorm.cu
│   │   ├── rope.cu
│   │   ├── swiglu.cu
│   │   ├── matmul.cu           # matmul_fp16_kernel + matmul_fp16_to_fp32_kernel (LM head)
│   │   └── attention.cu
│   └── int8/                   # Phase 4 INT8 kernels (reuses fp16/ for non-matmul ops)
│       ├── kernels.cuh
│       ├── matmul.cu           # Fused dequant + matmul (W8A16)
│       └── embedding.cu        # INT8 embedding lookup with per-row dequant
│
├── tools/
│   ├── convert.py              # Export Qwen2-0.5B weights → tinyllm.bin (fp32/fp16/int8)
│   ├── tokenizer.py            # Encode text → token IDs; decode IDs → text
│   ├── golden.py               # Dump golden reference logits + greedy continuation
│   └── bench_hf.py             # HuggingFace transformers baseline benchmark
│
├── benchmarks/
│   ├── golden.txt              # Human-readable golden reference output
│   └── golden_logits.bin       # Raw FP32 logit vector from HF model (first step)
│
└── details.md                  # Phase-by-phase deep-dive + learning log
```

---

## Prerequisites

### Python (weight export + tokenization)

```
pip install torch transformers numpy
```

Python 3.8+ recommended. The model (`Qwen/Qwen2-0.5B`) downloads automatically
from HuggingFace on first use (~1 GB).

### C++ / CPU build

- A 64-bit C++ compiler: `g++` (Linux/macOS) or MSVC `cl` (Windows)
- The FP32 weight file is ~2 GB — a 32-bit process cannot address it; the build
  **must** be 64-bit

### CUDA / GPU build

- CUDA Toolkit 11.0+ with `nvcc`
- An NVIDIA GPU (Makefile defaults to `-arch=sm_75` for T4 — change it for other
  GPUs, e.g. `sm_86` for RTX 3090, `sm_89` for RTX 4090)

---

## Setup

Run these once, in order:

```bash
# 1. Export FP32 weights (~2 GB)
python tools/convert.py --out tinyllm.bin

# 2. Export FP16 weights (~1 GB) — needed for Phase 3
python tools/convert.py --out tinyllm_fp16.bin --dtype fp16

# 3. Export INT8 weights (~500 MB) — needed for Phase 4
python tools/convert.py --out tinyllm_int8.bin --dtype int8

# 4. Generate the golden reference (needs a GPU or is slow on CPU)
python tools/golden.py
#   writes: benchmarks/golden.txt  (human-readable)
#           benchmarks/golden_logits.bin  (raw FP32 logits for numeric diff)
```

---

## Building

```bash
make            # CPU binaries (naive + OpenMP)  — uses g++
make gpu        # GPU FP32 binary               — uses nvcc, needs CUDA + NVIDIA GPU
make gpu_fp16   # GPU FP16 binary
make gpu_int8   # GPU INT8 W8A16 binary
make clean      # remove build/
```

> **Windows without `make`:** compile the four `src/*.cpp` files with 64-bit MSVC
> `cl` and add `/openmp` for the OpenMP build. For CUDA builds, `nvcc` is
> cross-platform — the Makefile commands translate directly.

All built binaries land in `build/`.

---

## Running

### Tokenize your prompt

```bash
python tools/tokenizer.py encode "The capital of France is"
# output: 785 6722 315 9625 374
```

### Run the engine

```bash
# CPU naive (single thread)
./build/tinyllm_naive tinyllm.bin --ids "785 6722 315 9625 374" --max-new 32

# CPU with OpenMP
./build/tinyllm_omp tinyllm.bin --ids "785 6722 315 9625 374" --max-new 32

# GPU FP32
./build/tinyllm_gpu tinyllm.bin --ids "785 6722 315 9625 374" --max-new 32

# GPU FP16
./build/tinyllm_gpu_fp16 tinyllm_fp16.bin --ids "785 6722 315 9625 374" --max-new 32

# GPU INT8 W8A16
./build/tinyllm_gpu_int8 tinyllm_int8.bin --ids "785 6722 315 9625 374" --max-new 32
```

Each binary prints generated token IDs to stdout and `tok/s` to stderr.

### Decode the output

```bash
python tools/tokenizer.py decode 12095 13 1084 374 279 7772 3283 ...
# output: " Paris. It is the largest city..."
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--ids "..."` | required | Space-separated prompt token IDs |
| `--max-new N` | 32 | Number of tokens to generate |
| `--temp T` | 0.0 | Sampling temperature (0 = greedy) |
| `--top-p P` | 0.9 | Nucleus sampling threshold |
| `--seed S` | 42 | RNG seed for reproducible sampling |
| `--dump-logits PATH` | — | Write first-step logits to a binary file |

---

## Roofline analysis

T4 peak DRAM bandwidth: **~320 GB/s**.
Profiled with `ncu --set basic --kernel-name matmul_fp16_kernel`:

| Matmul shape        | Used for              | DRAM utilization      |
|---------------------|-----------------------|-----------------------|
| `[4864 × 896]` × 24 | FFN gate / up / down  | **90.5% (~290 GB/s)** |
| `[896 × 896]` × 24  | Q / O projections     | 57.7% (~185 GB/s)     |
| `[128 × 896]` × 24  | K / V projections     | 15.6% (~50 GB/s)      |

The large FFN matmuls are at the memory-bandwidth roof. Arithmetic intensity for a
matrix-vector product is `~1 FLOP/byte` — the T4's FP16 compute roof (~65 TFLOPS)
is 200× higher, so **compute never becomes the bottleneck in single-token decode**.

FP16 → INT8 halves the bytes read per weight; the FFN matmul stays near 90% of
the (now INT8) roof, giving a 1.42× system-level speedup. The gap from the
theoretical 2× is the ~22% of GPU time in non-matmul kernels (RMSNorm, RoPE,
attention, SwiGLU) that are unchanged between precisions.

---

## Why we beat HuggingFace by 6–9×

HF `transformers` uses cuBLAS — a best-in-class GEMM library. The individual
kernels are fast. The bottleneck is **Python dispatch overhead**: each decode step
calls ~120+ separate Python-dispatched torch ops (one per projection, one per norm,
one for `apply_rotary_pos_emb`, etc.). At batch size 1, each CUDA kernel finishes
in tens of microseconds, then Python spends a similar amount re-entering the
dispatcher for the next op.

Our engine runs a single C++ `forward()` call per token; from there it is one
CUDA kernel after another on the default stream with zero Python in the loop.
**We win on the host-side cost path, not kernel speed.**

This is also why `torch.compile` and CUDA graphs exist in PyTorch 2.0+ — they
amortize or eliminate that dispatch overhead and close most of the gap.

---

## What I learned

**1. Decode is memory-bandwidth-bound.**
Each token reads every weight once (~990 MB at FP16). At T4's 320 GB/s that is
~3 ms/token, capping throughput at ~320 tok/s. We reach 263 tok/s at INT8 —
82% of the theoretical ceiling.

**2. Halving the bytes roughly doubles the speed.**
FP32 → FP16: measured 1.64× (theory: 2×; gap = non-matmul ops + launch overhead).
FP16 → INT8: measured 1.42× (same gap). Inside the big FFN matmul the bandwidth
win is close to 2× both times.

**3. Naive INT8 breaks on small models.**
The kernel is correct — the first-step argmax matches HuggingFace. But per-row
symmetric INT8 error accumulates through the 24-layer KV cache and the sequence
diverges at token 3. The same scheme works on 7B+ models. The fix (GPTQ, AWQ,
bitsandbytes) is calibration on top of the same W8A16 skeleton built here.

**4. Op fusion eliminates memory round-trips.**
SwiGLU fused `silu(gate) * up` avoids writing and re-reading a 4864-element
intermediate buffer every layer. Small win per layer, but free to get right.

**5. Python dispatch is the real enemy above raw kernel speed.**
HF's cuBLAS kernels are faster than the hand-written warp-per-row GEMV, yet this
engine is 6× faster end-to-end. The lesson: in production, the overhead layer
matters as much as the kernel.
