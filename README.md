# llm-engine.cu

A single-GPU, raw C++/CUDA inference engine for **Qwen2-0.5B** — no cuBLAS, no
CUTLASS. Every kernel is written by hand. The goal is to understand *why* LLM
inference is slow (memory bandwidth) and what actually fixes it (FP16, INT8,
op fusion).

See [details.md](details.md) for an in-depth, phase-by-phase explanation of how
it is built and why.

## Benchmark

Single-token decode throughput. Prompt: `"The capital of France is"`, 32 tokens,
greedy. Output verified token-exact against HuggingFace `transformers`.

All rows below are measured on the same cloud GPU VM (lightning.ai) for an
apples-to-apples comparison.

| Phase | Engine               | Hardware | tok/s  | vs HF FP16 |
|-------|----------------------|----------|--------|------------|
| 1     | CPU naive (1 thr)    | VM CPU   | 1.60   | 0.06×      |
| 1     | CPU + OpenMP         | VM CPU   | 5.13   | 0.18×      |
| 2     | GPU FP32             | VM GPU   | 112.43 | 3.89×      |
| 3     | GPU FP16             | VM GPU   | 184.66 | **6.39×**  |
| 4     | GPU INT8 (W8A16)     | VM GPU   | 263.04 | **9.11×**  |
| ref   | HF transformers FP16 | VM GPU   | 28.88  | 1.00×      |

## Usage

```
# one-time: export weights + golden reference (needs torch + transformers)
python tools/convert.py --out tinyllm.bin
python tools/golden.py

# build CPU engine (Phase 1)
make
# build GPU FP32 engine (Phase 2 — needs CUDA toolkit + NVIDIA GPU)
make gpu
# build GPU FP16 engine (Phase 3 — needs an extra `--dtype fp16` export)
python tools/convert.py --out tinyllm_fp16.bin --dtype fp16
make gpu_fp16
# build GPU INT8 W8A16 engine (Phase 4)
python tools/convert.py --out tinyllm_int8.bin --dtype int8
make gpu_int8

# tokenize, run, de-tokenize
python tools/tokenizer.py encode "The capital of France is"
./build/tinyllm_omp tinyllm.bin --ids "785 6722 315 9625 374" --max-new 32
python tools/tokenizer.py decode <ids...>
```

> **Note:** `make` is the primary build. On Windows, if `make`/`g++` is missing
> or 32-bit, compile the four `src/*.cpp` files with a 64-bit MSVC `cl` instead
> (add `/openmp` for the parallel build). The build must be 64-bit — the FP32
> model is ~2 GB.

## Roofline analysis

T4 peak DRAM bandwidth: **~320 GB/s**.
Profiled with `ncu --set basic --kernel-name matmul_fp16_kernel`:

| Matmul shape | Used for | DRAM utilization |
|---|---|---:|
| `[4864 × 896]` × 24 | FFN gate / up / down | **90.5% (~290 GB/s)** |
| `[896 × 896]` × 24  | Q / O projections    | 57.7% (~185 GB/s)     |
| `[128 × 896]` × 24  | K / V projections    | 15.6% (~50 GB/s)      |

The large FFN matmuls are at the memory-bandwidth roof. Arithmetic intensity for a
matrix-vector product is `~1 FLOP/byte` — the T4's FP16 compute roof (~65 TFLOPS)
is 200× higher, so compute never becomes the bottleneck in decode.

FP16 → INT8 halves the bytes read per weight; the FFN matmul stays near 90% of
the (now INT8) roof, and the system-level speedup is 1.42×. The gap from 2× is
the ~22% of GPU time in non-matmul kernels (RMSNorm, RoPE, attention, SwiGLU)
that are unchanged between FP16 and INT8.

## Why we beat HuggingFace by 6–9×

HF `transformers` uses cuBLAS — a best-in-class GEMM library. The individual
kernels are fast. The bottleneck is **Python dispatch overhead**: each decode step
calls ~120+ separate Python-dispatched torch ops (one per projection, one per norm,
one for `apply_rotary_pos_emb`, etc.). At batch size 1, each CUDA kernel finishes
in tens of microseconds, then Python spends a similar amount re-entering the
dispatcher for the next op. That overhead accumulates to dominate wall time.

Our engine runs a single C++ `forward()` call per token; from there it is one
CUDA kernel after another on the default stream with zero Python in the loop.
**We win on the host-side cost path, not the kernel speed.**

## What I learned

**1. Decode is memory-bandwidth-bound.** Each token reads every weight once
(~990 MB at FP16). At 320 GB/s that is ~3 ms/token, capping throughput at
~320 tok/s. We reach 263 tok/s at INT8 — 82% of the theoretical ceiling.

**2. Halving the bytes roughly doubles the speed.**
FP32→FP16: 1.64× (theory: 2×, gap = non-matmul ops + launch overhead).
FP16→INT8: 1.42× (same gap, same reason).
Inside the big FFN matmul itself, the bandwidth win is close to 2× both times.

**3. Naive INT8 breaks on small models.** The kernel is correct — first-step
argmax matches HuggingFace. But per-row symmetric INT8 error accumulates through
the 24-layer KV cache and the decode sequence diverges by token 3. The same
scheme works on 7B+ models. The fix (GPTQ, AWQ, bitsandbytes outlier protection)
is calibration on top of the same W8A16 skeleton we built.

**4. Op fusion eliminates memory round-trips.** SwiGLU fused
`silu(gate) * up` avoids writing and re-reading a 4864-element intermediate
buffer every layer. Small, but free to get right.

**5. Python dispatch is the real bottleneck above kernel speed.** HF's cuBLAS
kernels are faster than our hand-written warp-per-row GEMV, yet we are 6×
faster end-to-end. The lesson: in production, `torch.compile` and CUDA graphs
exist precisely to close this gap.

## Status

- [x] Phase 0 — weight export + golden reference
- [x] Phase 1 — CPU baseline (naive + OpenMP)
- [x] Phase 2 — GPU port, kernel by kernel
- [x] Phase 3 — FP16 + profiling
- [x] Phase 4 — INT8 weight quantization (speed verified; accuracy known limit on 0.5B — see details.md)
- [x] Phase 5 — roofline analysis + write-up
