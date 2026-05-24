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

| Phase | Engine            | Hardware          | tok/s  |
|-------|-------------------|-------------------|--------|
| 1     | CPU naive (1 thr) | VM CPU            | 1.60   |
| 1     | CPU + OpenMP      | VM CPU            | 5.13   |
| 2     | GPU FP32          | VM GPU            | 112.43 |
| 3     | GPU FP16          | VM GPU            | TBD    |
| 4     | GPU INT8 (W8A16)  | VM GPU            | TBD    |
| ref   | HF transformers   | VM GPU            | TBD    |

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

# tokenize, run, de-tokenize
python tools/tokenizer.py encode "The capital of France is"
./build/tinyllm_omp tinyllm.bin --ids "785 6722 315 9625 374" --max-new 32
python tools/tokenizer.py decode <ids...>
```

> **Note:** `make` is the primary build. On Windows, if `make`/`g++` is missing
> or 32-bit, compile the four `src/*.cpp` files with a 64-bit MSVC `cl` instead
> (add `/openmp` for the parallel build). The build must be 64-bit — the FP32
> model is ~2 GB.

## Status

- [x] Phase 0 — weight export + golden reference
- [x] Phase 1 — CPU baseline (naive + OpenMP)
- [x] Phase 2 — GPU port, kernel by kernel
- [~] Phase 3 — FP16 + profiling (code written; verify + profile on GPU)
- [ ] Phase 4 — INT8 weight quantization
- [ ] Phase 5 — measure & document
