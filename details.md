# TinyLLM — Implementation Details (phase by phase)

This document explains, in depth, **what** we built in each phase, **which files**
we created, and **why** each decision was made. It is a learning log — read it
top to bottom to understand how a from-scratch LLM inference engine comes together.

The project: a single-GPU C++/CUDA inference engine for **Qwen2-0.5B**, with no
cuBLAS and no CUTLASS. We write every kernel by hand. The point is to *understand*
how inference works and why it is slow, not just to make it run.

---

## How to use this doc + learning resources

You do **not** need to understand everything before building. Each phase below
ends with a **"Concepts to focus on (80/20)"** box: the small number of ideas
that explain most of what is going on. If a phase feels confusing, go study the
starred (★) item *first* — it is the highest-leverage one.

**Resources worth knowing about** (look them up by name when a phase points you
to them — no links, find the current best version yourself):

- **"The Illustrated Transformer" — Jay Alammar.** The single best first read for
  *what a transformer does*. Pictures, no math. Read this once, early.
- **"Let's build GPT from scratch" — Andrej Karpathy (video) / nanoGPT (repo).**
  Watch this if attention/embeddings/logits still feel abstract. It builds the
  exact same compute graph we built in Phase 1, in Python.
- **llama2.c — Andrej Karpathy (repo).** A ~1000-line C inference engine for a
  Llama-style model. Our Phase 1 is structurally the same idea. Great to read
  side-by-side with `infer_cpu.cpp`.
- **"Attention Is All You Need" (paper).** The original transformer. Skim it for
  vocabulary; don't get stuck on it.
- **Andrew Kchan's blog post on writing a fast LLM inference engine.** This is the
  post the project brief quotes. Most relevant for Phases 2–4.
- **"Programming Massively Parallel Processors" (book, Hwu/Kirk).** The standard
  intro to CUDA. Chapters on memory and tiling matter for Phase 2.
- **"Making Deep Learning Go Brrrr From First Principles" — Horace He.** Explains
  compute-bound vs memory-bound. Read before Phase 3.

---

## Phase 0 — Setup & weight export

**Goal:** Get the model's weights out of the Python/HuggingFace world and into a
format our own C++ code can read, and create a "golden" reference output we can
check every later phase against.

### Why this phase exists

Before we can write a single line of inference code, we need two things:

1. **The weights, in a format we control.** HuggingFace ships Qwen2-0.5B as
   SafeTensors files. We *could* parse SafeTensors in C++, but that means writing
   a JSON parser (the SafeTensors header is JSON). Instead we convert the weights
   once, in Python, into a dead-simple flat binary file. C++ then just does
   `fread` into arrays — no parsing logic, no bugs.
2. **A correctness oracle.** Our CPU engine, GPU engine, FP16 engine, and INT8
   engine must all produce the *same* text. We need one trusted answer from the
   real model to compare against. That is the "golden reference."

### Files created

#### `src/config.h`
The Qwen2-0.5B architecture as C++ constants, plus the binary file format.

- **Architecture constants** (`hidden_size=896`, `num_layers=24`,
  `num_heads=14`, `num_kv_heads=2`, etc.). These never change for this model, so
  they are `constexpr` — the compiler bakes them in, and there is no config file
  to load at runtime.
- **Why `num_heads=14` but `num_kv_heads=2`?** This is **GQA (Grouped Query
  Attention)**. The model has 14 query heads but only 2 key/value heads — every
  7 query heads share one KV head. This shrinks the KV cache 7x, which matters a
  lot because the KV cache is read on *every* token.
- `TinyllmHeader` — the 48-byte header that starts our binary file. It stores a
  magic number (so we can detect a corrupt/wrong file), a version, a dtype flag
  (fp32 now; fp16/int8 later), and a copy of the architecture constants. Storing
  the config *inside* the file means the file is self-describing.
- `RunConfig` — runtime sampling settings (greedy vs top-p, temperature, seed).

#### `tools/convert.py`
Loads Qwen2-0.5B with HuggingFace `transformers` and writes `tinyllm.bin`.

- **File layout:** `[48-byte header][embed_tokens][layer 0 ... layer 23][final norm]`.
  Everything is little-endian FP32, written in the exact order the forward pass
  will consume it. The docstring in the file lists the precise tensor order.
- **Why keep HF's `[out_features, in_features]` weight layout?** A linear layer
  computes `y = x @ W^T`. With weights stored row-major as `[out, in]`, all the
  numbers needed to produce one output value sit next to each other in memory.
  That makes the CPU matmul a simple, cache-friendly loop.
- **Tied embeddings:** Qwen2-0.5B uses the *same* weight matrix for the input
  token embedding and the final output projection (the "LM head"). So we store
  that matrix **once** and reuse it. Saves ~136M floats (~0.5 GB).
- **QKV bias:** Qwen2's query/key/value projections have a bias vector; the
  output projection and the MLP do not. We export the biases that exist and skip
  the ones that don't — getting this wrong silently corrupts the output.

#### `tools/tokenizer.py`
Converts text <-> token IDs using HuggingFace's tokenizer.

- **Why Python and not C++?** Qwen2 uses a BPE (byte-pair encoding) tokenizer.
  Implementing BPE correctly in C++ is a real project on its own and is *not*
  what we are here to learn. We decided the C++/CUDA engine only ever handles
  integer token IDs. This script is the bridge: `encode` text -> IDs before
  inference, `decode` IDs -> text after.

#### `tools/golden.py`
Runs the *real* HuggingFace model and saves its output as the reference.

- Uses a fixed prompt: `"The capital of France is"`.
- Saves `benchmarks/golden.txt` — human-readable: the prompt, its token IDs, the
  first-step argmax token, the top-5 logits, and a 32-token greedy continuation.
- Saves `benchmarks/golden_logits.bin` — the raw FP32 logit vector of the *first*
  forward step (151936 numbers). This lets a later phase do a precise numeric
  diff, not just a "does the text match" check.
- **Why both?** Text match tells us the *final* answer is right. The raw logits
  let us catch a small numeric bug (e.g. a slightly wrong RMSNorm) that might not
  yet change the argmax token but would drift later.

### Checkpoint 0 — result: PASSED

- `tinyllm.bin` written: **1,976,131,120 bytes**. We independently computed the
  expected size from the architecture constants and it matched exactly — proof
  that every tensor was written with the right shape and nothing is missing.
- Header read back correctly: `hidden=896, layers=24, heads=14, kv=2, vocab=151936`.
- Golden reference: prompt ids `785 6722 315 9625 374` -> first-step argmax token
  `12095` (`" Paris"`) -> continuation `" Paris. It is the largest city in
  France and the second largest in Europe..."`.

**This is now the target.** When our CPU engine prints token `12095` first and
then reproduces that continuation, Phase 1 is correct.

### Concepts to focus on (80/20)

- ★ **What a "token" and "logits" are.** A token is a chunk of text mapped to an
  integer id. The model's final output is *logits* — one raw score per vocabulary
  word. Greedy decoding just picks the highest. If you only learn one thing for
  this phase, learn this. (Resource: "The Illustrated Transformer".)
- **Tokenizer / BPE — just the idea, not the algorithm.** Know that text is split
  into sub-word pieces and that it is a lookup table + merge rules. You do *not*
  need to implement it.
- **Tensor = a multi-dimensional array of numbers.** A weight "matrix" `[out, in]`
  is just `out*in` floats laid out in a row. "Row-major" = row 0, then row 1, etc.
- **Floating point: FP32 vs FP16.** FP32 = 4 bytes per number, FP16 = 2 bytes.
  This single fact drives Phases 3 and 4. (Skip the bit-level details for now.)

---

## Phase 1 — CPU baseline

**Goal:** A complete, correct forward pass for Qwen2-0.5B in plain C++ — no GPU,
no external math library. Running this end-to-end forces us to understand the
*whole* compute graph before we touch a single CUDA kernel.

### Why this phase exists

It is tempting to jump straight to GPU kernels. That is a mistake. If you write
a CUDA RMSNorm kernel before you have ever seen a working RMSNorm, you have no
way to know whether a wrong answer is a *math* bug or a *CUDA* bug. The CPU
baseline removes that ambiguity: it is the reference implementation. Every GPU
kernel in Phase 2 will be checked against the CPU version of the same op.

It also gives us our first real number — tok/s — and teaches the central lesson
of the project: inference is **memory-bandwidth-bound**. Even on a CPU, the time
goes into *reading the 0.5B weights*, not into arithmetic.

### Files created

#### `src/model.h` / `src/model.cpp` — weight loading
Reads `tinyllm.bin` into one big `std::vector<float>` and sets up pointers.

- The file is one flat blob. `model.cpp` "walks" it with a `take(n)` lambda that
  hands out the next `n` floats and advances a cursor — in the exact order
  `convert.py` wrote them. At the end it asserts the cursor consumed the whole
  buffer, which catches any layout mismatch immediately.
- `LayerWeights` is just 12 `const float*` pointers per layer. No copying — the
  pointers index *into* the one buffer. Loading 2 GB twice would be wasteful.
- **Why 64-bit matters:** the FP32 model is ~1.98 GB. A 32-bit process cannot
  address that much memory — the build must be 64-bit.

#### `src/infer_cpu.h` / `src/infer_cpu.cpp` — the forward pass
`CpuRunner` holds the KV cache + scratch buffers and exposes `forward(token, pos)`.

The forward pass, one token at a time, does exactly this per layer:

1. **RMSNorm** — `out = x / sqrt(mean(x^2) + eps) * weight`. A normalization that,
   unlike LayerNorm, has no mean-subtraction and no bias. Cheap.
2. **QKV projection** — three matmuls turning the 896-dim hidden state into
   query (896), key (128), value (128) vectors. Biases are added here (Qwen2 has
   QKV bias). Note K/V are only 128-dim — that is GQA: 2 KV heads, not 14.
3. **RoPE** — rotary position embedding. Instead of *adding* a position vector,
   it *rotates* pairs of numbers inside each head by an angle proportional to the
   position. We use the "rotate-half" form: the head vector is split in half and
   `(x0, x1) -> (x0·cos - x1·sin, x1·cos + x0·sin)`. This is what lets attention
   know token order.
4. **KV cache write** — the freshly computed K and V for this position are
   copied into a big preallocated `[layers][4096][kv_dim]` array. This is the
   whole point of a cache: past tokens' K/V never change, so we compute them once
   and reread them forever.
5. **Causal attention** — for each of the 14 query heads: dot the query against
   every cached key up to the current position, scale by `1/sqrt(head_dim)`,
   softmax, then take that weighted sum of the cached values. "Causal" = a token
   only ever attends to earlier tokens (we only loop `t <= pos`). GQA means query
   head `h` reads KV head `h / 7`.
6. **Output projection + residual** — project attention output back to 896 dims
   and *add* it to the input (`x = x + attn`). Residual connections.
7. **SwiGLU FFN** — `down( silu(gate(x)) * up(x) )`. Three matmuls and a SiLU
   activation (`silu(z) = z·sigmoid(z)`). The gate and up projections both blow
   the 896-dim vector up to 4864 dims; down brings it back. Then another residual.

Finally: one more RMSNorm, then the **LM head** — a matmul against the
embedding matrix (tied weights) producing a 151936-long logit vector, one score
per vocabulary token.

#### `src/sampler.h` / `src/sampler.cpp` — picking the next token
- **Greedy** (temperature 0): just `argmax` of the logits. Deterministic. This is
  what we use for verification, because the golden reference is also greedy.
- **Top-p / nucleus** (temperature > 0): softmax with temperature, sort by
  probability, keep the smallest set of tokens whose probabilities sum to `top_p`,
  sample from that set. Uses a tiny xorshift RNG so a fixed seed is reproducible.

#### `src/main.cpp` — the CLI
Loads the model, **prefills** the prompt (runs `forward` on every prompt token to
populate the KV cache), then generates `--max-new` tokens, timing the loop and
printing tok/s. `--dump-logits` writes the first-step logit vector to disk so we
can numerically diff it against `golden_logits.bin`.

#### `Makefile` — two builds from one source
The `Makefile` is the primary way to build. It produces a **naive** binary and
an **OpenMP** binary *from identical source* — the only difference is the
`-fopenmp` flag. Without it, the compiler silently ignores the `#pragma omp`
lines and you get the single-threaded version.

> **Note (Windows):** the build must be 64-bit (see above). If `make`/`g++` is
> unavailable or 32-bit, compile the same four `src/*.cpp` files with 64-bit
> MSVC `cl` — add `/openmp` for the OpenMP build.

### The OpenMP optimization

There is exactly **one** kind of pragma in `infer_cpu.cpp`:
`#pragma omp parallel for` on the *output-row* loop of the matmul (and on the
per-head loop of attention). Each output row of a matmul is independent — row `o`
needs only `x` and weight row `o` — so different threads can compute different
rows with no coordination. One line, and the matmuls go parallel.

This works because the rows are *embarrassingly parallel* and share only
read-only data. No locks, no races.

### Checkpoint 1 — result: PASSED

Measured on the cloud GPU VM's CPU (same machine as all later phases):

| Build       | tok/s | Notes                                  |
|-------------|-------|----------------------------------------|
| CPU naive   | 1.60  | single thread (compiler still vectorizes the inner loop) |
| CPU OpenMP  | 5.13  | ~3.2x faster, all cores                |

- **Greedy output is token-exact** vs the golden reference — all 32 tokens match
  (` Paris. It is the largest city in France...`), for *both* the naive and
  OpenMP builds.
- **First-step logits** match the HuggingFace logits to a **max absolute
  difference of 3.8e-5** — that is pure floating-point rounding noise (different
  summation order), not a bug. The argmax token (`12095`) is identical.

**Takeaway:** the math is provably correct. From here on, when a GPU kernel
disagrees with this CPU engine, the bug is in the *kernel*, not the *model*. The
OpenMP speedup also previews Phase 2's lesson — throughput comes from doing the
weight reads in parallel, not from cleverer arithmetic.

### Concepts to focus on (80/20)

This is the phase with the most new ideas. Study them in this order:

- ★ **Self-attention — the core idea.** Every token produces a *query*, *key*, and
  *value*. A token's new representation is a weighted average of all *values*,
  where the weights come from how well its *query* matches each *key*. That's it.
  Everything else is plumbing. (Resource: Karpathy's "Let's build GPT" video — he
  derives exactly this. Or "The Illustrated Transformer".)
- ★ **Matmul is ~90% of the compute.** QKV, output, gate, up, down — five matmuls
  per layer, 24 layers. Attention and the norms are cheap by comparison. When we
  optimize later, we optimize matmul. Be very comfortable with "y = W·x": output
  element `o` is the dot product of weight row `o` with the input vector.
- **The KV cache — why inference isn't quadratic.** Past tokens' keys/values never
  change, so we compute them once and store them. Each new token only computes
  *its own* K/V and reads the rest. Without this, generation would recompute
  everything every step.
- **Residual connections + the "residual stream."** `x = x + sublayer(x)`. Each
  layer *adds* a correction to a running hidden state rather than replacing it.
  This mental model ("the residual stream") explains the whole layer loop.
- **RMSNorm, RoPE, SwiGLU — know the *role*, not the derivation.** RMSNorm = keep
  numbers from exploding/vanishing. RoPE = inject token *position* by rotating
  vectors. SwiGLU = the feed-forward network's particular flavor of nonlinearity.
  You can treat each as a labeled box for now; revisit the math only if curious.
- **GQA (Grouped Query Attention).** 14 query heads, 2 KV heads. Purely a memory
  optimization — it shrinks the KV cache 7x. (Resource: search "GQA" — the idea is
  one paragraph.)
- **Greedy vs sampling.** Greedy = always argmax (deterministic, what we verify
  with). Temperature/top-p = controlled randomness for non-repetitive text.

If short on time: master the two ★ items. Attention + matmul *is* the model.

---

## Phase 2 — GPU port, kernel by kernel

**Goal:** Move the entire forward pass onto the GPU. Every CPU op from Phase 1
gets a hand-written CUDA kernel — no cuBLAS, no CUTLASS. Still FP32.

> **Status: verified.** Authored on a machine with no GPU, then compiled with
> `nvcc` and run on the cloud GPU VM — it compiled and produced correct output
> on the first try. See "Checkpoint 2" below for the numbers.

### Why this phase exists

A GPU has thousands of cores. The CPU engine computes one matmul output row at a
time (or a few, with OpenMP). The GPU computes thousands at once. But the GPU
only helps if the *data movement* is right — and that is the real lesson here.
Inference is **memory-bandwidth-bound**: each generated token must read every
weight in the model exactly once, and does only a couple of FLOPs per weight.
So the kernels are designed around *how memory is read*, not how math is done.

### How the GPU build is structured

- **`kernels/kernels.cuh`** — declares the host-side launch wrappers
  (`rmsnorm_cuda`, `rope_cuda`, …) and a `CUDA_CHECK` macro that aborts on any
  CUDA error. This is the contract between `infer.cu` and the kernels.
- **`kernels/*.cu`** — one file per op. Each has a `__global__` kernel (runs on
  the GPU) and a small host wrapper that picks the launch grid.
- **`infer.cu`** — `GpuRunner`: same interface as `CpuRunner`. It uploads all
  weights to the GPU once, owns the device KV cache, and runs the per-layer
  sequence of kernel launches.
- **`src/infer_gpu.h`** — `GpuRunner`'s header. Deliberately contains *no* CUDA
  types (device pointers are plain `float*`), so `main.cpp` can include it
  without needing the CUDA compiler.
- **`main.cpp`** — one `#ifdef USE_CUDA` picks `GpuRunner` vs `CpuRunner`. The
  `gpu` Makefile target defines `USE_CUDA` and builds with `nvcc`.

### The kernels (ported easiest-to-hardest)

#### `rmsnorm.cu`
A **reduction** followed by an element-wise scale. One block handles the whole
896-dim vector: each thread sums the squares of its slice, then a tree reduction
in shared memory combines them into one sum-of-squares. The "tree reduction"
(halve the active threads each step) is the fundamental GPU pattern for "combine
N values into one" — worth understanding well.

#### `rope.cu`
Pure element-wise: one thread per `(head, pair)` index, each rotating one pair of
numbers. No reduction, no shared memory — the easiest kind of kernel.

#### `swiglu.cu`
One thread per element, computing `silu(gate[i]) * up[i]`. **Fused**: the
`silu(gate)` intermediate is never written to global memory — it lives in a
register for the one instruction between computing it and multiplying. Writing
it out and reading it back would cost two extra trips over the bandwidth bottleneck.

#### `matmul.cu` — the important one
For single-token decode this is a matrix-times-**vector** product. The design:
**one warp (32 threads) per output row**. The 32 lanes stride across the input
dimension so they read *consecutive* weights — a "coalesced" access, which is
what lets the GPU read memory at full bandwidth. The 32 partial sums are then
combined with a **warp-shuffle reduction** (`__shfl_down_sync`) — threads in a
warp exchange registers directly, no shared memory needed.
- *Why this matters:* five matmuls per layer × 24 layers — this kernel is where
  almost all the time goes. Phase 3's profiling should confirm it.
- *Known next step:* `float4` vectorized loads (read 4 weights per instruction).
  Left as the Phase 2 stretch optimization.

#### `attention.cu` — the hardest
One block per query head. It does, in order: (1) score = dot(query, each cached
key); (2) a block reduction for the max; (3) `exp` and a block reduction for the
sum — that is the softmax; (4) the weighted sum of the cached values. GQA is
handled by mapping query head `h` to key/value head `h / 7`. The scores live in
shared memory. *Known next step:* a single-pass "flash-attention style" fused
softmax that never materializes the full score array.

### Two design choices worth noting

- **K and V are written straight into the cache.** The `k`/`v` projection
  matmuls write their output *directly* into this token's slot in the KV cache,
  and RoPE rotates it in place. No separate buffer, no copy.
- **Default CUDA stream = correct ordering for free.** Every kernel launches on
  the default stream, so they run strictly one after another. Since each op
  depends on the previous one's output, that is exactly the ordering we need —
  no manual synchronization between kernels.

### Checkpoint 2 — result: PASSED

Built with `make gpu` and run on the cloud GPU VM:

| Engine        | Hardware | tok/s  |
|---------------|----------|--------|
| GPU FP32      | VM GPU   | 112.43 |
| CPU + OpenMP  | VM CPU   | 5.13   |

- **Greedy output token-exact** vs the golden reference — all 32 tokens match.
- **First-step logits** match HuggingFace to a **max absolute difference of
  1.6e-5** (even tighter than the CPU engine's 3.8e-5). Argmax `12095` identical.
- **~22x faster** than the CPU+OpenMP build on the same VM — and this is still
  the *naive* matmul (warp-per-row, no `float4`) in plain FP32. Phases 3–4 push
  it further.

To reproduce: `make gpu`, then run with `--dump-logits` and diff the dumped
vector against `golden_logits.bin` with NumPy, exactly as in Phase 1. If output
is ever wrong, bisect kernel by kernel — the GPU and CPU runners share an
interface, so one GPU kernel can be swapped back to CPU at a time.

### Concepts to focus on (80/20)

- ★ **Coalesced memory access.** The single most important GPU idea for this
  project. When 32 neighboring threads read 32 neighboring addresses, the
  hardware fetches it in one transaction at full bandwidth. Scattered reads waste
  most of the bandwidth. This is *why* `matmul.cu` is laid out warp-per-row.
- ★ **Memory-bandwidth-bound.** Decoding one token reads all ~0.5B weights once.
  The math is trivial next to the reading. So "make it faster" = "read less / read
  better", which is the whole point of Phases 3–4. (Resource: Horace He, "Making
  Deep Learning Go Brrrr".)
- **Thread / block / warp / grid.** A *thread* does scalar work; 32 threads form a
  *warp* that executes in lockstep; threads group into *blocks* that share fast
  shared memory; blocks form the *grid*. (Resource: PMPP book, early chapters.)
- **Reductions.** Combining N values into one (a sum, a max) in parallel — the
  tree reduction (shared memory) and the warp-shuffle reduction. Used in rmsnorm
  and attention. Learn the pattern once; it recurs everywhere.
- **Shared memory vs global memory.** Shared memory is small, per-block, and
  ~100x faster than global. Kernels stage hot data there (the scores in
  attention). Global memory is the big, slow space the weights live in.
- **Kernel launch / `<<<blocks, threads>>>`.** How the host starts GPU work, and
  why launches on one stream serialize. The grid dimensions are just "how many
  threads, grouped how".

If short on time: master the two ★ items. The rest of Phases 3–5 only makes
sense once "it's bandwidth-bound, so read memory well" is second nature.

---

## Phase 3 — FP16 + profiling

_Not started yet._

---

## Phase 4 — INT8 weight quantization

_Not started yet._

---

## Phase 5 — Measure & document

_Not started yet._
