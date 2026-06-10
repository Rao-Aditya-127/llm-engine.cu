"""TinyLLM FastAPI inference server.

A single background scheduler thread owns the engine and runs **continuous
batching**: many sequences decode together in one batched GPU step, and the
scheduler admits new requests / evicts finished ones every step. Both
/generate (blocking) and /generate/stream (SSE) feed the same scheduler.

  POST /generate          — blocking, full response
  POST /generate/stream   — SSE, tokens arrive one by one
  GET  /health            — slot occupancy + queue depth
  GET  /                  — demo UI

Start:
    cd <repo-root>
    uvicorn server.server:app --host 0.0.0.0 --port 8000

The llm_engine .so must be on sys.path.  The Makefile builds it at
server/llm_engine<ext>.so; running uvicorn from the repo root and having
the server/ directory in sys.path (added below) is enough.
"""

import itertools
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

# Allow `import llm_engine` to find the compiled .so in the server/ directory.
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from llm_engine import LLMEngine

# ---------------------------------------------------------------------------
# App + startup
# ---------------------------------------------------------------------------
app = FastAPI(title="TinyLLM", description="Qwen2-0.5B FP16 inference server")

MODEL_PATH = "tinyllm_fp16.bin"   # relative to cwd where uvicorn is launched
# Override with: export TINYLLM_MODEL=Qwen/Qwen2-1.5B-Instruct
MODEL_ID   = os.environ.get("TINYLLM_MODEL", "Qwen/Qwen2-0.5B-Instruct")

print(f"Loading tokenizer ({MODEL_ID}) ...")
_tok = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Loading engine ({MODEL_PATH}) ...")
_engine = LLMEngine(MODEL_PATH)
print(f"Engine ready. {_engine.max_slots()} batch slots.")

# Tokens that end generation: <|endoftext|> and the chat-turn terminator.
STOP_IDS = {
    _tok.eos_token_id,
    _tok.convert_tokens_to_ids("<|im_end|>"),
}

# ---------------------------------------------------------------------------
# Continuous-batch scheduler
# ---------------------------------------------------------------------------

MAX_SLOTS  = _engine.max_slots()
_seed_seq  = itertools.count(1)        # distinct seeds so sampled runs differ


class _Seq:
    """One in-flight sequence. Output goes to a queue (streaming) or a result
    list guarded by an event (blocking)."""
    def __init__(self, prompt_ids, max_tokens, temperature, top_p, streaming):
        self.prompt_ids  = prompt_ids
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.top_p       = top_p
        self.seed        = next(_seed_seq)
        self.streaming   = streaming

        # scheduler-owned state
        self.slot        = -1
        self.pos         = 0
        self.generated   = 0
        self.last_token  = 0
        self.finished    = False

        # output channels
        self.out_queue: queue.Queue = queue.Queue() if streaming else None
        self.result: list[int]      = []
        self.event   = threading.Event()

    def emit(self, token_id: int) -> None:
        if self.streaming:
            self.out_queue.put(token_id)
        else:
            self.result.append(token_id)

    def finish(self) -> None:
        self.finished = True
        if self.streaming:
            self.out_queue.put(None)   # sentinel — closes the SSE stream
        self.event.set()


_waiting: queue.Queue[_Seq] = queue.Queue()
_active: list[_Seq] = []
_free_slots: list[int] = list(range(MAX_SLOTS))


def _admit(seq: _Seq) -> None:
    """Assign a free slot, prefill the prompt, and emit the first token."""
    seq.slot = _free_slots.pop()
    first = _engine.prefill_slot(seq.prompt_ids, seq.slot,
                                 seq.temperature, seq.top_p, seq.seed)
    seq.pos        = len(seq.prompt_ids)   # next decode places last_token here
    seq.generated  = 1
    seq.last_token = first

    is_stop = first in STOP_IDS
    if not is_stop:
        seq.emit(first)
    if is_stop or seq.generated >= seq.max_tokens:
        seq.finish()
        _free_slots.append(seq.slot)
    else:
        _active.append(seq)


def _scheduler() -> None:
    """Continuous batching: admit waiting requests into free slots, then run
    one batched decode step over every active sequence, every iteration."""
    while True:
        # If nothing is running, block until at least one request arrives.
        if not _active:
            _admit(_waiting.get())
        # Fill any remaining free slots without blocking.
        while _free_slots and not _waiting.empty():
            try:
                _admit(_waiting.get_nowait())
            except queue.Empty:
                break
        if not _active:
            continue

        # One batched decode step over all active sequences.
        next_toks = _engine.decode_batch(
            [s.last_token for s in _active],
            [s.pos for s in _active],
            [s.slot for s in _active])

        for s, t in zip(_active, next_toks):
            s.pos       += 1
            s.generated += 1
            s.last_token = t
            is_stop = t in STOP_IDS
            if not is_stop:
                s.emit(t)
            if is_stop or s.generated >= s.max_tokens:
                s.finish()
                _free_slots.append(s.slot)

        _active[:] = [s for s in _active if not s.finished]


threading.Thread(target=_scheduler, daemon=True, name="scheduler").start()

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt:      str
    max_tokens:  int   = Field(default=100, ge=1, le=4000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p:       float = Field(default=1.0, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    response:         str
    tokens_generated: int
    time_ms:          float


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text()


# ---------------------------------------------------------------------------
# Phase 1 — blocking endpoint
# ---------------------------------------------------------------------------

def _apply_chat_template(prompt: str) -> list[int]:
    """Wrap a raw user message in the Instruct chat template."""
    text = _tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    return _tok.encode(text, add_special_tokens=False)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate a response for the given prompt. Blocks until complete."""
    prompt_ids = _apply_chat_template(req.prompt)
    t0 = time.perf_counter()

    seq = _Seq(prompt_ids, req.max_tokens, req.temperature, req.top_p,
               streaming=False)
    _waiting.put(seq)
    seq.event.wait()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return GenerateResponse(
        response         = _tok.decode(seq.result, skip_special_tokens=True),
        tokens_generated = len(seq.result),
        time_ms          = round(elapsed_ms, 1),
    )


# ---------------------------------------------------------------------------
# Streaming endpoint (Server-Sent Events)
# ---------------------------------------------------------------------------

@app.post("/generate/stream")
def generate_stream(req: GenerateRequest) -> StreamingResponse:
    """Stream tokens back as Server-Sent Events.

    Each event is: data: {"token": "<text>"}\\n\\n
    Final event is: data: [DONE]\\n\\n
    """
    prompt_ids = _apply_chat_template(req.prompt)
    seq = _Seq(prompt_ids, req.max_tokens, req.temperature, req.top_p,
               streaming=True)
    _waiting.put(seq)

    def event_stream():
        while True:
            token_id = seq.out_queue.get()
            if token_id is None:          # sentinel from finish()
                yield "data: [DONE]\n\n"
                break
            text = _tok.decode([token_id], skip_special_tokens=True)
            yield f"data: {json.dumps({'token': text})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Return server status and current scheduler occupancy."""
    return {
        "status":       "ok",
        "model":        MODEL_ID,
        "max_slots":    MAX_SLOTS,
        "active":       len(_active),
        "free_slots":   len(_free_slots),
        "queue_depth":  _waiting.qsize(),
    }
