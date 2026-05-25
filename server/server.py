"""TinyLLM FastAPI inference server.

Phases implemented here:
  Phase 1 — POST /generate          (blocking, full response)
  Phase 2 — POST /generate/stream   (SSE, tokens arrive one by one)
  Phase 3 — thread-safe request queue (one background thread owns the engine;
             concurrent HTTP requests queue up safely)

Start:
    cd <repo-root>
    uvicorn server.server:app --host 0.0.0.0 --port 8000

The llm_engine .so must be on sys.path.  The Makefile builds it at
server/llm_engine<ext>.so; running uvicorn from the repo root and having
the server/ directory in sys.path (added below) is enough.
"""

import json
import queue
import sys
import threading
import time
from pathlib import Path

# Allow `import llm_engine` to find the compiled .so in the server/ directory.
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from llm_engine import LLMEngine

# ---------------------------------------------------------------------------
# App + startup
# ---------------------------------------------------------------------------
app = FastAPI(title="TinyLLM", description="Qwen2-0.5B FP16 inference server")

MODEL_PATH = "tinyllm_fp16.bin"   # relative to cwd where uvicorn is launched
MODEL_ID   = "Qwen/Qwen2-0.5B"

print(f"Loading tokenizer ({MODEL_ID}) ...")
_tok = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Loading engine ({MODEL_PATH}) ...")
_engine = LLMEngine(MODEL_PATH)
print("Engine ready.")

# ---------------------------------------------------------------------------
# Phase 3 — single background worker owns the engine
# ---------------------------------------------------------------------------

class _PendingRequest:
    """Holds one HTTP request waiting for the engine worker."""
    def __init__(self, prompt_ids: list[int], max_tokens: int,
                 temperature: float, top_p: float,
                 streaming: bool = False,
                 on_token=None):
        self.prompt_ids  = prompt_ids
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.top_p       = top_p
        self.streaming   = streaming
        self.on_token    = on_token   # callable(token_id: int) — streaming only
        self.result: list[int] | None = None
        self.event = threading.Event()

_request_queue: queue.Queue[_PendingRequest] = queue.Queue()


def _engine_worker() -> None:
    """Single thread that serialises all calls into the engine."""
    while True:
        req = _request_queue.get()
        try:
            if req.streaming:
                _engine.generate_ids_streaming(
                    req.prompt_ids, req.max_tokens, req.on_token,
                    req.temperature, req.top_p)
            else:
                req.result = _engine.generate_ids(
                    req.prompt_ids, req.max_tokens,
                    req.temperature, req.top_p)
        finally:
            req.event.set()   # wake up the waiting HTTP handler


threading.Thread(target=_engine_worker, daemon=True, name="engine-worker").start()

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
# Phase 1 — blocking endpoint
# ---------------------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate a response for the given prompt. Blocks until complete."""
    prompt_ids = _tok.encode(req.prompt)
    t0 = time.perf_counter()

    pending = _PendingRequest(prompt_ids, req.max_tokens,
                              req.temperature, req.top_p)
    _request_queue.put(pending)
    pending.event.wait()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return GenerateResponse(
        response         = _tok.decode(pending.result),
        tokens_generated = len(pending.result),
        time_ms          = round(elapsed_ms, 1),
    )


# ---------------------------------------------------------------------------
# Phase 2 — streaming endpoint (Server-Sent Events)
# ---------------------------------------------------------------------------

@app.post("/generate/stream")
def generate_stream(req: GenerateRequest) -> StreamingResponse:
    """Stream tokens back as Server-Sent Events.

    Each event is: data: {"token": "<text>"}\\n\\n
    Final event is: data: [DONE]\\n\\n
    """
    prompt_ids = _tok.encode(req.prompt)
    token_queue: queue.Queue[int | None] = queue.Queue()

    def on_token(token_id: int) -> None:
        token_queue.put(token_id)

    pending = _PendingRequest(prompt_ids, req.max_tokens,
                              req.temperature, req.top_p,
                              streaming=True, on_token=on_token)
    _request_queue.put(pending)

    def event_stream():
        while True:
            token_id = token_queue.get()
            # pending.event is set after the last on_token call returns.
            # Check: if the event fired and the queue is now drained, we're done.
            if pending.event.is_set() and token_queue.empty():
                # token_id itself may still be a real token — yield it first.
                if token_id is not None:
                    yield f"data: {json.dumps({'token': _tok.decode([token_id])})}\n\n"
                yield "data: [DONE]\n\n"
                break
            if token_id is not None:
                yield f"data: {json.dumps({'token': _tok.decode([token_id])})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Return server status and current request queue depth."""
    return {
        "status":      "ok",
        "model":       MODEL_ID,
        "queue_depth": _request_queue.qsize(),
    }
