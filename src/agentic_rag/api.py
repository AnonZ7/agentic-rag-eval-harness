"""FastAPI surface. Typed request/response, health check, and a single /ask endpoint.

Security posture (see SECURITY.md for the full threat model):
- Input is length-capped by Pydantic and re-checked by the agent's input guardrail.
- CORS is closed by default — set ALLOWED_ORIGINS (comma-separated) to open specific origins.
- A lightweight in-process rate limiter caps abuse. NOTE: it is per-worker and resets on
  restart — front with a shared limiter (Redis / API gateway) in real production.
- The exception handler never leaks internals to the client.

Run:  uvicorn agentic_rag.api:app --reload
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .agent import answer

app = FastAPI(
    title="Agentic RAG Eval Harness",
    version=__version__,
    description="LangGraph agent + hybrid RAG + guardrails, with grounding scores on every answer.",
)

# --- CORS: closed by default (no wildcard with credentials) ----------------------
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,            # empty list = no cross-origin browser access
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    allow_credentials=False,
)

# --- Minimal in-process rate limiter (per client IP, sliding window) --------------
_RATE_MAX = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
_WINDOW = 60.0
_hits: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    q = _hits[ip]
    while q and now - q[0] > _WINDOW:
        q.popleft()
    if len(q) >= _RATE_MAX:
        return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
    q.append(now)
    return await call_next(request)


@app.exception_handler(Exception)
async def _safe_errors(request: Request, exc: Exception) -> JSONResponse:
    # never leak stack traces / internals to the caller
    return JSONResponse(status_code=500, content={"detail": "internal error"})


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, examples=["What is reciprocal rank fusion?"])


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    grounding: float
    blocked: str = ""


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return AskResponse(**answer(req.question))
