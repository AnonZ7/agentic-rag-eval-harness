"""FastAPI surface. Typed request/response, health check, and a single /ask endpoint.

Run:  uvicorn agentic_rag.api:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .agent import answer

app = FastAPI(
    title="Agentic RAG Eval Harness",
    version=__version__,
    description="LangGraph agent + hybrid RAG + guardrails, with grounding scores on every answer.",
)


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
