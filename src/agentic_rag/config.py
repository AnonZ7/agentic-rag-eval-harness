"""Central configuration. Everything is env-overridable so the same code runs
offline in CI (no keys) and live in production (real provider keys)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Provider: "fake" (offline, deterministic), "anthropic", "openai", or "gemini".
    # Defaults to "fake" so the repo runs with zero secrets — flip via LLM_PROVIDER.
    provider: str = os.getenv("LLM_PROVIDER", "fake")
    model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-5")

    # Retrieval
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    rrf_k: int = int(os.getenv("RAG_RRF_K", "60"))  # reciprocal-rank-fusion constant
    embed_dim: int = int(os.getenv("RAG_EMBED_DIM", "256"))

    # Guardrails
    max_question_chars: int = int(os.getenv("GUARD_MAX_Q", "2000"))
    min_grounding: float = float(os.getenv("GUARD_MIN_GROUNDING", "0.30"))

    docs_dir: str = os.getenv("RAG_DOCS_DIR", "data/docs")


settings = Settings()
