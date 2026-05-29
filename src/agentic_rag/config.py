"""Central configuration. Everything is env-overridable so the same code runs
offline in CI (no keys) and live in production (real provider keys)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


VALID_PROVIDERS = ("fake", "anthropic", "openai", "gemini")


def _validated_provider() -> str:
    """Coerce LLM_PROVIDER to a known value (defaults to 'fake'). An allowlist here both
    hardens config and removes the tainted env->log path (unknown values never reach output)."""
    p = os.getenv("LLM_PROVIDER", "fake").strip().lower()
    return p if p in VALID_PROVIDERS else "fake"


@dataclass(frozen=True)
class Settings:
    # Provider: "fake" (offline, deterministic), "anthropic", "openai", or "gemini".
    # Defaults to "fake" so the repo runs with zero secrets — flip via LLM_PROVIDER.
    provider: str = _validated_provider()
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
