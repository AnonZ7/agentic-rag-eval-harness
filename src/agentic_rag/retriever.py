"""Hybrid retriever: BM25 (lexical) + dense embeddings (semantic), fused with
Reciprocal Rank Fusion. The default embedder is a deterministic hashing embedder so
retrieval works offline with no model download; swap in a real embedder in production.

Hybrid retrieval consistently beats either signal alone — lexical catches exact terms
(names, codes), dense catches paraphrase. RRF needs no score normalization between them.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings

_WORD = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class Chunk:
    id: str
    text: str
    source: str


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder (no deps, no network).

    Good enough to demonstrate semantic fusion and to keep CI hermetic. In production,
    replace `.embed` with sentence-transformers or a provider embedding endpoint.
    """

    def __init__(self, dim: int):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _tok(text):
            # blake2b: fast, stable across runs (unlike built-in hash with PYTHONHASHSEED),
            # and not a weak-crypto primitive. Used only for feature bucketing, not security.
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v


def _chunk_markdown(text: str, source: str, size: int = 90) -> list[Chunk]:
    """Split on paragraphs, then pack into ~`size`-word chunks (cheap, sane defaults)."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], []
    for p in paras:
        buf.append(p)
        if sum(len(b.split()) for b in buf) >= size:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    return [Chunk(id=f"{source}#{i}", text=c, source=source) for i, c in enumerate(chunks)]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: HashEmbedder | None = None):
        if not chunks:
            raise ValueError("HybridRetriever needs at least one chunk")
        self.chunks = chunks
        self.embedder = embedder or HashEmbedder(settings.embed_dim)
        self._bm25 = BM25Okapi([_tok(c.text) for c in chunks])
        self._emb = np.vstack([self.embedder.embed(c.text) for c in chunks])

    @classmethod
    def from_dir(cls, docs_dir: str | None = None) -> "HybridRetriever":
        docs_dir = docs_dir or settings.docs_dir
        root = os.path.realpath(docs_dir)
        chunks: list[Chunk] = []
        for path in sorted(glob.glob(os.path.join(docs_dir, "**", "*.md"), recursive=True)):
            # path-traversal guard: only read files that resolve INSIDE the docs root
            # (defends against symlinks / crafted docs_dir escaping the corpus).
            real = os.path.realpath(path)
            if os.path.commonpath([root, real]) != root:
                continue
            with open(real, encoding="utf-8") as f:
                chunks += _chunk_markdown(f.read(), os.path.basename(real))
        return cls(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Chunk]:
        top_k = top_k or settings.top_k
        # lexical ranking
        bm = self._bm25.get_scores(_tok(query))
        bm_rank = np.argsort(bm)[::-1]
        # semantic ranking
        qv = self.embedder.embed(query)
        sim = self._emb @ qv
        sim_rank = np.argsort(sim)[::-1]
        # reciprocal rank fusion
        k = settings.rrf_k
        fused: dict[int, float] = {}
        for rank, idx in enumerate(bm_rank):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
        for rank, idx in enumerate(sim_rank):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
        order = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [self.chunks[i] for i in order]


def grounding_score(answer: str, contexts: list[str]) -> float:
    """Fraction of answer content-words that appear in retrieved context. Cheap, fast,
    offline proxy for faithfulness used by the guardrail + verify node (Ragas gives the
    LLM-judged version in the full eval path)."""
    aw = {w for w in _tok(answer) if len(w) > 3}
    if not aw:
        return 1.0
    cw = set()
    for c in contexts:
        cw |= set(_tok(c))
    return len(aw & cw) / len(aw)
