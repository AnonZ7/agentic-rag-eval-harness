# Agentic RAG Eval Harness

> A production-shaped **agentic RAG** service with a built-in **evaluation harness** —
> a LangGraph **plan→act→verify** agent, **hybrid retrieval** (BM25 + dense, fused with RRF),
> deterministic **guardrails**, a typed **FastAPI** surface, and an **eval gate in CI**.
> Provider-agnostic (Anthropic / OpenAI / Gemini) in production; **runs fully offline in CI with no API keys**.

![ci-and-evals](https://img.shields.io/badge/CI-tests%20%2B%20eval%20gate-1a7f5a)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## Why this exists

Most "RAG demos" call an LLM once and hope. Production agentic systems need three things
this repo demonstrates end to end:

1. **An agent that verifies itself** — a LangGraph graph that plans a query, retrieves,
   generates strictly from context, then **checks its own grounding and refines once** if
   weak, rather than confidently hallucinating.
2. **Guardrails on every request** — length caps, prompt-injection detection, PII
   redaction, and a **grounding floor that makes the agent refuse instead of fabricate**.
3. **Evaluation as a CI gate** — quality is a *number*, not a vibe. Retrieval
   precision/recall, grounding, and answer-F1 are computed on a reference dataset and
   **fail the build on regression**. Add Ragas LLM-judged faithfulness/answer-relevancy in live mode.

## Architecture

```
            ┌── guardrails (input: length · injection · PII) ──┐
question ─► │  plan ─► retrieve(BM25 ⊕ dense, RRF) ─► generate │ ─► answer + sources + grounding
            │     ▲                                   │        │
            │     └──── refine (verify failed) ◄──────┘        │
            └── guardrails (output: grounding floor · cite) ───┘
```

## Quickstart (offline — zero secrets)

```bash
pip install -e ".[dev]"
pytest -q                       # unit tests (fake LLM + hashing embedder)
python -m evals.run_evals       # eval gate — prints metrics, exits non-zero on regression
uvicorn agentic_rag.api:app     # POST /ask  {"question": "What is reciprocal rank fusion?"}
```

## Run it live (real models)

```bash
pip install -e ".[providers,evals]"
export LLM_PROVIDER=anthropic LLM_MODEL=claude-sonnet-4-5 ANTHROPIC_API_KEY=...
python -m evals.run_evals --ragas      # adds Ragas faithfulness / answer-relevancy / context-precision
```

Swap `LLM_PROVIDER` to `openai` or `gemini` with no code changes — that is the point of the
provider-agnostic client in [`llm.py`](src/agentic_rag/llm.py).

## What to read first

| File | What it shows |
|---|---|
| [`agent.py`](src/agentic_rag/agent.py) | The LangGraph plan→act→verify→refine graph |
| [`retriever.py`](src/agentic_rag/retriever.py) | Hybrid BM25 + dense retrieval, RRF fusion |
| [`guardrails.py`](src/agentic_rag/guardrails.py) | Input/output guardrails, grounding floor |
| [`evals/run_evals.py`](evals/run_evals.py) | Offline metric gate + optional Ragas |
| [`evals/metrics.md`](evals/metrics.md) | What each metric means and why |

## Security

The LLM is treated as **untrusted by design** — defenses live around it, not just in the prompt.
Full threat model + OWASP-LLM-Top-10 mapping in [SECURITY.md](SECURITY.md).

- **Input guardrails** — length caps + prompt-injection/jailbreak pattern detection.
- **Output guardrails** — PII redaction + a **grounding floor** that refuses rather than hallucinates.
- **Corpus isolation** — path-traversal guard; only files inside the docs root are ever read.
- **API hardening** — closed CORS by default, in-process rate limiting, leak-free error handler.
- **Bounded loops** — the verify→refine step runs at most once (no runaway agent consumption).

Verified with a SAST pass (foxguard): weak-crypto and log-injection findings remediated; the one
remaining `open()` flag is a documented false positive (mitigated by the `commonpath` containment guard).

## Tested & verified

- **19 tests** (retriever, guardrails, agent, API, config) — `pytest -q`
- **Eval gate** — fails CI if retrieval recall / grounding / answer-F1 regress
- **Ruff** clean · **CI** runs all three on every push
- LangGraph **1.x** / LangChain-core **1.x** (current majors)

## Design decisions

- **Offline-first CI.** A deterministic `FakeLLM` + hashing embedder make the test/eval path
  hermetic — no flaky network, no API spend on every PR. Real providers light up via one env var.
- **RRF over score-normalization.** Fusing BM25 and dense ranks by reciprocal rank avoids
  comparing incomparable score scales.
- **Bounded self-correction.** The verify node refines **at most once** — agents that loop
  forever are a production incident waiting to happen.

## License

MIT — see [LICENSE](LICENSE).

---
Built by **Zakaria Aichaoui** — Applied AI / Agent Engineer (Algiers, remote). Part of an
ongoing portfolio of agentic-systems work.
