# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions are date-tagged.

## [Unreleased]

### Added
- Optional real embeddings via `sentence-transformers` (`RAG_EMBEDDER=st`, `pip install .[embedder]`);
  offline `HashEmbedder` remains the default so CI stays hermetic.
- `CHANGELOG.md`, issue/PR templates, Dependabot (pip + github-actions), Dockerfile, `.dockerignore`,
  `CONTRIBUTING.md`, `SECURITY.md`, supply-chain audit (`pip-audit`) in CI.

### Changed
- Bumped dependency floors to **LangGraph 1.x / LangChain-core 1.x**.
- CI actions updated (checkout v6, setup-python v6).

### Security
- Replaced `md5` with `blake2b` in the embedder (weak-crypto finding).
- Added a path-traversal containment guard (`os.path.commonpath`) to the corpus loader.
- Provider allowlist removes a tainted env→log path.
- Expanded prompt-injection patterns; FastAPI gained closed CORS, in-process rate limiting,
  and a leak-free exception handler.

## [0.1.0]

### Added
- Initial release: LangGraph plan→act→verify→refine agent, hybrid BM25 + dense retrieval (RRF),
  input/output guardrails (injection, PII, grounding floor), typed FastAPI surface, and an offline
  evaluation gate (retrieval P/R, grounding, answer-F1) with optional Ragas. Provider-agnostic
  (Anthropic / OpenAI / Gemini); runs fully offline in CI with no keys.
