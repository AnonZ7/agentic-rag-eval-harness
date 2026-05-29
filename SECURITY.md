# Security & Threat Model

This service treats the LLM as **untrusted by design** — the security architecture assumes the
model can be manipulated, so defenses live around it (input filtering, context isolation, output
validation, grounding floor), not only in the prompt. Below is how the design maps to the
[OWASP Top 10 for LLM Applications (2025/2026)](https://owasp.org/www-project-top-10-for-large-language-model-applications/).

## Threat model (assumptions)

- **Untrusted input** — questions arrive from the public internet and may contain injection,
  oversized payloads, or PII.
- **Untrusted model output** — the LLM may hallucinate or be steered; nothing it returns is trusted
  until grounding is checked.
- **Trusted corpus** — `data/docs/` is operator-controlled; the retriever refuses to read outside it.

## OWASP LLM Top 10 coverage

| Risk | Mitigation in this repo |
|---|---|
| **LLM01 Prompt Injection** | Input guardrail regex ([`guardrails.py`](src/agentic_rag/guardrails.py)) blocks known override/jailbreak patterns; generation is constrained to retrieved context only; the model is never given tool/exec authority. |
| **LLM02 Sensitive Information Disclosure** | `redact_pii()` scrubs emails/SSN/card-shaped tokens from output; closed CORS; exception handler never leaks internals. |
| **LLM05 Improper Output Handling** | Output guardrail enforces a **grounding floor** — the agent refuses rather than emit ungrounded claims; responses are typed (Pydantic), not free-form passthrough. |
| **LLM04 Data/Model Poisoning** | Corpus is operator-controlled; retriever has a **path-traversal guard** (`os.path.commonpath`) so only files inside the docs root are read. |
| **LLM08 Vector/Embedding Weaknesses** | Retrieval is read-only over a fixed corpus; no user-controlled documents are ingested at query time. |
| **LLM10 Unbounded Consumption** | In-process **rate limiter** + Pydantic length caps + **bounded self-correction** (refine runs at most once — no infinite agent loops). |

## Hardening checklist for real production

This repo is a reference implementation. Before deploying for real:

- [ ] Replace the in-process rate limiter with a shared one (Redis / API gateway) — the bundled one is per-worker.
- [ ] Put the API behind auth (API keys / OAuth) — there is no authN/authZ here by design.
- [ ] Set `ALLOWED_ORIGINS` explicitly; never use `*` with credentials.
- [ ] Send guardrail blocks + 429s to your SIEM; alert on injection-pattern spikes.
- [ ] Pin and scan dependencies (`pip-audit`); enable Dependabot.
- [ ] Add output PII detection beyond regex (NER) if handling regulated data.

## Reporting

This is a portfolio/reference repo. For real disclosures, open a private issue or email the maintainer.
