# Evaluation metrics

Quality is measured, not asserted. Two layers.

## Offline layer (deterministic, runs in CI without keys)

| Metric | Definition | Threshold | Why it matters |
|---|---|---|---|
| `retrieval_precision` | fraction of retrieved chunks whose source is expected | report | noisy retrieval poisons generation |
| `retrieval_recall` | fraction of expected sources that were retrieved | **≥ 0.80** | if the right doc isn't retrieved, the answer can't be right |
| `grounding` | answer content-words that appear in retrieved context | **≥ 0.30** | cheap, fast faithfulness proxy; the guardrail uses the same signal |
| `answer_f1` | token-F1 between answer and ground-truth | **≥ 0.15** | catches answers that drift off the known-good response |

A metric below threshold **exits non-zero**, failing the PR — the eval is a gate, not a report.

## Live layer (LLM-judged, `--ragas` + provider key)

| Metric | Source | What it judges |
|---|---|---|
| `faithfulness` | Ragas | are the answer's claims supported by retrieved context |
| `answer_relevancy` | Ragas | does the answer actually address the question |
| `context_precision` | Ragas | are the retrieved passages relevant |

## Extending the dataset

Add rows to [`dataset.jsonl`](dataset.jsonl):

```json
{"question": "...", "ground_truth": "...", "expected_sources": ["file.md"]}
```

Keep questions answerable from `data/docs/` so retrieval recall stays meaningful. Grow the
set whenever a real failure is found — every production bug becomes a permanent eval case.
