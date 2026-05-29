"""Evaluation harness.

Two modes, chosen automatically:

  OFFLINE (default, no keys)  — deterministic metrics computed locally:
      * retrieval_precision@k / recall@k vs expected_sources
      * grounding (answer content-word overlap with retrieved context)
      * answer_overlap vs ground_truth (token F1)
    Hermetic, fast, CI-safe. Fails the build if a metric drops below threshold.

  LIVE (LLM_PROVIDER set + `pip install .[evals]`) — adds Ragas LLM-judged metrics:
      * faithfulness, answer_relevancy, context_precision

Run:  python -m evals.run_evals        (offline)
      LLM_PROVIDER=anthropic python -m evals.run_evals --ragas
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from agentic_rag.agent import build_agent
from agentic_rag.retriever import HybridRetriever, grounding_score

DATASET = Path(__file__).parent / "dataset.jsonl"
THRESHOLDS = {"retrieval_recall": 0.80, "grounding": 0.30, "answer_f1": 0.15}


def _tok(s: str) -> set[str]:
    import re

    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 3}


def token_f1(pred: str, gold: str) -> float:
    p, g = _tok(pred), _tok(gold)
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if not tp:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def load() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_offline() -> dict:
    retriever = HybridRetriever.from_dir()
    agent = build_agent(retriever=retriever)
    rows = load()
    prec_list, rec_list, ground_list, f1_list = [], [], [], []

    print(f"\n  Running OFFLINE eval on {len(rows)} cases (provider={os.getenv('LLM_PROVIDER', 'fake')})\n")
    for r in rows:
        hits = retriever.retrieve(r["question"])
        got = {c.source for c in hits}
        want = set(r["expected_sources"])
        prec = len(got & want) / len(got) if got else 0.0
        rec = len(got & want) / len(want) if want else 1.0

        out = agent.invoke({"question": r["question"]})
        ans = out.get("answer", "")
        ground = grounding_score(ans, [c.text for c in hits])
        f1 = token_f1(ans, r["ground_truth"])

        prec_list.append(prec)
        rec_list.append(rec)
        ground_list.append(ground)
        f1_list.append(f1)
        print(f"  - {r['question'][:54]:54}  P={prec:.2f} R={rec:.2f} grd={ground:.2f} f1={f1:.2f}")

    agg = {
        "retrieval_precision": round(sum(prec_list) / len(prec_list), 3),
        "retrieval_recall": round(sum(rec_list) / len(rec_list), 3),
        "grounding": round(sum(ground_list) / len(ground_list), 3),
        "answer_f1": round(sum(f1_list) / len(f1_list), 3),
    }
    return agg


def run_ragas() -> dict:
    """LLM-judged metrics. Requires LLM_PROVIDER + `pip install .[evals]`."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    retriever = HybridRetriever.from_dir()
    agent = build_agent(retriever=retriever)
    rows = load()
    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for r in rows:
        hits = retriever.retrieve(r["question"])
        out = agent.invoke({"question": r["question"]})
        records["question"].append(r["question"])
        records["answer"].append(out.get("answer", ""))
        records["contexts"].append([c.text for c in hits])
        records["ground_truth"].append(r["ground_truth"])
    result = evaluate(Dataset.from_dict(records), metrics=[faithfulness, answer_relevancy, context_precision])
    return {k: round(float(v), 3) for k, v in result.items()}


def main() -> int:
    use_ragas = "--ragas" in sys.argv
    agg = run_offline()
    print("\n  OFFLINE AGGREGATE:", json.dumps(agg, indent=2))

    failures = [m for m, thr in THRESHOLDS.items() if agg.get(m, 0) < thr]
    if use_ragas:
        try:
            ragas = run_ragas()
            print("\n  RAGAS (LLM-judged):", json.dumps(ragas, indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"\n  [ragas skipped] {e}")

    if failures:
        print(f"\n  FAIL - below threshold: {failures}")
        return 1
    print("\n  PASS - all offline metrics above threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
