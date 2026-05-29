from agentic_rag.retriever import HybridRetriever, grounding_score


def test_retrieves_relevant_source():
    r = HybridRetriever.from_dir("data/docs")
    hits = r.retrieve("What is reciprocal rank fusion?")
    assert any(c.source == "rag.md" for c in hits)


def test_hybrid_beats_empty():
    r = HybridRetriever.from_dir("data/docs")
    hits = r.retrieve("faithfulness metric")
    assert hits and any("faithful" in c.text.lower() for c in hits)


def test_grounding_score_bounds():
    assert grounding_score("", ["anything"]) == 1.0  # empty answer -> vacuously grounded
    assert 0.0 <= grounding_score("reciprocal rank fusion", ["reciprocal rank fusion k 60"]) <= 1.0
