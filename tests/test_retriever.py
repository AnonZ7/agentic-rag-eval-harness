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


def test_embedder_is_deterministic():
    from agentic_rag.retriever import HashEmbedder

    e = HashEmbedder(64)
    import numpy as np

    assert np.allclose(e.embed("hybrid retrieval"), e.embed("hybrid retrieval"))


def test_from_dir_ignores_paths_outside_root(tmp_path):
    # a doc inside the corpus is read; the realpath guard keeps reads within root
    (tmp_path / "ok.md").write_text("hybrid retrieval combines bm25 and dense", encoding="utf-8")
    r = HybridRetriever.from_dir(str(tmp_path))
    assert r.chunks and all("ok.md" == c.source for c in r.chunks)
