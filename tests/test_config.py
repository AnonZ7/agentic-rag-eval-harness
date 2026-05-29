import importlib

import agentic_rag.config as cfg


def test_provider_allowlist_coerces_unknown(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "evil; rm -rf /")
    importlib.reload(cfg)
    assert cfg.settings.provider == "fake"


def test_provider_allowlist_accepts_known(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    importlib.reload(cfg)
    assert cfg.settings.provider == "anthropic"
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    importlib.reload(cfg)  # restore default for other tests
