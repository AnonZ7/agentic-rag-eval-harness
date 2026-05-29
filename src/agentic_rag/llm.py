"""Provider-agnostic LLM client.

A tiny protocol (`complete`) keeps the agent decoupled from any vendor. Three real
backends (Anthropic / OpenAI / Gemini) plus a deterministic `FakeLLM` that needs no
network or keys — that is what makes the test + CI eval path runnable offline.

Design note: provider-agnostic routing is a deliberate production pattern — it lets you
swap models for cost/latency without touching the agent graph.
"""
from __future__ import annotations

from typing import Protocol

from .config import settings


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class FakeLLM:
    """Deterministic, offline stand-in.

    For generation it returns an extractive answer built from the retrieved context,
    so faithfulness/grounding metrics are meaningful without calling a real model.
    For planning it returns the question unchanged.
    """

    def complete(self, system: str, user: str) -> str:
        if "PLAN" in system:
            # echo the question as the single retrieval sub-query
            return user.strip().splitlines()[-1][:200]
        # GENERATE / VERIFY: extract the most question-overlapping sentences from context
        ctx = _between(user, "CONTEXT:", "QUESTION:")
        q = _after(user, "QUESTION:")
        if "VERIFY" in system:
            # return a grounding verdict the verify-node can parse
            return "GROUNDED" if ctx.strip() else "UNGROUNDED"
        sents = [s.strip() for s in ctx.replace("\n", " ").split(".") if s.strip()]
        qwords = {w.lower() for w in q.split() if len(w) > 3}
        scored = sorted(
            sents,
            key=lambda s: len(qwords & {w.lower() for w in s.split()}),
            reverse=True,
        )
        top = [s for s in scored[:2] if s]
        return (". ".join(top) + ".") if top else "I don't have enough grounded context to answer."


class AnthropicLLM:
    def __init__(self, model: str):
        import anthropic

        self._c = anthropic.Anthropic()
        self._model = model

    def complete(self, system: str, user: str) -> str:
        r = self._c.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class OpenAILLM:
    def __init__(self, model: str):
        from openai import OpenAI

        self._c = OpenAI()
        self._model = model

    def complete(self, system: str, user: str) -> str:
        r = self._c.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return r.choices[0].message.content or ""


class GeminiLLM:
    def __init__(self, model: str):
        from google import genai

        self._c = genai.Client()
        self._model = model

    def complete(self, system: str, user: str) -> str:
        r = self._c.models.generate_content(
            model=self._model, contents=f"{system}\n\n{user}"
        )
        return r.text or ""


def get_llm() -> LLM:
    p = settings.provider.lower()
    if p == "anthropic":
        return AnthropicLLM(settings.model)
    if p == "openai":
        return OpenAILLM(settings.model)
    if p == "gemini":
        return GeminiLLM(settings.model)
    return FakeLLM()


def _between(text: str, a: str, b: str) -> str:
    try:
        return text.split(a, 1)[1].split(b, 1)[0]
    except IndexError:
        return ""


def _after(text: str, a: str) -> str:
    return text.split(a, 1)[1] if a in text else text
