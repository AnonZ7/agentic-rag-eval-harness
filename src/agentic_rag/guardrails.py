"""Input + output guardrails. Cheap deterministic checks that run on every request —
the kind enterprises actually require before an agent touches production.

Input:  length cap, prompt-injection heuristics, basic PII flag.
Output: grounding floor (refuse rather than hallucinate), citation presence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import settings
from .retriever import grounding_score

_INJECTION = re.compile(
    r"("
    r"ignore (all )?(previous|prior|above)|"
    r"disregard (the )?(above|prior|previous|instructions)|"
    r"forget (everything|all|previous|prior)|"
    r"you are now|act as (a|an|if|though)|developer mode|do anything now|\bDAN\b|"
    r"new instructions\s*:|override (the )?(system|instructions|guardrails|rules)|"
    r"system prompt|reveal (your|the) (system )?(prompt|instructions)|"
    r"print (your|the) (system )?(prompt|instructions)|repeat (the|everything) above|"
    r"jailbreak"
    r")",
    re.I,
)
_PII = re.compile(
    r"(\b\d{3}-\d{2}-\d{4}\b|"                       # US SSN
    r"\b(?:\d[ -]?){13,16}\b|"                        # card-ish
    r"[\w.+-]+@[\w-]+\.[\w.-]+)"                      # email
)


@dataclass
class GuardResult:
    ok: bool
    reason: str = ""


def check_input(question: str) -> GuardResult:
    q = question.strip()
    if not q:
        return GuardResult(False, "empty question")
    if len(q) > settings.max_question_chars:
        return GuardResult(False, f"question exceeds {settings.max_question_chars} chars")
    if _INJECTION.search(q):
        return GuardResult(False, "possible prompt-injection pattern")
    return GuardResult(True)


def redact_pii(text: str) -> str:
    return _PII.sub("[REDACTED]", text)


def check_output(answer: str, contexts: list[str]) -> GuardResult:
    if not answer.strip():
        return GuardResult(False, "empty answer")
    score = grounding_score(answer, contexts)
    if score < settings.min_grounding:
        return GuardResult(False, f"low grounding {score:.2f} < {settings.min_grounding}")
    return GuardResult(True, f"grounding {score:.2f}")
