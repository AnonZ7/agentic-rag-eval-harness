"""The agent graph (LangGraph).

Flow:  guard_input -> plan -> retrieve -> generate -> verify -> [refine | guard_output | end]

- plan      : turn the question into a focused retrieval query (LLM)
- retrieve  : hybrid BM25 + dense retrieval
- generate  : answer strictly from retrieved context (LLM)
- verify    : self-check grounding; if weak, loop once to refine, else refuse
- guard     : input/output guardrails wrap the whole thing

This plan->act->verify->(refine) loop is the difference between "calls an LLM" and
"designs an agentic system" — verification + bounded self-correction is what keeps
production agents from confidently hallucinating.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .guardrails import check_input, check_output, redact_pii
from .llm import LLM, get_llm
from .retriever import Chunk, HybridRetriever, grounding_score

_PLAN_SYS = "You are a retrieval PLANNER. Rewrite the user question into one concise search query. Output only the query."
_GEN_SYS = (
    "You are a careful assistant. Answer the QUESTION using ONLY the CONTEXT. "
    "If the context is insufficient, say so. Be concise. Cite sources in [brackets]."
)
_VERIFY_SYS = "You VERIFY grounding. Reply GROUNDED if the answer is supported by context, else UNGROUNDED."


class AgentState(TypedDict, total=False):
    question: str
    query: str
    contexts: list[Chunk]
    answer: str
    grounding: float
    refined: bool
    blocked: str
    sources: list[str]


def build_agent(retriever: HybridRetriever | None = None, llm: LLM | None = None):
    retriever = retriever or HybridRetriever.from_dir()
    llm = llm or get_llm()

    def guard_input(state: AgentState) -> AgentState:
        g = check_input(state["question"])
        return {} if g.ok else {"blocked": g.reason, "answer": f"Request blocked: {g.reason}"}

    def plan(state: AgentState) -> AgentState:
        q = llm.complete(_PLAN_SYS + " PLAN", f"Question:\n{state['question']}")
        return {"query": (q or state["question"]).strip()}

    def retrieve(state: AgentState) -> AgentState:
        ctx = retriever.retrieve(state.get("query") or state["question"])
        return {"contexts": ctx, "sources": sorted({c.source for c in ctx})}

    def generate(state: AgentState) -> AgentState:
        ctx_txt = "\n".join(f"[{c.source}] {c.text}" for c in state["contexts"])
        user = f"CONTEXT:\n{ctx_txt}\n\nQUESTION: {state['question']}"
        ans = redact_pii(llm.complete(_GEN_SYS, user))
        score = grounding_score(ans, [c.text for c in state["contexts"]])
        return {"answer": ans, "grounding": score}

    def verify(state: AgentState) -> AgentState:
        # bounded self-correction: one refine attempt if grounding is weak
        if state.get("grounding", 0) >= 0.30 or state.get("refined"):
            return {}
        return {"refined": True}

    def refine(state: AgentState) -> AgentState:
        # widen retrieval and regenerate once
        ctx = retriever.retrieve(state["question"], top_k=8)
        ctx_txt = "\n".join(f"[{c.source}] {c.text}" for c in ctx)
        user = f"CONTEXT:\n{ctx_txt}\n\nQUESTION: {state['question']}"
        ans = redact_pii(llm.complete(_GEN_SYS, user))
        return {
            "contexts": ctx,
            "sources": sorted({c.source for c in ctx}),
            "answer": ans,
            "grounding": grounding_score(ans, [c.text for c in ctx]),
        }

    def guard_output(state: AgentState) -> AgentState:
        g = check_output(state.get("answer", ""), [c.text for c in state.get("contexts", [])])
        if not g.ok:
            return {
                "blocked": g.reason,
                "answer": "I don't have enough grounded context to answer that confidently.",
            }
        return {}

    def route_after_guard(state: AgentState) -> str:
        return END if state.get("blocked") else "plan"

    def route_after_verify(state: AgentState) -> str:
        return "refine" if state.get("refined") and state.get("grounding", 0) < 0.30 else "guard_output"

    g = StateGraph(AgentState)
    for name, fn in [
        ("guard_input", guard_input), ("plan", plan), ("retrieve", retrieve),
        ("generate", generate), ("verify", verify), ("refine", refine),
        ("guard_output", guard_output),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", route_after_guard, {"plan": "plan", END: END})
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"refine": "refine", "guard_output": "guard_output"})
    g.add_edge("refine", "guard_output")
    g.add_edge("guard_output", END)
    return g.compile()


_AGENT = None


def answer(question: str) -> dict:
    """One-shot convenience wrapper. Returns {answer, sources, grounding, blocked}."""
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    out = _AGENT.invoke({"question": question})
    return {
        "answer": out.get("answer", ""),
        "sources": out.get("sources", []),
        "grounding": round(out.get("grounding", 0.0), 3),
        "blocked": out.get("blocked", ""),
    }
