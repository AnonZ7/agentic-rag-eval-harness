"""Agentic RAG service with a built-in evaluation harness.

Public surface:
    build_agent()      -> compiled LangGraph agent
    answer(question)   -> convenience one-shot call
"""
from .agent import build_agent, answer

__all__ = ["build_agent", "answer"]
__version__ = "0.1.0"
