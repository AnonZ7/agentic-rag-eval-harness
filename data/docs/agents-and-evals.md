# Agentic patterns and evaluation

## Plan, act, verify

An agentic system does more than call a model once. A common loop is plan, act, then
verify. The agent first plans by turning a user request into a focused sub-task or query,
then acts by retrieving context and generating an answer, then verifies that the answer is
supported by evidence. Bounded self-correction means the agent may refine once if
verification fails, but does not loop forever.

## Guardrails

Guardrails are deterministic checks that wrap an agent. Input guardrails cap request length,
detect prompt-injection patterns (for example, instructions telling the model to disregard
its earlier directions), and flag personally identifiable information. Output guardrails
enforce a grounding floor so the agent refuses rather than hallucinates, and confirm that
sources are cited.

## Faithfulness

Faithfulness measures whether the claims in an answer are supported by the retrieved
context. An answer can be fluent and relevant yet unfaithful if it adds facts not present in
the sources. Faithfulness is the single most important metric for trustworthy RAG.

## Answer relevance and context precision

Answer relevance measures how well the answer addresses the question. Context precision
measures how many of the retrieved passages are actually relevant, and context recall
measures how much of the needed information was retrieved. Together these four metrics —
faithfulness, answer relevance, context precision, and context recall — form the core of a
RAG evaluation suite such as Ragas.

## Why evaluate

Evaluation turns agent quality from an opinion into a number. A reference dataset of
questions with known answers lets you measure regressions in continuous integration before
they reach production, the same way unit tests guard ordinary code.
