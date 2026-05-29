# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation grounds a language model's answer in documents fetched at
query time. Instead of relying only on parametric memory, the system retrieves relevant
passages from a knowledge base and conditions generation on them. This reduces
hallucination and lets the model answer about private or recent data it was never trained on.

## Hybrid retrieval

Hybrid retrieval combines a lexical retriever such as BM25 with a dense embedding
retriever. BM25 ranks documents by exact term overlap and is strong on names, codes, and
rare keywords. Dense retrieval encodes text into vectors and ranks by cosine similarity,
capturing paraphrase and meaning. Combining both recovers results that either method alone
would miss.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) merges several ranked lists without needing to normalize their
scores. Each document receives a score of one divided by a constant k plus its rank in each
list, and the per-list scores are summed. A typical value of k is 60. RRF is robust because
it depends only on ranks, not on the raw, incomparable scores of BM25 versus cosine
similarity.

## Chunking

Chunking splits source documents into passages small enough to retrieve precisely but large
enough to stay coherent. Paragraph-aware chunking that packs paragraphs up to a target word
count preserves meaning better than fixed-character splits.
