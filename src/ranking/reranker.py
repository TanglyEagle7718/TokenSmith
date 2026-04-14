"""
reranker.py

This module supports re-ranking strategies applied before the generative LLM call.
"""

from typing import Dict, List, Any

# -------------------------- Cross-Encoder Cache --------------------------
_CROSS_ENCODER_CACHE: Dict[str, Any] = {}

def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
    """
    Fetch the cached cross-encoder model to prevent reloading on every query.
    """
    from sentence_transformers import CrossEncoder
    if model_name not in _CROSS_ENCODER_CACHE:
        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_CACHE[model_name]


# -------------------------- Reranking Strategies -------------------------
def rerank_with_cross_encoder(query: str, chunks: List[str], top_n: int) -> List[str]:
    """
    Reranks a list of documents using the cross-encoder model.
    """
    if not chunks:
        return []

    model = get_cross_encoder()

    # Create pairs of [query, chunk] for the model
    pairs = [(query, chunk) for chunk in chunks]

    # Predict the scores
    scores = model.predict(pairs, show_progress_bar=False)

    # Combine chunks with their scores and sort
    chunk_with_scores = list(zip(chunks, scores))
    chunk_with_scores.sort(key=lambda x: x[1], reverse=True)

    #allow neg scores
    return [chunk for chunk, score in chunk_with_scores[:top_n]]


# -------------------------- Reranking Router -----------------------------
def rerank(query: str, chunks: List[str], mode: str, top_n: int) -> List[str]:
    """
    Routes to the appropriate reranker based on the mode in the config.
    """
    if mode == "cross_encoder":
        return rerank_with_cross_encoder(query, chunks, top_n)

    # We can add other re-ranking strategies to switch between them.
    return chunks