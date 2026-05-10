import logging
from typing import List, Dict, Any

from app.services.embeddings import embed_query
from app.services.vector_store import search_similar
from app.utils.constants import MMR_K, MMR_LAMBDA

logger = logging.getLogger(__name__)


def mmr_rerank(
    query_embedding: List[float],
    candidates: List[Dict[str, Any]],
    k: int = MMR_K,
    lambda_mult: float = MMR_LAMBDA
) -> List[Dict[str, Any]]:
    """
    Apply Maximal Marginal Relevance (MMR) to diversify results.
    lambda_mult: 1 = pure relevance, 0 = pure diversity
    """
    import numpy as np

    if not candidates:
        return []

    query_vec = np.array(query_embedding, dtype=np.float32)

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # Use pre-retrieved embeddings from FAISS (avoids wasteful API re-embedding).
    # Fall back to re-embedding only if embeddings are missing.
    if all(c.get("embedding") is not None for c in candidates):
        candidate_embeddings = [c["embedding"] for c in candidates]
    else:
        logger.warning("Candidate embeddings missing — falling back to re-embedding via API.")
        from app.services.embeddings import embed_texts
        contents = [c["content"] for c in candidates]
        candidate_embeddings = embed_texts(contents)

    selected = []
    selected_embeddings = []
    remaining = list(range(len(candidates)))

    for _ in range(min(k, len(candidates))):
        mmr_scores = []
        for i in remaining:
            relevance = cosine_sim(query_vec, candidate_embeddings[i])
            if selected_embeddings:
                redundancy = max(
                    cosine_sim(candidate_embeddings[i], se)
                    for se in selected_embeddings
                )
            else:
                redundancy = 0.0
            score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            mmr_scores.append((i, score))

        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(candidates[best_idx])
        selected_embeddings.append(candidate_embeddings[best_idx])
        remaining.remove(best_idx)

    logger.info(f"MMR selected {len(selected)} chunks.")
    return selected


def retrieve(query: str) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline:
    1. Embed query
    2. Search FAISS for top candidates
    3. Apply MMR reranking
    4. Return top-k diverse chunks
    """
    logger.info(f"Retrieving for query: {query}")

    query_embedding = embed_query(query)

    # Fetch more candidates than needed for MMR to work well
    candidates = search_similar(query_embedding, k=MMR_K * 3)

    if not candidates:
        logger.warning("No candidates found in vector store.")
        return []

    results = mmr_rerank(query_embedding, candidates, k=MMR_K)
    return results