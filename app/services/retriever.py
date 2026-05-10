import re
import logging
from typing import List, Dict, Any

import numpy as np

from app.services.embeddings import embed_query
from app.services.vector_store import search_similar, get_all_chunks
from app.utils.constants import MMR_K, MMR_LAMBDA

_UUID_PREFIX = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_',
    re.IGNORECASE
)

def _clean_source(name: str) -> str:
    return _UUID_PREFIX.sub("", name)

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


def _cosine_sim(a, b) -> float:
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def retrieve(query: str) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline:
    1. Embed query
    2. Search FAISS for top candidates
    3. Apply MMR reranking for relevance + diversity
    4. Coverage pass — guarantee at least 1 chunk per unique source file
    5. Return combined results (deduped)
    """
    logger.info(f"Retrieving for query: {query}")

    query_embedding = embed_query(query)

    # Fetch a wide candidate pool for MMR to work well
    candidates = search_similar(query_embedding, k=max(MMR_K * 5, 30))

    if not candidates:
        logger.warning("No candidates found in vector store.")
        return []

    # ── Step 1: MMR reranking ────────────────────────────────────────────────
    results = mmr_rerank(query_embedding, candidates, k=MMR_K)

    # ── Step 2: Build full chunk pool grouped by source ─────────────────────
    all_chunks = get_all_chunks()

    from collections import defaultdict
    source_pool: Dict[str, List[Dict]] = defaultdict(list)
    for chunk in all_chunks:
        src = _clean_source(chunk["metadata"].get("source", "unknown"))
        source_pool[src].append(chunk)

    # ── Step 3: Summary-guarantee pass ──────────────────────────────────────
    # For EVERY source that has a summary chunk (page=0 / type=summary),
    # ensure that summary chunk is always in the results — even if MMR already
    # picked other chunks from that source.  This guarantees broad "describe"
    # queries always have the document-level description.
    result_chunk_ids = {
        c.get("metadata", {}).get("chunk_id") for c in results
    }
    for src, pool in source_pool.items():
        summary_chunks = [
            c for c in pool
            if c.get("metadata", {}).get("type") == "summary"
            or c.get("metadata", {}).get("page") == 0
        ]
        for sc in summary_chunks:
            if sc.get("metadata", {}).get("chunk_id") not in result_chunk_ids:
                results.append(sc)
                result_chunk_ids.add(sc.get("metadata", {}).get("chunk_id"))
                logger.info(f"Summary-guarantee: added summary chunk for '{src}'")

    # ── Step 4: Coverage pass — sources with zero representation ────────────
    covered = {
        _clean_source(r["metadata"].get("source", ""))
        for r in results
    }

    added = 0
    for src, pool in source_pool.items():
        if src in covered:
            continue
        best = max(
            pool,
            key=lambda c: _cosine_sim(query_embedding, c["embedding"])
            if c.get("embedding") else 0.0
        )
        results.append(best)
        covered.add(src)
        added += 1
        logger.info(f"Coverage pass: added chunk from '{src}'")

    if added:
        logger.info(f"Coverage pass added {added} chunk(s) for under-represented source(s).")

    return results