import re
import logging
from typing import List, Dict, Any

import numpy as np

from app.services.embeddings import embed_query
from app.services.vector_store import search_similar, get_all_chunks
from app.utils.constants import MMR_K, MMR_LAMBDA, RELEVANCE_RESCUE_K, COVERAGE_MIN_SIMILARITY, SUMMARY_MIN_SIMILARITY

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


def retrieve(query: str, include_all_sources: bool = False,
             source_filter: List[str] | None = None) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline:
    1. Embed query
    2. Search FAISS for top candidates
    3. Apply MMR reranking for relevance + diversity
    4. Coverage / summary-guarantee pass
    5. Max-sources cap (skipped when include_all_sources=True)
    6. Return combined results (deduped)

    include_all_sources: when True (describe-all queries), disables similarity
    thresholds and the source cap so every indexed file appears in context.
    """
    logger.debug(f"\n[RETRIEVER] ---- retrieve() called ----")
    logger.debug(f"[RETRIEVER] query             : '{query}'")
    logger.debug(f"[RETRIEVER] include_all_sources: {include_all_sources}")
    logger.debug(f"[RETRIEVER] source_filter     : {source_filter}")
    logger.info(f"Retrieving for query: {query}")

    query_embedding = embed_query(query)
    logger.debug(f"[RETRIEVER] Embedding done    : dim={len(query_embedding)}")

    # Fetch a wide candidate pool for MMR to work well
    candidates = search_similar(query_embedding, k=max(MMR_K * 5, 30))
    raw_candidate_sources = list({_clean_source(c["metadata"].get("source","?")) for c in candidates})
    logger.debug(f"[RETRIEVER] FAISS candidates  : {len(candidates)} chunks from {len(raw_candidate_sources)} files")
    logger.debug(f"[RETRIEVER] Candidate sources : {raw_candidate_sources}")

    if not candidates:
        logger.warning("No candidates found in vector store.")
        logger.debug(f"[RETRIEVER] !! No candidates in vector store — returning []")
        return []

    # ── Step 0b: Source filter ───────────────────────────────────────────────
    # When the frontend passes a list of filenames (session uploads), restrict
    # the candidate pool to only those sources so the LLM never sees files the
    # user didn't upload in this session.
    if source_filter:
        allowed = set(source_filter)
        before_filter = len(candidates)
        candidates = [
            c for c in candidates
            if _clean_source(c["metadata"].get("source", "")) in allowed
        ]
        logger.debug(f"[RETRIEVER] Source filter     : {before_filter} -> {len(candidates)} candidates")
        logger.debug(f"[RETRIEVER] Allowed set       : {allowed}")
        filtered_sources = list({_clean_source(c["metadata"].get("source","?")) for c in candidates})
        logger.debug(f"[RETRIEVER] After filter srcs : {filtered_sources}")
        logger.info(f"Source filter applied: {len(candidates)} candidates kept from {allowed}")
        if not candidates:
            logger.warning("Source filter removed all candidates — no matching files indexed.")
            logger.debug(f"[RETRIEVER] !! Filter removed ALL candidates — check if files are indexed")
            return []

    # ── Step 1: MMR reranking ────────────────────────────────────────────────
    results = mmr_rerank(query_embedding, candidates, k=MMR_K)
    mmr_sources = list({_clean_source(c["metadata"].get("source","?")) for c in results})
    logger.debug(f"[RETRIEVER] After MMR         : {len(results)} chunks, sources={mmr_sources}")

    # ── Step 1b: Relevance rescue ────────────────────────────────────────────
    # MMR penalises chunks that are similar to already-selected ones, which
    # can drop highly relevant chunks (e.g. a specific exclusion clause) when
    # another chunk from the same document was already picked.
    # Rescue the top-RELEVANCE_RESCUE_K candidates by pure cosine similarity
    # that MMR did not select.
    mmr_chunk_ids = {c.get("metadata", {}).get("chunk_id") for c in results}
    rescue_pool = sorted(
        [c for c in candidates if c.get("metadata", {}).get("chunk_id") not in mmr_chunk_ids],
        key=lambda c: _cosine_sim(query_embedding, c["embedding"]) if c.get("embedding") else 0.0,
        reverse=True
    )
    rescued = 0
    for c in rescue_pool[:RELEVANCE_RESCUE_K]:
        results.append(c)
        mmr_chunk_ids.add(c.get("metadata", {}).get("chunk_id"))
        rescued += 1
        logger.info(
            f"Relevance rescue: added chunk from "
            f"'{_clean_source(c['metadata'].get('source', ''))}'"
        )
    logger.debug(f"[RETRIEVER] Relevance rescue  : +{rescued} chunks → {len(results)} total")

    # ── Step 2: Build full chunk pool grouped by source + best-sim map ─────
    all_chunks = get_all_chunks()
    logger.debug(f"[RETRIEVER] All indexed chunks: {len(all_chunks)} total")

    # Respect source_filter in ALL downstream passes (summary-guarantee,
    # coverage, max-sources).  Without this gate, source_pool contains every
    # indexed file and the summary-guarantee pass pulls in files the user never
    # uploaded this session — even when filter_sources is set.
    allowed_sources = set(source_filter) if source_filter else None

    from collections import defaultdict
    source_pool: Dict[str, List[Dict]] = defaultdict(list)
    skipped_sources = set()
    for chunk in all_chunks:
        src = _clean_source(chunk["metadata"].get("source", "unknown"))
        if allowed_sources and src not in allowed_sources:
            skipped_sources.add(src)
            continue   # ← skip files outside session scope
        source_pool[src].append(chunk)
    logger.debug(f"[RETRIEVER] source_pool files : {list(source_pool.keys())}")
    if skipped_sources:
        logger.debug(f"[RETRIEVER] Skipped (out of scope): {list(skipped_sources)}")

    # Build best cosine similarity per source from the wide candidate pool
    # (candidates come from search_similar, so they already have embeddings)
    source_best_sim: Dict[str, float] = {}
    for c in candidates:
        src = _clean_source(c["metadata"].get("source", "unknown"))
        sim = _cosine_sim(query_embedding, c["embedding"]) if c.get("embedding") else 0.0
        if sim > source_best_sim.get(src, 0.0):
            source_best_sim[src] = sim
    logger.debug(f"[RETRIEVER] Best sim per file :")
    for src, sim in sorted(source_best_sim.items(), key=lambda x: -x[1]):
        logger.debug(f"[RETRIEVER]   {sim:.4f}  {src}")

    # ── Step 3: Summary-guarantee pass ──────────────────────────────────────
    # For sources that MMR + rescue already deemed relevant AND whose best
    # candidate chunk meets the SUMMARY_MIN_SIMILARITY threshold, ensure their
    # summary chunk is in the results.  The similarity gate prevents ML papers
    # or unrelated files from sneaking in via low-scoring MMR diversity picks.
    relevant_sources = {
        _clean_source(c["metadata"].get("source", ""))
        for c in results  # MMR + rescue results only
    }
    logger.debug(f"[RETRIEVER] relevant_sources  : {relevant_sources}")

    result_chunk_ids = {
        c.get("metadata", {}).get("chunk_id") for c in results
    }
    # When source_filter is active the user explicitly chose these files —
    # never drop them by similarity threshold.
    skip_sim_gates = bool(source_filter)
    logger.debug(f"[RETRIEVER] skip_sim_gates    : {skip_sim_gates}")

    summary_added = 0
    for src, pool in source_pool.items():
        if not include_all_sources and not skip_sim_gates:
            if src not in relevant_sources:
                logger.debug(f"[RETRIEVER] Summary-guarantee : SKIP '{src}' — not in MMR results")
                continue  # not in MMR results at all
            if source_best_sim.get(src, 0.0) < SUMMARY_MIN_SIMILARITY:
                logger.debug(f"[RETRIEVER] Summary-guarantee : SKIP '{src}' — sim={source_best_sim.get(src,0):.3f} < {SUMMARY_MIN_SIMILARITY}")
                logger.info(
                    f"Summary-guarantee: skipping '{src}' "
                    f"(best sim {source_best_sim.get(src, 0.0):.3f} < {SUMMARY_MIN_SIMILARITY})"
                )
                continue
        summary_chunks = [
            c for c in pool
            if c.get("metadata", {}).get("type") == "summary"
            or c.get("metadata", {}).get("page") == 0
        ]
        logger.debug(f"[RETRIEVER] Summary-guarantee : '{src}' — {len(summary_chunks)} summary chunk(s) found")
        for sc in summary_chunks:
            if sc.get("metadata", {}).get("chunk_id") not in result_chunk_ids:
                results.append(sc)
                result_chunk_ids.add(sc.get("metadata", {}).get("chunk_id"))
                summary_added += 1
                logger.info(f"Summary-guarantee: added summary chunk for '{src}'")
    logger.debug(f"[RETRIEVER] Summary-guarantee : +{summary_added} chunks added → {len(results)} total")

    # ── Step 4: Coverage pass — sources with zero representation ────────────
    # Only add a chunk for an uncovered source if its best chunk has cosine
    # similarity ≥ COVERAGE_MIN_SIMILARITY.  Sources below the threshold are
    # unrelated to this query and forcing them in pollutes the context.
    covered = {
        _clean_source(r["metadata"].get("source", ""))
        for r in results
    }
    logger.debug(f"[RETRIEVER] Covered after summ: {covered}")

    added = 0
    for src, pool in source_pool.items():
        if src in covered:
            logger.debug(f"[RETRIEVER] Coverage pass     : '{src}' — already covered, skip")
            continue
        best = max(
            pool,
            key=lambda c: _cosine_sim(query_embedding, c["embedding"])
            if c.get("embedding") else 0.0
        )
        best_sim = _cosine_sim(query_embedding, best["embedding"]) if best.get("embedding") else 0.0
        if not include_all_sources and not skip_sim_gates and best_sim < COVERAGE_MIN_SIMILARITY:
            logger.debug(f"[RETRIEVER] Coverage pass     : SKIP '{src}' — sim={best_sim:.3f} < {COVERAGE_MIN_SIMILARITY}")
            logger.info(
                f"Coverage pass: skipping '{src}' (best similarity {best_sim:.3f} < {COVERAGE_MIN_SIMILARITY})"
            )
            continue
        results.append(best)
        covered.add(src)
        added += 1
        logger.debug(f"[RETRIEVER] Coverage pass     : ADDED '{src}' — sim={best_sim:.3f}")
        logger.info(f"Coverage pass: added chunk from '{src}' (similarity {best_sim:.3f})")

    logger.debug(f"[RETRIEVER] Coverage pass     : +{added} chunks → {len(results)} total")
    if added:
        logger.info(f"Coverage pass added {added} chunk(s) for under-represented source(s).")

    # ── Step 5: Max-sources cap ──────────────────────────────────────────────
    # Never send more than MAX_SOURCES distinct files to the LLM regardless of
    # how many passed the threshold above.  Pick the top sources by best
    # candidate similarity so the least-relevant files are dropped first.
    MAX_SOURCES = 3
    source_sims = {
        src: source_best_sim.get(src, 0.0)
        for src in {_clean_source(c["metadata"].get("source", "")) for c in results}
    }
    logger.debug(f"[RETRIEVER] source_sims (final): {source_sims}")
    # Only sources meeting the coverage threshold are eligible for a slot.
    # This prevents low-sim files that happened to appear in the FAISS top-30
    # from claiming a slot just because they scored slightly better than others.
    if skip_sim_gates:
        # User explicitly scoped to these files — keep all of them, no sim filtering
        eligible = source_sims
        logger.debug(f"[RETRIEVER] Max-sources cap   : SKIPPED (skip_sim_gates=True)")
    else:
        eligible = {
            src: sim for src, sim in source_sims.items()
            if sim >= COVERAGE_MIN_SIMILARITY
        }
        # Guarantee at least the single best source is always kept (edge case safety)
        if not eligible and source_sims:
            best_src = max(source_sims, key=source_sims.get)
            eligible = {best_src: source_sims[best_src]}
        logger.debug(f"[RETRIEVER] Eligible sources  : {eligible}")

    if not include_all_sources and not skip_sim_gates and (len(source_sims) > len(eligible) or len(eligible) > MAX_SOURCES):
        top_sources = set(
            sorted(eligible, key=eligible.get, reverse=True)[:MAX_SOURCES]
        )
        before = len(results)
        results = [
            c for c in results
            if _clean_source(c["metadata"].get("source", "")) in top_sources
        ]
        logger.debug(f"[RETRIEVER] Max-sources cap   : kept {top_sources}, dropped {before-len(results)} chunks")
        logger.info(
            f"Max-sources cap: kept {len(top_sources)} sources "
            f"(eligible={len(eligible)}, total={len(source_sims)}), "
            f"dropped {before - len(results)} chunks. Sources: {top_sources}"
        )
    elif include_all_sources:
        logger.debug(f"[RETRIEVER] Max-sources cap   : SKIPPED (include_all_sources=True)")

    final_sources = list({_clean_source(c["metadata"].get("source","?")) for c in results})
    logger.debug(f"[RETRIEVER] FINAL result      : {len(results)} chunks from {len(final_sources)} file(s)")
    logger.debug(f"[RETRIEVER] FINAL sources     : {final_sources}")
    logger.debug(f"[RETRIEVER] ---- retrieve() done ----\n")
    return results