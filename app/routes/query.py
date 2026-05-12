# import logging
# from fastapi import APIRouter
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel

# from app.services.retriever import retrieve
# from app.services.llm import stream_answer
# from app.services.memory import add_message, get_history_as_text
# from app.services.citation import generate_citations, format_citations_block

# logger = logging.getLogger(__name__)

# router = APIRouter()


# class QueryRequest(BaseModel):
#     query: str


# def response_generator(query: str):
#     """Generator that streams LLM response then appends citations."""

#     # Retrieve relevant chunks
#     chunks = retrieve(query)

#     if not chunks:
#         yield "No relevant information found in the uploaded documents."
#         return

#     # Get chat history
#     history = get_history_as_text()

#     # Store user message in memory
#     add_message("user", query)

#     # Collect full response for memory storage
#     full_response = ""

#     # Stream LLM response
#     for token in stream_answer(query, chunks, history):
#         full_response += token
#         yield token

#     # Generate and append citations
#     # Generate and append citations
#     citations = generate_citations(chunks)
#     citation_block = format_citations_block(citations)
#     if citation_block:
#         yield citation_block

#     # Store assistant response in memory (without duplicate citation block)
#     add_message("assistant", full_response)
#     logger.info(f"Query processed: {query[:50]}")


# @router.post("/query")
# async def query_documents(request: QueryRequest):
#     """
#     Accept a query, retrieve relevant chunks, stream LLM response with citations.
#     """
#     return StreamingResponse(
#         response_generator(request.query),
#         media_type="text/plain"
#     )


import re
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from app.services.retriever import retrieve
from app.services.llm import (
    stream_answer, rewrite_query, classify_query,
    MODE_DOCUMENT, MODE_ANALYTICAL, MODE_GENERAL
)
from app.services.memory import add_message, get_history_as_text
from app.services.citation import generate_citations, format_citations_block

# Queries that want a description/overview of ALL uploaded files
_DESCRIBE_ALL_RE = re.compile(
    r'\b(describe|explain|summarize|summarise|overview|what are|what is|'
    r'tell me about|list|show|walk\s+\w*\s*through|give\s+\w*\s*(overview|summary|me\s+a\s+summary))\b'
    r'.{0,60}\b(all|these|both|my|the|uploaded|each|every)?\s*(files?|docs?|documents?|pdfs?|uploads?)\b'
    r'|what (have i uploaded|did i upload|files? (are|have been|were) (indexed|uploaded|available))'
    r'|explain (the |my |these |those |all )?(files?|docs?|documents?|uploads?|pdfs?)'
    r'|^(all (the |my )?(files?|docs?|documents?|uploads?))$'
    r'|(all|every|each) (the |my )?(files?|docs?|documents?|uploads?|pdfs?)',
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    filter_sources: list[str] | None = None   # filenames to restrict retrieval to

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query must not be empty.")
        return v.strip()


_MODE_LABEL = {
    MODE_DOCUMENT:   "📄 Document",
    MODE_ANALYTICAL: "🔍 Analytical",
    MODE_GENERAL:    "🌐 General Knowledge",
}


def _detect_filename_mention(query: str, session_files: list[str] | None) -> list[str] | None:
    """
    If the user's query explicitly names one or more files from the session,
    return only those files as the effective source filter.
    This prevents semantic retrieval from returning the wrong file when the
    user asks about a specific file by name (e.g. 'explain improved_wheat_logo').
    """
    if not session_files:
        return None
    query_lower = query.lower()
    matched = []
    for fname in session_files:
        # Strip extension and split into individual words
        stem = re.sub(r'\.[^.]+$', '', fname).lower()
        # Full stem match (e.g. "improved_wheat_logo")
        stem_spaced = stem.replace('_', ' ').replace('-', ' ')
        # Individual meaningful words from the filename (len >= 3 to skip short noise)
        stem_words = [w for w in re.split(r'[_\-\s]+', stem) if len(w) >= 3]

        full_match = stem in query_lower or stem_spaced in query_lower
        # Partial: at least 2 stem words appear in the query (or 1 if stem has only 1 word)
        words_found = sum(1 for w in stem_words if w in query_lower)
        min_words = max(1, min(2, len(stem_words) - 1))
        partial_match = words_found >= min_words

        if full_match or partial_match:
            matched.append(fname)
    return matched if matched else None


def response_generator(query: str, filter_sources: list[str] | None = None):
    """
    Generator that:
    1. Detects if query names a specific file → pin retrieval to that file
    2. Rewrites vague follow-up questions for better retrieval
    3. Classifies the query (DOCUMENT / ANALYTICAL / GENERAL)
    4. Retrieves relevant chunks — scoped to filter_sources when provided
    5. Streams the LLM response using the appropriate prompt mode
    6. Saves to memory
    """
    logger.debug(f"\n{'='*60}")
    logger.debug(f"[QUERY] Incoming query : '{query}'")
    logger.debug(f"[QUERY] filter_sources : {filter_sources}")
    logger.debug(f"{'='*60}")

    history = get_history_as_text()
    logger.debug(f"[QUERY] History length : {len(history)} chars")

    # Step 1 — Detect describe-all intent before rewriting
    is_describe_all = bool(_DESCRIBE_ALL_RE.search(query))
    logger.debug(f"[QUERY] is_describe_all : {is_describe_all}")
    if is_describe_all:
        logger.info("Describe-all query detected — summaries-first, no source cap")

    # Step 1b — Detect explicit filename mention → narrow scope to that file
    # e.g. "explain improved_wheat_logo image" → only retrieve from that file
    if not is_describe_all:
        filename_match = _detect_filename_mention(query, filter_sources)
        logger.debug(f"[QUERY] filename_match  : {filename_match}")
        if filename_match:
            logger.info(f"Filename mention detected — pinning scope to: {filename_match}")
            filter_sources = filename_match
            logger.debug(f"[QUERY] Scope pinned to : {filter_sources}")
    else:
        logger.debug(f"[QUERY] Filename detection skipped (describe-all mode)")

    # Step 2 — Rewrite query for better retrieval
    # For describe-all, use a neutral query so no file is unfairly deprioritised
    if is_describe_all:
        rewritten_query = "overview summary description of every uploaded file and document"
        logger.debug(f"[QUERY] Rewrite skipped  : using neutral describe-all query")
    else:
        rewritten_query = rewrite_query(query, history)
    logger.debug(f"[QUERY] Original query  : '{query}'")
    logger.debug(f"[QUERY] Rewritten query : '{rewritten_query}'")
    if filter_sources:
        logger.debug(f"[QUERY] Active filter   : {filter_sources}")
        logger.info(f"Source filter active: {filter_sources}")

    # Step 3 — Retrieve relevant chunks (scoped to session files when provided)
    logger.debug(f"\n[QUERY] >>> Calling retriever ...")
    chunks = retrieve(
        rewritten_query,
        include_all_sources=is_describe_all,
        source_filter=filter_sources if filter_sources else None,
    )
    has_docs = len(chunks) > 0
    retrieved_sources = list({c["metadata"].get("source", "?") for c in chunks})
    logger.debug(f"[QUERY] Chunks returned : {len(chunks)}")
    logger.debug(f"[QUERY] Sources in ctx  : {retrieved_sources}")
    logger.debug(f"[QUERY] has_docs        : {has_docs}")

    # Step 4 — Classify query mode (uses original question for intent, not rewritten)
    logger.debug(f"\n[QUERY] >>> Classifying query mode ...")
    mode = classify_query(query, history, has_documents=has_docs)
    logger.debug(f"[QUERY] Mode selected   : {mode}")
    logger.info(f"Query mode: {mode}")

    if not has_docs and mode == MODE_DOCUMENT:
        logger.debug(f"[QUERY] No docs + DOCUMENT mode -> returning empty response")
        yield "No relevant information found in the uploaded documents."
        return

    # Step 4 — Store user message
    add_message("user", query)

    # Step 5 — Emit the mode indicator so the user knows which mode was used
    mode_label = _MODE_LABEL.get(mode, mode)
    yield f"[{mode_label} mode]\n\n"

    # Step 6 — Stream LLM response
    # For describe-all queries, suppress history — the LLM would otherwise
    # refuse to re-describe files it described in a previous turn.
    effective_history = "" if is_describe_all else history
    logger.debug(f"\n[QUERY] >>> Streaming LLM answer (mode={mode}, summaries_first={is_describe_all}) ...")
    full_response = ""
    try:
        for token in stream_answer(query, chunks, effective_history, mode=mode,
                                    summaries_first=is_describe_all):
            full_response += token
            yield token
    except GeneratorExit:
        logger.warning(f"Client disconnected mid-stream for query: '{query[:50]}'")
        if full_response:
            add_message("assistant", full_response + " [truncated]")
        return

    # Citations suppressed — chunk-level source metadata not shown to user

    # Step 8 — Persist to memory
    add_message("assistant", full_response)
    logger.debug(f"[QUERY] Response length : {len(full_response)} chars")
    logger.debug(f"[QUERY] Query complete  : [{mode}] '{query[:60]}'")
    logger.debug(f"{'='*60}\n")
    logger.info(f"Query processed [{mode}]: {query[:50]}")


@router.post("/query")
async def query_documents(request: QueryRequest):
    """
    Accept a query, rewrite it using memory, retrieve chunks,
    stream LLM response with citations.
    """
    return StreamingResponse(
        response_generator(request.query, filter_sources=request.filter_sources),
        media_type="text/plain"
    )


#. /history endpoint
@router.get("/history")
def get_history():
    """Return current conversation memory as JSON."""
    from app.services.memory import get_recent_history, get_memory_size
    return {
        "total_messages": get_memory_size(),
        "messages": get_recent_history()
    }

@router.delete("/history")
def clear_history():
    """Clear conversation memory."""
    from app.services.memory import clear_memory
    clear_memory()
    return {"status": "success", "message": "Conversation memory cleared."}