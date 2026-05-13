import re
import logging
from fastapi import APIRouter
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
    filter_sources: list[str] | None = None

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
    if not session_files:
        return None
    query_lower = query.lower()
    matched = []
    for fname in session_files:
        stem = re.sub(r'\.[^.]+$', '', fname).lower()
        stem_spaced = stem.replace('_', ' ').replace('-', ' ')
        stem_words = [w for w in re.split(r'[_\-\s]+', stem) if len(w) >= 3]
        full_match = stem in query_lower or stem_spaced in query_lower
        words_found = sum(1 for w in stem_words if w in query_lower)
        min_words = max(1, min(2, len(stem_words) - 1))
        partial_match = words_found >= min_words
        if full_match or partial_match:
            matched.append(fname)
    return matched if matched else None


def response_generator(query: str, filter_sources: list[str] | None = None):
    from app.main import set_debug
    from app.services.llm import _clean_source

    history = get_history_as_text()

    # Step 1 — describe-all detection
    is_describe_all = bool(_DESCRIBE_ALL_RE.search(query))
    logger.info(f"Query='{query}' | describe_all={is_describe_all} | filter={filter_sources}")

    # Step 1b — filename pin
    pinned_to = None
    if not is_describe_all:
        filename_match = _detect_filename_mention(query, filter_sources)
        if filename_match:
            filter_sources = filename_match
            pinned_to = filename_match
            logger.info(f"Filename pin -> {filename_match}")

    # Step 2 — rewrite
    if is_describe_all:
        rewritten_query = "overview summary description of every uploaded file and document"
    else:
        rewritten_query = rewrite_query(query, history)
    logger.info(f"Rewritten: '{rewritten_query}'")

    # Step 3 — retrieve
    chunks = retrieve(
        rewritten_query,
        include_all_sources=is_describe_all,
        source_filter=filter_sources if filter_sources else None,
    )
    has_docs = bool(chunks)

    # Build per-file stats for debug panel
    chunk_stats = {}
    sim_stats = {}
    for c in chunks:
        src = _clean_source(c["metadata"].get("source", "?"))
        chunk_stats[src] = chunk_stats.get(src, 0) + 1

    # Step 4 — classify
    mode = classify_query(query, history, has_documents=has_docs)
    logger.info(f"Mode={mode} | chunks={len(chunks)} | sources={list(chunk_stats.keys())}")

    # Save debug info — readable by Streamlit via /debug-last
    set_debug({
        "query": query,
        "rewritten_query": rewritten_query,
        "is_describe_all": is_describe_all,
        "session_filter": filter_sources,
        "pinned_to_file": pinned_to,
        "mode": mode,
        "total_chunks": len(chunks),
        "chunks_per_file": chunk_stats,
        "files_in_response": list(chunk_stats.keys()),
        "files_missing": [
            f for f in (filter_sources or [])
            if _clean_source(f) not in chunk_stats
        ],
    })

    if not has_docs and mode == MODE_DOCUMENT:
        yield "No relevant information found in the uploaded documents."
        return

    add_message("user", query)
    mode_label = _MODE_LABEL.get(mode, mode)
    yield f"[{mode_label} mode]\n\n"

    effective_history = "" if is_describe_all else history
    full_response = ""
    try:
        for token in stream_answer(query, chunks, effective_history, mode=mode,
                                   summaries_first=is_describe_all):
            full_response += token
            yield token
    except GeneratorExit:
        logger.warning(f"Client disconnected: '{query[:50]}'")
        if full_response:
            add_message("assistant", full_response + " [truncated]")
        return

    add_message("assistant", full_response)
    logger.info(f"Done [{mode}]: '{query[:60]}'")


@router.post("/query")
async def query_documents(request: QueryRequest):
    return StreamingResponse(
        response_generator(request.query, filter_sources=request.filter_sources),
        media_type="text/plain"
    )


@router.get("/history")
def get_history():
    from app.services.memory import get_recent_history, get_memory_size
    return {
        "total_messages": get_memory_size(),
        "messages": get_recent_history()
    }

@router.delete("/history")
def clear_history():
    from app.services.memory import clear_memory
    clear_memory()
    return {"status": "success", "message": "Conversation memory cleared."}
