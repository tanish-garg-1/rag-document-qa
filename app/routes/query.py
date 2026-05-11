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
    r'tell me about|list|show|walk\s+\w*\s*through|give\s+\w*\s*(overview|summary))\b'
    r'.{0,60}\b(all|these|both|my|the|uploaded|each|every)?\s*(files?|docs?|documents?|pdfs?|uploads?)\b'
    r'|what (have i uploaded|did i upload|files? (are|have been|were) (indexed|uploaded|available))'
    r'|explain (the |my |these |those |all )?(files?|docs?|documents?|uploads?|pdfs?)',
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


def response_generator(query: str, filter_sources: list[str] | None = None):
    """
    Generator that:
    1. Rewrites vague follow-up questions for better retrieval
    2. Classifies the query (DOCUMENT / ANALYTICAL / GENERAL)
    3. Retrieves relevant chunks — scoped to filter_sources when provided
    4. Streams the LLM response using the appropriate prompt mode
    5. Saves to memory
    """
    history = get_history_as_text()

    # Step 1 — Detect describe-all intent before rewriting
    is_describe_all = bool(_DESCRIBE_ALL_RE.search(query))
    if is_describe_all:
        logger.info("Describe-all query detected — summaries-first, no source cap")

    # Step 2 — Rewrite query for better retrieval
    # For describe-all, use a neutral query so no file is unfairly deprioritised
    if is_describe_all:
        rewritten_query = "overview summary description of every uploaded file and document"
    else:
        rewritten_query = rewrite_query(query, history)
    logger.info(f"Original: '{query}' | Rewritten: '{rewritten_query}'")
    if filter_sources:
        logger.info(f"Source filter active: {filter_sources}")

    # Step 3 — Retrieve relevant chunks (scoped to session files when provided)
    chunks = retrieve(
        rewritten_query,
        include_all_sources=is_describe_all,
        source_filter=filter_sources if filter_sources else None,
    )
    has_docs = len(chunks) > 0

    # Step 4 — Classify query mode (uses original question for intent, not rewritten)
    mode = classify_query(query, history, has_documents=has_docs)
    logger.info(f"Query mode: {mode}")

    if not has_docs and mode == MODE_DOCUMENT:
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