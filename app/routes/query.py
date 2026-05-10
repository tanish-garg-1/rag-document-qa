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

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    query: str

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


def response_generator(query: str):
    """
    Generator that:
    1. Rewrites vague follow-up questions for better retrieval
    2. Classifies the query (DOCUMENT / ANALYTICAL / GENERAL)
    3. Retrieves relevant chunks from FAISS
    4. Streams the LLM response using the appropriate prompt mode
    5. Appends citations and saves to memory
    """
    history = get_history_as_text()

    # Step 1 — Rewrite query for better retrieval
    rewritten_query = rewrite_query(query, history)
    logger.info(f"Original: '{query}' | Rewritten: '{rewritten_query}'")

    # Step 2 — Retrieve relevant chunks
    chunks = retrieve(rewritten_query)
    has_docs = len(chunks) > 0

    # Step 3 — Classify query mode (uses original question for intent, not rewritten)
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
    full_response = ""
    try:
        for token in stream_answer(query, chunks, history, mode=mode):
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
        response_generator(request.query),
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