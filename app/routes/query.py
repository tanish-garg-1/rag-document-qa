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
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.retriever import retrieve
from app.services.llm import stream_answer, rewrite_query
from app.services.memory import add_message, get_history_as_text
from app.services.citation import generate_citations, format_citations_block

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


def response_generator(query: str):
    """Generator that rewrites query, streams LLM response then appends citations."""

    # Get chat history
    history = get_history_as_text()

    # Step 1 — Rewrite query using memory for better retrieval
    rewritten_query = rewrite_query(query, history)
    logger.info(f"Original: '{query}' | Rewritten: '{rewritten_query}'")

    # Step 2 — Retrieve relevant chunks using rewritten query
    chunks = retrieve(rewritten_query)

    if not chunks:
        yield "No relevant information found in the uploaded documents."
        return

    # Step 3 — Store user message in memory
    add_message("user", query)

    # Step 4 — Stream LLM response using original question
    full_response = ""
    for token in stream_answer(query, chunks, history):
        full_response += token
        yield token

    # Step 5 — Generate and append citations
    citations = generate_citations(chunks)
    citation_block = format_citations_block(citations)
    if citation_block:
        yield citation_block

    # Step 6 — Store assistant response in memory
    add_message("assistant", full_response)
    logger.info(f"Query processed: {query[:50]}")


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