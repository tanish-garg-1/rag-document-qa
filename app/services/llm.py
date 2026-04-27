import logging
from typing import List, Dict, Generator

from groq import Groq

from app.utils.constants import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


def build_prompt(context: str, history: str, question: str) -> str:
    """Build the full prompt for the LLM."""
    return f"""Answer the question using ONLY the provided context.
Always include sources in your answer.

Context:
{context}

Chat History:
{history}

Question:
{question}"""


def format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a single context string."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("metadata", {}).get("source", "unknown")
        page = chunk.get("metadata", {}).get("page", "N/A")
        content = chunk.get("content", "")
        parts.append(f"[{i}] (Source: {source}, Page: {page})\n{content}")
    return "\n\n".join(parts)


def stream_answer(
    question: str,
    chunks: List[Dict],
    history: str
) -> Generator[str, None, None]:
    """
    Stream answer tokens from Groq LLM.
    Yields text chunks as they arrive.
    """
    context = format_context(chunks)
    prompt = build_prompt(context, history, question)

    try:
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.2,
            max_tokens=1024
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

        logger.info("Streaming complete.")

    except Exception as e:
        logger.error(f"Groq LLM error: {e}")
        yield f"Error generating response: {str(e)}"


# query rewriting for better retrieval
def rewrite_query(question: str, history: str) -> str:
    """
    Rewrite a vague follow-up question into a standalone search query
    using the chat history for context.
    """
    if not history.strip():
        return question  # No history, no rewriting needed

    prompt = f"""Given the chat history below, rewrite the user's latest question into a 
complete, standalone search query that captures the full context and intent.
Return ONLY the rewritten query, nothing else. No explanation, no quotes.

Chat History:
{history}

Latest Question: {question}

Rewritten Query:"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0.0,
            max_tokens=150
        )
        rewritten = response.choices[0].message.content.strip()
        logger.info(f"Query rewritten: '{question}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.error(f"Query rewriting failed: {e}")
        return question  # Fall back to original question