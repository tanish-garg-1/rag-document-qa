import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def generate_citations(chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Generate citation strings from retrieved chunk metadata.
    Format: filename — Page X — Chunk Y
    """
    citations = []
    seen = set()

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown")
        page = meta.get("page", "N/A")
        chunk_id = meta.get("chunk_id", f"chunk-{i}")

        # Short chunk_id for display (first 8 chars)
        short_id = str(chunk_id)[:8]

        citation = f"{source} — Page {page} — Chunk {short_id}"

        if citation not in seen:
            citations.append(citation)
            seen.add(citation)

    logger.info(f"Generated {len(citations)} citations.")
    return citations


def format_citations_block(citations: List[str]) -> str:
    """Format citations as a readable block to append to response."""
    if not citations:
        return ""
    lines = ["\n\n---\n📚 **Sources:**"]
    for i, citation in enumerate(citations, start=1):
        lines.append(f"{i}. {citation}")
    return "\n".join(lines)