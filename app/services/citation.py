import re
import logging
from collections import defaultdict, OrderedDict
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

_UUID_PREFIX = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_',
    re.IGNORECASE
)


def _clean_source(name: str) -> str:
    """Strip a UUID prefix from a filename."""
    return _UUID_PREFIX.sub("", name)


def generate_citations(chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Generate citation strings in the SAME ORDER as format_context() groups them
    (by cleaned source filename), so that citation numbers [1], [2], …
    match exactly what the LLM sees in its context window.

    Format: filename — Page X — Chunk Y
    """
    # Mirror format_context() grouping: group by cleaned source, preserve insertion order
    grouped: Dict[str, List[Dict]] = OrderedDict()
    for chunk in chunks:
        raw_source = chunk.get("metadata", {}).get("source", "unknown")
        clean = _clean_source(raw_source)
        grouped.setdefault(clean, []).append(chunk)

    citations = []
    for source, source_chunks in grouped.items():
        for chunk in source_chunks:
            meta = chunk.get("metadata", {})
            page = meta.get("page", "N/A")
            chunk_id = str(meta.get("chunk_id", ""))[:8]
            citations.append(f"{source} — Page {page} — Chunk {chunk_id}")

    logger.info(f"Generated {len(citations)} citations.")
    return citations


def format_citations_block(citations: List[str]) -> str:
    """Format citations as a readable block to append to the streamed response."""
    if not citations:
        return ""
    lines = ["\n\n---\n📚 **Sources:**"]
    for i, citation in enumerate(citations, start=1):
        lines.append(f"{i}. {citation}")
    return "\n".join(lines)