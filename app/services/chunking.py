import uuid
import logging
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.constants import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# Initialize splitter once
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""]
)


def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split documents into chunks with metadata.
    Each chunk gets: source, page, chunk_id, type
    """
    all_chunks = []

    for doc in documents:
        content = doc.get("content", "").strip()
        metadata = doc.get("metadata", {})

        if not content:
            continue

        splits = splitter.split_text(content)

        for i, chunk_text in enumerate(splits):
            chunk = {
                "content": chunk_text,
                "metadata": {
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page", 1),
                    "chunk_id": str(uuid.uuid4()),
                    "type": metadata.get("type", "text")
                }
            }
            all_chunks.append(chunk)

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks