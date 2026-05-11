import os
import json
import hashlib
import logging
import numpy as np
import faiss
from typing import List, Dict, Any

from app.utils.constants import FAISS_INDEX_DIR, EMBEDDING_DIM

logger = logging.getLogger(__name__)

INDEX_FILE    = os.path.join(FAISS_INDEX_DIR, "index.faiss")
METADATA_FILE = os.path.join(FAISS_INDEX_DIR, "metadata.json")
HASHES_FILE   = os.path.join(FAISS_INDEX_DIR, "hashes.json")


def _load_or_create_index() -> faiss.IndexFlatIP:
    """Load existing FAISS index or create a new one."""
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        logger.info("FAISS index loaded from disk.")
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        logger.info("New FAISS index created.")
    return index


def _load_metadata() -> List[Dict[str, Any]]:
    """Load metadata from disk."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return []


def _load_hashes() -> set:
    """Load the set of already-indexed content hashes."""
    if os.path.exists(HASHES_FILE):
        with open(HASHES_FILE, "r") as f:
            return set(json.load(f))
    return set()


def _save_hashes(hashes: set):
    """Atomically save content hashes to disk."""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    tmp_path = HASHES_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(list(hashes), f)
    os.replace(tmp_path, HASHES_FILE)


def compute_file_hash(file_bytes: bytes) -> str:
    """Return SHA-256 hex digest of raw file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def is_content_indexed(file_hash: str) -> bool:
    """Return True if this file's content is already in the index."""
    return file_hash in _load_hashes()


def mark_content_indexed(file_hash: str):
    """Record a hash so the same content won't be indexed twice."""
    hashes = _load_hashes()
    hashes.add(file_hash)
    _save_hashes(hashes)


def _save_index(index: faiss.IndexFlatIP):
    """Save FAISS index to disk using atomic write (temp file + rename)."""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    tmp_path = INDEX_FILE + ".tmp"
    faiss.write_index(index, tmp_path)
    os.replace(tmp_path, INDEX_FILE)   # atomic on POSIX; best-effort on Windows
    logger.info("FAISS index saved.")


def _save_metadata(metadata: List[Dict[str, Any]]):
    """Save metadata to disk using atomic write (temp file + rename)."""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    tmp_path = METADATA_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(metadata, f)
    os.replace(tmp_path, METADATA_FILE)  # atomic on POSIX; best-effort on Windows
    logger.info("Metadata saved.")


def add_chunks_to_store(chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
    """Add chunks and their embeddings to FAISS index."""
    index = _load_or_create_index()
    metadata = _load_metadata()

    vectors = np.array(embeddings, dtype=np.float32)
    index.add(vectors)

    for chunk in chunks:
        metadata.append({
            "content": chunk["content"],
            "metadata": chunk["metadata"]
        })

    _save_index(index)
    _save_metadata(metadata)
    logger.info(f"Added {len(chunks)} chunks to vector store.")


def search_similar(query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
    """Search FAISS index for top-k similar chunks."""
    index = _load_or_create_index()
    metadata = _load_metadata()

    if index.ntotal == 0:
        logger.warning("FAISS index is empty.")
        return []

    query_vec = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_vec, min(k, index.ntotal))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        entry = metadata[idx]
        # Reconstruct the stored embedding vector so MMR can reuse it
        # without making extra API calls
        try:
            embedding = index.reconstruct(int(idx)).tolist()
        except Exception:
            embedding = None
        results.append({
            "content": entry["content"],
            "metadata": entry["metadata"],
            "score": float(dist),
            "embedding": embedding,
        })

    return results


def get_all_chunks() -> List[Dict[str, Any]]:
    """
    Return every chunk in the index with its embedding.
    Used to guarantee per-source coverage during retrieval.
    """
    index = _load_or_create_index()
    metadata = _load_metadata()

    if index.ntotal == 0:
        return []

    results = []
    for idx, entry in enumerate(metadata):
        try:
            embedding = index.reconstruct(idx).tolist()
        except Exception:
            embedding = None
        results.append({
            "content": entry["content"],
            "metadata": entry["metadata"],
            "score": 0.0,
            "embedding": embedding,
        })
    return results


def get_unique_sources() -> List[str]:
    """Return sorted list of unique source filenames in the index."""
    import re
    _UUID_PREFIX = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_',
        re.IGNORECASE
    )
    metadata = _load_metadata()
    seen = set()
    sources = []
    for entry in metadata:
        raw = entry.get("metadata", {}).get("source", "unknown")
        clean = _UUID_PREFIX.sub("", raw)
        if clean not in seen:
            seen.add(clean)
            sources.append(clean)
    return sorted(sources)


def get_index_stats() -> Dict[str, Any]:
    """Return basic stats about the FAISS index."""
    index = _load_or_create_index()
    metadata = _load_metadata()
    return {
        "total_vectors": index.ntotal,
        "total_chunks": len(metadata),
        "embedding_dim": EMBEDDING_DIM
    }


def get_indexed_sources() -> List[str]:
    """Return sorted list of unique source filenames currently in the index."""
    import re
    _UUID_PREFIX = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_',
        re.IGNORECASE
    )
    metadata = _load_metadata()
    seen = set()
    sources = []
    for m in metadata:
        raw = m.get("source", "")
        clean = _UUID_PREFIX.sub("", raw)
        if clean and clean not in seen:
            seen.add(clean)
            sources.append(clean)
    return sorted(sources)