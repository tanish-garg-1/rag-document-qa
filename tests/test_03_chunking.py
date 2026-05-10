"""
TEST 03: TEXT CHUNKING
Tests the RecursiveCharacterTextSplitter chunking logic, chunk sizes,
metadata assignment, and edge cases.
Run: python tests/test_03_chunking.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("TEST 03: TEXT CHUNKING")
print("=" * 60)

# --- 1. Import ---
print("\n--- Import chunking ---")
try:
    from app.services.chunking import chunk_documents
    print("  [PASS] chunking module imported")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

from app.utils.constants import CHUNK_SIZE, CHUNK_OVERLAP
print(f"  [INFO] CHUNK_SIZE    = {CHUNK_SIZE}")
print(f"  [INFO] CHUNK_OVERLAP = {CHUNK_OVERLAP}")

# --- 2. Basic chunking test ---
print("\n--- Basic Chunking Test ---")
short_text = "This is a short sentence."
docs_short = [{"content": short_text, "metadata": {"source": "test.txt", "page": 1, "type": "text"}}]
try:
    chunks = chunk_documents(docs_short)
    print(f"  Input text length: {len(short_text)} chars")
    print(f"  Chunks produced: {len(chunks)}")
    if len(chunks) >= 1:
        print(f"  [PASS] At least 1 chunk produced from short text")
        print(f"  Chunk 0 content: '{chunks[0]['content'][:80]}'")
        print(f"  Chunk 0 metadata: {chunks[0]['metadata']}")
    else:
        print(f"  [FAIL] No chunks produced from non-empty text")
except Exception as e:
    print(f"  [FAIL] chunk_documents error: {e}")
    import traceback
    traceback.print_exc()

# --- 3. Long text chunking ---
print("\n--- Long Text Chunking ---")
long_text = " ".join([f"Sentence number {i} with some extra words to fill space." for i in range(100)])
docs_long = [{"content": long_text, "metadata": {"source": "long_doc.txt", "page": 1, "type": "text"}}]
try:
    chunks = chunk_documents(docs_long)
    print(f"  Input text length: {len(long_text)} chars")
    print(f"  Chunks produced: {len(chunks)}")
    expected_min_chunks = len(long_text) // CHUNK_SIZE
    print(f"  Expected min chunks (~{expected_min_chunks})")

    for i, chunk in enumerate(chunks):
        clen = len(chunk["content"])
        print(f"    Chunk {i}: {clen} chars  — '{chunk['content'][:40]}...'")

    # Check chunk sizes
    oversized = [c for c in chunks if len(c["content"]) > CHUNK_SIZE * 1.2]
    if oversized:
        print(f"  [WARN] {len(oversized)} chunks exceed expected size by >20%")
    else:
        print(f"  [PASS] All chunks within expected size range")

    # Check metadata keys
    required_keys = {"source", "page", "chunk_id", "type"}
    for i, chunk in enumerate(chunks[:3]):
        meta = chunk.get("metadata", {})
        missing = required_keys - set(meta.keys())
        if missing:
            print(f"  [FAIL] Chunk {i} missing metadata keys: {missing}")
        else:
            print(f"  [PASS] Chunk {i} has all required metadata: {set(meta.keys())}")

except Exception as e:
    print(f"  [FAIL] Long text chunking error: {e}")
    import traceback
    traceback.print_exc()

# --- 4. Empty content test ---
print("\n--- Empty Content Test ---")
docs_empty = [{"content": "", "metadata": {"source": "empty.txt", "page": 1, "type": "text"}}]
try:
    chunks = chunk_documents(docs_empty)
    print(f"  Chunks from empty content: {len(chunks)}")
    if len(chunks) == 0:
        print("  [PASS] Empty content produces 0 chunks (expected behavior)")
    else:
        print(f"  [WARN] {len(chunks)} chunks from empty content — may cause issues downstream")
        for c in chunks:
            print(f"         Content: '{c['content']}'")
except Exception as e:
    print(f"  [FAIL] Empty content error: {e}")

# --- 5. Multiple documents ---
print("\n--- Multiple Documents ---")
multi_docs = [
    {"content": "Document one content. " * 20, "metadata": {"source": "doc1.txt", "page": 1, "type": "text"}},
    {"content": "Document two content. " * 20, "metadata": {"source": "doc2.txt", "page": 2, "type": "text"}},
    {"content": "Document three. " * 30, "metadata": {"source": "doc3.txt", "page": 1, "type": "text"}},
]
try:
    chunks = chunk_documents(multi_docs)
    print(f"  3 documents -> {len(chunks)} total chunks")

    sources_seen = {}
    for c in chunks:
        src = c["metadata"]["source"]
        sources_seen[src] = sources_seen.get(src, 0) + 1

    for src, count in sources_seen.items():
        print(f"    '{src}': {count} chunks")

    if len(sources_seen) == 3:
        print(f"  [PASS] All 3 source documents represented")
    else:
        print(f"  [WARN] Only {len(sources_seen)} sources found, expected 3")

except Exception as e:
    print(f"  [FAIL] Multi-doc chunking error: {e}")

# --- 6. Chunk ID uniqueness ---
print("\n--- Chunk ID Uniqueness ---")
try:
    text = "Unique chunk test. " * 50
    docs = [{"content": text, "metadata": {"source": "test.txt", "page": 1, "type": "text"}}]
    chunks = chunk_documents(docs)

    ids = [c["metadata"].get("chunk_id") for c in chunks]
    ids_none = [i for i in ids if i is None]
    ids_valid = [i for i in ids if i is not None]
    unique_ids = set(ids_valid)

    print(f"  Total chunks: {len(chunks)}")
    print(f"  Chunks with chunk_id: {len(ids_valid)}")
    print(f"  Chunks missing chunk_id: {len(ids_none)}")
    print(f"  Unique chunk_ids: {len(unique_ids)}")

    if ids_none:
        print(f"  [FAIL] Some chunks missing chunk_id")
    elif len(unique_ids) == len(ids_valid):
        print(f"  [PASS] All chunk IDs are unique")
    else:
        dup_count = len(ids_valid) - len(unique_ids)
        print(f"  [FAIL] {dup_count} duplicate chunk IDs found!")

except Exception as e:
    print(f"  [FAIL] Chunk ID test error: {e}")

# --- 7. Whitespace-only text ---
print("\n--- Whitespace-Only Text ---")
docs_ws = [{"content": "    \n\n\t   \n   ", "metadata": {"source": "ws.txt", "page": 1, "type": "text"}}]
try:
    chunks = chunk_documents(docs_ws)
    print(f"  Chunks from whitespace-only content: {len(chunks)}")
    if len(chunks) == 0:
        print("  [PASS] Whitespace-only content yields 0 chunks")
    else:
        print(f"  [WARN] {len(chunks)} chunks from whitespace-only content")
        for c in chunks:
            print(f"         Content repr: {repr(c['content'])}")
except Exception as e:
    print(f"  [FAIL] Whitespace text error: {e}")

print("\n" + "=" * 60)
print("TEST 03 COMPLETE")
print("=" * 60)
