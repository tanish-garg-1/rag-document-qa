"""
TEST 09: CITATION GENERATION
Tests citation creation, deduplication, and formatting.
No API key required.
Run: python tests/test_09_citation.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("TEST 09: CITATION GENERATION")
print("=" * 60)

# --- 1. Import ---
print("\n--- Import citation ---")
try:
    from app.services.citation import generate_citations, format_citations_block
    print("  [PASS] citation module imported")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

# --- 2. Basic citation generation ---
print("\n--- generate_citations() Basic ---")
chunks = [
    {
        "content": "RAG systems combine retrieval with generation.",
        "metadata": {
            "source": "ai_guide.pdf",
            "page": 3,
            "chunk_id": "abc12345-1111-2222-3333-444444444444",
            "type": "text",
        }
    },
    {
        "content": "FAISS is used for efficient vector search.",
        "metadata": {
            "source": "faiss_docs.txt",
            "page": 1,
            "chunk_id": "def67890-5555-6666-7777-888888888888",
            "type": "text",
        }
    },
    {
        "content": "Gemini provides powerful embeddings.",
        "metadata": {
            "source": "gemini_overview.pdf",
            "page": 7,
            "chunk_id": "ghi11111-aaaa-bbbb-cccc-dddddddddddd",
            "type": "text",
        }
    },
]

try:
    citations = generate_citations(chunks)
    print(f"  Citations returned: {len(citations)}")
    print(f"  Citations:")
    for i, c in enumerate(citations):
        print(f"    [{i}] {c}")

    if len(citations) == 3:
        print("  [PASS] 3 citations for 3 unique chunks")
    else:
        print(f"  [FAIL] Expected 3 citations, got {len(citations)}")

    # Check format
    for c in citations:
        if "ai_guide.pdf" in c or "faiss_docs.txt" in c or "gemini_overview.pdf" in c:
            print(f"  [PASS] Source filename present in citation")
        else:
            print(f"  [WARN] Source not found in citation: '{c}'")
            break

    # Check page number
    for c in citations:
        if "Page" in c or "page" in c:
            print(f"  [PASS] Page number present in citation")
            break
    else:
        print(f"  [WARN] 'Page' not found in any citation")

    # Check chunk_id (at least partial)
    for c in citations:
        if "abc123" in c or "def678" in c or "ghi111" in c:
            print(f"  [PASS] chunk_id prefix present in citation")
            break
    else:
        print(f"  [WARN] chunk_id prefix not found in citations")

except Exception as e:
    print(f"  [FAIL] generate_citations error: {e}")
    import traceback
    traceback.print_exc()

# --- 3. Deduplication test ---
print("\n--- Deduplication ---")
dup_chunks = [
    {
        "content": "First occurrence of this chunk.",
        "metadata": {"source": "doc.pdf", "page": 1, "chunk_id": "same-chunk-id-001", "type": "text"}
    },
    {
        "content": "Second occurrence of same chunk.",
        "metadata": {"source": "doc.pdf", "page": 1, "chunk_id": "same-chunk-id-001", "type": "text"}
    },
    {
        "content": "Third occurrence of same chunk.",
        "metadata": {"source": "doc.pdf", "page": 1, "chunk_id": "same-chunk-id-001", "type": "text"}
    },
    {
        "content": "A different chunk.",
        "metadata": {"source": "doc.pdf", "page": 2, "chunk_id": "different-id-002", "type": "text"}
    },
]
try:
    citations = generate_citations(dup_chunks)
    print(f"  4 chunks (3 duplicates) → {len(citations)} citation(s)")
    for c in citations:
        print(f"    '{c}'")
    if len(citations) == 2:
        print("  [PASS] Duplicates removed correctly (2 unique citations)")
    elif len(citations) == 4:
        print("  [FAIL] No deduplication — all 4 citations returned")
    else:
        print(f"  [INFO] Got {len(citations)} citations — check dedup logic")
except Exception as e:
    print(f"  [FAIL] Dedup test error: {e}")

# --- 4. format_citations_block ---
print("\n--- format_citations_block() ---")
test_citations = [
    "ai_guide.pdf — Page 3 — Chunk abc12345",
    "faiss_docs.txt — Page 1 — Chunk def67890",
    "gemini.pdf — Page 7 — Chunk ghi11111",
]
try:
    block = format_citations_block(test_citations)
    print(f"  Formatted block:\n{block}")

    if "Sources" in block or "sources" in block.lower():
        print("  [PASS] 'Sources' header present")
    else:
        print("  [WARN] 'Sources' header not found")

    for i, c in enumerate(test_citations, 1):
        # Check citation appears or its parts
        if c in block:
            print(f"  [PASS] Citation {i} present in block")
        else:
            print(f"  [WARN] Citation {i} may be reformatted in block")

    if "---" in block or "📚" in block:
        print("  [PASS] Block has expected formatting elements")
    else:
        print("  [INFO] Block format may differ from expected")

except Exception as e:
    print(f"  [FAIL] format_citations_block error: {e}")
    import traceback
    traceback.print_exc()

# --- 5. Empty chunks ---
print("\n--- Empty Chunks ---")
try:
    citations = generate_citations([])
    print(f"  Citations from empty list: {len(citations)}")
    if len(citations) == 0:
        print("  [PASS] Empty chunks returns empty citations")
    else:
        print(f"  [FAIL] Expected 0 citations, got {len(citations)}")
except Exception as e:
    print(f"  [FAIL] Empty chunks error: {e}")

# --- 6. format_citations_block with empty list ---
print("\n--- format_citations_block with empty ---")
try:
    block = format_citations_block([])
    print(f"  Block from empty: '{block}'")
    if not block or block.strip() == "":
        print("  [INFO] Empty citations returns empty block")
    else:
        print(f"  [INFO] Non-empty block for empty citations: '{block[:60]}'")
except Exception as e:
    print(f"  [FAIL] Empty format_citations_block error: {e}")

# --- 7. Missing metadata keys ---
print("\n--- Missing Metadata Keys ---")
bad_chunks = [
    {"content": "Chunk with no source", "metadata": {"page": 1, "chunk_id": "xxx", "type": "text"}},
    {"content": "Chunk with no page", "metadata": {"source": "doc.pdf", "chunk_id": "yyy", "type": "text"}},
    {"content": "Chunk with no chunk_id", "metadata": {"source": "doc.pdf", "page": 1, "type": "text"}},
]
try:
    citations = generate_citations(bad_chunks)
    print(f"  Citations from chunks with missing metadata: {len(citations)}")
    for c in citations:
        print(f"    '{c}'")
    print("  [INFO] Check how missing fields appear in citations")
except Exception as e:
    print(f"  [FAIL] Missing metadata error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 09 COMPLETE")
print("=" * 60)
