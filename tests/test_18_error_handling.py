"""
TEST 18: ERROR HANDLING & EDGE CASES
Tests how the system handles errors, corrupted files, empty inputs,
and boundary conditions throughout the pipeline. No API key required for most tests.
Run: python tests/test_18_error_handling.py
"""

import sys
import os
import tempfile
import json
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("TEST 18: ERROR HANDLING & EDGE CASES")
print("=" * 60)

# --- 1. Document loader with nonexistent file ---
print("\n--- load_document with nonexistent file ---")
try:
    from app.services.document_loader import load_document
    docs = load_document("/nonexistent/path/file.txt", "file.txt")
    print(f"  Result: {docs}")
    print("  [WARN] No error raised for nonexistent file — check error handling")
except FileNotFoundError as e:
    print(f"  [PASS] FileNotFoundError raised: {e}")
except Exception as e:
    print(f"  [INFO] Exception type {type(e).__name__}: {e}")

# --- 2. load_txt with nonexistent file ---
print("\n--- load_txt with nonexistent file ---")
try:
    from app.services.document_loader import load_txt
    docs = load_txt("/nonexistent/path/file.txt")
    print(f"  Result: {docs}")
    print("  [WARN] No error raised")
except FileNotFoundError as e:
    print(f"  [PASS] FileNotFoundError: {e}")
except Exception as e:
    print(f"  [INFO] {type(e).__name__}: {e}")

# --- 3. load_pdf with corrupted bytes ---
print("\n--- load_pdf with corrupted bytes ---")
try:
    from app.services.document_loader import load_pdf
    fd, bad_pdf = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(b"this is not a valid PDF file at all")
    docs = load_pdf(bad_pdf)
    print(f"  Result: {docs}")
    print("  [WARN] No error raised for corrupted PDF")
    os.unlink(bad_pdf)
except Exception as e:
    print(f"  [PASS] Error raised for corrupted PDF: {type(e).__name__}: {str(e)[:80]}")
    try:
        os.unlink(bad_pdf)
    except Exception:
        pass

# --- 4. chunk_documents with None content ---
print("\n--- chunk_documents with None content ---")
try:
    from app.services.chunking import chunk_documents
    docs_none = [{"content": None, "metadata": {"source": "x.txt", "page": 1, "type": "text"}}]
    chunks = chunk_documents(docs_none)
    print(f"  Result: {len(chunks)} chunks")
    print("  [WARN] None content not handled — may cause issues")
except TypeError as e:
    print(f"  [PASS] TypeError raised for None content: {e}")
except Exception as e:
    print(f"  [INFO] {type(e).__name__}: {e}")

# --- 5. chunk_documents with missing metadata ---
print("\n--- chunk_documents with missing/partial metadata ---")
try:
    docs_no_meta = [{"content": "Some text content here for testing.", "metadata": {}}]
    chunks = chunk_documents(docs_no_meta)
    print(f"  Result: {len(chunks)} chunks")
    if chunks:
        meta = chunks[0].get("metadata", {})
        print(f"  Chunk metadata: {meta}")
        print("  [PASS] Chunks produced despite empty metadata")
    else:
        print("  [INFO] No chunks from partial metadata doc")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")

# --- 6. vector_store with mismatched embedding sizes ---
print("\n--- vector_store add with mismatched sizes ---")
try:
    from app.services.vector_store import add_chunks_to_store
    from app.utils.constants import EMBEDDING_DIM, FAISS_INDEX_DIR

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    faiss_path = os.path.join(root, FAISS_INDEX_DIR)
    backup_path = faiss_path + "_backup_err"
    if os.path.exists(faiss_path):
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.copytree(faiss_path, backup_path)
        shutil.rmtree(faiss_path)
        os.makedirs(faiss_path)

    # Wrong dimension embedding
    wrong_dim = 128  # should be EMBEDDING_DIM (3072)
    chunks = [{"content": "test chunk", "metadata": {"source": "x.txt", "page": 1, "chunk_id": "x001", "type": "text"}}]
    wrong_embeds = [np.random.randn(wrong_dim).tolist()]

    add_chunks_to_store(chunks, wrong_embeds)
    print(f"  [WARN] Wrong-dim embedding ({wrong_dim}) accepted — may cause FAISS issues")

    # Restore
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)
    if os.path.exists(backup_path):
        shutil.copytree(backup_path, faiss_path)
        shutil.rmtree(backup_path)
except Exception as e:
    print(f"  [PASS] Error raised for wrong embedding dim: {type(e).__name__}: {str(e)[:80]}")
    try:
        if os.path.exists(faiss_path):
            shutil.rmtree(faiss_path)
        if os.path.exists(backup_path):
            shutil.copytree(backup_path, faiss_path)
            shutil.rmtree(backup_path)
    except Exception:
        pass

# --- 7. vector_store search on empty index ---
print("\n--- vector_store search on empty index ---")
try:
    from app.services.vector_store import search_similar
    from app.utils.constants import EMBEDDING_DIM, FAISS_INDEX_DIR

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    faiss_path = os.path.join(root, FAISS_INDEX_DIR)
    backup_path = faiss_path + "_backup_empty"
    if os.path.exists(faiss_path):
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.copytree(faiss_path, backup_path)
        shutil.rmtree(faiss_path)
        os.makedirs(faiss_path)

    query_emb = np.random.randn(EMBEDDING_DIM).tolist()
    results = search_similar(query_emb, k=5)
    print(f"  Results from empty index: {len(results)}")
    if len(results) == 0:
        print("  [PASS] Empty index returns 0 results")
    else:
        print(f"  [WARN] Got {len(results)} results from empty index")

    # Restore
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)
    if os.path.exists(backup_path):
        shutil.copytree(backup_path, faiss_path)
        shutil.rmtree(backup_path)
    else:
        os.makedirs(faiss_path, exist_ok=True)
except Exception as e:
    print(f"  [INFO] Exception on empty index search: {type(e).__name__}: {str(e)[:80]}")
    print("  [WARN] Need to handle empty index gracefully in retriever.py")
    try:
        if os.path.exists(faiss_path):
            shutil.rmtree(faiss_path)
        if os.path.exists(backup_path):
            shutil.copytree(backup_path, faiss_path)
            shutil.rmtree(backup_path)
        else:
            os.makedirs(faiss_path, exist_ok=True)
    except Exception:
        pass

# --- 8. metadata.json desync check ---
print("\n--- metadata.json Desync Potential ---")
print("  [INFO] Known architectural risk: FAISS index and metadata.json may desync if:")
print("         - Server crashes mid-write")
print("         - FAISS index saved but metadata.json save fails")
print("         - Manual edits to metadata.json")
print("         - /clear deletes FAISS but metadata survives (or vice versa)")

# Check current state
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from app.utils.constants import FAISS_INDEX_DIR
faiss_path = os.path.join(root, FAISS_INDEX_DIR)
index_file = os.path.join(faiss_path, "index.faiss")
meta_file = os.path.join(faiss_path, "metadata.json")

if os.path.exists(index_file) and os.path.exists(meta_file):
    try:
        import faiss
        idx = faiss.read_index(index_file)
        with open(meta_file) as f:
            meta = json.load(f)
        faiss_count = idx.ntotal
        meta_count = len(meta)
        print(f"  FAISS index vectors: {faiss_count}")
        print(f"  metadata.json entries: {meta_count}")
        if faiss_count == meta_count:
            print("  [PASS] FAISS and metadata counts match")
        else:
            print(f"  [FAIL] COUNTS MISMATCH! FAISS={faiss_count}, metadata={meta_count}")
            print("          This will cause incorrect metadata on search results!")
    except Exception as e:
        print(f"  [WARN] Could not verify sync: {e}")
elif not os.path.exists(index_file) and not os.path.exists(meta_file):
    print("  [INFO] No index exists yet — sync is N/A")
elif os.path.exists(index_file) and not os.path.exists(meta_file):
    print("  [FAIL] index.faiss exists but metadata.json does NOT — desync!")
elif not os.path.exists(index_file) and os.path.exists(meta_file):
    print("  [FAIL] metadata.json exists but index.faiss does NOT — desync!")

# --- 9. Memory add_message with empty content ---
print("\n--- memory add_message edge cases ---")
try:
    from app.services.memory import add_message, get_recent_history, clear_memory, get_memory_size

    clear_memory()
    add_message("user", "")
    size = get_memory_size()
    print(f"  add_message('user', '') → size: {size}")
    if size == 1:
        print("  [INFO] Empty message accepted — may pollute history")
    elif size == 0:
        print("  [PASS] Empty message rejected")

    clear_memory()
    add_message("unknown_role", "Some content")
    hist = get_recent_history(10)
    if hist:
        role = hist[0].get("role")
        print(f"  add_message('unknown_role', ...) → stored role: '{role}'")
        print("  [INFO] Unknown role accepted — check if validation needed")
    clear_memory()
except Exception as e:
    print(f"  [FAIL] Memory edge case error: {e}")

# --- 10. citation with special characters ---
print("\n--- citation with special characters in source ---")
try:
    from app.services.citation import generate_citations
    chunks = [
        {
            "content": "test",
            "metadata": {
                "source": "doc with spaces & special <chars>.pdf",
                "page": 1,
                "chunk_id": "special_001",
                "type": "text",
            }
        }
    ]
    citations = generate_citations(chunks)
    print(f"  Citation: '{citations[0] if citations else 'None'}'")
    print("  [PASS] Special chars in filename handled")
except Exception as e:
    print(f"  [FAIL] Special chars citation error: {e}")

print("\n" + "=" * 60)
print("TEST 18 COMPLETE")
print("=" * 60)
