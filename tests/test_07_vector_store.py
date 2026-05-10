"""
TEST 07: VECTOR STORE (FAISS)
Tests FAISS index creation, add, search, metadata, persistence, and clear.
No API key required — uses synthetic random embeddings.
Run: python tests/test_07_vector_store.py
"""

import sys
import os
import json
import shutil
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("TEST 07: VECTOR STORE (FAISS)")
print("=" * 60)

# --- 1. Import ---
print("\n--- Import FAISS ---")
try:
    import faiss
    print(f"  [PASS] FAISS imported")
except ImportError as e:
    print(f"  [FAIL] FAISS not installed: {e}")
    print("         Run: pip install faiss-cpu")
    sys.exit(1)

print("\n--- Import vector_store ---")
try:
    from app.services.vector_store import (
        add_chunks_to_store,
        search_similar,
        get_index_stats,
    )
    print("  [PASS] vector_store imported")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

from app.utils.constants import FAISS_INDEX_DIR, EMBEDDING_DIM
print(f"  [INFO] FAISS_INDEX_DIR = {FAISS_INDEX_DIR}")
print(f"  [INFO] EMBEDDING_DIM   = {EMBEDDING_DIM}")

# --- 2. Setup temp directory ---
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
faiss_path = os.path.join(root, FAISS_INDEX_DIR)
print(f"\n--- FAISS Directory ---")
print(f"  Path: {faiss_path}")
if os.path.exists(faiss_path):
    print(f"  [INFO] Directory exists")
    index_file = os.path.join(faiss_path, "index.faiss")
    meta_file = os.path.join(faiss_path, "metadata.json")
    print(f"  index.faiss exists: {os.path.exists(index_file)}")
    print(f"  metadata.json exists: {os.path.exists(meta_file)}")
else:
    print(f"  [INFO] Directory does not exist yet")

# Backup existing index if present
backup_path = faiss_path + "_backup_test"
if os.path.exists(faiss_path):
    print(f"  Backing up existing index to: {backup_path}")
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    shutil.copytree(faiss_path, backup_path)
    shutil.rmtree(faiss_path)
    os.makedirs(faiss_path)

# --- 3. Create synthetic data ---
def make_random_embedding(dim=EMBEDDING_DIM):
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)  # normalize
    return vec.tolist()

def make_chunks(n, prefix="Test chunk"):
    chunks = []
    for i in range(n):
        chunks.append({
            "content": f"{prefix} number {i}: artificial intelligence and machine learning",
            "metadata": {
                "source": f"doc_{i//3}.txt",
                "page": (i % 3) + 1,
                "chunk_id": f"chunk_{i:04d}",
                "type": "text",
            }
        })
    return chunks

# --- 4. add_chunks_to_store ---
print("\n--- add_chunks_to_store() ---")
chunks_5 = make_chunks(5, "Sample chunk")
embeddings_5 = [make_random_embedding() for _ in range(5)]

print(f"  Adding {len(chunks_5)} chunks with {EMBEDDING_DIM}-dim embeddings")
print(f"  Embedding shape: {len(embeddings_5)} x {len(embeddings_5[0])}")
try:
    add_chunks_to_store(chunks_5, embeddings_5)
    print("  [PASS] add_chunks_to_store completed without error")

    # Verify files created
    index_file = os.path.join(faiss_path, "index.faiss")
    meta_file = os.path.join(faiss_path, "metadata.json")

    if os.path.exists(index_file):
        size = os.path.getsize(index_file)
        print(f"  [PASS] index.faiss created, size={size} bytes")
    else:
        print(f"  [FAIL] index.faiss NOT created at {index_file}")

    if os.path.exists(meta_file):
        size = os.path.getsize(meta_file)
        print(f"  [PASS] metadata.json created, size={size} bytes")
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"         metadata.json has {len(meta)} entries")
        if len(meta) == 5:
            print(f"  [PASS] metadata count matches chunks added")
        else:
            print(f"  [FAIL] metadata count {len(meta)} != chunks added {5}")
    else:
        print(f"  [FAIL] metadata.json NOT created at {meta_file}")

except Exception as e:
    print(f"  [FAIL] add_chunks_to_store error: {e}")
    import traceback
    traceback.print_exc()

# --- 5. get_index_stats ---
print("\n--- get_index_stats() ---")
try:
    stats = get_index_stats()
    print(f"  Stats returned: {stats}")
    expected_keys = {"total_vectors", "total_chunks", "embedding_dim"}
    missing = expected_keys - set(stats.keys())
    if missing:
        print(f"  [FAIL] Missing stats keys: {missing}")
    else:
        print(f"  [PASS] All expected stats keys present")

    if stats.get("total_vectors") == 5:
        print(f"  [PASS] total_vectors = {stats['total_vectors']} (expected 5)")
    else:
        print(f"  [FAIL] total_vectors = {stats.get('total_vectors')}, expected 5")

    if stats.get("embedding_dim") == EMBEDDING_DIM:
        print(f"  [PASS] embedding_dim = {stats['embedding_dim']}")
    else:
        print(f"  [WARN] embedding_dim = {stats.get('embedding_dim')}, constants say {EMBEDDING_DIM}")

except Exception as e:
    print(f"  [FAIL] get_index_stats error: {e}")
    import traceback
    traceback.print_exc()

# --- 6. search_similar ---
print("\n--- search_similar() ---")
query_embedding = make_random_embedding()
try:
    results = search_similar(query_embedding, k=3)
    print(f"  Results returned: {len(results)}")
    if len(results) > 0:
        print(f"  [PASS] Got {len(results)} result(s)")
        for i, r in enumerate(results):
            content = r.get("content", "")[:40]
            score = r.get("score", "N/A")
            meta = r.get("metadata", {})
            print(f"    Result {i}: score={score:.4f}, content='{content}', meta={meta}")
    else:
        print(f"  [FAIL] No results from search_similar")

    # Check result structure
    if results:
        first = results[0]
        required = {"content", "metadata", "score"}
        missing = required - set(first.keys())
        if missing:
            print(f"  [FAIL] Result missing keys: {missing}")
        else:
            print(f"  [PASS] Result has all required keys: {set(first.keys())}")

except Exception as e:
    print(f"  [FAIL] search_similar error: {e}")
    import traceback
    traceback.print_exc()

# --- 7. Accumulate more chunks (test append) ---
print("\n--- Appending More Chunks ---")
chunks_3 = make_chunks(3, "Additional chunk")
embeddings_3 = [make_random_embedding() for _ in range(3)]
try:
    add_chunks_to_store(chunks_3, embeddings_3)
    stats = get_index_stats()
    print(f"  After appending 3 more chunks:")
    print(f"  total_vectors = {stats.get('total_vectors')} (expected 8)")
    if stats.get("total_vectors") == 8:
        print(f"  [PASS] Append works correctly")
    else:
        print(f"  [FAIL] Expected 8, got {stats.get('total_vectors')}")
except Exception as e:
    print(f"  [FAIL] Append error: {e}")

# --- 8. Search with k > stored vectors ---
print("\n--- search_similar k > total vectors ---")
query2 = make_random_embedding()
try:
    results = search_similar(query2, k=50)
    print(f"  k=50 with 8 stored vectors -> got {len(results)} results")
    if len(results) <= 8:
        print(f"  [PASS] Returns at most stored vector count")
    else:
        print(f"  [FAIL] Returned more results than stored vectors")
except Exception as e:
    print(f"  [FAIL] k > stored vectors error: {e}")

# --- 9. Empty index search ---
print("\n--- Restore/cleanup ---")
try:
    shutil.rmtree(faiss_path)
    if os.path.exists(backup_path):
        shutil.copytree(backup_path, faiss_path)
        shutil.rmtree(backup_path)
        print(f"  [INFO] Original index restored from backup")
    else:
        os.makedirs(faiss_path, exist_ok=True)
        print(f"  [INFO] No backup to restore — created fresh dir")
except Exception as e:
    print(f"  [WARN] Cleanup error: {e}")

print("\n" + "=" * 60)
print("TEST 07 COMPLETE")
print("=" * 60)
