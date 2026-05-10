"""
TEST 10: MMR RERANKING ALGORITHM
Tests the Maximal Marginal Relevance algorithm in isolation.
Uses synthetic embeddings — no API key required.
Run: python tests/test_10_mmr_algorithm.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("TEST 10: MMR RERANKING ALGORITHM")
print("=" * 60)

# --- 1. Import retriever ---
print("\n--- Import retriever ---")
try:
    from app.services.retriever import mmr_rerank
    print("  [PASS] mmr_rerank imported from retriever")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

from app.utils.constants import MMR_K, MMR_LAMBDA, EMBEDDING_DIM
print(f"  [INFO] MMR_K      = {MMR_K}")
print(f"  [INFO] MMR_LAMBDA = {MMR_LAMBDA}")
print(f"  [INFO] EMBEDDING_DIM = {EMBEDDING_DIM}")

# --- 2. Helper to make synthetic embeddings ---
def make_normalized_embedding(dim=EMBEDDING_DIM, seed=None):
    if seed is not None:
        np.random.seed(seed)
    vec = np.random.randn(dim).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()

def make_similar_embedding(base_vec, noise=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    arr = np.array(base_vec, dtype=np.float32)
    noisy = arr + noise * np.random.randn(len(arr)).astype(np.float32)
    return (noisy / np.linalg.norm(noisy)).tolist()

# --- 3. Basic MMR test ---
print("\n--- Basic MMR Test ---")

query_emb = make_normalized_embedding(seed=42)

# Make 8 candidate chunks with embeddings
candidates = []
for i in range(8):
    emb = make_normalized_embedding(seed=i + 100)
    candidates.append({
        "content": f"Candidate chunk {i}: information about topic {i % 3}",
        "metadata": {"source": f"doc{i}.txt", "page": i + 1, "chunk_id": f"chunk_{i:04d}", "type": "text"},
        "score": 0.9 - i * 0.05,
        "embedding": emb,
    })

print(f"  Input: {len(candidates)} candidates")
print(f"  Query embedding shape: {len(query_emb)}")

try:
    selected = mmr_rerank(query_emb, candidates, k=MMR_K)
    print(f"  MMR selected: {len(selected)} chunks")
    if len(selected) == MMR_K:
        print(f"  [PASS] Correct number of results returned: {MMR_K}")
    elif len(selected) < MMR_K and len(candidates) < MMR_K:
        print(f"  [PASS] Fewer candidates than k — returned all available")
    else:
        print(f"  [FAIL] Expected {MMR_K} results, got {len(selected)}")

    for i, chunk in enumerate(selected):
        content = chunk.get("content", "")[:50]
        print(f"    Selected {i}: '{content}'")

except Exception as e:
    print(f"  [FAIL] mmr_rerank error: {e}")
    import traceback
    traceback.print_exc()

# --- 4. MMR with highly similar candidates ---
print("\n--- MMR Diversity Test (Similar Candidates) ---")
base_emb = make_normalized_embedding(seed=99)
similar_candidates = []
for i in range(6):
    emb = make_similar_embedding(base_emb, noise=0.05, seed=i + 200)
    similar_candidates.append({
        "content": f"Very similar chunk {i}: almost identical content about RAG systems",
        "metadata": {"source": "same_doc.pdf", "page": 1, "chunk_id": f"sim_{i:04d}", "type": "text"},
        "score": 0.95 - i * 0.02,
        "embedding": emb,
    })

try:
    selected = mmr_rerank(query_emb, similar_candidates, k=4)
    print(f"  6 highly-similar candidates -> {len(selected)} selected")
    print(f"  [INFO] MMR should still select k items even with similar content")
    if len(selected) <= min(4, len(similar_candidates)):
        print(f"  [PASS] Returns at most min(k, candidates) results")
    for i, c in enumerate(selected):
        print(f"    Selected {i}: '{c['content'][:50]}'")
except Exception as e:
    print(f"  [FAIL] Similarity test error: {e}")

# --- 5. MMR with k > candidates ---
print("\n--- MMR k > candidates ---")
small_candidates = [
    {
        "content": f"Small candidate {i}",
        "metadata": {"source": "doc.txt", "page": 1, "chunk_id": f"small_{i}", "type": "text"},
        "score": 0.9,
        "embedding": make_normalized_embedding(seed=i + 300),
    }
    for i in range(2)
]
try:
    selected = mmr_rerank(query_emb, small_candidates, k=10)
    print(f"  k=10, candidates=2 -> {len(selected)} selected")
    if len(selected) <= 2:
        print(f"  [PASS] Returns at most available candidates")
    else:
        print(f"  [FAIL] Returned more than available candidates")
except Exception as e:
    print(f"  [FAIL] k > candidates error: {e}")

# --- 6. MMR with empty candidates ---
print("\n--- MMR empty candidates ---")
try:
    selected = mmr_rerank(query_emb, [], k=4)
    print(f"  Empty candidates -> {len(selected)} selected")
    if len(selected) == 0:
        print("  [PASS] Empty input returns empty output")
    else:
        print(f"  [FAIL] Expected 0, got {len(selected)}")
except Exception as e:
    print(f"  [FAIL] Empty candidates error: {e}")
    import traceback
    traceback.print_exc()

# --- 7. Check MMR re-embedding behavior ---
print("\n--- MMR Embedding Behavior Check ---")
print("  Checking if mmr_rerank re-embeds candidates via Gemini API...")
print("  [INFO] If candidates already have 'embedding' key, they should NOT be re-embedded.")
print("  [INFO] If mmr_rerank ignores 'embedding' and calls embed_texts(), it wastes API calls.")
print()

# Read the retriever source to check
try:
    retriever_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "app", "services", "retriever.py"
    )
    with open(retriever_path) as f:
        source = f.read()

    if "embed_texts" in source:
        lines_with_embed = [l.strip() for l in source.split("\n") if "embed_texts" in l]
        print(f"  [WARN] embed_texts() is called in retriever.py:")
        for l in lines_with_embed:
            print(f"         {l}")
        print(f"  [WARN] This means MMR re-embeds candidates — wasteful API calls!")
        print(f"  [INFO] Fix: Store embeddings with FAISS index and retrieve them for MMR")
    else:
        print("  [PASS] embed_texts() NOT called in retriever.py")

    if "cosine" in source.lower() or "dot" in source.lower():
        print("  [INFO] Cosine/dot product similarity found in retriever")
    else:
        print("  [INFO] No explicit cosine similarity — check scoring method")

except Exception as e:
    print(f"  [WARN] Could not read retriever.py: {e}")

# --- 8. MMR lambda sensitivity ---
print("\n--- MMR Lambda Sensitivity ---")
print(f"  Current lambda = {MMR_LAMBDA}")
print(f"  lambda=1.0 -> pure relevance (no diversity)")
print(f"  lambda=0.0 -> pure diversity (no relevance)")
print(f"  lambda=0.5 -> balanced (current setting)")
print(f"  [INFO] At lambda=0.5, relevance and diversity weighted equally")

print("\n" + "=" * 60)
print("TEST 10 COMPLETE")
print("=" * 60)
