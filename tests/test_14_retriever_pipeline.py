"""
TEST 14: RETRIEVER PIPELINE (End-to-End Retrieval)
Tests the full retrieve() function: embed → FAISS search → MMR.
REQUIRES GEMINI_API_KEY and a populated FAISS index.
Run: python tests/test_14_retriever_pipeline.py
"""

import sys
import os
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("TEST 14: RETRIEVER PIPELINE")
print("=" * 60)

# --- 1. Check API key ---
print("\n--- API Key Check ---")
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    print("  [SKIP] GEMINI_API_KEY not set")
    print("\n" + "=" * 60)
    print("TEST 14 SKIPPED (no API key)")
    print("=" * 60)
    sys.exit(0)

masked = gemini_key[:6] + "..." + gemini_key[-4:]
print(f"  [PASS] GEMINI_API_KEY: {masked}")

# --- 2. Import retriever ---
print("\n--- Import retriever ---")
try:
    from app.services.retriever import retrieve, mmr_rerank
    print("  [PASS] retriever imported")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

from app.services.vector_store import add_chunks_to_store, get_index_stats
from app.services.embeddings import embed_texts
from app.utils.constants import FAISS_INDEX_DIR, EMBEDDING_DIM, MMR_K

print(f"  [INFO] MMR_K = {MMR_K}")

# --- 3. Populate index with test data ---
print("\n--- Populate FAISS with Test Chunks ---")

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
faiss_path = os.path.join(root, FAISS_INDEX_DIR)

# Backup existing
backup_path = faiss_path + "_backup_test14"
if os.path.exists(faiss_path):
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    shutil.copytree(faiss_path, backup_path)
    shutil.rmtree(faiss_path)
    os.makedirs(faiss_path)
else:
    os.makedirs(faiss_path, exist_ok=True)

test_chunks_text = [
    "Retrieval-Augmented Generation (RAG) combines document retrieval with LLM generation to produce accurate answers.",
    "FAISS (Facebook AI Similarity Search) provides efficient similarity search for high-dimensional vectors.",
    "Gemini is Google's multimodal AI model capable of processing text, images, and code.",
    "Groq provides ultra-fast LLM inference using custom hardware accelerators.",
    "LangChain provides tools for building applications powered by language models.",
    "Vector embeddings represent text as dense numerical vectors for semantic similarity search.",
    "The chunking process splits long documents into smaller pieces that fit in LLM context windows.",
    "Maximal Marginal Relevance (MMR) balances relevance and diversity in document retrieval.",
    "Conversation memory stores previous exchanges to enable context-aware follow-up questions.",
    "Citations help users trace AI-generated answers back to their source documents.",
]

test_chunks = [
    {
        "content": text,
        "metadata": {
            "source": f"rag_guide.pdf",
            "page": i + 1,
            "chunk_id": f"test_chunk_{i:04d}",
            "type": "text",
        }
    }
    for i, text in enumerate(test_chunks_text)
]

try:
    print(f"  Embedding {len(test_chunks)} test chunks...")
    import time
    t0 = time.time()
    embeddings = embed_texts([c["content"] for c in test_chunks])
    elapsed = time.time() - t0
    print(f"  Embedding time: {elapsed:.2f}s")
    print(f"  Embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
    add_chunks_to_store(test_chunks, embeddings)
    stats = get_index_stats()
    print(f"  Index stats: {stats}")
    print(f"  [PASS] {stats['total_vectors']} vectors added to index")
except Exception as e:
    print(f"  [FAIL] Failed to populate index: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 4. retrieve() test ---
print("\n--- retrieve() Function Test ---")
test_queries = [
    "What is RAG and how does it work?",
    "How does FAISS search work?",
    "What is the purpose of embeddings?",
]

for query in test_queries:
    print(f"\n  Query: '{query}'")
    try:
        t0 = time.time()
        chunks = retrieve(query)
        elapsed = time.time() - t0

        print(f"  Retrieve time: {elapsed:.2f}s")
        print(f"  Chunks returned: {len(chunks)}")

        if len(chunks) > 0:
            print(f"  [PASS] Got {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                content = chunk.get("content", "")[:70]
                score = chunk.get("score", "N/A")
                print(f"    [{i}] score={score if isinstance(score, str) else f'{score:.4f}'} | '{content}'")
        else:
            print(f"  [WARN] No chunks returned — check index population or retrieval logic")

        if len(chunks) <= MMR_K:
            print(f"  [PASS] Returned <= MMR_K ({MMR_K}) chunks")
        else:
            print(f"  [FAIL] Returned {len(chunks)} > MMR_K ({MMR_K})")

    except Exception as e:
        print(f"  [FAIL] retrieve() error: {e}")
        import traceback
        traceback.print_exc()

# --- 5. Retrieve with unrelated query ---
print("\n--- Retrieve with Unrelated Query ---")
query = "What are the best pizza toppings?"
print(f"  Query: '{query}' (unrelated to indexed content)")
try:
    chunks = retrieve(query)
    print(f"  Chunks returned: {len(chunks)}")
    if chunks:
        print(f"  [INFO] FAISS returns closest matches even for unrelated queries")
        for c in chunks:
            print(f"    - '{c['content'][:60]}'")
    else:
        print(f"  [INFO] No chunks — empty index or retrieval issue")
except Exception as e:
    print(f"  [FAIL] Error: {e}")

# --- 6. Check return structure ---
print("\n--- Chunk Return Structure ---")
try:
    chunks = retrieve("What is machine learning?")
    if chunks:
        c = chunks[0]
        required_keys = {"content", "metadata"}
        missing = required_keys - set(c.keys())
        if missing:
            print(f"  [FAIL] Chunk missing keys: {missing}")
        else:
            print(f"  [PASS] Chunk has required keys: {set(c.keys())}")

        meta_keys = {"source", "page", "chunk_id", "type"}
        meta = c.get("metadata", {})
        missing_meta = meta_keys - set(meta.keys())
        if missing_meta:
            print(f"  [FAIL] Metadata missing keys: {missing_meta}")
        else:
            print(f"  [PASS] Metadata has required keys")
            print(f"         {meta}")
except Exception as e:
    print(f"  [FAIL] Structure check error: {e}")

# --- 7. Restore index ---
print("\n--- Restore Original Index ---")
try:
    shutil.rmtree(faiss_path)
    if os.path.exists(backup_path):
        shutil.copytree(backup_path, faiss_path)
        shutil.rmtree(backup_path)
        print("  [INFO] Original index restored")
    else:
        os.makedirs(faiss_path, exist_ok=True)
        print("  [INFO] Fresh index directory created")
except Exception as e:
    print(f"  [WARN] Restore error: {e}")

print("\n" + "=" * 60)
print("TEST 14 COMPLETE")
print("=" * 60)
