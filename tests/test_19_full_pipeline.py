"""
TEST 19: FULL END-TO-END PIPELINE
Tests the complete flow: upload -> index -> query -> stream -> citations.
REQUIRES both GEMINI_API_KEY and GROQ_API_KEY.
Run: python tests/test_19_full_pipeline.py
"""

import sys
import os
import io
import shutil
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("TEST 19: FULL END-TO-END PIPELINE")
print("=" * 60)

# --- 1. Check API keys ---
print("\n--- API Keys Check ---")
gemini_key = os.getenv("GEMINI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if not gemini_key or not groq_key:
    missing = [k for k, v in [("GEMINI_API_KEY", gemini_key), ("GROQ_API_KEY", groq_key)] if not v]
    print(f"  [SKIP] Missing: {', '.join(missing)}")
    print("\n" + "=" * 60)
    print("TEST 19 SKIPPED")
    print("=" * 60)
    sys.exit(0)

print(f"  [PASS] All API keys set")

# --- 2. Setup ---
print("\n--- Setup ---")
try:
    from fastapi.testclient import TestClient
    from app.main import app
    from app.services.memory import clear_memory
    from app.utils.constants import FAISS_INDEX_DIR

    client = TestClient(app, raise_server_exceptions=False)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    faiss_path = os.path.join(root, FAISS_INDEX_DIR)

    # Backup
    backup_path = faiss_path + "_backup_test19"
    if os.path.exists(faiss_path):
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.copytree(faiss_path, backup_path)
        shutil.rmtree(faiss_path)
        os.makedirs(faiss_path)

    clear_memory()
    print("  [PASS] Setup complete — fresh state")
except Exception as e:
    print(f"  [FAIL] Setup error: {e}")
    sys.exit(1)

# --- 3. Upload multiple documents ---
print("\n--- STEP 1: Upload Documents ---")

doc1 = b"""Machine Learning Fundamentals

Machine learning is a method of data analysis that automates analytical model building. It is based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.

Types of machine learning:
- Supervised learning: trained on labeled data
- Unsupervised learning: finds patterns in unlabeled data
- Reinforcement learning: learns through trial and error with rewards

Neural networks are the backbone of deep learning, inspired by biological neural networks in the human brain."""

doc2 = b"""Vector Databases and Embeddings

A vector database stores data as high-dimensional vectors (numerical representations). These vectors capture semantic meaning, allowing similarity-based search.

Key concepts:
- Embeddings: dense numerical representations of text or images
- Cosine similarity: measures angle between vectors
- FAISS: Facebook AI Similarity Search library for fast vector search
- Approximate nearest neighbor (ANN): faster but approximate similarity search

Embeddings enable semantic search, finding conceptually related content even when exact words differ."""

doc3 = b"""Large Language Models

Large language models (LLMs) are AI models trained on vast amounts of text data. They can generate human-like text, answer questions, translate languages, and more.

Popular LLMs include:
- GPT-4 by OpenAI
- Claude by Anthropic
- Gemini by Google
- Llama by Meta

LLMs work by predicting the next token in a sequence, trained using transformer architecture with attention mechanisms."""

uploads = [
    ("ml_fundamentals.txt", doc1),
    ("vector_databases.txt", doc2),
    ("large_language_models.txt", doc3),
]

upload_success = 0
total_chunks = 0
for filename, content in uploads:
    t0 = time.time()
    response = client.post(
        "/upload",
        files=[("files", (filename, io.BytesIO(content), "text/plain"))],
    )
    elapsed = time.time() - t0
    if response.status_code == 200:
        data = response.json()
        r = data.get("results", [{}])[0]
        chunks = r.get("chunks_added", 0)
        total_chunks += chunks
        status = r.get("status")
        print(f"  [{status.upper()}] {filename}: {chunks} chunks ({elapsed:.1f}s)")
        if status == "success":
            upload_success += 1
    else:
        print(f"  [FAIL] {filename}: HTTP {response.status_code}")

print(f"\n  Uploads: {upload_success}/{len(uploads)} succeeded, {total_chunks} total chunks")
if upload_success == len(uploads):
    print("  [PASS] All documents uploaded")
else:
    print(f"  [WARN] {len(uploads) - upload_success} uploads failed")

# --- 4. Check index stats ---
print("\n--- STEP 2: Verify Index ---")
stats_resp = client.get("/stats")
if stats_resp.status_code == 200:
    stats = stats_resp.json()
    print(f"  Index stats: {stats}")
    print(f"  [PASS] /stats endpoint working")
else:
    print(f"  [WARN] /stats status {stats_resp.status_code}")

# --- 5. Query pipeline ---
print("\n--- STEP 3: Query Pipeline ---")
queries = [
    ("What is machine learning?", ["machine learning", "supervised", "data"]),
    ("How do vector databases work?", ["vector", "embedding", "FAISS", "similarity"]),
    ("What are examples of large language models?", ["GPT", "Claude", "Gemini", "Llama"]),
]

query_results = []
for question, expected_terms in queries:
    print(f"\n  Q: '{question}'")
    t0 = time.time()
    response = client.post("/query", json={"query": question}, timeout=90)
    elapsed = time.time() - t0

    if response.status_code == 200:
        answer = response.text
        print(f"  Time: {elapsed:.2f}s, Length: {len(answer)} chars")
        print(f"  Answer preview: '{answer[:200].encode('ascii', 'replace').decode()}'")

        # Check expected terms
        found_terms = [t for t in expected_terms if t.lower() in answer.lower()]
        print(f"  Terms found {len(found_terms)}/{len(expected_terms)}: {found_terms}")

        if len(found_terms) >= len(expected_terms) // 2:
            print(f"  [PASS] Answer contains relevant content")
            query_results.append(True)
        else:
            print(f"  [WARN] Few relevant terms in answer")
            query_results.append(False)

        if "---" in answer or "Sources" in answer or "📚" in answer:
            print(f"  [PASS] Citations present in answer")
        else:
            print(f"  [WARN] No citations found in answer")
    else:
        print(f"  [FAIL] Query HTTP {response.status_code}: {response.text[:100]}")
        query_results.append(False)

print(f"\n  Query results: {sum(query_results)}/{len(query_results)} passed")

# --- 6. Follow-up with conversation context ---
print("\n--- STEP 4: Follow-up Query (Context Awareness) ---")
followup = "How does it compare to supervised learning?"
print(f"  Follow-up: '{followup}'")
print(f"  (Expects context rewriting to clarify 'it' from previous conversation)")
t0 = time.time()
response = client.post("/query", json={"query": followup}, timeout=90)
elapsed = time.time() - t0
if response.status_code == 200:
    answer = response.text
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Answer: '{answer[:300].encode('ascii', 'replace').decode()}'")
    if answer.strip():
        print("  [PASS] Follow-up answered")
    else:
        print("  [FAIL] Empty follow-up answer")
else:
    print(f"  [FAIL] Follow-up HTTP {response.status_code}")

# --- 7. Check conversation history ---
print("\n--- STEP 5: Verify Conversation History ---")
response = client.get("/history")
if response.status_code == 200:
    hist = response.json()
    total = hist.get("total_messages", 0)
    messages = hist.get("messages", [])
    print(f"  Total messages: {total}")
    print(f"  Expected: ~{(len(queries) + 1) * 2} messages (q+a pairs)")
    if total >= 2:
        print(f"  [PASS] History tracking works")
        # Show last 4
        for m in messages[-4:]:
            role = m.get("role")
            content = str(m.get("content", ""))[:60]
            print(f"    {role}: '{content}'")
    else:
        print(f"  [WARN] Few messages in history: {total}")
else:
    print(f"  [WARN] /history status {response.status_code}")

# --- 8. Summary ---
print("\n--- PIPELINE TEST SUMMARY ---")
print(f"  Documents uploaded:  {upload_success}/{len(uploads)}")
print(f"  Chunks indexed:      {total_chunks}")
print(f"  Queries answered:    {sum(query_results)}/{len(query_results)}")
print(f"  History messages:    {total if 'total' in dir() else 'N/A'}")

all_pass = (upload_success == len(uploads)) and all(query_results)
if all_pass:
    print("\n  [PASS] Full end-to-end pipeline working!")
else:
    print("\n  [WARN] Some issues found — check individual test results above")

# --- Restore ---
print("\n--- Restore ---")
try:
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)
    if os.path.exists(backup_path):
        shutil.copytree(backup_path, faiss_path)
        shutil.rmtree(backup_path)
        print("  [INFO] Index restored")
    else:
        os.makedirs(faiss_path, exist_ok=True)
    clear_memory()
    print("  [INFO] Memory cleared")
except Exception as e:
    print(f"  [WARN] Restore error: {e}")

print("\n" + "=" * 60)
print("TEST 19 COMPLETE")
print("=" * 60)
