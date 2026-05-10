"""
TEST 17: QUERY ENDPOINT
Tests POST /query with streaming, GET/DELETE /history.
REQUIRES both GEMINI_API_KEY and GROQ_API_KEY and a populated FAISS index.
Run: python tests/test_17_query_endpoint.py
"""

import sys
import os
import io
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("TEST 17: QUERY ENDPOINT")
print("=" * 60)

# --- 1. Check API keys ---
print("\n--- API Keys Check ---")
gemini_key = os.getenv("GEMINI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if not gemini_key or not groq_key:
    missing = []
    if not gemini_key:
        missing.append("GEMINI_API_KEY")
    if not groq_key:
        missing.append("GROQ_API_KEY")
    print(f"  [SKIP] Missing: {', '.join(missing)}")
    print("\n" + "=" * 60)
    print("TEST 17 SKIPPED (missing API keys)")
    print("=" * 60)
    sys.exit(0)

print(f"  [PASS] Both API keys set")

# --- 2. Setup client ---
print("\n--- Setup TestClient ---")
try:
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app, raise_server_exceptions=False)
    print("  [PASS] TestClient ready")
except Exception as e:
    print(f"  [FAIL] Setup error: {e}")
    sys.exit(1)

from app.utils.constants import FAISS_INDEX_DIR
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
faiss_path = os.path.join(root, FAISS_INDEX_DIR)

# Backup
backup_path = faiss_path + "_backup_test17"
if os.path.exists(faiss_path):
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    shutil.copytree(faiss_path, backup_path)
    shutil.rmtree(faiss_path)
    os.makedirs(faiss_path)

# --- 3. Upload test document first ---
print("\n--- Upload Test Document ---")
txt_content = b"""Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by retrieving relevant information from external knowledge bases before generating responses.

RAG systems work in three main steps:
1. Indexing: Documents are chunked, embedded, and stored in a vector database.
2. Retrieval: When a query arrives, it is embedded and used to find similar document chunks.
3. Generation: The retrieved chunks are provided as context to the LLM to generate an answer.

FAISS (Facebook AI Similarity Search) is widely used as the vector store in RAG systems due to its efficiency.

Gemini provides state-of-the-art text embeddings with 3072 dimensions.

Groq offers ultra-fast LLM inference, making it ideal for streaming responses."""

try:
    import time
    t0 = time.time()
    response = client.post(
        "/upload",
        files=[("files", ("rag_guide.txt", io.BytesIO(txt_content), "text/plain"))],
    )
    elapsed = time.time() - t0
    print(f"  Upload time: {elapsed:.2f}s, status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        r = data.get("results", [{}])[0]
        print(f"  Chunks indexed: {r.get('chunks_added', 'N/A')}")
        print(f"  [PASS] Test document uploaded")
    else:
        print(f"  [FAIL] Upload failed: {response.text[:200]}")
        sys.exit(1)
except Exception as e:
    print(f"  [FAIL] Upload error: {e}")
    sys.exit(1)

# --- 4. POST /query basic ---
print("\n--- POST /query Basic Test ---")
try:
    t0 = time.time()
    response = client.post(
        "/query",
        json={"query": "What is RAG?"},
        timeout=60,
    )
    elapsed = time.time() - t0
    print(f"  Query time: {elapsed:.2f}s")
    print(f"  Status code: {response.status_code}")
    print(f"  Content-Type: {response.headers.get('content-type', 'N/A')}")

    if response.status_code == 200:
        text = response.text
        print(f"  Response length: {len(text)} chars")
        print(f"  Response preview:\n  {'='*40}")
        print(f"  {text[:500].encode('ascii', 'replace').decode()}")
        print(f"  {'='*40}")

        if text.strip():
            print("  [PASS] Got non-empty response")
        else:
            print("  [FAIL] Empty response")

        if "RAG" in text or "retrieval" in text.lower():
            print("  [PASS] Response contains relevant content")
        else:
            print("  [WARN] Response may not mention RAG")

        if "Sources" in text or "📚" in text or "---" in text:
            print("  [PASS] Citations/sources block found in response")
        else:
            print("  [WARN] No citations block found — check citation generation")

    else:
        print(f"  [FAIL] Query failed: {response.text[:300]}")
except Exception as e:
    print(f"  [FAIL] Query error: {e}")
    import traceback
    traceback.print_exc()

# --- 5. POST /query follow-up ---
print("\n--- POST /query Follow-up (Context Rewriting) ---")
try:
    response = client.post(
        "/query",
        json={"query": "How does it work step by step?"},
        timeout=60,
    )
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        text = response.text
        print(f"  Response: '{text[:300]}'")
        if text.strip():
            print("  [PASS] Follow-up query answered")
        else:
            print("  [FAIL] Empty follow-up response")
    else:
        print(f"  [FAIL] Follow-up failed: {response.text[:200]}")
except Exception as e:
    print(f"  [FAIL] Follow-up error: {e}")

# --- 6. GET /history ---
print("\n--- GET /history ---")
try:
    response = client.get("/history")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  total_messages: {data.get('total_messages')}")
        messages = data.get("messages", [])
        print(f"  Messages in history: {len(messages)}")
        for i, m in enumerate(messages[-4:]):
            role = m.get("role")
            content = str(m.get("content", ""))[:60]
            print(f"    [{i}] {role}: '{content}'")
        if len(messages) >= 2:
            print("  [PASS] History has at least 2 messages (query + response)")
        else:
            print(f"  [WARN] Expected >=2 messages, got {len(messages)}")
    else:
        print(f"  [FAIL] /history status {response.status_code}: {response.text[:100]}")
except Exception as e:
    print(f"  [FAIL] GET /history error: {e}")

# --- 7. DELETE /history ---
print("\n--- DELETE /history ---")
try:
    response = client.delete("/history")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Response: {data}")
        print("  [PASS] History deleted")
        # Verify cleared
        check = client.get("/history")
        if check.status_code == 200:
            hist = check.json()
            if hist.get("total_messages", 1) == 0:
                print("  [PASS] History confirmed empty after delete")
            else:
                print(f"  [WARN] History still has {hist.get('total_messages')} messages after delete")
    else:
        print(f"  [FAIL] DELETE /history status {response.status_code}")
except Exception as e:
    print(f"  [FAIL] DELETE /history error: {e}")

# --- 8. POST /query with empty query string ---
print("\n--- POST /query Empty Query ---")
try:
    response = client.post("/query", json={"query": ""}, timeout=30)
    print(f"  Status: {response.status_code}")
    print(f"  Response: '{response.text[:200]}'")
    if response.status_code in (400, 422):
        print("  [PASS] Empty query rejected with proper error code")
    elif response.status_code == 200:
        print("  [WARN] Empty query accepted — consider adding validation")
    else:
        print(f"  [INFO] Empty query status: {response.status_code}")
except Exception as e:
    print(f"  [FAIL] Empty query error: {e}")

# --- 9. POST /query missing field ---
print("\n--- POST /query Missing 'query' Field ---")
try:
    response = client.post("/query", json={}, timeout=10)
    print(f"  Status: {response.status_code}")
    if response.status_code == 422:
        print("  [PASS] Missing field returns 422 Unprocessable Entity")
    else:
        print(f"  [INFO] Missing field status: {response.status_code}, body: {response.text[:100]}")
except Exception as e:
    print(f"  [FAIL] Missing field error: {e}")

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
except Exception as e:
    print(f"  [WARN] Restore error: {e}")

print("\n" + "=" * 60)
print("TEST 17 COMPLETE")
print("=" * 60)
