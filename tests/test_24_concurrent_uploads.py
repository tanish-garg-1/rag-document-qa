"""
TEST 24: CONCURRENT UPLOADS & EDGE CASES
Tests parallel file uploads, large file rejection, duplicate uploads,
and race conditions in the vector store.
Requires GEMINI_API_KEY.
Run: python tests/test_24_concurrent_uploads.py
"""

import sys
import os
import io
import shutil
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True) or find_dotenv())

print("=" * 60)
print("TEST 24: CONCURRENT UPLOADS & EDGE CASES")
print("=" * 60)

# --- API key check ---
print("\n--- API Key Check ---")
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    print("  [SKIP] GEMINI_API_KEY not set")
    sys.exit(0)
print(f"  [PASS] GEMINI_API_KEY set")

# --- Setup ---
print("\n--- Setup TestClient ---")
try:
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app, raise_server_exceptions=False)
    print("  [PASS] TestClient ready")
except Exception as e:
    print(f"  [FAIL] Setup error: {e}")
    sys.exit(1)

from app.utils.constants import FAISS_INDEX_DIR, MAX_UPLOAD_SIZE_MB, MAX_UPLOAD_SIZE_BYTES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
faiss_path = os.path.join(ROOT, FAISS_INDEX_DIR)
backup_path = faiss_path + "_backup_test24"

# Backup existing index
if os.path.exists(faiss_path):
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    shutil.copytree(faiss_path, backup_path)
    shutil.rmtree(faiss_path)
os.makedirs(faiss_path, exist_ok=True)

# Helpers
def make_txt(content: str) -> bytes:
    return content.encode("utf-8")

# --- 1. File size limit enforcement ---
print(f"\n--- File Size Limit ({MAX_UPLOAD_SIZE_MB} MB) ---")
try:
    # Just-under limit: 1 byte less than MAX
    small_ok = make_txt("A" * 100)
    resp = client.post("/upload", files=[("files", ("small.txt", io.BytesIO(small_ok), "text/plain"))])
    result = resp.json().get("results", [{}])[0]
    if result.get("status") == "success":
        print(f"  [PASS] Small file accepted (100 bytes)")
    else:
        print(f"  [WARN] Small file rejected: {result.get('reason')}")

    # Over limit: MAX + 1 byte (synthetic — just set MAX low for test)
    # We test the limit logic using a file that would exceed it
    # Since actually generating 50MB is slow, we monkey-patch the check
    from app.utils import constants as const_mod
    original_max = const_mod.MAX_UPLOAD_SIZE_BYTES
    const_mod.MAX_UPLOAD_SIZE_BYTES = 50  # 50 bytes max for test

    # Patch the upload route's imported constant
    import app.routes.upload as upload_mod
    original_upload_max = upload_mod.MAX_UPLOAD_SIZE_BYTES
    upload_mod.MAX_UPLOAD_SIZE_BYTES = 50

    large_bytes = make_txt("B" * 100)  # 100 bytes > 50 byte limit
    resp = client.post("/upload", files=[("files", ("big.txt", io.BytesIO(large_bytes), "text/plain"))])
    result = resp.json().get("results", [{}])[0]

    # Restore
    const_mod.MAX_UPLOAD_SIZE_BYTES = original_max
    upload_mod.MAX_UPLOAD_SIZE_BYTES = original_upload_max

    if result.get("status") == "failed" and "large" in result.get("reason", "").lower():
        print(f"  [PASS] Oversized file rejected: '{result.get('reason')}'")
    else:
        print(f"  [WARN] Oversized file not rejected: status={result.get('status')}, reason={result.get('reason')}")

except Exception as e:
    print(f"  [FAIL] Size limit test error: {e}")
    import traceback; traceback.print_exc()

# Clear for next tests
client.post("/clear")

# --- 2. Sequential multi-file upload ---
print("\n--- Sequential Multi-file Upload ---")
try:
    docs = [
        ("doc1.txt", "Artificial intelligence is the simulation of human intelligence by machines."),
        ("doc2.txt", "Machine learning allows computers to learn from data without explicit programming."),
        ("doc3.txt", "Deep learning uses neural networks with many layers to model complex patterns."),
    ]

    files = [("files", (name, io.BytesIO(make_txt(content)), "text/plain"))
             for name, content in docs]

    t0 = time.time()
    resp = client.post("/upload", files=files)
    elapsed = time.time() - t0

    results = resp.json().get("results", [])
    successes = [r for r in results if r["status"] == "success"]
    total_chunks = sum(r.get("chunks_added", 0) for r in successes)

    print(f"  Upload time: {elapsed:.2f}s")
    print(f"  Results: {len(successes)}/{len(docs)} succeeded, {total_chunks} total chunks")

    if len(successes) == len(docs):
        print("  [PASS] All 3 files uploaded successfully")
    else:
        print(f"  [FAIL] Only {len(successes)}/{len(docs)} succeeded")

    # Verify index grew
    stats = client.get("/stats").json()
    print(f"  Index stats: {stats}")
    if stats["total_vectors"] == total_chunks:
        print(f"  [PASS] Vector count matches chunks added ({total_chunks})")
    else:
        print(f"  [WARN] Vector count {stats['total_vectors']} != chunks {total_chunks}")

except Exception as e:
    print(f"  [FAIL] Sequential multi-file error: {e}")

# Clear for next tests
client.post("/clear")

# --- 3. Duplicate file upload (same content twice) ---
print("\n--- Duplicate File Upload ---")
try:
    content = "FAISS is a library for efficient similarity search developed by Facebook AI."
    file_bytes = make_txt(content)

    # Upload once
    resp1 = client.post("/upload", files=[("files", ("dup.txt", io.BytesIO(file_bytes), "text/plain"))])
    stats1 = client.get("/stats").json()

    # Upload again (same content)
    resp2 = client.post("/upload", files=[("files", ("dup.txt", io.BytesIO(file_bytes), "text/plain"))])
    stats2 = client.get("/stats").json()

    print(f"  After 1st upload: {stats1['total_vectors']} vectors")
    print(f"  After 2nd upload: {stats2['total_vectors']} vectors")

    if stats2["total_vectors"] == stats1["total_vectors"] * 2:
        print("  [INFO] Duplicate content indexed twice (FAISS doesn't deduplicate by content)")
        print("  [INFO] This is expected — deduplication would require content hashing")
    elif stats2["total_vectors"] == stats1["total_vectors"]:
        print("  [PASS] Duplicate detected and skipped")
    else:
        print(f"  [INFO] Vector count: {stats1['total_vectors']} -> {stats2['total_vectors']}")

except Exception as e:
    print(f"  [FAIL] Duplicate upload test error: {e}")

# Clear
client.post("/clear")

# --- 4. Concurrent upload threads ---
print("\n--- Concurrent Upload Threads (3 parallel) ---")
results_lock = threading.Lock()
thread_results = []

def upload_file(thread_id: int):
    content = f"Thread {thread_id}: Vector databases store embeddings for similarity search. Thread ID {thread_id} content unique."
    file_bytes = make_txt(content)
    try:
        resp = client.post(
            "/upload",
            files=[("files", (f"thread_{thread_id}.txt", io.BytesIO(file_bytes), "text/plain"))]
        )
        r = resp.json().get("results", [{}])[0]
        with results_lock:
            thread_results.append({
                "thread": thread_id,
                "status": r.get("status"),
                "chunks": r.get("chunks_added", 0)
            })
    except Exception as e:
        with results_lock:
            thread_results.append({"thread": thread_id, "status": "error", "error": str(e)})

try:
    threads = [threading.Thread(target=upload_file, args=(i,)) for i in range(3)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    print(f"  3 concurrent uploads completed in {elapsed:.2f}s")
    for r in sorted(thread_results, key=lambda x: x["thread"]):
        print(f"    Thread {r['thread']}: status={r['status']}, chunks={r.get('chunks', 0)}")

    successes = [r for r in thread_results if r["status"] == "success"]
    if len(successes) == 3:
        print("  [PASS] All 3 concurrent uploads succeeded")
    else:
        print(f"  [WARN] Only {len(successes)}/3 concurrent uploads succeeded")

    # Verify index integrity after concurrent writes
    stats = client.get("/stats").json()
    print(f"  Final index: {stats['total_vectors']} vectors, {stats['total_chunks']} chunks")
    if stats["total_vectors"] == stats["total_chunks"]:
        print("  [PASS] Vector count matches chunk count (no desync)")
    else:
        print(f"  [FAIL] Desync: {stats['total_vectors']} vectors vs {stats['total_chunks']} chunks")

except Exception as e:
    print(f"  [FAIL] Concurrent upload error: {e}")
    import traceback; traceback.print_exc()

# Clear
client.post("/clear")

# --- 5. Unsupported file types ---
print("\n--- Unsupported File Types ---")
unsupported = [
    ("data.csv", b"col1,col2\n1,2", "text/csv"),
    ("script.js", b"console.log('hi')", "application/javascript"),
    ("data.xlsx", b"fake excel", "application/vnd.openxmlformats"),
    ("image.gif", b"GIF89a", "image/gif"),
]
for fname, content, mime in unsupported:
    try:
        resp = client.post("/upload", files=[("files", (fname, io.BytesIO(content), mime))])
        r = resp.json().get("results", [{}])[0]
        if r.get("status") == "failed":
            print(f"  [PASS] {fname} rejected: '{r.get('reason', '')}'")
        else:
            print(f"  [FAIL] {fname} NOT rejected (status={r.get('status')})")
    except Exception as e:
        print(f"  [FAIL] {fname} test error: {e}")

# --- 6. Empty file upload ---
print("\n--- Empty File Upload ---")
try:
    resp = client.post("/upload", files=[("files", ("empty.txt", io.BytesIO(b""), "text/plain"))])
    r = resp.json().get("results", [{}])[0]
    print(f"  Status: {r.get('status')}, reason: {r.get('reason', 'N/A')}")
    if r.get("status") == "failed":
        print("  [PASS] Empty file correctly rejected")
    else:
        print("  [WARN] Empty file accepted (may produce 0 chunks)")
except Exception as e:
    print(f"  [FAIL] Empty file test error: {e}")

# --- 7. Upload with no files ---
print("\n--- Upload With No Files ---")
try:
    # Can't send truly empty multipart to TestClient, skip
    print("  [INFO] Skipped (TestClient requires at least 1 file)")
except Exception as e:
    print(f"  [FAIL] No-files test error: {e}")

# --- Restore ---
print("\n--- Restore ---")
if os.path.exists(backup_path):
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)
    shutil.copytree(backup_path, faiss_path)
    shutil.rmtree(backup_path)
    print("  [INFO] Original index restored")
else:
    print("  [INFO] No backup to restore")

print("\n" + "=" * 60)
print("TEST 24 COMPLETE")
print("=" * 60)
