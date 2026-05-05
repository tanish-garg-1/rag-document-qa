"""
TEST 20: KNOWN ISSUES AUDIT
Probes each known bug and architectural issue. Produces a clear pass/fail/warn
for each one so you know exactly what to fix and in what order.
No API key required for most checks.
Run: python tests/test_20_known_issues.py
"""

import sys
import os
import inspect
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("TEST 20: KNOWN ISSUES AUDIT")
print("=" * 60)
print("This test documents all known bugs and their current status.")
print()

issues = []

def check(issue_id, title, severity, passed, detail):
    status = "[PASS]" if passed else "[FAIL]"
    issues.append((issue_id, title, severity, passed, detail))
    print(f"{status} #{issue_id:02d} [{severity:8}] {title}")
    print(f"         {detail}")

# ─────────────────────────────────────────────────────────────────
# ISSUE 1: MMR Re-embedding inefficiency
# ─────────────────────────────────────────────────────────────────
print("\n--- Issue 1: MMR Re-embedding ---")
try:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    retriever_path = os.path.join(root, "app", "services", "retriever.py")
    with open(retriever_path) as f:
        retriever_src = f.read()

    has_reembed = "embed_texts" in retriever_src
    lines = [l.strip() for l in retriever_src.split("\n") if "embed_texts" in l]
    detail = f"embed_texts() called in retriever.py: {lines}" if has_reembed else "embed_texts() NOT called in mmr_rerank"
    check(1, "MMR Re-embedding (API waste)", "HIGH", not has_reembed, detail)
except Exception as e:
    check(1, "MMR Re-embedding (API waste)", "HIGH", False, f"Could not check: {e}")

# ─────────────────────────────────────────────────────────────────
# ISSUE 2: Conversation memory NOT persistent
# ─────────────────────────────────────────────────────────────────
print("\n--- Issue 2: Memory Persistence ---")
try:
    memory_path = os.path.join(root, "app", "services", "memory.py")
    with open(memory_path) as f:
        memory_src = f.read()

    uses_global_list = "_memory = []" in memory_src
    uses_redis = "redis" in memory_src.lower()
    uses_sqlite = "sqlite" in memory_src.lower()
    uses_file = "open(" in memory_src and "json" in memory_src

    if uses_global_list and not uses_redis and not uses_sqlite:
        check(2, "Memory NOT persistent (in-memory list)", "HIGH", False,
              "Global Python list — all history lost on server restart. Fix: Redis/SQLite/file")
    else:
        check(2, "Memory NOT persistent", "HIGH", True,
              "Memory uses persistent storage")
except Exception as e:
    check(2, "Memory NOT persistent", "HIGH", False, f"Could not check: {e}")

# ─────────────────────────────────────────────────────────────────
# ISSUE 3: FAISS index-metadata desync risk
# ─────────────────────────────────────────────────────────────────
print("\n--- Issue 3: FAISS-Metadata Desync Risk ---")
try:
    vs_path = os.path.join(root, "app", "services", "vector_store.py")
    with open(vs_path) as f:
        vs_src = f.read()

    # Check if atomic write or transaction used
    uses_atomic = "atomic" in vs_src.lower() or "tempfile" in vs_src or "tmp" in vs_src
    uses_transaction = "transaction" in vs_src.lower()
    faiss_save_before_meta = True  # Check order of saves

    if not uses_atomic and not uses_transaction:
        check(3, "FAISS-metadata desync (no atomic write)", "MEDIUM", False,
              "vector_store.py has no atomic write — crash between FAISS save and metadata save causes desync")
    else:
        check(3, "FAISS-metadata desync", "MEDIUM", True,
              "Atomic write or transaction in place")
except Exception as e:
    check(3, "FAISS-metadata desync", "MEDIUM", False, f"Could not check: {e}")

# ─────────────────────────────────────────────────────────────────
# ISSUE 4: Image failures crash entire upload
# ─────────────────────────────────────────────────────────────────
print("\n--- Issue 4: Image Failure Crash ---")
try:
    loader_path = os.path.join(root, "app", "services", "document_loader.py")
    with open(loader_path) as f:
        loader_src = f.read()

    # Check for try/except around vision API call in load_pdf
    # Look for pattern: try ... gemini ... except
    has_vision_error_handling = False
    lines = loader_src.split("\n")
    in_try_block = False
    for i, line in enumerate(lines):
        if "try:" in line:
            in_try_block = True
        if in_try_block and ("generate_content" in line or "gemini" in line.lower()):
            # Check if followed by except within reasonable range
            nearby = "\n".join(lines[i:min(i+10, len(lines))])
            if "except" in nearby:
                has_vision_error_handling = True
                break
        if "except" in line:
            in_try_block = False

    check(4, "Image vision failure crashes upload", "MEDIUM", has_vision_error_handling,
          "Gemini Vision call has try/except" if has_vision_error_handling else
          "No try/except around Gemini Vision — single bad image breaks entire PDF upload")
except Exception as e:
    check(4, "Image failure crash", "MEDIUM", False, f"Could not check: {e}")

# ─────────────────────────────────────────────────────────────────
# ISSUE 5: No rate limiting on endpoints
# ─────────────────────────────────────────────────────────────────
print("\n--- Issue 5: Rate Limiting ---")
try:
    upload_path = os.path.join(root, "app", "routes", "upload.py")
    query_path = os.path.join(root, "app", "routes", "query.py")
    main_path = os.path.join(root, "app", "main.py")

    with open(upload_path) as f:
        upload_src = f.read()
    with open(query_path) as f:
        query_src = f.read()
    with open(main_path) as f:
        main_src = f.read()

    has_rate_limit = any("rate" in s.lower() or "slowapi" in s.lower() or "limiter" in s.lower()
                         for s in [upload_src, query_src, main_src])
    check(5, "No rate limiting on API endpoints", "LOW", has_rate_limit,
          "Rate limiting found" if has_rate_limit else
          "No rate limiting — simultaneous requests can exhaust Gemini/Groq API quotas")
except Exception as e:
    check(5, "Rate limiting", "LOW", False, f"Could not check: {e}")

# ─────────────────────────────────────────────────────────────────
# ISSUE 6: FAISS L2 distance vs cosine in MMR
# ─────────────────────────────────────────────────────────────────
print("\n--- Issue 6: FAISS L2 vs Cosine Mismatch ---")
try:
    with open(vs_path) as f:
        vs_src = f.read()
    with open(retriever_path) as f:
        ret_src = f.read()

    uses_l2 = "IndexFlatL2" in vs_src
    uses_cosine_mmr = "cosine" in ret_src.lower() or "dot" in ret_src.lower() or "@" in ret_src

    if uses_l2 and uses_cosine_mmr:
        check(6, "FAISS L2 vs cosine similarity mismatch", "LOW", False,
              "FAISS uses L2 distance for retrieval, MMR uses cosine similarity — inconsistent metric")
    elif uses_l2:
        check(6, "FAISS uses L2 distance", "LOW", False,
              "FAISS IndexFlatL2 used — consider IndexFlatIP with normalized vectors for cosine")
    else:
        check(6, "FAISS distance metric", "LOW", True, "Cosine similarity used consistently")
except Exception as e:
    check(6, "FAISS distance metric", "LOW", False, f"Could not check: {e}")

# ─────────────────────────────────────────────────────────────────
# ISSUE 7: No input validation on query endpoint
# ─────────────────────────────────────────────────────────────────
print("\n--- Issue 7: Query Input Validation ---")
try:
    with open(query_path) as f:
        query_src = f.read()

    has_query_validation = ("if not" in query_src and "query" in query_src and
                            ("empty" in query_src.lower() or "strip" in query_src or
                             "len(" in query_src))
    has_pydantic_validator = "validator" in query_src or "field_validator" in query_src
    has_min_length = "min_length" in query_src

    has_validation = has_query_validation or has_pydantic_validator or has_min_length
    check(7, "No empty/null query validation", "MEDIUM", has_validation,
          "Query validation present" if has_validation else
          "Empty string query not validated — will call LLM with empty input")
except Exception as e:
    check(7, "Query validation", "MEDIUM", False, f"Could not check: {e}")

# ─────────────────────────────────────────────────────────────────
# ISSUE 8: No CORS headers
# ─────────────────────────────────────────────────────────────────
print("\n--- Issue 8: CORS Configuration ---")
try:
    with open(main_path) as f:
        main_src = f.read()

    has_cors = "CORSMiddleware" in main_src or "cors" in main_src.lower()
    check(8, "CORS headers", "LOW", has_cors,
          "CORSMiddleware configured" if has_cors else
          "No CORS headers — browser-based clients will be blocked by CORS policy")
except Exception as e:
    check(8, "CORS", "LOW", False, f"Could not check: {e}")

# ─────────────────────────────────────────────────────────────────
# ISSUE 9: Hardcoded model names (not configurable)
# ─────────────────────────────────────────────────────────────────
print("\n--- Issue 9: Hardcoded Model Names ---")
try:
    constants_path = os.path.join(root, "app", "utils", "constants.py")
    with open(constants_path) as f:
        const_src = f.read()

    models_from_env = "os.getenv" in const_src and ("GEMINI_VISION_MODEL" in const_src or "GROQ_MODEL" in const_src)
    check(9, "Hardcoded model names", "LOW", models_from_env,
          "Models configurable via env vars" if models_from_env else
          "Model names hardcoded in constants.py — not configurable without code change")
except Exception as e:
    check(9, "Hardcoded models", "LOW", False, f"Could not check: {e}")

# ─────────────────────────────────────────────────────────────────
# ISSUE 10: Streaming response not stored on disconnect
# ─────────────────────────────────────────────────────────────────
print("\n--- Issue 10: Streaming Disconnect Handling ---")
try:
    with open(query_path) as f:
        query_src = f.read()

    has_disconnect_handler = "disconnect" in query_src.lower() or "cancel" in query_src.lower()
    has_try_in_generator = "try:" in query_src and "except" in query_src
    check(10, "Streaming disconnect handling", "MEDIUM", has_disconnect_handler,
          "Disconnect handling present" if has_disconnect_handler else
          "No disconnect handler — partial responses NOT saved to memory if user disconnects")
except Exception as e:
    check(10, "Streaming disconnect", "MEDIUM", False, f"Could not check: {e}")

# ─────────────────────────────────────────────────────────────────
# ISSUE 11: No chunking validation (empty chunks accepted)
# ─────────────────────────────────────────────────────────────────
print("\n--- Issue 11: Empty Chunk Validation ---")
try:
    with open(upload_path) as f:
        upload_src = f.read()

    has_chunk_check = "len(chunks)" in upload_src or "not chunks" in upload_src or "if chunks" in upload_src
    check(11, "No empty chunks validation", "MEDIUM", has_chunk_check,
          "Upload validates chunk count" if has_chunk_check else
          "Upload succeeds even if 0 chunks produced — user sees 'success' but nothing indexed")
except Exception as e:
    check(11, "Chunk validation", "MEDIUM", False, f"Could not check: {e}")

# ─────────────────────────────────────────────────────────────────
# ISSUE 12: Dockerfile copies .env file (security risk)
# ─────────────────────────────────────────────────────────────────
print("\n--- Issue 12: Dockerfile .env copy ---")
try:
    dockerfile_path = os.path.join(root, "Dockerfile")
    with open(dockerfile_path) as f:
        df_src = f.read()

    copies_env = "COPY .env" in df_src
    check(12, "Dockerfile copies .env (security risk)", "MEDIUM", not copies_env,
          ".env NOT copied into image" if not copies_env else
          "Dockerfile copies .env into image — API keys embedded in Docker layers")
except Exception as e:
    check(12, "Dockerfile .env", "MEDIUM", False, f"Could not check: {e}")

# ─────────────────────────────────────────────────────────────────
# ISSUE 13: No file size limit on upload
# ─────────────────────────────────────────────────────────────────
print("\n--- Issue 13: Upload File Size Limit ---")
try:
    with open(upload_path) as f:
        upload_src = f.read()

    has_size_limit = "max_size" in upload_src.lower() or "content_length" in upload_src.lower() or "size" in upload_src.lower()
    check(13, "No file size limit on upload", "LOW", has_size_limit,
          "File size checked" if has_size_limit else
          "No file size limit — huge files will cause OOM or timeout during embedding")
except Exception as e:
    check(13, "File size limit", "LOW", False, f"Could not check: {e}")

# ─────────────────────────────────────────────────────────────────
# ISSUE 14: Embeddings not batched (one API call per chunk)
# ─────────────────────────────────────────────────────────────────
print("\n--- Issue 14: Embedding Batching ---")
try:
    embeddings_path = os.path.join(root, "app", "services", "embeddings.py")
    with open(embeddings_path) as f:
        emb_src = f.read()

    # Check if embed_texts calls API once per text or batches
    has_loop = "for " in emb_src and "embed_content" in emb_src
    has_batch = "contents=" in emb_src and "[" in emb_src
    check(14, "Embeddings not batched (1 API call/chunk)", "MEDIUM", not has_loop or has_batch,
          "Embeddings appear batched" if (not has_loop or has_batch) else
          "embed_texts() loops per text — N chunks = N API calls = slow + expensive for large docs")
except Exception as e:
    check(14, "Embedding batching", "MEDIUM", False, f"Could not check: {e}")

# ─────────────────────────────────────────────────────────────────
# ISSUE 15: Gemini SDK version (google-genai vs google-generativeai)
# ─────────────────────────────────────────────────────────────────
print("\n--- Issue 15: Gemini SDK version conflict ---")
try:
    req_path = os.path.join(root, "requirements.txt")
    with open(req_path) as f:
        req_src = f.read()

    has_old_sdk = "google-generativeai" in req_src
    has_new_sdk = "google-genai" in req_src

    if has_old_sdk and has_new_sdk:
        check(15, "Both Gemini SDKs installed (conflict risk)", "MEDIUM", False,
              "Both google-generativeai AND google-genai in requirements — may cause import conflicts")
    elif has_new_sdk and not has_old_sdk:
        check(15, "Gemini SDK version", "MEDIUM", True,
              "Only new google-genai SDK — correct")
    elif has_old_sdk and not has_new_sdk:
        check(15, "Old Gemini SDK (google-generativeai)", "MEDIUM", False,
              "Only old google-generativeai SDK — may need to migrate to google-genai for newer models")
    else:
        check(15, "No Gemini SDK found", "HIGH", False,
              "Neither google-generativeai nor google-genai in requirements.txt")
except Exception as e:
    check(15, "Gemini SDK", "MEDIUM", False, f"Could not check: {e}")

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ISSUE AUDIT SUMMARY")
print("=" * 60)

passed = [i for i in issues if i[3]]
failed = [i for i in issues if not i[3]]
critical = [i for i in failed if i[2] == "HIGH"]
medium = [i for i in failed if i[2] == "MEDIUM"]
low = [i for i in failed if i[2] == "LOW"]

print(f"\nTotal issues checked: {len(issues)}")
print(f"  Passed (no issue):  {len(passed)}")
print(f"  Failed (has issue): {len(failed)}")
print(f"    HIGH severity:    {len(critical)}")
print(f"    MEDIUM severity:  {len(medium)}")
print(f"    LOW severity:     {len(low)}")

if critical:
    print(f"\n{'─'*60}")
    print("HIGH SEVERITY — FIX FIRST:")
    for i in critical:
        print(f"  #{i[0]:02d}: {i[1]}")
        print(f"        {i[4]}")

if medium:
    print(f"\n{'─'*60}")
    print("MEDIUM SEVERITY — FIX NEXT:")
    for i in medium:
        print(f"  #{i[0]:02d}: {i[1]}")
        print(f"        {i[4]}")

if low:
    print(f"\n{'─'*60}")
    print("LOW SEVERITY — FIX WHEN TIME ALLOWS:")
    for i in low:
        print(f"  #{i[0]:02d}: {i[1]}")

print("\n" + "=" * 60)
print("TEST 20 COMPLETE — Run individual tests to fix each issue")
print("=" * 60)
