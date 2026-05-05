"""
TEST 12: LLM — GROQ API
Tests query rewriting and streaming answer generation. REQUIRES GROQ_API_KEY.
Run: python tests/test_12_llm_groq.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("TEST 12: LLM — GROQ API")
print("=" * 60)

# --- 1. Check API key ---
print("\n--- API Key Check ---")
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    print("  [SKIP] GROQ_API_KEY not set — skipping live API tests")
    print("         Add GROQ_API_KEY to .env and rerun")
    print("\n" + "=" * 60)
    print("TEST 12 SKIPPED (no API key)")
    print("=" * 60)
    sys.exit(0)
else:
    masked = groq_key[:6] + "..." + groq_key[-4:]
    print(f"  [PASS] GROQ_API_KEY found: {masked}")

# --- 2. Import groq ---
print("\n--- Import Groq SDK ---")
try:
    from groq import Groq
    print("  [PASS] Groq SDK imported")
except ImportError as e:
    print(f"  [FAIL] Groq not installed: {e}")
    print("         Run: pip install groq")
    sys.exit(1)

print("\n--- Import llm module ---")
try:
    from app.services.llm import rewrite_query, stream_answer
    print("  [PASS] llm module imported")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

from app.utils.constants import GROQ_MODEL
print(f"  [INFO] GROQ_MODEL = {GROQ_MODEL}")

# --- 3. rewrite_query with empty history ---
print("\n--- rewrite_query() with empty history ---")
query = "What is machine learning?"
history = ""
try:
    import time
    t0 = time.time()
    rewritten = rewrite_query(query, history)
    elapsed = time.time() - t0
    print(f"  Input query:    '{query}'")
    print(f"  Rewritten query: '{rewritten}'")
    print(f"  Time: {elapsed:.2f}s")
    if rewritten == query:
        print("  [PASS] Empty history returns original query (expected shortcut)")
    else:
        print("  [INFO] Query was rewritten even with empty history")
except Exception as e:
    print(f"  [FAIL] rewrite_query error: {e}")
    import traceback
    traceback.print_exc()

# --- 4. rewrite_query with context ---
print("\n--- rewrite_query() with conversation history ---")
history_text = "User: What is AI?\nAssistant: AI stands for artificial intelligence, the simulation of human intelligence by machines."
followup = "How is it used?"
try:
    t0 = time.time()
    rewritten = rewrite_query(followup, history_text)
    elapsed = time.time() - t0
    print(f"  History: '{history_text[:80]}...'")
    print(f"  Follow-up: '{followup}'")
    print(f"  Rewritten: '{rewritten}'")
    print(f"  Time: {elapsed:.2f}s")
    if "AI" in rewritten or "artificial intelligence" in rewritten.lower():
        print("  [PASS] Rewritten query incorporates context")
    elif followup not in rewritten:
        print("  [PASS] Query was transformed (rewritten)")
    else:
        print("  [INFO] Query unchanged — may be expected or query rewriting failed")
except Exception as e:
    print(f"  [FAIL] rewrite_query with history error: {e}")
    import traceback
    traceback.print_exc()

# --- 5. stream_answer basic ---
print("\n--- stream_answer() Basic Test ---")
test_chunks = [
    {
        "content": "Retrieval-Augmented Generation (RAG) is a technique that combines document retrieval with language model generation. It first retrieves relevant context from a knowledge base, then uses that context to generate accurate answers.",
        "metadata": {"source": "rag_guide.pdf", "page": 1, "chunk_id": "chunk_001", "type": "text"}
    },
    {
        "content": "RAG systems use vector embeddings to find semantically similar documents. FAISS is commonly used as the vector store for efficient similarity search.",
        "metadata": {"source": "rag_guide.pdf", "page": 2, "chunk_id": "chunk_002", "type": "text"}
    },
]
question = "What is RAG and how does it work?"
history = ""
try:
    import time
    print(f"  Question: '{question}'")
    print(f"  Chunks: {len(test_chunks)}")
    print(f"  Streaming response:")
    print("  " + "-" * 40)

    t0 = time.time()
    full_response = ""
    token_count = 0

    for token in stream_answer(question, test_chunks, history):
        full_response += token
        token_count += 1
        print(token, end="", flush=True)

    elapsed = time.time() - t0
    print()
    print("  " + "-" * 40)
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Total tokens: {token_count}")
    print(f"  Total length: {len(full_response)} chars")

    if full_response.strip():
        print("  [PASS] Got non-empty streaming response")
    else:
        print("  [FAIL] Empty streaming response")

    if "RAG" in full_response or "retrieval" in full_response.lower():
        print("  [PASS] Response mentions RAG (context used)")
    else:
        print("  [WARN] Response may not have used the provided context")

    if token_count > 1:
        print("  [PASS] Response was streamed in multiple tokens")
    else:
        print("  [WARN] Only 1 token received — streaming may not be working")

except Exception as e:
    print(f"\n  [FAIL] stream_answer error: {e}")
    import traceback
    traceback.print_exc()

# --- 6. stream_answer with history ---
print("\n--- stream_answer() with conversation history ---")
hist = "User: What is FAISS?\nAssistant: FAISS is a library for efficient similarity search on dense vectors."
followup = "How is it used in RAG?"
try:
    print(f"  Follow-up question with history:")
    print(f"  History: '{hist[:60]}...'")
    print(f"  Question: '{followup}'")
    print("  Response:")
    print("  " + "-" * 40)
    full = ""
    for token in stream_answer(followup, test_chunks, hist):
        full += token
        print(token, end="", flush=True)
    print()
    print("  " + "-" * 40)
    print(f"  Response length: {len(full)} chars")
    if full.strip():
        print("  [PASS] Got response with history context")
    else:
        print("  [FAIL] Empty response with history")
except Exception as e:
    print(f"\n  [FAIL] stream_answer with history error: {e}")

# --- 7. stream_answer with no chunks ---
print("\n--- stream_answer() with no context chunks ---")
try:
    print(f"  Question: 'What is the capital of France?'")
    print("  Response (no context):")
    print("  " + "-" * 40)
    full = ""
    for token in stream_answer("What is the capital of France?", [], ""):
        full += token
        print(token, end="", flush=True)
    print()
    print("  " + "-" * 40)
    print(f"  Response: '{full[:100]}'")
    if full.strip():
        print("  [PASS] LLM responds even with no context (prompt instructs this)")
    else:
        print("  [FAIL] Empty response")
except Exception as e:
    print(f"\n  [FAIL] No-chunks stream_answer error: {e}")

# --- 8. Groq model check ---
print("\n--- Groq Model Configuration ---")
print(f"  Model: {GROQ_MODEL}")
known_groq_models = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "llama3-70b-8192",
    "gemma2-9b-it",
]
if GROQ_MODEL in known_groq_models:
    print(f"  [PASS] Model is a known Groq model")
else:
    print(f"  [WARN] Model '{GROQ_MODEL}' not in known list — may work or be new")

print("\n" + "=" * 60)
print("TEST 12 COMPLETE")
print("=" * 60)
