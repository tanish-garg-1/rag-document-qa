"""
TEST 11: EMBEDDINGS API (GEMINI)
Tests the Gemini embedding service. REQUIRES GEMINI_API_KEY in .env.
Run: python tests/test_11_embeddings_api.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("TEST 11: EMBEDDINGS API (GEMINI)")
print("=" * 60)

# --- 1. Check API key ---
print("\n--- API Key Check ---")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("  [SKIP] GEMINI_API_KEY not set — skipping live API tests")
    print("         Add GEMINI_API_KEY to .env and rerun")
    print("\n" + "=" * 60)
    print("TEST 11 SKIPPED (no API key)")
    print("=" * 60)
    sys.exit(0)
else:
    masked = api_key[:6] + "..." + api_key[-4:]
    print(f"  [PASS] GEMINI_API_KEY found: {masked}")

# --- 2. Import embeddings ---
print("\n--- Import embeddings ---")
try:
    from app.services.embeddings import embed_texts, embed_query
    print("  [PASS] embeddings module imported")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

from app.utils.constants import GEMINI_EMBEDDING_MODEL, EMBEDDING_DIM
print(f"  [INFO] Model: {GEMINI_EMBEDDING_MODEL}")
print(f"  [INFO] Expected dim: {EMBEDDING_DIM}")

# --- 3. Single text embedding ---
print("\n--- embed_query() Single Text ---")
test_query = "What is retrieval-augmented generation?"
try:
    import time
    t0 = time.time()
    embedding = embed_query(test_query)
    elapsed = time.time() - t0

    print(f"  Query: '{test_query}'")
    print(f"  API call time: {elapsed:.2f}s")
    print(f"  Embedding type: {type(embedding)}")
    print(f"  Embedding length: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    print(f"  Last 5 values: {embedding[-5:]}")

    if len(embedding) == EMBEDDING_DIM:
        print(f"  [PASS] Embedding dimension matches EMBEDDING_DIM ({EMBEDDING_DIM})")
    else:
        print(f"  [FAIL] Embedding dim {len(embedding)} != expected {EMBEDDING_DIM}")
        print(f"         Update EMBEDDING_DIM in constants.py to match actual dim")

    if all(isinstance(v, float) for v in embedding[:10]):
        print(f"  [PASS] Embedding values are floats")
    else:
        print(f"  [WARN] Some values may not be floats")

    # Check magnitude (roughly unit-normalized or within reasonable range)
    import math
    mag = math.sqrt(sum(v**2 for v in embedding))
    print(f"  Vector magnitude: {mag:.4f}")
    if 0.5 < mag < 2.0:
        print(f"  [PASS] Magnitude in expected range")
    else:
        print(f"  [WARN] Unusual magnitude: {mag}")

except Exception as e:
    print(f"  [FAIL] embed_query error: {e}")
    import traceback
    traceback.print_exc()

# --- 4. embed_texts batch ---
print("\n--- embed_texts() Multiple Texts ---")
texts = [
    "Machine learning is a subset of AI.",
    "Natural language processing handles text data.",
    "Vector databases store embeddings for similarity search.",
]
try:
    import time
    t0 = time.time()
    embeddings = embed_texts(texts)
    elapsed = time.time() - t0

    print(f"  Input texts: {len(texts)}")
    print(f"  API call time: {elapsed:.2f}s")
    print(f"  Embeddings returned: {len(embeddings)}")

    if len(embeddings) == len(texts):
        print(f"  [PASS] Correct count of embeddings returned")
    else:
        print(f"  [FAIL] Expected {len(texts)}, got {len(embeddings)}")

    for i, emb in enumerate(embeddings):
        print(f"    Embedding {i}: dim={len(emb)}, first_val={emb[0]:.6f}")
        if len(emb) != EMBEDDING_DIM:
            print(f"    [FAIL] Unexpected dim for embedding {i}: {len(emb)}")

except Exception as e:
    print(f"  [FAIL] embed_texts error: {e}")
    import traceback
    traceback.print_exc()

# --- 5. Similarity sanity check ---
print("\n--- Embedding Similarity Sanity Check ---")
try:
    import numpy as np

    emb1 = embed_query("What is machine learning?")
    emb2 = embed_query("Machine learning is a field of AI.")  # similar
    emb3 = embed_query("What is the weather today?")  # dissimilar

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_12 = cosine_sim(emb1, emb2)
    sim_13 = cosine_sim(emb1, emb3)

    print(f"  Query 1: 'What is machine learning?'")
    print(f"  Query 2: 'Machine learning is a field of AI.'  (similar)")
    print(f"  Query 3: 'What is the weather today?'          (dissimilar)")
    print(f"  Cosine sim(1,2) = {sim_12:.4f}  (should be HIGH)")
    print(f"  Cosine sim(1,3) = {sim_13:.4f}  (should be LOW)")

    if sim_12 > sim_13:
        print("  [PASS] Similar texts have higher similarity score")
    else:
        print("  [FAIL] Similarity ordering incorrect — embeddings may not work correctly")

    if sim_12 > 0.7:
        print(f"  [PASS] Similar pair cosine similarity > 0.7")
    else:
        print(f"  [WARN] Similar pair cosine similarity low: {sim_12:.4f}")

except Exception as e:
    print(f"  [FAIL] Similarity check error: {e}")
    import traceback
    traceback.print_exc()

# --- 6. Empty text embedding ---
print("\n--- Empty Text Embedding ---")
try:
    emb = embed_query("")
    print(f"  Empty text embedding length: {len(emb)}")
    print(f"  [WARN] Empty text produced embedding — check if this causes issues")
except Exception as e:
    print(f"  [INFO] Empty text raises error: {type(e).__name__}: {e}")
    print(f"  [INFO] This is expected behavior — empty text cannot be embedded")

# --- 7. API model name check ---
print("\n--- Model Name Validation ---")
print(f"  Using model: {GEMINI_EMBEDDING_MODEL}")
known_valid_models = [
    "models/text-embedding-004",
    "models/gemini-embedding-exp-03-07",
    "models/gemini-embedding-2",
]
print(f"  Known valid Gemini embedding models:")
for m in known_valid_models:
    print(f"    {m}")
if GEMINI_EMBEDDING_MODEL in known_valid_models:
    print(f"  [PASS] Model name is a known valid model")
else:
    print(f"  [WARN] Model name '{GEMINI_EMBEDDING_MODEL}' not in known list — may fail or be new")

print("\n" + "=" * 60)
print("TEST 11 COMPLETE")
print("=" * 60)
