"""
TEST 01: ENVIRONMENT & CONFIGURATION CHECK
Tests .env file, API keys presence, constants, and directory setup.
Run: python tests/test_01_env_config.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("TEST 01: ENVIRONMENT & CONFIGURATION CHECK")
print("=" * 60)

# --- 1. .env file existence ---
print("\n--- .env File ---")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(root, ".env")
env_example_path = os.path.join(root, ".env.example")

if os.path.exists(env_path):
    print(f"  [PASS] .env file found at: {env_path}")
else:
    print(f"  [FAIL] .env file NOT found at: {env_path}")
    print("         Create it from .env.example and add your API keys")

if os.path.exists(env_example_path):
    print(f"  [PASS] .env.example found at: {env_example_path}")
    print("         Contents of .env.example:")
    with open(env_example_path) as f:
        for line in f:
            print(f"           {line.rstrip()}")
else:
    print(f"  [WARN] .env.example not found")

# --- 2. Load dotenv ---
print("\n--- Loading .env ---")
try:
    from dotenv import load_dotenv
    load_dotenv(env_path)
    print("  [PASS] dotenv loaded successfully")
except Exception as e:
    print(f"  [FAIL] dotenv load error: {e}")

# --- 3. API Keys ---
print("\n--- API Keys ---")
gemini_key = os.getenv("GEMINI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if gemini_key:
    masked = gemini_key[:6] + "..." + gemini_key[-4:] if len(gemini_key) > 10 else "***"
    print(f"  [PASS] GEMINI_API_KEY is set — value: {masked}")
else:
    print("  [FAIL] GEMINI_API_KEY is NOT set or empty")
    print("         Add GEMINI_API_KEY=your_key to .env")

if groq_key:
    masked = groq_key[:6] + "..." + groq_key[-4:] if len(groq_key) > 10 else "***"
    print(f"  [PASS] GROQ_API_KEY is set — value: {masked}")
else:
    print("  [FAIL] GROQ_API_KEY is NOT set or empty")
    print("         Add GROQ_API_KEY=your_key to .env")

# --- 4. Constants module ---
print("\n--- Constants Module ---")
try:
    from app.utils.constants import (
        GEMINI_API_KEY, GROQ_API_KEY,
        UPLOAD_DIR, FAISS_INDEX_DIR,
        CHUNK_SIZE, CHUNK_OVERLAP,
        MMR_K, MMR_LAMBDA,
        MAX_MEMORY_MESSAGES, INFERENCE_MEMORY_MESSAGES,
        GEMINI_EMBEDDING_MODEL, GEMINI_VISION_MODEL, GROQ_MODEL,
        EMBEDDING_DIM
    )
    print("  [PASS] constants.py imported successfully")

    print(f"\n  UPLOAD_DIR            = {UPLOAD_DIR}")
    print(f"  FAISS_INDEX_DIR       = {FAISS_INDEX_DIR}")
    print(f"  CHUNK_SIZE            = {CHUNK_SIZE}")
    print(f"  CHUNK_OVERLAP         = {CHUNK_OVERLAP}")
    print(f"  MMR_K                 = {MMR_K}")
    print(f"  MMR_LAMBDA            = {MMR_LAMBDA}")
    print(f"  MAX_MEMORY_MESSAGES   = {MAX_MEMORY_MESSAGES}")
    print(f"  INFERENCE_MEMORY_MSGS = {INFERENCE_MEMORY_MESSAGES}")
    print(f"  GEMINI_EMBEDDING_MODEL= {GEMINI_EMBEDDING_MODEL}")
    print(f"  GEMINI_VISION_MODEL   = {GEMINI_VISION_MODEL}")
    print(f"  GROQ_MODEL            = {GROQ_MODEL}")
    print(f"  EMBEDDING_DIM         = {EMBEDDING_DIM}")

    if GEMINI_API_KEY:
        print(f"\n  [PASS] constants.GEMINI_API_KEY is set")
    else:
        print(f"\n  [FAIL] constants.GEMINI_API_KEY is None (API key not loaded into constants)")

    if GROQ_API_KEY:
        print(f"  [PASS] constants.GROQ_API_KEY is set")
    else:
        print(f"  [FAIL] constants.GROQ_API_KEY is None (API key not loaded into constants)")

except ImportError as e:
    print(f"  [FAIL] Cannot import constants: {e}")
except Exception as e:
    print(f"  [FAIL] Unexpected error loading constants: {e}")

# --- 5. Directory structure ---
print("\n--- Directory Structure ---")
expected_dirs = [
    "app",
    "app/routes",
    "app/services",
    "app/utils",
    "frontend",
    "data",
    "data/uploads",
    "data/faiss_index",
]
for d in expected_dirs:
    full = os.path.join(root, d)
    if os.path.isdir(full):
        print(f"  [PASS] {d}/")
    else:
        print(f"  [MISS] {d}/ — will be created at runtime")

# --- 6. Requirements file ---
print("\n--- Requirements File ---")
req_path = os.path.join(root, "requirements.txt")
if os.path.exists(req_path):
    print(f"  [PASS] requirements.txt found")
    with open(req_path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    print(f"         {len(lines)} packages listed:")
    for l in lines:
        print(f"           {l}")
else:
    print(f"  [FAIL] requirements.txt not found")

print("\n" + "=" * 60)
print("TEST 01 COMPLETE")
print("=" * 60)
