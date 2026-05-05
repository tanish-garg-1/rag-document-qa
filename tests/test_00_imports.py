"""
TEST 00: IMPORT & DEPENDENCY CHECK
Tests that all required packages are installed and importable.
Run: python tests/test_00_imports.py
"""

import sys
import os

# Add project root to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("TEST 00: IMPORT & DEPENDENCY CHECK")
print("=" * 60)

failures = []
successes = []

def try_import(module_name, package_label=None):
    label = package_label or module_name
    try:
        __import__(module_name)
        print(f"  [PASS] {label}")
        successes.append(label)
    except ImportError as e:
        print(f"  [FAIL] {label} — {e}")
        failures.append((label, str(e)))

print("\n--- Standard Library ---")
try_import("os")
try_import("sys")
try_import("json")
try_import("uuid")
try_import("pathlib")
try_import("typing")
try_import("io")

print("\n--- Web Framework ---")
try_import("fastapi", "fastapi")
try_import("uvicorn", "uvicorn")
try_import("multipart", "python-multipart")
try_import("dotenv", "python-dotenv")

print("\n--- Document Processing ---")
try_import("fitz", "pymupdf (fitz)")
try_import("docx", "python-docx")
try_import("PIL", "Pillow")
try_import("PIL.Image", "PIL.Image")

print("\n--- AI / ML ---")
try_import("langchain", "langchain")
try_import("langchain_text_splitters", "langchain_text_splitters")
try_import("langchain_community", "langchain_community")
try_import("google.generativeai", "google-generativeai")
try_import("google.genai", "google-genai")
try_import("groq", "groq")
try_import("faiss", "faiss-cpu")
try_import("numpy", "numpy")

print("\n--- Frontend ---")
try_import("streamlit", "streamlit")
try_import("requests", "requests")

print("\n--- App Modules ---")
try_import("app.utils.constants", "app.utils.constants")
try_import("app.utils.file_utils", "app.utils.file_utils")
try_import("app.services.document_loader", "app.services.document_loader")
try_import("app.services.chunking", "app.services.chunking")
try_import("app.services.embeddings", "app.services.embeddings")
try_import("app.services.vector_store", "app.services.vector_store")
try_import("app.services.retriever", "app.services.retriever")
try_import("app.services.llm", "app.services.llm")
try_import("app.services.memory", "app.services.memory")
try_import("app.services.citation", "app.services.citation")
try_import("app.routes.upload", "app.routes.upload")
try_import("app.routes.query", "app.routes.query")
try_import("app.main", "app.main")

print("\n" + "=" * 60)
print(f"RESULTS: {len(successes)} passed, {len(failures)} failed")
if failures:
    print("\nFAILED IMPORTS:")
    for label, err in failures:
        print(f"  - {label}: {err}")
    print("\nFIX: pip install -r requirements.txt")
else:
    print("All imports successful!")
print("=" * 60)
