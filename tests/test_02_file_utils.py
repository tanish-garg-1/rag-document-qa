"""
TEST 02: FILE UTILITIES
Tests directory creation, file saving, extension detection, and type validation.
Run: python tests/test_02_file_utils.py
"""

import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("TEST 02: FILE UTILITIES")
print("=" * 60)

# --- 1. Import ---
print("\n--- Import file_utils ---")
try:
    from app.utils.file_utils import (
        ensure_directories,
        save_uploaded_file,
        get_file_extension,
        is_supported_file,
    )
    print("  [PASS] file_utils imported")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

# --- 2. ensure_directories ---
print("\n--- ensure_directories() ---")
try:
    ensure_directories()
    from app.utils.constants import UPLOAD_DIR, FAISS_INDEX_DIR
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    upload_path = os.path.join(root, UPLOAD_DIR)
    faiss_path = os.path.join(root, FAISS_INDEX_DIR)

    if os.path.isdir(upload_path):
        print(f"  [PASS] UPLOAD_DIR exists: {upload_path}")
    else:
        print(f"  [FAIL] UPLOAD_DIR not created: {upload_path}")

    if os.path.isdir(faiss_path):
        print(f"  [PASS] FAISS_INDEX_DIR exists: {faiss_path}")
    else:
        print(f"  [FAIL] FAISS_INDEX_DIR not created: {faiss_path}")

except Exception as e:
    print(f"  [FAIL] ensure_directories error: {e}")
    import traceback
    traceback.print_exc()

# --- 3. get_file_extension ---
print("\n--- get_file_extension() ---")
test_cases = [
    ("document.pdf", "pdf"),
    ("report.PDF", "pdf"),
    ("data.DOCX", "docx"),
    ("image.JPG", "jpg"),
    ("photo.webp", "webp"),
    ("file.txt", "txt"),
    ("no_extension", ""),
    ("multiple.dots.file.png", "png"),
]
ext_pass = 0
ext_fail = 0
for filename, expected in test_cases:
    try:
        result = get_file_extension(filename)
        if result == expected:
            print(f"  [PASS] '{filename}' → '{result}'")
            ext_pass += 1
        else:
            print(f"  [FAIL] '{filename}' → got '{result}', expected '{expected}'")
            ext_fail += 1
    except Exception as e:
        print(f"  [FAIL] '{filename}' → exception: {e}")
        ext_fail += 1

print(f"         Extension tests: {ext_pass} passed, {ext_fail} failed")

# --- 4. is_supported_file ---
print("\n--- is_supported_file() ---")
supported_cases = [
    ("doc.pdf", True),
    ("doc.docx", True),
    ("doc.txt", True),
    ("img.png", True),
    ("img.jpg", True),
    ("img.jpeg", True),
    ("img.webp", True),
    ("data.csv", False),
    ("script.py", False),
    ("archive.zip", False),
    ("data.xlsx", False),
]
sup_pass = 0
sup_fail = 0
for filename, expected in supported_cases:
    try:
        result = is_supported_file(filename)
        status = "[PASS]" if result == expected else "[FAIL]"
        label = "supported" if result else "unsupported"
        if result == expected:
            print(f"  [PASS] '{filename}' → {label}")
            sup_pass += 1
        else:
            print(f"  [FAIL] '{filename}' → got {label}, expected {'supported' if expected else 'unsupported'}")
            sup_fail += 1
    except Exception as e:
        print(f"  [FAIL] '{filename}' → exception: {e}")
        sup_fail += 1

print(f"         Supported-type tests: {sup_pass} passed, {sup_fail} failed")

# --- 5. save_uploaded_file ---
print("\n--- save_uploaded_file() ---")
try:
    test_content = b"Hello, this is test file content for RAG system testing."
    test_filename = "test_upload.txt"
    saved_path = save_uploaded_file(test_content, test_filename)

    print(f"  Saved to: {saved_path}")
    if os.path.exists(saved_path):
        print(f"  [PASS] File saved successfully")
        size = os.path.getsize(saved_path)
        print(f"  [PASS] File size: {size} bytes (expected {len(test_content)})")
        if size == len(test_content):
            print(f"  [PASS] File size matches")
        else:
            print(f"  [FAIL] File size mismatch")

        content_read = open(saved_path, "rb").read()
        if content_read == test_content:
            print(f"  [PASS] File content matches")
        else:
            print(f"  [FAIL] File content mismatch")

        # Check UUID prefix
        basename = os.path.basename(saved_path)
        print(f"  [INFO] Saved filename: {basename}")
        if "_" in basename and len(basename) > len(test_filename):
            print(f"  [PASS] UUID prefix added to filename")
        else:
            print(f"  [WARN] No UUID prefix detected — collision risk")

        # Cleanup
        os.remove(saved_path)
        print(f"  [INFO] Test file cleaned up")
    else:
        print(f"  [FAIL] File was not created at {saved_path}")

except Exception as e:
    print(f"  [FAIL] save_uploaded_file error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 02 COMPLETE")
print("=" * 60)
