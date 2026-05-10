"""
TEST 04: DOCUMENT LOADER — TXT FILES
Tests loading plain text documents. No API key required.
Run: python tests/test_04_document_loader_txt.py
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("TEST 04: DOCUMENT LOADER — TXT FILES")
print("=" * 60)

# --- 1. Import ---
print("\n--- Import document_loader ---")
try:
    from app.services.document_loader import load_document, load_txt
    print("  [PASS] document_loader imported")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

# --- 2. Create a temp .txt file ---
def make_txt(content, suffix=".txt"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path

# --- 3. Simple TXT load ---
print("\n--- Simple TXT Load ---")
sample_text = """Hello World!
This is a test document for the RAG system.
It contains multiple lines of text.
Each line will be part of the extracted content.
Testing the document loader functionality."""

path = make_txt(sample_text)
print(f"  Temp file: {path}")
try:
    docs = load_txt(path)
    print(f"  Documents returned: {len(docs)}")
    if len(docs) >= 1:
        print(f"  [PASS] Got {len(docs)} document(s)")
        doc = docs[0]
        print(f"  Content length: {len(doc['content'])} chars")
        print(f"  Content preview: '{doc['content'][:100]}'")
        print(f"  Metadata: {doc['metadata']}")

        # Check metadata keys
        required = {"source", "page", "type"}
        meta = doc.get("metadata", {})
        missing = required - set(meta.keys())
        if missing:
            print(f"  [FAIL] Missing metadata keys: {missing}")
        else:
            print(f"  [PASS] All required metadata keys present")

        if doc["content"].strip() == sample_text.strip():
            print(f"  [PASS] Content matches input text")
        else:
            print(f"  [WARN] Content differs from input — check encoding/stripping")
            print(f"         Input length: {len(sample_text)}, Output length: {len(doc['content'])}")

    else:
        print("  [FAIL] No documents returned from TXT file")
except Exception as e:
    print(f"  [FAIL] load_txt error: {e}")
    import traceback
    traceback.print_exc()
finally:
    os.unlink(path)

# --- 4. TXT via load_document dispatcher ---
print("\n--- load_document() Dispatcher for TXT ---")
path2 = make_txt("Testing load_document dispatcher with TXT file.")
try:
    docs = load_document(path2, "test.txt")
    print(f"  load_document returned {len(docs)} doc(s)")
    if docs:
        print(f"  [PASS] Dispatcher works for .txt")
        print(f"  Content: '{docs[0]['content'][:60]}'")
    else:
        print("  [FAIL] No docs returned")
except Exception as e:
    print(f"  [FAIL] load_document error: {e}")
    import traceback
    traceback.print_exc()
finally:
    os.unlink(path2)

# --- 5. Unicode / Special characters ---
print("\n--- Unicode & Special Characters ---")
unicode_text = "Hello 你好 Héllo Привет 🌍\nSpecial chars: <>&\"'\nTab:\there"
path3 = make_txt(unicode_text)
try:
    docs = load_txt(path3)
    if docs:
        content = docs[0]["content"]
        print(f"  [PASS] Unicode TXT loaded, length: {len(content)}")
        print(f"  Content: {ascii(content[:80])}")
        chinese_ok = "你好" in content
        cyrillic_ok = "Привет" in content
        print(f"  [{'PASS' if chinese_ok else 'WARN'}] Chinese characters {'preserved' if chinese_ok else 'may have been lost'}")
        print(f"  [{'PASS' if cyrillic_ok else 'WARN'}] Cyrillic characters {'preserved' if cyrillic_ok else 'may have been lost'}")
    else:
        print("  [FAIL] No docs from unicode text")
except Exception as e:
    print(f"  [FAIL] Unicode TXT error: {e}")
finally:
    os.unlink(path3)

# --- 6. Empty TXT file ---
print("\n--- Empty TXT File ---")
path4 = make_txt("")
try:
    docs = load_txt(path4)
    print(f"  Docs from empty file: {len(docs)}")
    if len(docs) == 0:
        print("  [PASS] Empty file returns no documents")
    elif len(docs) == 1 and docs[0]["content"] == "":
        print("  [WARN] Empty file returns 1 doc with empty content — will produce 0 chunks")
    else:
        print(f"  [INFO] Got {len(docs)} docs from empty file")
        for d in docs:
            print(f"         Content repr: {repr(d['content'][:50])}")
except Exception as e:
    print(f"  [FAIL] Empty TXT error: {e}")
finally:
    os.unlink(path4)

# --- 7. Very large TXT ---
print("\n--- Very Large TXT File ---")
large_text = ("This is line {n} of a large test document with plenty of content. " * 3 + "\n")
large_content = "".join(large_text.format(n=i) for i in range(500))
path5 = make_txt(large_content)
print(f"  Large file size: {len(large_content.encode('utf-8'))} bytes")
try:
    docs = load_txt(path5)
    print(f"  Docs returned: {len(docs)}")
    if docs:
        print(f"  Content length: {len(docs[0]['content'])}")
        print(f"  [PASS] Large TXT loaded")
    else:
        print("  [FAIL] No docs from large file")
except Exception as e:
    print(f"  [FAIL] Large TXT error: {e}")
finally:
    os.unlink(path5)

# --- 8. load_document with wrong extension ---
print("\n--- load_document with unsupported extension ---")
path6 = make_txt("Some content", suffix=".csv")
try:
    docs = load_document(path6, "data.csv")
    print(f"  Docs returned: {len(docs)}")
    print(f"  [WARN] No error raised for unsupported type — check dispatcher")
except Exception as e:
    print(f"  [PASS] Error raised for unsupported type: {type(e).__name__}: {e}")
finally:
    os.unlink(path6)

print("\n" + "=" * 60)
print("TEST 04 COMPLETE")
print("=" * 60)
