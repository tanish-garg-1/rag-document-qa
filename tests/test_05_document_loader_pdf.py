"""
TEST 05: DOCUMENT LOADER — PDF FILES
Tests PDF loading using PyMuPDF (fitz). Requires no API key for text extraction.
Image-within-PDF description requires GEMINI_API_KEY.
Run: python tests/test_05_document_loader_pdf.py
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("TEST 05: DOCUMENT LOADER — PDF FILES")
print("=" * 60)

# --- 1. Import fitz ---
print("\n--- Import PyMuPDF (fitz) ---")
try:
    import fitz
    print(f"  [PASS] PyMuPDF imported, version: {fitz.version}")
except ImportError as e:
    print(f"  [FAIL] PyMuPDF not installed: {e}")
    print("         Run: pip install pymupdf")
    sys.exit(1)

# --- 2. Import document loader ---
print("\n--- Import document_loader ---")
try:
    from app.services.document_loader import load_document, load_pdf
    print("  [PASS] document_loader imported")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

# --- 3. Create a minimal in-memory PDF ---
def create_test_pdf(pages_text):
    """Creates a PDF with given pages. Returns bytes."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

# --- 4. Single-page PDF test ---
print("\n--- Single-Page PDF ---")
pdf_bytes = create_test_pdf(["Hello from page one. This is a test PDF document for the RAG system."])
fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
with os.fdopen(fd, "wb") as f:
    f.write(pdf_bytes)

try:
    docs = load_pdf(pdf_path)
    print(f"  Documents returned: {len(docs)}")
    if len(docs) >= 1:
        print(f"  [PASS] Got {len(docs)} document(s) from 1-page PDF")
        for i, doc in enumerate(docs):
            print(f"    Doc {i}: type={doc['metadata'].get('type')}, page={doc['metadata'].get('page')}")
            print(f"           content: '{doc['content'][:80]}'")
    else:
        print("  [FAIL] No documents from single-page PDF")
except Exception as e:
    print(f"  [FAIL] load_pdf error: {e}")
    import traceback
    traceback.print_exc()
finally:
    os.unlink(pdf_path)

# --- 5. Multi-page PDF test ---
print("\n--- Multi-Page PDF ---")
pages = [
    "Page 1: Introduction to artificial intelligence and machine learning.",
    "Page 2: Deep learning and neural networks overview.",
    "Page 3: Natural language processing fundamentals.",
    "Page 4: Retrieval-augmented generation (RAG) systems.",
    "Page 5: Conclusion and future directions.",
]
pdf_bytes2 = create_test_pdf(pages)
fd, pdf_path2 = tempfile.mkstemp(suffix=".pdf")
with os.fdopen(fd, "wb") as f:
    f.write(pdf_bytes2)

try:
    docs = load_pdf(pdf_path2)
    print(f"  Pages in PDF: {len(pages)}")
    print(f"  Documents returned: {len(docs)}")

    page_nums = [d["metadata"].get("page") for d in docs]
    print(f"  Page numbers in output: {sorted(set(page_nums))}")

    text_docs = [d for d in docs if d["metadata"].get("type") == "text"]
    image_docs = [d for d in docs if d["metadata"].get("type") == "image"]
    print(f"  Text docs: {len(text_docs)}")
    print(f"  Image docs: {len(image_docs)}")

    if len(text_docs) >= len(pages):
        print(f"  [PASS] At least 1 text doc per page")
    elif len(text_docs) > 0:
        print(f"  [WARN] Got {len(text_docs)} text docs from {len(pages)} pages")
    else:
        print(f"  [FAIL] No text documents extracted")

    for i, doc in enumerate(docs[:5]):
        print(f"    Doc {i}: page={doc['metadata'].get('page')}, type={doc['metadata'].get('type')}, len={len(doc['content'])}")
        print(f"           '{doc['content'][:60]}'")

except Exception as e:
    print(f"  [FAIL] Multi-page PDF error: {e}")
    import traceback
    traceback.print_exc()
finally:
    os.unlink(pdf_path2)

# --- 6. load_document dispatcher for PDF ---
print("\n--- load_document() Dispatcher for PDF ---")
pdf_bytes3 = create_test_pdf(["Dispatcher test content."])
fd, pdf_path3 = tempfile.mkstemp(suffix=".pdf")
with os.fdopen(fd, "wb") as f:
    f.write(pdf_bytes3)
try:
    docs = load_document(pdf_path3, "test.pdf")
    if docs:
        print(f"  [PASS] Dispatcher returned {len(docs)} doc(s) for PDF")
    else:
        print("  [FAIL] Dispatcher returned no docs")
except Exception as e:
    print(f"  [FAIL] Dispatcher error: {e}")
finally:
    os.unlink(pdf_path3)

# --- 7. Empty PDF ---
print("\n--- Empty PDF ---")
doc_empty = fitz.open()
doc_empty.new_page()  # blank page
fd, pdf_path4 = tempfile.mkstemp(suffix=".pdf")
with os.fdopen(fd, "wb") as f:
    f.write(doc_empty.tobytes())
doc_empty.close()
try:
    docs = load_pdf(pdf_path4)
    print(f"  Docs from blank PDF: {len(docs)}")
    for d in docs:
        print(f"    Content repr: {repr(d['content'][:40])}, type={d['metadata'].get('type')}")
    if len(docs) == 0:
        print("  [PASS] Blank page produces no documents")
    else:
        empty_content = all(d["content"].strip() == "" for d in docs)
        if empty_content:
            print("  [WARN] Blank page produces empty-content docs — may affect chunking")
        else:
            print("  [INFO] Got docs from blank page with non-empty content")
except Exception as e:
    print(f"  [FAIL] Empty PDF error: {e}")
finally:
    os.unlink(pdf_path4)

# --- 8. PDF metadata keys check ---
print("\n--- Metadata Keys Check ---")
pdf_bytes5 = create_test_pdf(["Metadata test page."])
fd, pdf_path5 = tempfile.mkstemp(suffix=".pdf")
with os.fdopen(fd, "wb") as f:
    f.write(pdf_bytes5)
try:
    docs = load_pdf(pdf_path5)
    required_keys = {"source", "page", "type"}
    for i, doc in enumerate(docs):
        meta = doc.get("metadata", {})
        missing = required_keys - set(meta.keys())
        if missing:
            print(f"  [FAIL] Doc {i} missing metadata: {missing}")
        else:
            print(f"  [PASS] Doc {i} has all metadata: {meta}")
        if "content" not in doc:
            print(f"  [FAIL] Doc {i} missing 'content' key")
        else:
            print(f"  [PASS] Doc {i} has 'content' key")
except Exception as e:
    print(f"  [FAIL] Metadata check error: {e}")
finally:
    os.unlink(pdf_path5)

print("\n" + "=" * 60)
print("TEST 05 COMPLETE")
print("=" * 60)
