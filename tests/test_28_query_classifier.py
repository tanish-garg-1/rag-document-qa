"""
Test 28 — Query classifier + rewriter logic
Tests classify_query(), rewrite_query(), and _detect_filename_mention()
without hitting the FAISS index.
Run: python tests/test_28_query_classifier.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.llm import classify_query, rewrite_query, MODE_DOCUMENT, MODE_ANALYTICAL, MODE_GENERAL
from app.routes.query import _detect_filename_mention


def section(title):
    print(f"\n-- {title} {'-'*(55-len(title))}")


def check(label, got, expected, strict=True):
    if strict:
        ok = got == expected
    else:
        ok = got in expected
    status = "[PASS] PASS" if ok else "[FAIL] FAIL"
    print(f"  {status}  {label}")
    print(f"         expected={expected}  got={got}")
    return ok


def run():
    print(f"\n{'='*60}")
    print("Test 28 — Query Classifier & Rewriter")
    print(f"{'='*60}")
    passed = failed = 0

    # -- 1. classify_query ------------------------------------------
    section("classify_query — DOCUMENT queries")
    cases_doc = [
        ("What are Tanish's skills?", True),
        ("Summarise the resume", True),
        ("What does the policy say about grace period?", True),
        ("Does the insurance cover LASIK?", True),
    ]
    for q, has_docs in cases_doc:
        mode = classify_query(q, "", has_documents=has_docs)
        ok = check(f'"{q}"', mode, MODE_DOCUMENT)
        passed += ok; failed += not ok

    section("classify_query — ANALYTICAL queries")
    cases_ana = [
        ("What job should he apply for?", True),
        ("Is the SBI policy a good deal?", True),
        ("What project should he build next?", True),
    ]
    for q, has_docs in cases_ana:
        mode = classify_query(q, "", has_documents=has_docs)
        ok = check(f'"{q}"', mode, MODE_ANALYTICAL)
        passed += ok; failed += not ok

    section("classify_query — GENERAL queries (no docs)")
    cases_gen = [
        ("Explain transformer architecture", False),
        ("What is LangChain?", False),
    ]
    for q, has_docs in cases_gen:
        mode = classify_query(q, "", has_documents=has_docs)
        ok = check(f'"{q}"', mode, MODE_GENERAL)
        passed += ok; failed += not ok

    # -- 2. rewrite_query -------------------------------------------
    section("rewrite_query — self-contained (should NOT rewrite)")
    history = "User: What are Tanish's skills?\nAssistant: Tanish knows Python and C++."

    self_contained = [
        "What is his GPA?",               # pronoun but long enough?  actually short
        "What technologies does Tanish use?",   # no pronoun, >5 words → skip
        "Summarise his work experience",   # pronoun → rewrite
    ]
    for q in self_contained:
        rewritten = rewrite_query(q, history)
        print(f"  original  : \"{q}\"")
        print(f"  rewritten : \"{rewritten}\"")
        print()

    # -- 3. _detect_filename_mention --------------------------------
    section("_detect_filename_mention")
    session_files = ["improved_wheat_logo.png", "Lhr_Driving_Lic.jpg", "Resume.pdf"]
    fn_cases = [
        ("explain the improved_wheat_logo image", ["improved_wheat_logo.png"]),
        ("what is in the driving lic", ["Lhr_Driving_Lic.jpg"]),
        ("tell me about resume",      ["Resume.pdf"]),
        ("what are all the files",    None),   # no specific file → None
        ("explain machine learning",  None),
    ]
    for q, expected in fn_cases:
        result = _detect_filename_mention(q, session_files)
        ok = result == expected
        status = "[PASS] PASS" if ok else "[FAIL] FAIL"
        print(f"  {status}  \"{q}\"")
        print(f"         expected={expected}  got={result}")
        passed += ok; failed += not ok

    print(f"\n{'='*60}")
    print(f"Result: {passed}/{passed+failed} passed, {failed} failed")
    print(f"{'='*60}\n")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
