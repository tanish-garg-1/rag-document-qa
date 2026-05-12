"""
Test 27 — _DESCRIBE_ALL_RE regex robustness
Checks that describe-all intent is detected correctly including typos,
bare phrases, and edge cases that should NOT match.
Run: python tests/test_27_describe_all_regex.py
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.routes.query import _DESCRIBE_ALL_RE

SHOULD_MATCH = [
    # Standard
    "describe all the files",
    "describe my documents",
    "summarize all docs",
    "explain the files",
    "list all documents",
    "tell me about my files",
    "what are the files",
    "what have i uploaded",
    "what did i upload",
    "show me all documents",
    # Bare phrases (no verb)
    "all the files",
    "all my docs",
    "all files",
    "all documents",
    "every file",
    "each document",
    # Variations
    "give me an overview of the files",
    "give me a summary of my docs",
    "walk me through the files",
    "what files are indexed",
    "what files have been uploaded",
]

SHOULD_NOT_MATCH = [
    # Specific file queries — must NOT trigger describe-all
    "what does the resume say",
    "explain transformer architecture",
    "what is AML",
    "summarize the SBI policy",
    "explain the policy exclusions",
    "describe Tanish's skills",
    "what is in the driving license",
    "tell me about his internship",
    "explain machine learning",
    "what is RAG",
]


def run():
    print(f"\n{'='*60}")
    print("Test 27 — _DESCRIBE_ALL_RE Regex Robustness")
    print(f"{'='*60}\n")

    passed = failed = 0

    print("-- SHOULD MATCH (describe-all intent) ----------------------")
    for q in SHOULD_MATCH:
        match = bool(_DESCRIBE_ALL_RE.search(q))
        ok = match
        status = "[PASS] PASS" if ok else "[FAIL] FAIL"
        if not ok: failed += 1
        else: passed += 1
        print(f"  {status}  [{('MATCH' if match else 'no-match')}]  \"{q}\"")

    print("\n-- SHOULD NOT MATCH (specific queries) ---------------------")
    for q in SHOULD_NOT_MATCH:
        match = bool(_DESCRIBE_ALL_RE.search(q))
        ok = not match
        status = "[PASS] PASS" if ok else "[FAIL] FAIL"
        if not ok: failed += 1
        else: passed += 1
        print(f"  {status}  [{('MATCH' if match else 'no-match')}]  \"{q}\"")

    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Result: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}\n")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
