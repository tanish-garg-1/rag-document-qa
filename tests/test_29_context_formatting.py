"""
Test 29 — format_context() budget, ordering, and truncation
Tests that:
  - MAX_CHUNK_CHARS truncation works
  - MAX_CONTEXT_CHARS budget stops adding chunks
  - summaries_first=True puts summary chunks before content chunks
  - summaries_first=False puts summary chunks after content chunks
  - UUID prefixes are stripped from source names
Run: python tests/test_29_context_formatting.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.llm import format_context
from app.utils.constants import MAX_CHUNK_CHARS, MAX_CONTEXT_CHARS


def section(title):
    print(f"\n-- {title} {'-'*(55-len(title))}")


def make_chunk(source, page, content, chunk_type="text"):
    return {
        "content": content,
        "metadata": {
            "source": source,
            "page": page,
            "chunk_id": f"{source}-p{page}",
            "type": chunk_type,
        }
    }


def run():
    print(f"\n{'='*60}")
    print("Test 29 — format_context() Formatting & Budget")
    print(f"{'='*60}")
    passed = failed = 0

    # -- 1. UUID prefix stripping -----------------------------------
    section("UUID prefix stripping")
    chunks = [make_chunk("a1b2c3d4-1234-1234-1234-abcdef123456_resume.pdf", 1, "Skills: Python")]
    ctx = format_context(chunks)
    ok = "resume.pdf" in ctx and "a1b2c3d4" not in ctx
    status = "[PASS] PASS" if ok else "[FAIL] FAIL"
    print(f"  {status}  UUID prefix stripped from source name")
    print(f"  Context preview: {ctx[:100]}")
    passed += ok; failed += not ok

    # -- 2. MAX_CHUNK_CHARS truncation -----------------------------
    section(f"Chunk truncation at MAX_CHUNK_CHARS={MAX_CHUNK_CHARS}")
    long_content = "X" * (MAX_CHUNK_CHARS + 500)
    chunks = [make_chunk("test.pdf", 1, long_content)]
    ctx = format_context(chunks)
    ok = "[truncated]" in ctx
    status = "[PASS] PASS" if ok else "[FAIL] FAIL"
    print(f"  {status}  Long chunk truncated with '[truncated]' marker")
    passed += ok; failed += not ok

    # -- 3. MAX_CONTEXT_CHARS budget -------------------------------
    section(f"Context budget cap at MAX_CONTEXT_CHARS={MAX_CONTEXT_CHARS}")
    # Create enough chunks to exceed the budget
    big_chunks = [
        make_chunk("file_a.pdf", i, "A" * (MAX_CHUNK_CHARS - 10))
        for i in range(1, 30)
    ]
    ctx = format_context(big_chunks)
    ok = len(ctx) <= MAX_CONTEXT_CHARS + 500  # small tolerance for headers
    status = "[PASS] PASS" if ok else "[FAIL] FAIL"
    print(f"  {status}  Context capped (len={len(ctx)}, limit≈{MAX_CONTEXT_CHARS})")
    passed += ok; failed += not ok

    # -- 4. summaries_first=True ordering -------------------------
    section("summaries_first=True — summary before content chunks")
    chunks = [
        make_chunk("doc.pdf", 3, "Content on page 3", "text"),
        make_chunk("doc.pdf", 0, "This is the summary", "summary"),
        make_chunk("doc.pdf", 2, "Content on page 2", "text"),
    ]
    ctx = format_context(chunks, summaries_first=True)
    summary_pos = ctx.find("This is the summary")
    content_pos = ctx.find("Content on page 2")
    ok = summary_pos < content_pos
    status = "[PASS] PASS" if ok else "[FAIL] FAIL"
    print(f"  {status}  Summary at pos {summary_pos}, content at pos {content_pos}")
    passed += ok; failed += not ok

    # -- 5. summaries_first=False ordering -------------------------
    section("summaries_first=False — content before summary chunks")
    ctx = format_context(chunks, summaries_first=False)
    summary_pos = ctx.find("This is the summary")
    content_pos = ctx.find("Content on page 2")
    ok = content_pos < summary_pos
    status = "[PASS] PASS" if ok else "[FAIL] FAIL"
    print(f"  {status}  Content at pos {content_pos}, summary at pos {summary_pos}")
    passed += ok; failed += not ok

    # -- 6. Multi-source grouping ----------------------------------
    section("Multiple sources grouped with file headers")
    chunks = [
        make_chunk("resume.pdf", 1, "Python developer"),
        make_chunk("policy.pdf", 1, "Insurance clause"),
    ]
    ctx = format_context(chunks)
    ok = "--- File: resume.pdf ---" in ctx and "--- File: policy.pdf ---" in ctx
    status = "[PASS] PASS" if ok else "[FAIL] FAIL"
    print(f"  {status}  Both file headers present in context")
    passed += ok; failed += not ok

    print(f"\n{'='*60}")
    print(f"Result: {passed}/{passed+failed} passed, {failed} failed")
    print(f"{'='*60}\n")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
