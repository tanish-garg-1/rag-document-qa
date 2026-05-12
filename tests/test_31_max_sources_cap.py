"""
Test 31 — MAX_SOURCES cap behaviour
Verifies:
  1. Normal queries cap at MAX_SOURCES=3 files
  2. include_all_sources=True bypasses the cap (describe-all)
  3. source_filter=N files with skip_sim_gates returns all N files
Run: python tests/test_31_max_sources_cap.py
"""
import sys, os, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_chunk(source, page, content, sim=0.8):
    """sim controls the fake embedding magnitude to simulate cosine similarity."""
    import numpy as np
    # Embeddings are unit vectors; cosine sim against query depends on direction.
    # For this test we just need them non-zero.
    emb = np.full(10, sim / 10.0, dtype=float)
    return {
        "content": content,
        "metadata": {
            "source": source,
            "page": page,
            "chunk_id": str(uuid.uuid4()),
            "type": "text",
        },
        "embedding": list(emb)
    }


def run():
    import numpy as np
    import unittest.mock as mock

    # Pre-import so mock.patch can resolve "app.services.retriever.*" as attributes
    import app.services.retriever  # noqa: F401
    from app.services.retriever import retrieve  # noqa: F401

    print(f"\n{'='*60}")
    print("Test 31 — MAX_SOURCES Cap Behaviour")
    print(f"{'='*60}\n")
    passed = failed = 0

    def section(title):
        print(f"-- {title} {'-'*(55-len(title))}")

    # 5 fake files — more than MAX_SOURCES=3
    file_names = [
        "Resume.pdf",
        "SBI_Policy.pdf",
        "Blue_Presentation.pdf",
        "FY23_Resume.pdf",
        "Wheat_Logo.png",
    ]
    all_chunks = []
    for fname in file_names:
        all_chunks.append(make_chunk(fname, 0, f"Summary of {fname}", sim=0.75))
        all_chunks.append(make_chunk(fname, 1, f"Content of {fname}", sim=0.72))

    query_embedding = list(np.ones(10, dtype=float) * 0.1)
    candidates = [c.copy() for c in all_chunks]

    # -- Test 1: Normal query caps at MAX_SOURCES -------------------
    section("Normal query (include_all_sources=False) caps at MAX_SOURCES=3")
    with mock.patch("app.services.retriever.embed_query", return_value=query_embedding), \
         mock.patch("app.services.retriever.search_similar", return_value=candidates), \
         mock.patch("app.services.retriever.get_all_chunks", return_value=all_chunks):
        results = retrieve("what are Tanish's skills", source_filter=None)

    sources = {c["metadata"]["source"] for c in results}
    print(f"  Files in results: {len(sources)} — {sources}")
    ok = len(sources) <= 3
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {status}  Capped at ≤3 sources (got {len(sources)})")
    passed += ok; failed += not ok

    # -- Test 2: include_all_sources=True bypasses cap --------------
    print()
    section("describe-all (include_all_sources=True) bypasses MAX_SOURCES cap")
    with mock.patch("app.services.retriever.embed_query", return_value=query_embedding), \
         mock.patch("app.services.retriever.search_similar", return_value=candidates), \
         mock.patch("app.services.retriever.get_all_chunks", return_value=all_chunks):
        results_all = retrieve(
            "describe all files",
            include_all_sources=True,
            source_filter=file_names
        )

    sources_all = {c["metadata"]["source"] for c in results_all}
    print(f"  Files in results: {len(sources_all)} — {sources_all}")
    ok2 = len(sources_all) == 5
    status = "✅ PASS" if ok2 else "❌ FAIL"
    print(f"  {status}  All 5 sources returned (got {len(sources_all)})")
    passed += ok2; failed += not ok2

    # -- Test 3: source_filter with skip_sim_gates returns all ------
    print()
    section("source_filter set → sim thresholds bypassed → all scoped files returned")
    with mock.patch("app.services.retriever.embed_query", return_value=query_embedding), \
         mock.patch("app.services.retriever.search_similar", return_value=candidates), \
         mock.patch("app.services.retriever.get_all_chunks", return_value=all_chunks):
        results_scoped = retrieve(
            "explain the image",   # vague query — low similarity expected
            source_filter=["Wheat_Logo.png"]
        )

    sources_scoped = {c["metadata"]["source"] for c in results_scoped}
    print(f"  Files in results: {sources_scoped}")
    ok3 = "Wheat_Logo.png" in sources_scoped
    status = "✅ PASS" if ok3 else "❌ FAIL"
    print(f"  {status}  Wheat_Logo.png returned despite vague query")
    passed += ok3; failed += not ok3

    print(f"\n{'='*60}")
    print(f"Result: {passed}/{passed+failed} passed, {failed} failed")
    print(f"{'='*60}\n")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
