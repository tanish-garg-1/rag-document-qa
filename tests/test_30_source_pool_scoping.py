"""
Test 30 — source_pool scoping fix verification
Verifies that when source_filter is provided, the retriever's
source_pool, summary-guarantee pass, and coverage pass ONLY
operate on allowed files — not the full index.

This test does NOT hit the FAISS index. It patches get_all_chunks()
and search_similar() with fake data to isolate the logic.
Run: python tests/test_30_source_pool_scoping.py
"""
import sys, os, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_chunk(source, page, content, chunk_type="text", embedding=None):
    import numpy as np
    return {
        "content": content,
        "metadata": {
            "source": source,
            "page": page,
            "chunk_id": str(uuid.uuid4()),
            "type": chunk_type,
        },
        "embedding": embedding if embedding is not None else list(np.random.rand(10).astype(float))
    }


def run():
    print(f"\n{'='*60}")
    print("Test 30 — source_pool Scoping Fix")
    print(f"{'='*60}\n")
    passed = failed = 0

    import numpy as np
    import unittest.mock as mock

    # Pre-import so mock.patch can resolve "app.services.retriever.*" as attributes
    import app.services.retriever  # noqa: F401
    from app.services.retriever import retrieve  # noqa: F401

    # Fake chunks for 3 "indexed" files
    resume_chunks = [
        make_chunk("Resume.pdf", 0, "Summary of resume", "summary"),
        make_chunk("Resume.pdf", 1, "Skills: Python, C++"),
    ]
    policy_chunks = [
        make_chunk("SBI_Policy.pdf", 0, "Insurance summary", "summary"),
        make_chunk("SBI_Policy.pdf", 1, "Exclusion clause content"),
    ]
    ml_paper_chunks = [
        make_chunk("transformer_paper.pdf", 0, "ML paper summary", "summary"),
        make_chunk("ml_paper.pdf", 1, "Transformer architecture"),
    ]
    all_chunks = resume_chunks + policy_chunks + ml_paper_chunks

    # Query embedding (random, doesn't matter for this logic test)
    query_embedding = list(np.random.rand(10).astype(float))

    # Candidates returned by FAISS (all files, unfiltered)
    candidates = [c.copy() for c in all_chunks]

    def section(title):
        print(f"-- {title} {'-'*(55-len(title))}")

    # -- Test: source_filter restricts source_pool -----------------
    section("source_filter restricts source_pool to session files")

    with mock.patch("app.services.retriever.embed_query", return_value=query_embedding), \
         mock.patch("app.services.retriever.search_similar", return_value=candidates), \
         mock.patch("app.services.retriever.get_all_chunks", return_value=all_chunks):

        # Only allow Resume.pdf — transformer_paper should NEVER appear
        results = retrieve(
            "what are the skills",
            source_filter=["Resume.pdf"]
        )

    result_sources = {c["metadata"]["source"] for c in results}
    print(f"  Source filter: ['Resume.pdf']")
    print(f"  Sources in results: {result_sources}")

    ok_no_policy = "SBI_Policy.pdf" not in result_sources
    ok_no_ml = "transformer_paper.pdf" not in result_sources and "ml_paper.pdf" not in result_sources
    ok_has_resume = "Resume.pdf" in result_sources

    for label, ok in [
        ("SBI_Policy.pdf excluded", ok_no_policy),
        ("transformer_paper.pdf excluded", ok_no_ml),
        ("Resume.pdf included", ok_has_resume),
    ]:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {label}")
        passed += ok; failed += not ok

    # -- Test: describe-all with source_filter ---------------------
    print()
    section("describe-all (include_all_sources=True) still respects source_filter")

    with mock.patch("app.services.retriever.embed_query", return_value=query_embedding), \
         mock.patch("app.services.retriever.search_similar", return_value=candidates), \
         mock.patch("app.services.retriever.get_all_chunks", return_value=all_chunks):

        results_all = retrieve(
            "describe all files",
            include_all_sources=True,
            source_filter=["Resume.pdf", "SBI_Policy.pdf"]
        )

    result_sources_all = {c["metadata"]["source"] for c in results_all}
    print(f"  Source filter: ['Resume.pdf', 'SBI_Policy.pdf']")
    print(f"  Sources in results: {result_sources_all}")

    ok_resume = "Resume.pdf" in result_sources_all
    ok_policy = "SBI_Policy.pdf" in result_sources_all
    ok_no_ml2 = "transformer_paper.pdf" not in result_sources_all

    for label, ok in [
        ("Resume.pdf included", ok_resume),
        ("SBI_Policy.pdf included", ok_policy),
        ("transformer_paper.pdf excluded even with include_all_sources", ok_no_ml2),
    ]:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {label}")
        passed += ok; failed += not ok

    # -- Test: no source_filter uses full index ---------------------
    print()
    section("No source_filter → all files eligible (no artificial restriction)")

    with mock.patch("app.services.retriever.embed_query", return_value=query_embedding), \
         mock.patch("app.services.retriever.search_similar", return_value=candidates), \
         mock.patch("app.services.retriever.get_all_chunks", return_value=all_chunks):

        results_no_filter = retrieve(
            "describe all files",
            include_all_sources=True,
            source_filter=None
        )

    result_sources_nf = {c["metadata"]["source"] for c in results_no_filter}
    print(f"  Sources in results: {result_sources_nf}")
    ok_all = len(result_sources_nf) >= 2  # at least some files found
    status = "✅ PASS" if ok_all else "❌ FAIL"
    print(f"  {status}  Multiple sources returned without filter")
    passed += ok_all; failed += not ok_all

    print(f"\n{'='*60}")
    print(f"Result: {passed}/{passed+failed} passed, {failed} failed")
    print(f"{'='*60}\n")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
