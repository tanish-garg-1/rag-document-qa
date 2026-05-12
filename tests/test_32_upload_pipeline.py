"""
Test 32 — Upload pipeline end-to-end (requires running backend)
Tests the full upload flow: file → index → sources list
Checks each file type separately.
Run: python tests/test_32_upload_pipeline.py
"""
import sys, os, requests, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
FILES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "files_for_test")

TEST_FILES = {
    "Resume.pdf":           "application/pdf",
    "README.txt":           "text/plain",
    "Lhr_Driving_Lic.jpg":  "image/jpeg",
    "improved_wheat_logo.png": "image/png",
}


def section(title):
    print(f"\n-- {title} {'-'*(55-len(title))}")


def check_backend():
    try:
        r = requests.get(f"{API_BASE}/", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def run():
    print(f"\n{'='*60}")
    print("Test 32 — Upload Pipeline End-to-End")
    print(f"{'='*60}")
    passed = failed = 0

    if not check_backend():
        print(f"\n❌ Backend not reachable at {API_BASE} — start it first.")
        print("   uvicorn app.main:app --reload --port 8000")
        sys.exit(1)
    print(f"\n✅ Backend reachable at {API_BASE}")

    # -- Check existing index size — warn before clearing ----------
    section("Checking existing index before clearing")
    existing = requests.get(f"{API_BASE}/stats", timeout=5).json()
    existing_count = existing.get("total_vectors", 0)
    if existing_count > 0:
        print(f"  ⚠️  WARNING: {existing_count} vectors currently in live index!")
        print(f"       This test will CLEAR the index. Re-upload your files after.")
        if sys.stdin.isatty():
            try:
                ans = input("  Continue and clear? [y/N]: ").strip().lower()
                if ans != "y":
                    print("  Aborted -- index not cleared.")
                    sys.exit(0)
            except EOFError:
                print("  Non-interactive mode (stdin closed) -- proceeding with clear.")
        else:
            print("  Non-interactive mode -- proceeding with clear.")

    r = requests.post(f"{API_BASE}/clear", timeout=10)
    ok = r.json().get("status") == "success"
    print(f"  {'✅' if ok else '❌'}  Index cleared")

    # -- Upload each file and check result -------------------------
    for fname, mime in TEST_FILES.items():
        fpath = os.path.join(FILES_DIR, fname)
        section(f"Uploading: {fname}")

        if not os.path.exists(fpath):
            print(f"  ⚠️  File not found: {fpath} — skipping")
            continue

        with open(fpath, "rb") as f:
            file_bytes = f.read()

        try:
            r = requests.post(
                f"{API_BASE}/upload",
                files=[("files", (fname, file_bytes, mime))],
                timeout=60
            )
            results = r.json().get("results", [])
            if not results:
                print(f"  ❌  No results returned")
                failed += 1
                continue

            result = results[0]
            status = result.get("status")
            chunks = result.get("chunks_added", 0)
            reason = result.get("reason", "")

            if status == "success":
                print(f"  ✅  Indexed successfully — {chunks} chunks")
                passed += 1
            elif status == "duplicate":
                print(f"  ↩️  Duplicate — already indexed (counts as pass)")
                passed += 1
            else:
                print(f"  ❌  FAILED — {reason}")
                failed += 1

        except Exception as e:
            print(f"  ❌  Exception: {e}")
            failed += 1

        time.sleep(1)  # avoid hammering the API

    # -- Check /sources lists all uploaded files -------------------
    section("Checking /sources endpoint")
    try:
        r = requests.get(f"{API_BASE}/sources", timeout=10)
        sources = r.json().get("sources", [])
        print(f"  Sources in index: {sources}")
        for fname in TEST_FILES:
            base = os.path.splitext(fname)[0]
            found = any(fname in s or base in s for s in sources)
            status = "✅" if found else "❌"
            print(f"  {status}  {fname} in /sources")
            passed += found; failed += not found
    except Exception as e:
        print(f"  ❌  /sources error: {e}")
        failed += 1

    print(f"\n{'='*60}")
    print(f"Result: {passed}/{passed+failed} passed, {failed} failed")
    print(f"{'='*60}\n")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
