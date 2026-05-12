"""
Run all new tests (27-32) and print a summary.
Usage: python tests/run_all_new_tests.py
"""
import sys
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Prefer the project venv Python so all packages (google, groq, etc.) are available.
# Fall back to sys.executable if no venv found.
_venv_python = os.path.join(ROOT, "rag_proj_env", "Scripts", "python.exe")  # Windows
if not os.path.exists(_venv_python):
    _venv_python = os.path.join(ROOT, "rag_proj_env", "bin", "python")       # Linux/Mac
PYTHON = _venv_python if os.path.exists(_venv_python) else sys.executable

TESTS = [
    ("27", "tests/test_27_describe_all_regex.py",    "Describe-all regex robustness",  60),
    ("28", "tests/test_28_query_classifier.py",       "Query classifier & rewriter",    60),
    ("29", "tests/test_29_context_formatting.py",     "Context formatting & budget",    30),
    ("30", "tests/test_30_source_pool_scoping.py",    "Source pool scoping fix",        30),
    ("31", "tests/test_31_max_sources_cap.py",        "MAX_SOURCES cap behaviour",      30),
    ("32", "tests/test_32_upload_pipeline.py",        "Upload pipeline end-to-end",    360),  # image upload via Gemini
]

# Force UTF-8 in subprocesses on Windows
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"

print("=" * 65)
print("  RAG Pipeline -- New Test Suite (Tests 27-32)")
print("=" * 65)
print()

results = []
for num, path, label, timeout in TESTS:
    full_path = os.path.join(ROOT, path)
    print(f"Running Test {num}: {label} ...")
    try:
        proc = subprocess.run(
            [PYTHON, "-X", "utf8", full_path],
            capture_output=True,
            stdin=subprocess.DEVNULL,   # cuts off stdin so isatty()=False
            text=True,
            encoding="utf-8",
            cwd=ROOT,
            env=env,
            timeout=timeout,
        )
        ok = proc.returncode == 0
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        ok = False
        stdout, stderr = "", f"TIMEOUT: test took longer than {timeout} seconds"
    results.append((num, label, ok, stdout, stderr))
    print(f"  {'PASSED' if ok else 'FAILED'}\n")

print("=" * 65)
print("  SUMMARY")
print("=" * 65)
all_pass = True
for num, label, ok, stdout, stderr in results:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}]  Test {num}: {label}")
    if not ok:
        all_pass = False
        output = (stdout + stderr).strip().split("\n")
        for line in output[-25:]:
            print(f"         {line}")

print()
print("=" * 65)
total = len(results)
passed = sum(1 for _, _, ok, _, _ in results if ok)
print(f"  {passed}/{total} test files passed")
print("=" * 65)

sys.exit(0 if all_pass else 1)
