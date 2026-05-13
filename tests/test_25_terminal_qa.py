"""
TEST 25: INTERACTIVE TERMINAL Q&A
Uploads all files from  files_for_test/  into the RAG system,
then opens an interactive question-answering session in the terminal.

No browser or running server needed — runs everything in-process.

Usage:
  python tests/test_25_terminal_qa.py

Commands inside the session:
  <any question>   Ask about your documents
  /stats           Show how many chunks are indexed
  /history         View conversation history
  /clear-history   Wipe conversation memory
  /clear-index     Wipe the vector store and re-upload files
  /reupload        Re-upload all files from files_for_test/
  /files           List files currently in files_for_test/
  /quit  or /exit  Exit the session
"""

import sys
import os
import io
import time
import shutil

# -- path setup ----------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FILES_DIR = os.path.join(ROOT, "files_for_test")

# -- load .env -----------------------------------------------------------------
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True) or find_dotenv())

# -- check API keys -------------------------------------------------------------
gemini_key = os.getenv("GEMINI_API_KEY")
groq_key   = os.getenv("GROQ_API_KEY")

if not gemini_key or not groq_key:
    missing = [k for k, v in [("GEMINI_API_KEY", gemini_key), ("GROQ_API_KEY", groq_key)] if not v]
    print(f"\n[ERROR] Missing API keys: {', '.join(missing)}")
    print("        Add them to your .env file and retry.")
    sys.exit(1)

import re as _re

# -- supported extensions -------------------------------------------------------
SUPPORTED = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".webp"}

# UUID prefix pattern: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx_
_UUID_PREFIX = _re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_',
    _re.IGNORECASE
)

def display_name(fname: str) -> str:
    """Strip a leading UUID prefix from a filename for nicer display."""
    return _UUID_PREFIX.sub("", fname)

MIME_MAP = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt":  "text/plain",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

# -- force UTF-8 output so tick/cross/emoji print on Windows -------------------
if sys.platform == "win32":
    import ctypes
    try:
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7)   # enable ANSI
    except Exception:
        pass
    # Reconfigure stdout/stderr to UTF-8 (Python 3.7+)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# -- colour helpers -------------------------------------------------------------
def green(t):  return f"\033[92m{t}\033[0m"
def yellow(t): return f"\033[93m{t}\033[0m"
def cyan(t):   return f"\033[96m{t}\033[0m"
def red(t):    return f"\033[91m{t}\033[0m"
def bold(t):   return f"\033[1m{t}\033[0m"
def dim(t):    return f"\033[2m{t}\033[0m"

# -- boot FastAPI in-process ----------------------------------------------------
print(bold("\n  RAG Terminal Q&A — booting..."))
try:
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app, raise_server_exceptions=False)
    print(green("  [OK] FastAPI app loaded"))
except Exception as e:
    print(red(f"  [FAIL] Could not load app: {e}"))
    sys.exit(1)

# -- helpers --------------------------------------------------------------------

def list_test_files():
    """Return list of (filename, full_path) for supported files in files_for_test/."""
    if not os.path.exists(FILES_DIR):
        return []
    return [
        (f, os.path.join(FILES_DIR, f))
        for f in sorted(os.listdir(FILES_DIR))
        if os.path.splitext(f)[1].lower() in SUPPORTED
    ]


def friendly_reason(reason: str) -> str:
    """Turn a raw backend reason into a human-friendly failure message."""
    if not reason:
        return "Unknown error"
    r = reason.lower()
    if "429" in r or "rate" in r or "quota" in r or "resource_exhausted" in r or "resource exhausted" in r:
        return "Rate limit hit (API quota) — try again later"
    if "no content" in r or "empty" in r or "no text" in r:
        return "Empty file — no text or content could be extracted"
    if "unsupported" in r or "not supported" in r:
        return "Unsupported file type"
    if "too large" in r or "size" in r:
        return "File too large — exceeds upload limit"
    if "timeout" in r:
        return "Request timed out — file may be too complex"
    if "permission" in r or "access" in r:
        return "Permission denied reading file"
    if "corrupt" in r or "invalid" in r:
        return "File appears corrupted or invalid"
    # Return the original reason, capitalised, if nothing matched
    return reason.strip().capitalize()


def upload_files():
    """Upload all supported files from files_for_test/ into FAISS."""
    files_found = list_test_files()

    if not files_found:
        print(yellow(f"\n  No supported files found in files_for_test/"))
        print(dim( f"  Supported: {', '.join(sorted(SUPPORTED))}"))
        print(dim( f"  Path: {FILES_DIR}"))
        return 0

    print(bold(f"\n  Uploading {len(files_found)} file(s)...\n"))
    total_chunks  = 0
    success_count = 0
    fail_count    = 0

    dup_count = 0

    for fname, fpath in files_found:
        ext  = os.path.splitext(fname)[1].lower()
        mime = MIME_MAP.get(ext, "application/octet-stream")
        dname = display_name(fname)   # strip UUID prefix for display

        with open(fpath, "rb") as f:
            file_bytes = f.read()

        t0 = time.time()
        try:
            resp = client.post(
                "/upload",
                files=[("files", (fname, io.BytesIO(file_bytes), mime))]
            )
            elapsed = time.time() - t0
            result  = resp.json().get("results", [{}])[0]
            status  = result.get("status", "")

            if status == "success":
                chunks = result.get("chunks_added", 0)
                total_chunks  += chunks
                success_count += 1
                print(f"  {green('✔')} {bold(dname)}")
                print(dim(f"       {chunks} chunk(s)  •  {elapsed:.1f}s"))

            elif status == "duplicate":
                dup_count += 1
                print(f"  {yellow('<-')} {bold(dname)}")
                print(dim(f"       Already indexed — skipped (identical content)"))

            else:
                raw_reason = result.get("reason", "")
                nice       = friendly_reason(raw_reason)
                fail_count += 1
                print(f"  {red('✘')} {bold(dname)}")
                print(dim(f"       {nice}"))
        except Exception as e:
            fail_count += 1
            print(f"  {red('✘')} {bold(dname)}")
            print(dim(f"       Exception: {e}"))

    # -- summary line ----------------------------------------------------------
    total = success_count + fail_count + dup_count
    parts = []
    if success_count:
        parts.append(green(f"{success_count} loaded"))
    if dup_count:
        parts.append(yellow(f"{dup_count} already indexed"))
    if fail_count:
        parts.append(red(f"{fail_count} failed"))

    summary = f"  {total} file(s): " + "  •  ".join(parts)
    print(f"\n{summary}")

    # -- index stats -----------------------------------------------------------
    stats = client.get("/stats").json()
    print(f"  {cyan('Index:')} {stats['total_vectors']} vectors  •  "
          f"{stats['total_chunks']} chunks  •  dim={stats['embedding_dim']}")
    return total_chunks


def ask_question(question: str):
    """Send question to /query and print streamed response."""
    print(f"\n{cyan('Assistant: ')}", end="", flush=True)
    full = ""
    try:
        with client.stream(
            "POST", "/query",
            json={"query": question},
            timeout=120
        ) as r:
            if r.status_code == 422:
                print(yellow("[empty or invalid query]"))
                return
            for chunk in r.iter_text():
                if chunk:
                    # Print raw; encode safely for Windows CP1252
                    safe = chunk.encode("utf-8", "replace").decode("utf-8")
                    print(safe, end="", flush=True)
                    full += safe
    except Exception as e:
        print(red(f"\n[stream error: {e}]"))
    print()   # newline after response


def show_stats():
    stats = client.get("/stats").json()
    print(f"\n  {cyan('Vectors:')}  {stats['total_vectors']}")
    print(f"  {cyan('Chunks:')}   {stats['total_chunks']}")
    print(f"  {cyan('Emb dim:')}  {stats['embedding_dim']}")


def show_history():
    resp = client.get("/history").json()
    msgs = resp.get("messages", [])
    total = resp.get("total_messages", 0)
    if not msgs:
        print(dim("\n  No conversation history yet."))
        return
    print(f"\n  {cyan(f'History ({total} messages):')}")
    for m in msgs:
        role = "You" if m["role"] == "user" else "Bot"
        colour = yellow if m["role"] == "user" else green
        preview = m["content"][:120].replace("\n", " ")
        print(f"  {colour(role+':')} {preview}{'...' if len(m['content'])>120 else ''}")


def show_files():
    files = list_test_files()
    if not files:
        print(dim(f"\n  No supported files in files_for_test/"))
        return
    print(f"\n  {cyan('Files in files_for_test/')}  ({len(files)} found)")
    for fname, fpath in files:
        size_kb = os.path.getsize(fpath) / 1024
        dname   = display_name(fname)
        print(f"    • {dname}  {dim(f'({size_kb:.1f} KB)')}")


def print_help():
    print(f"""
  {bold('Commands:')}
  {cyan('/stats')}          — show vector store stats
  {cyan('/history')}        — view conversation history
  {cyan('/clear-history')}  — wipe conversation memory
  {cyan('/clear-index')}    — wipe index then re-upload files
  {cyan('/reupload')}       — re-upload files without clearing
  {cyan('/files')}          — list files in files_for_test/
  {cyan('/help')}           — show this help
  {cyan('/quit')} or {cyan('/exit')}  — exit
    """)


# -- banner ---------------------------------------------------------------------

def print_banner():
    print("\n" + "=" * 60)
    print(bold("   RAG TERMINAL Q&A"))
    print("   Ask questions about files in  files_for_test/")
    print("=" * 60)
    print_help()


# -- main -----------------------------------------------------------------------

def main():
    print_banner()

    # Upload files on startup
    chunks = upload_files()

    if chunks == 0 and not list_test_files():
        print(yellow("\n  Add files to files_for_test/ and run again."))
        print(yellow(f"  Folder: {FILES_DIR}"))
        sys.exit(0)

    print(f"\n{bold('  Ready! Type your question below.')}")
    print(dim("  (type /help for commands, /quit to exit)\n"))

    while True:
        try:
            user_input = input(f"{yellow('You: ')}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{dim('  Goodbye!')}")
            break

        if not user_input:
            continue

        # -- commands -----------------------------------------------------------
        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            print(dim("\n  Goodbye!"))
            break

        elif cmd == "/help":
            print_help()

        elif cmd == "/stats":
            show_stats()

        elif cmd == "/history":
            show_history()

        elif cmd == "/files":
            show_files()

        elif cmd == "/clear-history":
            client.delete("/history")
            print(green("  Conversation memory cleared."))

        elif cmd == "/clear-index":
            client.post("/clear")
            client.delete("/history")
            print(yellow("  Index and memory cleared. Re-uploading files..."))
            upload_files()

        elif cmd == "/reupload":
            print(yellow("  Re-uploading files (existing index kept)..."))
            upload_files()

        # -- question -----------------------------------------------------------
        else:
            ask_question(user_input)

        print()   # blank line between turns


if __name__ == "__main__":
    main()
