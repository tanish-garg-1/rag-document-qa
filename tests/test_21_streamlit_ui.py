"""
TEST 21: STREAMLIT FRONTEND AUDIT
Tests the Streamlit app structure, logic, and API wiring WITHOUT
needing a running browser or server.
No API key required.
Run: python tests/test_21_streamlit_ui.py
"""

import sys
import os
import ast
import py_compile
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("TEST 21: STREAMLIT FRONTEND AUDIT")
print("=" * 60)

FRONTEND_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend", "streamlit_app.py"
)

# --- 1. File existence ---
print("\n--- File Existence ---")
if os.path.exists(FRONTEND_PATH):
    print(f"  [PASS] frontend/streamlit_app.py found")
else:
    print(f"  [FAIL] frontend/streamlit_app.py NOT found at {FRONTEND_PATH}")
    sys.exit(1)

with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
    source = f.read()

print(f"  [INFO] File size: {len(source)} chars, {len(source.splitlines())} lines")

# --- 2. Syntax check ---
print("\n--- Syntax Check ---")
try:
    py_compile.compile(FRONTEND_PATH, doraise=True)
    print("  [PASS] No syntax errors in streamlit_app.py")
except py_compile.PyCompileError as e:
    print(f"  [FAIL] Syntax error: {e}")
    sys.exit(1)

# --- 3. AST parse ---
print("\n--- AST Parse ---")
try:
    tree = ast.parse(source)
    print(f"  [PASS] AST parsed successfully")
    # Count function defs and class defs
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    print(f"  [INFO] Function definitions: {len(funcs)}")
except SyntaxError as e:
    print(f"  [FAIL] AST parse failed: {e}")

# --- 4. Required imports ---
print("\n--- Required Imports ---")
required_imports = ["streamlit", "requests", "os"]
for imp in required_imports:
    if imp in source:
        print(f"  [PASS] '{imp}' imported")
    else:
        print(f"  [FAIL] '{imp}' not found in imports")

# --- 5. API_BASE configuration ---
print("\n--- API_BASE Configuration ---")
if "API_BASE" in source:
    print("  [PASS] API_BASE defined")
if 'os.getenv("API_BASE"' in source or "os.getenv('API_BASE'" in source:
    print("  [PASS] API_BASE loaded from environment variable")
else:
    print("  [WARN] API_BASE may be hardcoded")
if "http://127.0.0.1:8000" in source or "http://localhost:8000" in source:
    print("  [INFO] Default fallback API_BASE = localhost:8000")
if "http://backend:8000" in source:
    print("  [INFO] Docker service name 'backend' referenced (from env var)")

# --- 6. API endpoints referenced ---
print("\n--- API Endpoints Referenced ---")
endpoints = {
    "/upload": "Document upload",
    "/query": "Question answering",
    "/history": "Conversation history",
    "/stats": "Index statistics",
    "/clear": "Clear vector store",
}
for endpoint, desc in endpoints.items():
    if endpoint in source:
        print(f"  [PASS] '{endpoint}' endpoint used ({desc})")
    else:
        print(f"  [WARN] '{endpoint}' endpoint NOT referenced ({desc})")

# --- 7. Streamlit UI elements ---
print("\n--- Streamlit UI Elements ---")
ui_elements = {
    "st.set_page_config": "Page config (title/layout)",
    "st.sidebar": "Sidebar panel",
    "st.file_uploader": "File upload widget",
    "st.chat_input": "Chat input box",
    "st.chat_message": "Chat message display",
    "st.session_state": "Session state (chat history)",
    "st.spinner": "Loading spinner",
    "st.success": "Success notification",
    "st.error": "Error notification",
    "st.button": "Button widget",
    "st.markdown": "Markdown renderer",
}
for element, desc in ui_elements.items():
    if element in source:
        print(f"  [PASS] {element} ({desc})")
    else:
        print(f"  [WARN] {element} NOT found ({desc})")

# --- 8. Streaming response handling ---
print("\n--- Streaming Response Handling ---")
if "stream=True" in source:
    print("  [PASS] HTTP streaming enabled (stream=True)")
else:
    print("  [FAIL] stream=True NOT found — responses won't stream")

if "iter_content" in source or "iter_lines" in source:
    print("  [PASS] Chunk iteration found (iter_content/iter_lines)")
else:
    print("  [WARN] No chunk iteration — streaming may not work correctly")

if "placeholder" in source and ("markdown" in source or "write" in source):
    print("  [PASS] Streaming placeholder pattern found (live token update)")
else:
    print("  [WARN] No streaming placeholder pattern — tokens may not display live")

# --- 9. Supported file types match backend ---
print("\n--- Supported File Types (Frontend vs Backend) ---")
backend_types = {"pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"}
# Extract type= list from source
import re
type_match = re.search(r'type=\[([^\]]+)\]', source)
if type_match:
    raw = type_match.group(1)
    frontend_types = set(t.strip().strip('"').strip("'") for t in raw.split(","))
    print(f"  Frontend types: {sorted(frontend_types)}")
    print(f"  Backend types:  {sorted(backend_types)}")
    missing = backend_types - frontend_types
    extra = frontend_types - backend_types
    if not missing and not extra:
        print("  [PASS] File types match exactly between frontend and backend")
    if missing:
        print(f"  [WARN] Backend supports these but frontend hides them: {missing}")
    if extra:
        print(f"  [WARN] Frontend shows these but backend may reject: {extra}")
else:
    print("  [WARN] Could not parse type= list from file_uploader")

# --- 10. Error handling ---
print("\n--- Error Handling ---")
try_count = source.count("try:")
except_count = source.count("except")
print(f"  try/except blocks: {try_count}")
if try_count >= 3:
    print("  [PASS] Multiple error handlers found")
elif try_count >= 1:
    print("  [INFO] Some error handling present")
else:
    print("  [FAIL] No try/except found — unhandled exceptions will crash UI")

if "except Exception as e" in source or "except:" in source:
    print("  [PASS] Generic exception handler found")

# --- 11. Session state for messages ---
print("\n--- Session State Chat History ---")
if '"messages" not in st.session_state' in source or "'messages' not in st.session_state" in source:
    print("  [PASS] messages initialized in session_state")
else:
    print("  [WARN] messages initialization not found in session_state")

if "st.session_state.messages.append" in source:
    print("  [PASS] Messages appended to session_state")
else:
    print("  [WARN] No session_state.messages.append found")

# --- 12. Page config check ---
print("\n--- Page Config ---")
if 'page_title' in source:
    title_match = re.search(r'page_title=["\']([^"\']+)["\']', source)
    if title_match:
        print(f"  [INFO] Page title: '{title_match.group(1)}'")
if 'layout="wide"' in source or "layout='wide'" in source:
    print("  [PASS] Wide layout enabled")
else:
    print("  [INFO] Not using wide layout")

# --- 13. Docker frontend integration ---
print("\n--- Docker Frontend Integration ---")
dockerfile_st = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Dockerfile.streamlit"
)
if os.path.exists(dockerfile_st):
    with open(dockerfile_st) as f:
        df_src = f.read()
    print("  [PASS] Dockerfile.streamlit exists")
    if "streamlit" in df_src.lower():
        print("  [PASS] streamlit referenced in Dockerfile.streamlit")
    if "8501" in df_src:
        print("  [PASS] Port 8501 exposed in Dockerfile.streamlit")
    else:
        print("  [WARN] Port 8501 not found in Dockerfile.streamlit")
    if "COPY .env" in df_src:
        print("  [FAIL] Dockerfile.streamlit copies .env (security risk)")
    else:
        print("  [PASS] .env NOT copied into Streamlit image")
else:
    print("  [WARN] Dockerfile.streamlit not found")

print("\n" + "=" * 60)
print("TEST 21 COMPLETE")
print("=" * 60)
