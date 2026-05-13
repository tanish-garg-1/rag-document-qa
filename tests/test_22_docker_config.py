"""
TEST 22: DOCKER & DEPLOYMENT CONFIGURATION
Tests Dockerfile, Dockerfile.streamlit, docker-compose.yml,
.dockerignore, and .env.example for correctness and security.
No API key required.
Run: python tests/test_22_docker_config.py
"""

import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("TEST 22: DOCKER & DEPLOYMENT CONFIGURATION")
print("=" * 60)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None

# --- 1. Check all deployment files exist ---
print("\n--- Deployment Files Exist ---")
files = {
    "Dockerfile": os.path.join(ROOT, "Dockerfile"),
    "Dockerfile.streamlit": os.path.join(ROOT, "Dockerfile.streamlit"),
    "docker-compose.yml": os.path.join(ROOT, "docker-compose.yml"),
    ".dockerignore": os.path.join(ROOT, ".dockerignore"),
    ".env.example": os.path.join(ROOT, ".env.example"),
    "requirements.txt": os.path.join(ROOT, "requirements.txt"),
}
sources = {}
for name, path in files.items():
    src = read_file(path)
    if src is not None:
        sources[name] = src
        print(f"  [PASS] {name} exists ({len(src.splitlines())} lines)")
    else:
        print(f"  [WARN] {name} not found")
        sources[name] = ""

# --- 2. Dockerfile (backend) checks ---
print("\n--- Dockerfile (Backend) ---")
df = sources.get("Dockerfile", "")
if df:
    if "FROM python:" in df:
        m = re.search(r"FROM python:([\d.]+)", df)
        print(f"  [PASS] Python base image: {m.group(0) if m else 'found'}")
    else:
        print("  [WARN] No Python base image found")

    if "COPY .env" in df:
        print("  [FAIL] Dockerfile copies .env into image (API keys embedded in layers!)")
    else:
        print("  [PASS] .env NOT copied into backend image")

    if "requirements.txt" in df:
        print("  [PASS] requirements.txt installed")

    if "EXPOSE 8000" in df:
        print("  [PASS] Port 8000 exposed")
    else:
        print("  [WARN] Port 8000 not explicitly exposed")

    if "uvicorn" in df or "CMD" in df:
        print("  [PASS] CMD/entrypoint defined")
    else:
        print("  [WARN] No CMD found")

    if "WORKDIR" in df:
        print("  [PASS] WORKDIR set")

# --- 3. Dockerfile.streamlit checks ---
print("\n--- Dockerfile.streamlit (Frontend) ---")
df_st = sources.get("Dockerfile.streamlit", "")
if df_st:
    if "FROM python:" in df_st:
        m = re.search(r"FROM python:([\d.]+)", df_st)
        print(f"  [PASS] Python base image: {m.group(0) if m else 'found'}")

    if "COPY .env" in df_st:
        print("  [FAIL] Dockerfile.streamlit copies .env (security risk)")
    else:
        print("  [PASS] .env NOT copied into Streamlit image")

    if "EXPOSE 8501" in df_st:
        print("  [PASS] Port 8501 exposed")
    else:
        print("  [WARN] Port 8501 not exposed")

    # CMD can be shell form ("streamlit run ...") or JSON array (["streamlit","run",...])
    if "streamlit run" in df_st or ('"streamlit"' in df_st and '"run"' in df_st):
        print("  [PASS] streamlit run command found")
    else:
        print("  [FAIL] streamlit run not found in CMD")

    if "frontend/" in df_st or "streamlit_app.py" in df_st:
        print("  [PASS] Frontend app file referenced")

# --- 4. docker-compose.yml checks ---
print("\n--- docker-compose.yml ---")
dc = sources.get("docker-compose.yml", "")
if dc:
    # Services
    if "backend:" in dc:
        print("  [PASS] 'backend' service defined")
    else:
        print("  [FAIL] No 'backend' service")

    if "frontend:" in dc:
        print("  [PASS] 'frontend' service defined")
    else:
        print("  [FAIL] No 'frontend' service")

    # Ports
    if "8000:8000" in dc:
        print("  [PASS] Backend port 8000 mapped")
    else:
        print("  [WARN] Backend port 8000 not mapped")

    if "8501:8501" in dc:
        print("  [PASS] Frontend port 8501 mapped")
    else:
        print("  [WARN] Frontend port 8501 not mapped")

    # env_file (secure API key injection)
    env_file_count = dc.count("env_file:")
    if env_file_count >= 2:
        print(f"  [PASS] env_file used in both services ({env_file_count} times)")
    elif env_file_count == 1:
        print(f"  [WARN] env_file used in only 1 service")
    else:
        print("  [FAIL] env_file not used — API keys not injected via docker-compose")

    # API_BASE for inter-service communication
    if "API_BASE=http://backend:8000" in dc:
        print("  [PASS] Frontend API_BASE points to backend service name")
    else:
        print("  [WARN] API_BASE not set to backend service name")

    # depends_on
    if "depends_on:" in dc:
        print("  [PASS] depends_on defined (frontend waits for backend)")
    else:
        print("  [WARN] No depends_on — frontend may start before backend")

    # Data volume
    if "volumes:" in dc:
        print("  [PASS] Volume mount defined (data persistence)")
    else:
        print("  [WARN] No volume mount — FAISS data lost on container restart")

    # restart policy
    if "restart:" in dc:
        print("  [PASS] Restart policy defined")

# --- 5. .dockerignore checks ---
print("\n--- .dockerignore ---")
di = sources.get(".dockerignore", "")
if di:
    important_ignores = {
        ".env": "API keys",
        "__pycache__": "Python cache",
        "*.pyc": "Compiled Python",
        ".git": "Git history",
    }
    for pattern, desc in important_ignores.items():
        if pattern in di:
            print(f"  [PASS] '{pattern}' ignored ({desc})")
        else:
            print(f"  [WARN] '{pattern}' not in .dockerignore ({desc} may be copied)")
else:
    print("  [WARN] No .dockerignore — all files copied into image including caches")

# --- 6. .env.example checks ---
print("\n--- .env.example ---")
env_ex = sources.get(".env.example", "")
if env_ex:
    required_keys = ["GEMINI_API_KEY", "GROQ_API_KEY"]
    for key in required_keys:
        if key in env_ex:
            print(f"  [PASS] {key} documented in .env.example")
        else:
            print(f"  [WARN] {key} not in .env.example")

    # Should not contain real keys
    if "AIzaSy" in env_ex or "gsk_" in env_ex:
        print("  [FAIL] Real API keys found in .env.example!")
    else:
        print("  [PASS] No real API keys in .env.example (only placeholders)")
else:
    print("  [WARN] No .env.example found")

# --- 7. requirements.txt sanity ---
print("\n--- requirements.txt ---")
req = sources.get("requirements.txt", "")
if req:
    required_packages = [
        "fastapi", "uvicorn", "python-multipart", "python-dotenv",
        "pymupdf", "python-docx", "Pillow", "langchain",
        "faiss-cpu", "google-genai", "groq", "streamlit", "requests"
    ]
    missing = []
    for pkg in required_packages:
        if pkg.lower() in req.lower():
            print(f"  [PASS] {pkg}")
        else:
            missing.append(pkg)
            print(f"  [FAIL] {pkg} MISSING from requirements.txt")

    # Check for duplicate/conflicting Gemini SDKs
    has_old_sdk = "google-generativeai" in req
    has_new_sdk = "google-genai" in req
    if has_old_sdk and has_new_sdk:
        print("  [FAIL] Both google-generativeai AND google-genai in requirements (conflict!)")
    elif has_new_sdk:
        print("  [PASS] Only new google-genai SDK (correct)")
    elif has_old_sdk:
        print("  [WARN] Old google-generativeai SDK — migrate to google-genai")

# --- 8. Data directory structure ---
print("\n--- Data Directory Structure ---")
data_dir = os.path.join(ROOT, "data")
subdirs = {"uploads": "uploaded files", "faiss_index": "FAISS index"}
for subdir, desc in subdirs.items():
    path = os.path.join(data_dir, subdir)
    if os.path.exists(path):
        files_in = os.listdir(path)
        print(f"  [PASS] data/{subdir}/ exists ({len(files_in)} files — {desc})")
    else:
        print(f"  [WARN] data/{subdir}/ not found (will be created on first use)")

print("\n" + "=" * 60)
print("TEST 22 COMPLETE")
print("=" * 60)
