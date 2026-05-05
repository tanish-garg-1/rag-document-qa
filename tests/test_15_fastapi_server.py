"""
TEST 15: FASTAPI SERVER — STARTUP & HEALTH
Tests FastAPI app creation, routing, and health endpoint.
Does NOT start the server — tests app object directly.
Run: python tests/test_15_fastapi_server.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("TEST 15: FASTAPI SERVER — STARTUP & HEALTH")
print("=" * 60)

# --- 1. Import FastAPI ---
print("\n--- Import FastAPI ---")
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    print("  [PASS] FastAPI and TestClient imported")
except ImportError as e:
    print(f"  [FAIL] FastAPI not installed: {e}")
    print("         Run: pip install fastapi httpx")
    sys.exit(1)

# --- 2. Import app ---
print("\n--- Import app.main ---")
try:
    from app.main import app
    print("  [PASS] FastAPI app imported from app.main")
    print(f"  App title: {app.title}")
    print(f"  App routes: {[r.path for r in app.routes]}")
except Exception as e:
    print(f"  [FAIL] Cannot import app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 3. Create test client ---
print("\n--- Create TestClient ---")
try:
    client = TestClient(app, raise_server_exceptions=True)
    print("  [PASS] TestClient created")
except Exception as e:
    print(f"  [FAIL] TestClient creation error: {e}")
    sys.exit(1)

# --- 4. Health check GET / ---
print("\n--- GET / Health Check ---")
try:
    response = client.get("/")
    print(f"  Status code: {response.status_code}")
    print(f"  Response body: {response.json()}")
    if response.status_code == 200:
        print("  [PASS] Health check returns 200")
        body = response.json()
        if "message" in body:
            print(f"  [PASS] Response has 'message' key: '{body['message']}'")
        else:
            print(f"  [WARN] Response missing 'message' key: {body}")
    else:
        print(f"  [FAIL] Unexpected status code: {response.status_code}")
except Exception as e:
    print(f"  [FAIL] GET / error: {e}")
    import traceback
    traceback.print_exc()

# --- 5. GET /stats ---
print("\n--- GET /stats ---")
try:
    response = client.get("/stats")
    print(f"  Status code: {response.status_code}")
    print(f"  Response: {response.text[:200]}")
    if response.status_code == 200:
        print("  [PASS] /stats returns 200")
        try:
            data = response.json()
            print(f"  JSON: {data}")
            expected_keys = {"total_vectors", "total_chunks", "embedding_dim"}
            if all(k in data for k in expected_keys):
                print("  [PASS] Stats has expected keys")
            else:
                missing = expected_keys - set(data.keys())
                print(f"  [WARN] Missing stats keys: {missing}")
        except Exception as je:
            print(f"  [WARN] Response is not JSON: {je}")
    elif response.status_code == 404:
        print("  [WARN] /stats returns 404 — check router registration in main.py")
    else:
        print(f"  [INFO] /stats status: {response.status_code}")
except Exception as e:
    print(f"  [FAIL] GET /stats error: {e}")
    import traceback
    traceback.print_exc()

# --- 6. GET /history ---
print("\n--- GET /history ---")
try:
    response = client.get("/history")
    print(f"  Status code: {response.status_code}")
    if response.status_code == 200:
        print("  [PASS] /history returns 200")
        data = response.json()
        print(f"  Data: {data}")
        if "total_messages" in data and "messages" in data:
            print("  [PASS] Response has 'total_messages' and 'messages'")
        else:
            print(f"  [WARN] Unexpected response structure: {data}")
    elif response.status_code == 404:
        print("  [WARN] /history returns 404 — may not be registered at root level")
        print("         Check if it's under /query or /history prefix")
    else:
        print(f"  [INFO] /history status: {response.status_code}, body: {response.text[:100]}")
except Exception as e:
    print(f"  [FAIL] GET /history error: {e}")
    import traceback
    traceback.print_exc()

# --- 7. DELETE /history ---
print("\n--- DELETE /history ---")
try:
    response = client.delete("/history")
    print(f"  Status code: {response.status_code}")
    if response.status_code == 200:
        print("  [PASS] DELETE /history returns 200")
        data = response.json()
        print(f"  Response: {data}")
    elif response.status_code == 404:
        print("  [WARN] DELETE /history returns 404 — check router")
    else:
        print(f"  [INFO] DELETE /history status: {response.status_code}")
except Exception as e:
    print(f"  [FAIL] DELETE /history error: {e}")

# --- 8. POST /clear ---
print("\n--- POST /clear ---")
try:
    response = client.post("/clear")
    print(f"  Status code: {response.status_code}")
    if response.status_code == 200:
        print("  [PASS] POST /clear returns 200")
        data = response.json()
        print(f"  Response: {data}")
    elif response.status_code == 404:
        print("  [WARN] POST /clear returns 404")
    else:
        print(f"  [INFO] POST /clear status: {response.status_code}, body: {response.text[:100]}")
except Exception as e:
    print(f"  [FAIL] POST /clear error: {e}")

# --- 9. List all routes ---
print("\n--- All Registered Routes ---")
for route in app.routes:
    methods = getattr(route, "methods", {"GET"})
    path = getattr(route, "path", str(route))
    name = getattr(route, "name", "")
    print(f"  {str(methods):30} {path:30} ({name})")

# --- 10. Check router registration ---
print("\n--- Router Registration Check ---")
routes_by_path = {r.path: r for r in app.routes if hasattr(r, "path")}
expected_routes = ["/upload", "/query", "/clear", "/history", "/stats"]
for route in expected_routes:
    if route in routes_by_path:
        print(f"  [PASS] Route '{route}' registered")
    else:
        # Check with /upload and /query prefixes from routers
        print(f"  [WARN] Route '{route}' not found at root — may have prefix")

print("\n" + "=" * 60)
print("TEST 15 COMPLETE")
print("=" * 60)
