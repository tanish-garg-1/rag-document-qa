"""
TEST 13: GEMINI VISION — IMAGE DESCRIPTION
Tests image loading and Gemini Vision API calls. REQUIRES GEMINI_API_KEY.
Run: python tests/test_13_gemini_vision.py
"""

import sys
import os
import tempfile
import struct
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("TEST 13: GEMINI VISION — IMAGE DESCRIPTION")
print("=" * 60)

# --- 1. Check API key ---
print("\n--- API Key Check ---")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("  [SKIP] GEMINI_API_KEY not set — skipping live Vision API tests")
    print("         Add GEMINI_API_KEY to .env and rerun")
    print("\n" + "=" * 60)
    print("TEST 13 SKIPPED (no API key)")
    print("=" * 60)
    sys.exit(0)
else:
    masked = api_key[:6] + "..." + api_key[-4:]
    print(f"  [PASS] GEMINI_API_KEY found: {masked}")

# --- 2. Import Pillow ---
print("\n--- Import Pillow ---")
try:
    from PIL import Image
    import io
    print("  [PASS] Pillow imported")
except ImportError as e:
    print(f"  [FAIL] Pillow not installed: {e}")
    print("         Run: pip install Pillow")
    sys.exit(1)

# --- 3. Import document loader ---
print("\n--- Import document_loader ---")
try:
    from app.services.document_loader import load_document, load_image
    print("  [PASS] document_loader imported")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

from app.utils.constants import GEMINI_VISION_MODEL
print(f"  [INFO] Vision model: {GEMINI_VISION_MODEL}")

# --- 4. Create a minimal valid PNG ---
def create_minimal_png(width=100, height=100, color=(255, 0, 0)):
    """Creates a simple colored PNG image in memory."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def save_temp_image(data, suffix=".png"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path

# --- 5. Basic image load + vision API ---
print("\n--- load_image() with PNG ---")
png_data = create_minimal_png(100, 100, color=(200, 100, 50))
path = save_temp_image(png_data, ".png")
print(f"  Image: 100x100 PNG, size={len(png_data)} bytes, path={path}")

try:
    import time
    t0 = time.time()
    docs = load_image(path)
    elapsed = time.time() - t0

    print(f"  API call time: {elapsed:.2f}s")
    print(f"  Documents returned: {len(docs)}")

    if len(docs) >= 1:
        print(f"  [PASS] Got {len(docs)} document(s)")
        doc = docs[0]
        content = doc.get("content", "")
        meta = doc.get("metadata", {})

        print(f"  Content length: {len(content)}")
        print(f"  Content: '{content[:200]}'")
        print(f"  Metadata: {meta}")

        if content.strip():
            print("  [PASS] Non-empty description from Gemini Vision")
        else:
            print("  [FAIL] Empty description returned by Vision API")

        if meta.get("type") == "image":
            print("  [PASS] type='image' in metadata")
        else:
            print(f"  [WARN] type='{meta.get('type')}', expected 'image'")

    else:
        print("  [FAIL] No documents returned from image")

except Exception as e:
    print(f"  [FAIL] load_image error: {e}")
    import traceback
    traceback.print_exc()
finally:
    os.unlink(path)

# --- 6. load_document dispatcher for image ---
print("\n--- load_document() Dispatcher for Image ---")
png_data2 = create_minimal_png(50, 50, color=(0, 200, 100))
path2 = save_temp_image(png_data2, ".png")
try:
    docs = load_document(path2, "test_image.png")
    if docs:
        print(f"  [PASS] Dispatcher returned {len(docs)} doc(s) for PNG")
        print(f"  Content: '{docs[0]['content'][:100]}'")
    else:
        print("  [FAIL] Dispatcher returned no docs for image")
except Exception as e:
    print(f"  [FAIL] Dispatcher error: {e}")
finally:
    os.unlink(path2)

# --- 7. JPEG image ---
print("\n--- JPEG Image ---")
jpg_img = Image.new("RGB", (80, 80), color=(50, 100, 200))
buf = io.BytesIO()
jpg_img.save(buf, format="JPEG")
jpg_data = buf.getvalue()
path3 = save_temp_image(jpg_data, ".jpg")
print(f"  Image: 80x80 JPEG, size={len(jpg_data)} bytes")
try:
    docs = load_image(path3)
    if docs:
        print(f"  [PASS] JPEG loaded, {len(docs)} doc(s)")
        print(f"  Content: '{docs[0]['content'][:100]}'")
    else:
        print("  [FAIL] No docs from JPEG")
except Exception as e:
    print(f"  [FAIL] JPEG image error: {e}")
finally:
    os.unlink(path3)

# --- 8. Vision model name check ---
print("\n--- Gemini Vision Model Check ---")
print(f"  Configured model: {GEMINI_VISION_MODEL}")
known_vision_models = [
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]
if GEMINI_VISION_MODEL in known_vision_models:
    print(f"  [PASS] Model is a known Gemini vision model")
else:
    print(f"  [WARN] '{GEMINI_VISION_MODEL}' not in known vision models list")
    print(f"         May still work if it's a new model variant")

# --- 9. Error handling for corrupted image ---
print("\n--- Corrupted Image Handling ---")
path4 = save_temp_image(b"not a valid image file at all", ".png")
try:
    docs = load_image(path4)
    if docs:
        print(f"  [WARN] Got {len(docs)} doc(s) from corrupted image — check error handling")
        print(f"         Content: '{docs[0]['content'][:100]}'")
    else:
        print(f"  [INFO] No docs from corrupted image — check if exception was swallowed")
except Exception as e:
    print(f"  [PASS] Exception raised for corrupted image: {type(e).__name__}: {str(e)[:100]}")
    print(f"  [INFO] Should be caught gracefully in upload pipeline to not break entire upload")
finally:
    os.unlink(path4)

print("\n" + "=" * 60)
print("TEST 13 COMPLETE")
print("=" * 60)
