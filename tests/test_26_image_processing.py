"""
Test 26 — Image Processing (Gemini Vision)
Directly calls describe_image_with_gemini() on improved_wheat_logo.png
from files_for_test/ and prints the description.
Run from project root: python tests/test_26_image_processing.py
"""

import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "files_for_test",
    "improved_wheat_logo.png"
)


def test_image_processing():
    print(f"\n{'='*60}")
    print("Test 26 — Gemini Vision Image Processing")
    print(f"{'='*60}")
    print(f"Image: {IMAGE_PATH}")

    # Check file exists
    if not os.path.exists(IMAGE_PATH):
        print(f"FAIL — File not found: {IMAGE_PATH}")
        return False

    file_size = os.path.getsize(IMAGE_PATH) / 1024
    print(f"File size: {file_size:.1f} KB")

    # Load image bytes
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    # Show which key + model is actually loaded
    from app.utils.constants import GEMINI_API_KEY, GEMINI_VISION_MODEL
    masked = f"{GEMINI_API_KEY[:8]}...{GEMINI_API_KEY[-4:]}" if GEMINI_API_KEY else "NOT SET"
    print(f"API Key loaded : {masked}")
    print(f"Vision model   : {GEMINI_VISION_MODEL}")

    print("\nCalling Gemini Vision API...")

    try:
        from app.services.document_loader import describe_image_with_gemini
        description = describe_image_with_gemini(image_bytes, mime_type="image/png")

        print(f"\n[PASS] SUCCESS — Gemini Vision responded!\n")
        print(f"Description ({len(description)} chars):")
        print("-" * 60)
        print(description)
        print("-" * 60)
        return True

    except Exception as e:
        print(f"\n[FAIL] FAILED — {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    success = test_image_processing()
    sys.exit(0 if success else 1)
