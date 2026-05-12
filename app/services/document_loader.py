import os
import logging
import base64
from typing import List, Dict, Any

import fitz  # PyMuPDF
from docx import Document
from PIL import Image
import io

from google import genai
from google.genai import types

from app.utils.constants import GEMINI_API_KEY, GEMINI_VISION_MODEL
from app.utils.file_utils import get_file_extension

logger = logging.getLogger(__name__)

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


def describe_image_with_gemini(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Use Gemini Vision to describe an image."""
    try:
        response = client.models.generate_content(
            model=GEMINI_VISION_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type
                        ),
                        types.Part.from_text(
                            text="Describe this image in detail including any text, diagrams, logos, tables, colors and structure."
                        )
                    ]
                )
            ]
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini Vision error: {e}")
        raise


def load_pdf(file_path: str, original_filename: str = None) -> List[Dict[str, Any]]:
    """Extract text and images from PDF."""
    docs = []
    try:
        pdf = fitz.open(file_path)
        filename = original_filename or os.path.basename(file_path)

        for page_num, page in enumerate(pdf, start=1):
            # Extract text
            text = page.get_text().strip()
            if text:
                docs.append({
                    "content": text,
                    "metadata": {
                        "source": filename,
                        "page": page_num,
                        "type": "text"
                    }
                })

            # Extract images
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                # description = describe_image_with_gemini(image_bytes)
                img_mime = base_image.get("ext", "png")
                mime_type = f"image/{img_mime}" if img_mime in ["png", "jpeg", "webp"] else "image/png"
                description = describe_image_with_gemini(image_bytes, mime_type)
                docs.append({
                    "content": f"[Image on page {page_num}]: {description}",
                    "metadata": {
                        "source": filename,
                        "page": page_num,
                        "type": "image"
                    }
                })

        pdf.close()
        logger.info(f"PDF loaded: {filename}, {len(docs)} sections extracted.")
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
    return docs


def load_docx(file_path: str, original_filename: str = None) -> List[Dict[str, Any]]:
    """Extract text from DOCX."""
    docs = []
    try:
        filename = original_filename or os.path.basename(file_path)
        doc = Document(file_path)
        full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        if not full_text.strip():
            logger.warning(f"DOCX file is empty or has no text: {filename}")
            return docs
        docs.append({
            "content": full_text,
            "metadata": {
                "source": filename,
                "page": 1,
                "type": "text"
            }
        })
        logger.info(f"DOCX loaded: {filename}")
    except Exception as e:
        logger.error(f"Error loading DOCX {file_path}: {e}")
    return docs


def load_txt(file_path: str, original_filename: str = None) -> List[Dict[str, Any]]:
    """Read raw text file."""
    docs = []
    try:
        filename = original_filename or os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            logger.warning(f"TXT file is empty: {filename}")
            return docs
        docs.append({
            "content": text,
            "metadata": {
                "source": filename,
                "page": 1,
                "type": "text"
            }
        })
        logger.info(f"TXT loaded: {filename}")
    except Exception as e:
        logger.error(f"Error loading TXT {file_path}: {e}")
    return docs


def load_image(file_path: str, original_filename: str = None) -> List[Dict[str, Any]]:
    """Process image file using Gemini Vision."""
    docs = []
    try:
        filename = original_filename or os.path.basename(file_path)
        ext = get_file_extension(file_path)
        mime_map = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp"
        }
        mime_type = mime_map.get(ext, "image/png")

        with open(file_path, "rb") as f:
            image_bytes = f.read()

        description = describe_image_with_gemini(image_bytes, mime_type)
        docs.append({
            "content": description,
            "metadata": {
                "source": filename,
                "page": 1,
                "type": "image"
            }
        })
        logger.info(f"Image loaded: {filename}")
    except Exception as e:
        logger.error(f"Error loading image {file_path}: {e}")
        raise   # bubble up so upload route shows the real error, not silent ✅
    return docs


def load_document(file_path: str, original_filename: str = None) -> List[Dict[str, Any]]:
    """Route file to correct loader based on extension."""
    ext = get_file_extension(original_filename or file_path)
    if ext == "pdf":
        return load_pdf(file_path, original_filename)
    elif ext == "docx":
        return load_docx(file_path, original_filename)
    elif ext == "txt":
        return load_txt(file_path, original_filename)
    elif ext in {"png", "jpg", "jpeg", "webp"}:
        return load_image(file_path, original_filename)
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return []