import os
import uuid
import logging
from app.utils.constants import UPLOAD_DIR, FAISS_INDEX_DIR

logger = logging.getLogger(__name__)

def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    logger.info("Directories ensured.")

def save_uploaded_file(file_bytes: bytes, filename: str) -> str:
    """Save uploaded file to uploads directory, return full path."""
    ensure_directories()
    unique_name = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    logger.info(f"Saved file: {file_path}")
    return file_path

def get_file_extension(filename: str) -> str:
    """Return lowercase file extension without dot."""
    return os.path.splitext(filename)[-1].lower().strip(".")

def is_supported_file(filename: str) -> bool:
    """Check if file type is supported."""
    supported = {"pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"}
    return get_file_extension(filename) in supported