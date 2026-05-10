import logging
import os
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.utils.file_utils import save_uploaded_file, is_supported_file
from app.services.document_loader import load_document
from app.services.chunking import chunk_documents
from app.services.embeddings import embed_texts
from app.services.vector_store import add_chunks_to_store
from app.utils.constants import MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Accept multiple files, process them and update FAISS index.
    """
    results = []

    for file in files:
        filename = file.filename

        # Validate file type
        if not is_supported_file(filename):
            results.append({
                "filename": filename,
                "status": "failed",
                "reason": f"Unsupported file type."
            })
            continue

        try:
            # Read file and enforce size limit
            file_bytes = await file.read()
            if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
                results.append({
                    "filename": filename,
                    "status": "failed",
                    "reason": f"File too large. Max size is {MAX_UPLOAD_SIZE_MB} MB."
                })
                continue

            file_path = save_uploaded_file(file_bytes, filename)
            logger.info(f"File saved: {file_path}")

            # Load and extract content (pass original filename so metadata is clean)
            documents = load_document(file_path, original_filename=filename)
            if not documents:
                results.append({
                    "filename": filename,
                    "status": "failed",
                    "reason": "No content extracted."
                })
                continue

            # Chunk documents
            chunks = chunk_documents(documents)
            if not chunks:
                results.append({
                    "filename": filename,
                    "status": "failed",
                    "reason": "No chunks created."
                })
                continue

            # Embed chunks
            texts = [c["content"] for c in chunks]
            embeddings = embed_texts(texts)

            # Store in FAISS
            add_chunks_to_store(chunks, embeddings)

            results.append({
                "filename": filename,
                "status": "success",
                "chunks_added": len(chunks)
            })
            logger.info(f"File processed successfully: {filename}")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            results.append({
                "filename": filename,
                "status": "failed",
                "reason": str(e)
            })

    return {"results": results}





# Add a /clear API endpoint to reset FAISS from the UI
@router.post("/clear")
def clear_index():
    """Delete all FAISS index files and reset the vector store."""
    import shutil
    from app.utils.constants import FAISS_INDEX_DIR
    from app.utils.file_utils import ensure_directories

    try:
        # Delete all files in faiss_index folder
        if os.path.exists(FAISS_INDEX_DIR):
            shutil.rmtree(FAISS_INDEX_DIR)

        # Recreate empty directory
        ensure_directories()

        logger.info("FAISS index cleared successfully.")
        return {"status": "success", "message": "Vector store cleared. You can now upload new documents."}

    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(status_code=500, detail=str(e))