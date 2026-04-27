import logging
from fastapi import FastAPI
from app.routes.upload import router as upload_router
from app.routes.query import router as query_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Document QA System",
    description="Multimodal conversational RAG with FAISS, Gemini and Groq",
    version="1.0.0"
)

app.include_router(upload_router)
app.include_router(query_router)


@app.get("/")
def root():
    return {"message": "RAG System is running!"}


@app.get("/stats")
def stats():
    from app.services.vector_store import get_index_stats
    return get_index_stats()