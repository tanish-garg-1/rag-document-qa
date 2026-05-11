import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

# Allow cross-origin requests (frontend on different port / domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production to your Streamlit origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get("/sources")
def sources():
    """Return the list of unique source files currently indexed."""
    from app.services.vector_store import get_indexed_sources
    return {"sources": get_indexed_sources()}