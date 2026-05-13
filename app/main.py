import logging
from collections import deque
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.upload import router as upload_router
from app.routes.query import router as query_router

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ── In-memory log buffer (last 200 lines) — readable via /logs endpoint ───────
class _BufferHandler(logging.Handler):
    def __init__(self, maxlen: int = 200):
        super().__init__()
        self._buf: deque[str] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord):
        self._buf.append(self.format(record))

    def get_logs(self):
        return list(self._buf)


_buffer_handler = _BufferHandler()
_buffer_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                      datefmt="%H:%M:%S")
)
logging.getLogger().addHandler(_buffer_handler)

logger = logging.getLogger(__name__)

# ── Global debug state — updated by query pipeline, read via /debug-last ──────
# This is the single source of truth for what happened in the last query.
# Streamlit reads this after every response to show a live debug panel.
_last_debug: dict = {}

def set_debug(data: dict):
    """Called by the query pipeline to record what happened."""
    global _last_debug
    _last_debug = data

def get_debug() -> dict:
    return _last_debug

app = FastAPI(
    title="RAG Document QA System",
    description="Multimodal conversational RAG with FAISS, Gemini and Groq",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    from app.services.vector_store import get_indexed_sources
    return {"sources": get_indexed_sources()}


@app.get("/debug-last")
def debug_last():
    """Return debug info from the most recent query — shown in Streamlit panel."""
    return get_debug()


@app.get("/logs")
def get_logs(n: int = 50):
    lines = _buffer_handler.get_logs()
    return {"logs": lines[-n:]}


@app.get("/list-models")
def list_models():
    from google import genai
    from app.utils.constants import GEMINI_API_KEY
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        models = [
            m.name for m in client.models.list()
            if "generateContent" in (m.supported_actions or [])
        ]
        return {"models": sorted(models)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/test-gemini")
def test_gemini():
    import base64
    from google import genai
    from google.genai import types
    from app.utils.constants import GEMINI_API_KEY, GEMINI_VISION_MODEL

    ONE_PX_PNG = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
    )
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        resp = client.models.generate_content(
            model=GEMINI_VISION_MODEL,
            contents=[
                types.Content(role="user", parts=[
                    types.Part.from_bytes(data=ONE_PX_PNG, mime_type="image/png"),
                    types.Part.from_text(text="What colour is this image?")
                ])
            ]
        )
        return {"status": "ok", "response": resp.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}
