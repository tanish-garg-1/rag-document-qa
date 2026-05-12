import sys
import os
import logging
from collections import deque
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.upload import router as upload_router
from app.routes.query import router as query_router

# ── Debug log file — readable with: Get-Content -Wait backend.log ─────────────
_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend.log")

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                          datefmt="%H:%M:%S")

# File handler — writes everything to backend.log (max 2 MB, 1 backup)
_file_handler = RotatingFileHandler(_LOG_FILE, maxBytes=2_000_000, backupCount=1,
                                     encoding="utf-8")
_file_handler.setFormatter(_fmt)
_file_handler.setLevel(logging.DEBUG)

# Stream handler — also try to print to stderr (more reliable than stdout under reload)
_stream_handler = logging.StreamHandler(sys.stderr)
_stream_handler.setFormatter(_fmt)
_stream_handler.setLevel(logging.DEBUG)

# Root logger picks up all app.* loggers
logging.basicConfig(level=logging.DEBUG, handlers=[_file_handler, _stream_handler])

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
_buffer_handler.setFormatter(_fmt)
logging.getLogger().addHandler(_buffer_handler)

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


@app.get("/logs")
def get_logs(n: int = 50):
    """Return the last n backend log lines (for in-app debugging)."""
    lines = _buffer_handler.get_logs()
    return {"logs": lines[-n:]}


@app.get("/list-models")
def list_models():
    """List all available Gemini models that support generateContent."""
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
    """Test whether Gemini Vision API is reachable and within quota."""
    import base64
    from google import genai
    from google.genai import types
    from app.utils.constants import GEMINI_API_KEY, GEMINI_VISION_MODEL

    # Tiny 1x1 white PNG — minimal API call to check quota
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