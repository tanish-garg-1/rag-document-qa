import os
from dotenv import load_dotenv, find_dotenv

# find_dotenv() walks up the directory tree until it finds a .env file
# This works whether running from the worktree, the project root, or Docker
load_dotenv(find_dotenv(usecwd=True) or find_dotenv())

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "data", "faiss_index")

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
MMR_K = 6          # chunks per query (coverage pass may add more for missing sources)
MMR_LAMBDA = 0.6   # slightly more relevance-weighted (was 0.5)

# Memory
MAX_MEMORY_MESSAGES = 20
INFERENCE_MEMORY_MESSAGES = 8

# Models
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-2"
GEMINI_VISION_MODEL = "gemini-2.5-flash-lite"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Embedding dimension for text-embedding-004
EMBEDDING_DIM = 3072

# Upload limits
MAX_UPLOAD_SIZE_MB = 50
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024