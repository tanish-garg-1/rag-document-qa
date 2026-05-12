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
CHUNK_OVERLAP = 100   # was 50; larger overlap keeps section headers in adjacent chunks

# Retrieval
MMR_K = 10         # chunks per query (coverage pass may add more for missing sources)
MMR_LAMBDA = 0.6   # slightly more relevance-weighted (was 0.5)
RELEVANCE_RESCUE_K = 8  # was 4; raised to 8 so dense-list exclusion chunks aren't missed

# Memory
MAX_MEMORY_MESSAGES = 20
INFERENCE_MEMORY_MESSAGES = 4   # was 8; reduced to keep history tokens ≤ ~600

# Models
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-2"
GEMINI_VISION_MODEL = "gemini-2.5-flash"
#GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL = "llama-3.1-8b-instant"  # 500K tokens/day free tier

# Embedding dimension for text-embedding-004
EMBEDDING_DIM = 3072

# Upload limits
MAX_UPLOAD_SIZE_MB = 50
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# LLM context limits
# llama-3.1-8b-instant has 128K context window; 12 000 chars ~ 3 000 tokens,
# leaving plenty of room for the system prompt, history, and question.
MAX_CONTEXT_CHARS = 12_000       # raised from 5 500 — was too tight for 3+ files
MAX_CHUNK_CHARS   = 600          # slightly wider chunks for richer content

# Coverage pass — minimum cosine similarity for a source to be force-added
# Sources whose best chunk scores below this are unrelated to the query and
# should not pollute the context (avoids "describe all files" syndrome)
COVERAGE_MIN_SIMILARITY  = 0.60   # high bar — only clearly relevant sources force-added
SUMMARY_MIN_SIMILARITY   = 0.65   # raised — ML papers score ~0.5 on insurance queries; block them