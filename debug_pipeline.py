"""
debug_pipeline.py — runs the full pipeline end-to-end from Python,
printing every internal step so we can see exactly where files drop out.
Run: rag_proj_env\Scripts\python debug_pipeline.py
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

QUERY          = "explain the files in detail"
SESSION_FILES  = [
    "Blue Minimalist Project Presentation.pdf",
    "improved_wheat_logo.png",
    "Resume.pdf",
]

print(f"\n{'#'*65}")
print(f"  DEBUG PIPELINE")
print(f"  Query        : {QUERY!r}")
print(f"  Session files: {SESSION_FILES}")
print(f"{'#'*65}\n")

# ── 1. Check what's in the index ─────────────────────────────────────────────
print("=== STEP 0: Vector store contents ===")
from app.services.vector_store import get_all_chunks, get_indexed_sources
all_chunks = get_all_chunks()
sources_in_index = get_indexed_sources()
print(f"Total chunks in index : {len(all_chunks)}")
print(f"Sources in index      : {sources_in_index}")

chunk_counts = {}
for c in all_chunks:
    src = c["metadata"].get("source", "?")
    chunk_counts[src] = chunk_counts.get(src, 0) + 1
print(f"Chunks per file:")
for src, n in sorted(chunk_counts.items()):
    print(f"  {n:3d}  {src}")

# Check which session files are actually indexed
print(f"\nSession file index check:")
for f in SESSION_FILES:
    found = any(c["metadata"].get("source","") == f for c in all_chunks)
    tag = "OK" if found else "MISSING -- NOT INDEXED"
    print(f"  [{tag}]  {f}")

print()

# ── 2. Embed query ────────────────────────────────────────────────────────────
print("=== STEP 1: Embed query ===")
from app.services.embeddings import embed_query
query_text = "overview summary description of every uploaded file and document"
q_emb = embed_query(query_text)
print(f"Embedding dim : {len(q_emb)}")
print()

# ── 3. FAISS search ───────────────────────────────────────────────────────────
print("=== STEP 2: FAISS search_similar (k=30) ===")
from app.services.vector_store import search_similar
import numpy as np

candidates = search_similar(q_emb, k=30)
print(f"Total candidates returned: {len(candidates)}")

import re
_UUID = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_', re.I)
def clean(s): return _UUID.sub("", s)

def cosine(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0

print(f"\nAll candidates with similarity scores:")
for c in candidates:
    src  = clean(c["metadata"].get("source", "?"))
    page = c["metadata"].get("page", "?")
    typ  = c["metadata"].get("type", "text")
    sim  = cosine(q_emb, c["embedding"]) if c.get("embedding") else 0.0
    print(f"  sim={sim:.4f}  page={page}  type={typ:8s}  {src}")

# Per-file best sim
print(f"\nBest similarity per file:")
best_sim = {}
for c in candidates:
    src = clean(c["metadata"].get("source","?"))
    sim = cosine(q_emb, c["embedding"]) if c.get("embedding") else 0.0
    if sim > best_sim.get(src, 0.0):
        best_sim[src] = sim
for src, sim in sorted(best_sim.items(), key=lambda x: -x[1]):
    in_session = src in SESSION_FILES
    tag = "IN SESSION" if in_session else "NOT in session"
    print(f"  sim={sim:.4f}  [{tag}]  {src}")

print()

# ── 4. Apply source filter ────────────────────────────────────────────────────
print("=== STEP 3: Apply source_filter ===")
allowed = set(SESSION_FILES)
filtered = [c for c in candidates if clean(c["metadata"].get("source","")) in allowed]
print(f"After filter: {len(filtered)} / {len(candidates)} candidates kept")
for f in SESSION_FILES:
    count = sum(1 for c in filtered if clean(c["metadata"].get("source","")) == f)
    print(f"  {count:2d} chunks  {f}")

print()

# ── 5. Full retrieve() call ───────────────────────────────────────────────────
print("=== STEP 4: Full retrieve() call ===")
from app.services.retriever import retrieve
chunks = retrieve(
    query_text,
    include_all_sources=True,
    source_filter=SESSION_FILES,
)
print(f"\nretrieve() returned {len(chunks)} chunks")
result_sources = {}
for c in chunks:
    src = clean(c["metadata"].get("source","?"))
    result_sources[src] = result_sources.get(src, 0) + 1
print(f"Chunks per file in final result:")
for src, n in sorted(result_sources.items()):
    print(f"  {n:3d}  {src}")

print(f"\nFiles PRESENT in result:")
for f in SESSION_FILES:
    found = clean(f) in {clean(c["metadata"].get("source","")) for c in chunks}
    tag = "YES" if found else "NO  <--- DROPPED HERE"
    print(f"  [{tag}]  {f}")

print()

# ── 6. Context formatting ─────────────────────────────────────────────────────
print("=== STEP 5: format_context() ===")
from app.services.llm import format_context
context = format_context(chunks, summaries_first=True)
print(f"Context length: {len(context)} chars")
print(f"\nFiles mentioned in formatted context:")
for f in SESSION_FILES:
    base = os.path.splitext(f)[0]
    found = f in context or base in context
    tag = "YES" if found else "NO  <--- MISSING FROM CONTEXT"
    print(f"  [{tag}]  {f}")

print(f"\n{'#'*65}")
print(f"  DONE — check lines above for [NO] or [DROPPED HERE] markers")
print(f"{'#'*65}\n")
