import re
import logging
from collections import OrderedDict
from typing import List, Dict, Generator, Tuple

from groq import Groq

from app.utils.constants import GROQ_API_KEY, GROQ_MODEL

# Strip leading UUID prefix (e.g. "a1b2c3d4-..._") from source filenames
_UUID_PREFIX = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_',
    re.IGNORECASE
)

def _clean_source(name: str) -> str:
    return _UUID_PREFIX.sub("", name)

logger = logging.getLogger(__name__)

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


# ── Query mode constants ───────────────────────────────────────────────────────
MODE_DOCUMENT   = "DOCUMENT"    # answer strictly from uploaded files
MODE_ANALYTICAL = "ANALYTICAL"  # file context + LLM world knowledge combined
MODE_GENERAL    = "GENERAL"     # primarily LLM knowledge; files referenced where relevant


# ── Query classifier ───────────────────────────────────────────────────────────

def classify_query(question: str, history: str, has_documents: bool) -> str:
    """
    Use Groq to decide which answering mode to use.

    Returns one of: "DOCUMENT", "ANALYTICAL", "GENERAL"

    - DOCUMENT   : question can be answered from uploaded files alone
                   (e.g. "What are his skills?", "Summarise the PDF")
    - ANALYTICAL : question needs file content AS CONTEXT but also requires
                   AI reasoning / world knowledge to give a useful answer
                   (e.g. "What jobs should he apply for?",
                         "What project should he build next?",
                         "How does this compare to industry standards?")
    - GENERAL    : question goes entirely beyond file content and needs
                   general AI knowledge
                   (e.g. follow-up questions on a research paper's topic,
                         "Explain transformers", "What is AML?")
    """
    if not has_documents:
        return MODE_GENERAL

    history_snippet = history[-800:] if history else ""

    prompt = f"""You are a query router for a RAG (document Q&A) system.

Classify the user's question into EXACTLY ONE of these three modes:

DOCUMENT   — The answer exists in the uploaded files. The user is asking about file
             content, facts in the documents, or wants a description/summary of the files.
             Examples: "What are his skills?", "Summarise the PDF", "What is on page 3?"

ANALYTICAL — The question uses the uploaded files as context but ALSO requires
             real-world knowledge, recommendations, comparisons, or reasoning that
             goes beyond the text in the files.
             Examples: "What jobs should he apply for based on his resume?",
                       "What project should he build to strengthen his profile?",
                       "Is this research methodology sound?",
                       "How does his experience compare to industry requirements?"

GENERAL    — The question is not (or barely) covered by the uploaded files and
             needs broad AI knowledge to answer properly.
             Examples: "Explain transformer architecture",
                       "What are current trends in AML detection?",
                       "What is LangChain?", follow-up deep-dives on paper topics.

Recent chat history (for context):
{history_snippet}

User question: {question}

Reply with EXACTLY one word — DOCUMENT, ANALYTICAL, or GENERAL:"""

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0.0,
            max_tokens=10
        )
        mode = resp.choices[0].message.content.strip().upper()
        if mode not in (MODE_DOCUMENT, MODE_ANALYTICAL, MODE_GENERAL):
            mode = MODE_DOCUMENT   # safe fallback
        logger.info(f"Query classified as: {mode}")
        return mode
    except Exception as e:
        logger.error(f"Query classification failed: {e}")
        return MODE_DOCUMENT


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_document_prompt(context: str, history: str, question: str) -> str:
    """Strict RAG mode — answer only from uploaded files."""
    return f"""You are a helpful assistant answering questions about uploaded documents.

RULES:
1. Answer using ONLY the information in the context below.
2. When the user asks to describe or summarise the uploaded files, write ONE short
   paragraph per file (2-4 sentences). Format exactly like this:

   **<filename>**
   <One natural prose paragraph: what type of doc, who it's about, what it covers,
    any key facts like name/role/company/topic/skills. No bullet points. No headers
    like "Key Identifiers". Sound like a human wrote it.>

3. For specific factual questions, answer directly and concisely.
4. Never make up information not in the context.
5. Do NOT include citation numbers like [1], [2] anywhere in your answer.
6. If information is missing from the retrieved chunks, say so briefly — do not
   pad the answer with generic filler.

Context (grouped by source):
{context}

Chat History:
{history}

Question: {question}

Answer:"""


def _build_analytical_prompt(context: str, history: str, question: str) -> str:
    """Hybrid mode — use file content + Groq world knowledge together."""
    return f"""You are an expert AI assistant. You have access to content from the user's
uploaded documents AND your general training knowledge. Use both to give the best answer.

RULES:
1. Ground your answer in the uploaded documents first — state the relevant facts from
   the files naturally in your answer without citation numbers.
2. Then EXTEND your answer with your general knowledge, reasoning, and recommendations
   that go beyond what the files contain. Clearly label this section with:
   "💡 AI Recommendation:" or similar.
3. Be specific and actionable — for career/project questions, give concrete suggestions
   tailored to the person's actual skills and experience shown in the documents.
4. Never fabricate facts about the documents. If information is not in the context, say so.
5. Do NOT include citation numbers like [1], [2] anywhere in your answer.

Context from uploaded documents (grouped by source):
{context}

Chat History:
{history}

Question: {question}

Answer:"""


def _build_general_prompt(context: str, history: str, question: str) -> str:
    """General knowledge mode — LLM knowledge primary, files referenced where relevant."""
    return f"""You are a knowledgeable AI assistant. The user's question goes beyond the
content of uploaded documents. Answer using your general training knowledge.

RULES:
1. Answer the question fully using your general knowledge.
2. If any part of the uploaded context below is relevant, weave it naturally into your answer.
3. Clearly indicate at the start: "ℹ️ This answer uses general AI knowledge" so the
   user knows it is not sourced exclusively from their files.
4. Be accurate, detailed, and helpful.
5. Do NOT include citation numbers like [1], [2] anywhere in your answer.

Uploaded document context (use where relevant):
{context}

Chat History:
{history}

Question: {question}

Answer:"""


def build_prompt(context: str, history: str, question: str,
                 mode: str = MODE_DOCUMENT) -> str:
    """Route to the correct prompt builder based on query mode."""
    if mode == MODE_ANALYTICAL:
        return _build_analytical_prompt(context, history, question)
    elif mode == MODE_GENERAL:
        return _build_general_prompt(context, history, question)
    else:
        return _build_document_prompt(context, history, question)


# ── Context formatter ──────────────────────────────────────────────────────────

def format_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks grouped by source file.
    Summary chunks (page=0 / type=summary) are sorted first within each file.
    Uses OrderedDict so citation numbers match what citation.py generates.
    """
    def _chunk_sort_key(c):
        meta = c.get("metadata", {})
        is_summary = (meta.get("type") == "summary" or meta.get("page") == 0)
        return (0 if is_summary else 1, meta.get("page", 999))

    grouped: dict = OrderedDict()
    for chunk in sorted(chunks, key=_chunk_sort_key):
        raw_source = chunk.get("metadata", {}).get("source", "unknown")
        source = _clean_source(raw_source)
        page = chunk.get("metadata", {}).get("page", "N/A")
        content = chunk.get("content", "")
        grouped.setdefault(source, []).append((page, content))

    parts = []
    ref_num = 1
    for source, entries in grouped.items():
        file_parts = [f"--- File: {source} ---"]
        for page, content in entries:
            file_parts.append(f"[{ref_num}] Page {page}:\n{content}")
            ref_num += 1
        parts.append("\n".join(file_parts))

    return "\n\n".join(parts)


# ── Streaming answer ───────────────────────────────────────────────────────────

def stream_answer(
    question: str,
    chunks: List[Dict],
    history: str,
    mode: str = MODE_DOCUMENT
) -> Generator[str, None, None]:
    """
    Stream answer tokens from Groq LLM.
    mode controls which prompt template and knowledge sources are used.
    Yields text chunks as they arrive.
    """
    context = format_context(chunks)
    prompt = build_prompt(context, history, question, mode=mode)

    try:
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.3 if mode == MODE_ANALYTICAL else 0.2,
            max_tokens=1536 if mode in (MODE_ANALYTICAL, MODE_GENERAL) else 1024
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

        logger.info(f"Streaming complete (mode={mode}).")

    except Exception as e:
        logger.error(f"Groq LLM error: {e}")
        yield f"Error generating response: {str(e)}"


# ── Document summariser (called at upload time) ────────────────────────────────

def generate_document_summary(filename: str, full_text: str) -> str:
    """
    Use Groq to generate a structured summary of the entire document at upload time.
    Stored as a priority summary chunk so broad queries always have complete context.
    """
    truncated = full_text[:12_000]
    if len(full_text) > 12_000:
        truncated += "\n[... content truncated for summary ...]"

    prompt = f"""You are a document analyst. Read the content below and write a structured summary.

Rules:
- Identify the DOCUMENT TYPE first (resume, presentation, report, image description, text file, etc.)
- For a RESUME: state the candidate's full name, current role/degree, institution, key technical
  skills, notable projects, and work experience in detail.
- For a PRESENTATION: state the exact title, presenter(s), main topic, and 2-3 key takeaways.
- For a REPORT or ARTICLE: state the subject, author (if present), and main findings.
- For any other type: describe what the document contains and its purpose.
- Write in plain prose, 4-6 sentences. Be specific — include names, numbers, technologies.
- Do NOT say "the document says" or "the text mentions" — just state the facts directly.

Filename: {filename}

Document Content:
{truncated}

Summary:"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0.1,
            max_tokens=350
        )
        summary = response.choices[0].message.content.strip()
        logger.info(f"Document summary generated for '{filename}'")
        return summary
    except Exception as e:
        logger.error(f"Document summary generation failed for '{filename}': {e}")
        return ""


# ── Query rewriter ─────────────────────────────────────────────────────────────

def rewrite_query(question: str, history: str) -> str:
    """
    Rewrite a vague follow-up question into a standalone search query
    using the chat history for context.
    """
    if not history.strip():
        return question

    prompt = f"""Given the chat history below, rewrite the user's latest question into a
complete, standalone search query that captures the full context and intent.
Return ONLY the rewritten query, nothing else. No explanation, no quotes.

Chat History:
{history}

Latest Question: {question}

Rewritten Query:"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0.0,
            max_tokens=150
        )
        rewritten = response.choices[0].message.content.strip()
        logger.info(f"Query rewritten: '{question}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.error(f"Query rewriting failed: {e}")
        return question
