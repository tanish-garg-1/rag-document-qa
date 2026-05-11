import re
import logging
from collections import OrderedDict
from typing import List, Dict, Generator, Tuple

from groq import Groq

from app.utils.constants import GROQ_API_KEY, GROQ_MODEL, MAX_CONTEXT_CHARS, MAX_CHUNK_CHARS

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

# Questions matching these patterns almost always need reasoning beyond the docs
_ANALYTICAL_RE = re.compile(
    r'\b(good deal|worth it|suitable|is it (good|worth|right)|'
    r'should (i|he|she|they|tanish)|recommend|better (option|choice|plan)|'
    r'compare|pros and cons|evaluate|assess|would you suggest|'
    r'strengthen(s)? his|improve his|next (project|step|job)|'
    r'what (job|role|career|project)s? should)\b',
    re.IGNORECASE,
)

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

    # Fast keyword pre-screen: if question clearly needs reasoning, skip LLM call
    if _ANALYTICAL_RE.search(question):
        logger.info("Query pre-classified as ANALYTICAL (keyword match)")
        return MODE_ANALYTICAL

    history_snippet = history[-800:] if history else ""

    prompt = f"""You are a query router for a RAG (document Q&A) system.

Classify the user's question into EXACTLY ONE of these three modes:

DOCUMENT   — The answer exists in the uploaded files. Use this when:
             • The user asks about content, facts, or details IN the documents
             • The user mentions a specific document by name or type (policy, resume, PDF, file)
             • The question asks what a document "says", "states", "mentions", or "covers"
             • The question asks whether something is "covered", "included", or "excluded" in a policy
             • The user wants a description or summary of the uploaded files
             Examples: "What are his skills?", "Summarise the PDF",
                       "What does the policy say about grace period?",
                       "Does the insurance cover LASIK?",
                       "What is the cumulative bonus in the SBI policy?",
                       "What does the resume mention about his internship?"

ANALYTICAL — The question uses the uploaded files as context but ALSO requires
             real-world knowledge, recommendations, comparisons, or reasoning that
             goes beyond the text in the files.
             Examples: "What jobs should he apply for based on his resume?",
                       "What project should he build to strengthen his profile?",
                       "Is the SBI policy a good deal for Tanish?",
                       "Would this insurance policy suit a young professional?",
                       "Is this research methodology sound?",
                       "How does his experience compare to industry requirements?"

GENERAL    — The question has nothing to do with the uploaded files and needs
             broad AI knowledge to answer. Only use this if the question cannot
             be answered from the documents at all.
             Examples: "Explain transformer architecture",
                       "What is LangChain?",
                       "How does RAG work in general?"

IMPORTANT RULES:
- If the question asks WHAT a document says/contains/covers → DOCUMENT
- If the question asks WHETHER something is a good deal, suitable, recommended,
  or requires evaluation/judgment/comparison → ANALYTICAL (even if it names a document)
- If in doubt between DOCUMENT and ANALYTICAL, prefer DOCUMENT
- Only use GENERAL for questions completely unrelated to the uploaded files

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
1. Answer using ONLY the information in the context below. Always give a complete,
   fresh answer — never skip or defer because of anything in the chat history.
2. Answer the user's SPECIFIC question directly. Do NOT list or describe every file
   unless the user explicitly asks to "describe", "summarise", or "list" the files.
3. For explicit describe/summarise requests, write ONE short paragraph per file:

   **<filename>**
   <One natural prose paragraph: what type of doc, who it's about, what it covers,
    any key facts like name/role/company/topic/skills. No bullet points.>

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
1. Answer the user's SPECIFIC question directly. Do NOT list or describe each document
   separately. Do NOT use bold filename headers like **filename.pdf**. Synthesize the
   relevant information from all files into flowing, natural prose.
2. Ground your answer in the uploaded documents first — weave in relevant facts
   naturally without citation numbers.
3. Then EXTEND with your general knowledge, reasoning, and recommendations beyond
   what the files contain. Label this section "💡 AI Recommendation:".
4. Be specific and actionable — tailor suggestions to the person's actual
   skills and experience shown in the documents.
5. Never fabricate facts about the documents. If information is missing, say so.
6. Do NOT include citation numbers like [1], [2] anywhere in your answer.

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

def format_context(chunks: List[Dict], summaries_first: bool = False) -> str:
    """
    Format retrieved chunks grouped by source file.
    Summary chunks (page=0 / type=summary) are sorted first within each file.
    Uses OrderedDict so citation numbers match what citation.py generates.

    Enforces two limits to stay within the LLM's token budget:
      - Each chunk's content is truncated to MAX_CHUNK_CHARS characters.
      - Total context is capped at MAX_CONTEXT_CHARS; chunks are dropped
        (least-relevant last, since they're added in relevance order) once
        the budget is exhausted.
    """
    def _chunk_sort_key(c):
        meta = c.get("metadata", {})
        is_summary = (meta.get("type") == "summary" or meta.get("page") == 0)
        # For describe-all queries summaries go FIRST (we want the file overview
        # in budget before content chunks). For targeted queries summaries go
        # LAST so specific clauses/facts reach the LLM before summaries crowd them out.
        summary_rank = 0 if summaries_first else 1
        return (summary_rank if is_summary else (1 - summary_rank), meta.get("page", 999))

    grouped: dict = OrderedDict()
    for chunk in sorted(chunks, key=_chunk_sort_key):
        raw_source = chunk.get("metadata", {}).get("source", "unknown")
        source = _clean_source(raw_source)
        page = chunk.get("metadata", {}).get("page", "N/A")
        content = chunk.get("content", "")
        # Truncate individual chunk to avoid a single huge chunk blowing budget
        if len(content) > MAX_CHUNK_CHARS:
            content = content[:MAX_CHUNK_CHARS] + " …[truncated]"
        grouped.setdefault(source, []).append((page, content))

    parts = []
    ref_num = 1
    total_chars = 0
    for source, entries in grouped.items():
        file_parts = [f"--- File: {source} ---"]
        for page, content in entries:
            entry_text = f"[{ref_num}] Page {page}:\n{content}"
            if total_chars + len(entry_text) > MAX_CONTEXT_CHARS:
                logger.info(
                    f"Context budget reached at ref [{ref_num}] "
                    f"({total_chars}/{MAX_CONTEXT_CHARS} chars) — dropping remaining chunks."
                )
                # Flush whatever we've collected for this file so far
                if len(file_parts) > 1:
                    parts.append("\n".join(file_parts))
                return "\n\n".join(parts)
            file_parts.append(entry_text)
            total_chars += len(entry_text)
            ref_num += 1
        parts.append("\n".join(file_parts))

    return "\n\n".join(parts)


# ── Streaming answer ───────────────────────────────────────────────────────────

def stream_answer(
    question: str,
    chunks: List[Dict],
    history: str,
    mode: str = MODE_DOCUMENT,
    summaries_first: bool = False,
) -> Generator[str, None, None]:
    """
    Stream answer tokens from Groq LLM.
    mode controls which prompt template and knowledge sources are used.
    Yields text chunks as they arrive.
    """
    context = format_context(chunks, summaries_first=summaries_first)
    prompt = build_prompt(context, history, question, mode=mode)

    try:
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.3 if mode == MODE_ANALYTICAL else 0.2,
            max_tokens=1024 if mode in (MODE_ANALYTICAL, MODE_GENERAL) else 768
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

# Patterns that signal the question needs prior context to be understood
_NEEDS_REWRITE_RE = re.compile(
    r'\b(it|this|that|these|those|he|she|they|them|their|his|her|its)\b',
    re.IGNORECASE,
)

def rewrite_query(question: str, history: str) -> str:
    """
    Rewrite ONLY vague follow-up questions into standalone search queries.
    Self-contained questions (> 5 words AND no ambiguous pronouns) are
    returned unchanged — rewriting them causes the LLM to inject concepts
    from previous AI answers, polluting retrieval.
    """
    if not history.strip():
        return question

    # Skip rewriting if the question is already self-contained
    word_count = len(question.split())
    has_ambiguous_ref = bool(_NEEDS_REWRITE_RE.search(question))
    if word_count > 5 and not has_ambiguous_ref:
        logger.info(f"Query rewrite skipped (self-contained, {word_count} words): '{question}'")
        return question

    prompt = f"""Given the chat history below, rewrite the user's latest question into a
complete, standalone search query that captures the full context and intent.

RULES:
- Keep the rewritten query SHORT (under 12 words).
- Resolve pronouns ("it", "he", "that file") using history.
- Do NOT add biographical details, skills, or concepts from AI answers — only
  resolve what the pronouns/references refer to.
- Return ONLY the rewritten query. No explanation, no quotes.

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
