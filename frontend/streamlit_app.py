import time
import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_files" not in st.session_state:
    st.session_state.session_files = []
if "scope_initialized" not in st.session_state:
    try:
        srcs = requests.get(f"{API_BASE}/sources", timeout=3).json().get("sources", [])
        st.session_state.session_files = list(srcs)
    except Exception:
        pass
    st.session_state.scope_initialized = True
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

# ── Fetch backend data (runs every render) ─────────────────────────────────────
try:
    backend_online = requests.get(f"{API_BASE}/", timeout=1).status_code == 200
except Exception:
    backend_online = False

try:
    dbg = requests.get(f"{API_BASE}/debug-last", timeout=1).json() if backend_online else {}
except Exception:
    dbg = {}

try:
    raw_logs = (
        requests.get(f"{API_BASE}/logs?n=60", timeout=1).json().get("logs", [])
        if backend_online else []
    )
except Exception:
    raw_logs = []

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Document QA")
    st.caption("Powered by Gemini · Groq · FAISS")
    st.divider()

    # Upload
    st.subheader("⬆️ Upload Documents")
    uploaded_files = st.file_uploader(
        "Supported: PDF, DOCX, TXT, PNG, JPG, WEBP",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"]
    )
    if st.button("🚀 Process & Index Files", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Select at least one file first.")
        else:
            with st.spinner("Indexing..."):
                files = [("files", (f.name, f.read(), f.type)) for f in uploaded_files]
                try:
                    results = requests.post(f"{API_BASE}/upload", files=files).json().get("results", [])
                    for r in results:
                        fname = r["filename"]
                        if r["status"] == "success":
                            st.success(f"✅ {fname} — {r['chunks_added']} chunks")
                            if fname not in st.session_state.session_files:
                                st.session_state.session_files.append(fname)
                        elif r["status"] == "duplicate":
                            st.info(f"↩ {fname} — already indexed")
                            if fname not in st.session_state.session_files:
                                st.session_state.session_files.append(fname)
                        else:
                            st.error(f"❌ {fname} — {r.get('reason', 'Error')}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    st.divider()

    # Index stats
    st.subheader("📊 Index Stats")
    if st.button("🔄 Refresh Stats", use_container_width=True):
        try:
            s = requests.get(f"{API_BASE}/stats").json()
            c1, c2 = st.columns(2)
            c1.metric("Vectors", s["total_vectors"])
            c2.metric("Chunks", s["total_chunks"])
            for src in requests.get(f"{API_BASE}/sources").json().get("sources", []):
                st.caption(f"• {src}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    # Memory
    st.subheader("🧠 Memory")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        try:
            requests.delete(f"{API_BASE}/history")
        except Exception:
            pass
        st.rerun()
    if st.button("👁️ View Memory", use_container_width=True):
        try:
            for msg in requests.get(f"{API_BASE}/history").json().get("messages", []):
                icon = "🧑" if msg["role"] == "user" else "🤖"
                with st.expander(f"{icon} {msg['content'][:45]}..."):
                    st.write(msg["content"])
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    # Active scope
    st.subheader("🗂️ Active Scope")
    if st.session_state.session_files:
        st.caption(f"**{len(st.session_state.session_files)} file(s) in scope:**")
        for f in st.session_state.session_files:
            st.caption(f"• {f}")
        if st.button("🔄 Reload from index", use_container_width=True):
            try:
                srcs = requests.get(f"{API_BASE}/sources", timeout=3).json().get("sources", [])
                st.session_state.session_files = list(srcs)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        if st.button("🔓 Clear scope", use_container_width=True):
            st.session_state.session_files = []
            st.rerun()
    else:
        st.warning("No scope — ALL indexed files searched.")
        if st.button("📂 Load files into scope", use_container_width=True):
            try:
                srcs = requests.get(f"{API_BASE}/sources", timeout=3).json().get("sources", [])
                if srcs:
                    st.session_state.session_files = list(srcs)
                    st.rerun()
                else:
                    st.info("No files indexed yet.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # Vector store
    st.subheader("🗄️ Vector Store")
    if st.button("⚠️ Clear Index", use_container_width=True):
        try:
            if requests.post(f"{API_BASE}/clear").json().get("status") == "success":
                st.session_state.session_files = []
                st.success("Index cleared.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    # Auto-refresh toggle
    st.subheader("⚙️ Monitor Settings")
    live = st.toggle("🔄 Live refresh (3 s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = live
    if live:
        st.caption("Monitor panel auto-refreshes every 3 s.")


# ── Main area: chat (left 60%) + monitor (right 40%) ──────────────────────────
chat_col, monitor_col = st.columns([3, 2], gap="large")

# ── RIGHT: Monitor panel ───────────────────────────────────────────────────────
with monitor_col:
    st.subheader("💻 Backend Monitor")

    # Status badge
    if backend_online:
        st.success("● Online")
    else:
        st.error("● Offline")

    st.divider()

    # Last query trace
    st.markdown("**🔍 Last Query Trace**")
    if dbg:
        m1, m2, m3 = st.columns(3)
        m1.metric("Mode", dbg.get("mode", "—"))
        m2.metric("Chunks", dbg.get("total_chunks", 0))
        m3.metric("Files", len(dbg.get("files_in_response", [])))

        st.markdown(f"**Query:** {dbg.get('query', '—')[:120]}")

        rewritten = dbg.get("rewritten_query", "")
        if rewritten and rewritten != dbg.get("query", ""):
            st.markdown(f"**Rewritten:** :blue[{rewritten[:120]}]")

        is_da = dbg.get("is_describe_all", False)
        da_label = ":green[True]" if is_da else ":red[False]"
        st.markdown(f"**Describe-All:** {da_label}")

        # Files in context
        cpf = dbg.get("chunks_per_file", {})
        missing = dbg.get("files_missing", [])
        if cpf or missing:
            st.markdown("**Files in context:**")
            for fname, count in cpf.items():
                st.markdown(f"✅ `{fname}` — {count} chunk(s)")
            for m in missing:
                st.markdown(f"❌ **MISSING:** `{m}`")
        else:
            st.caption("No chunks retrieved yet.")
    else:
        st.caption("No query sent yet — send a question to see trace data.")

    st.divider()

    # Backend logs
    st.markdown("**📄 Backend Logs**")
    if raw_logs:
        log_text = "\n".join(raw_logs)
    else:
        log_text = "No logs yet — send a query to see activity."

    st.code(log_text, language=None)


# ── LEFT: Chat area ────────────────────────────────────────────────────────────
with chat_col:
    st.header("💬 Ask Questions About Your Documents")
    st.caption("Upload documents in the sidebar, then start chatting.")

    # Scope / onboarding hints
    if not st.session_state.session_files:
        st.warning(
            "⚠️ **No session scope active.** Queries will search ALL indexed files. "
            "Upload files and click **Process & Index Files**, or use "
            "**📂 Load files into scope** in the sidebar."
        )
    elif not st.session_state.messages:
        st.info("👈 Upload a document in the sidebar, then ask your first question below!")

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # If the last message is from the user, stream the assistant reply
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        pending = st.session_state.messages[-1]["content"]
        with st.chat_message("assistant"):
            full_response = ""
            placeholder = st.empty()
            try:
                payload = {"query": pending}
                if st.session_state.session_files:
                    payload["filter_sources"] = st.session_state.session_files
                with requests.post(f"{API_BASE}/query", json=payload, stream=True) as r:
                    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            placeholder.markdown(full_response + " ▌")
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"⚠️ Error contacting backend: {e}"
                placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()


# ── Chat input — always at the bottom of the page ─────────────────────────────
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# ── Auto-refresh ───────────────────────────────────────────────────────────────
if st.session_state.auto_refresh:
    time.sleep(3)
    st.rerun()
