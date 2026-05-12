import streamlit as st
import requests

# API_BASE = "http://127.0.0.1:8000"
import os
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state init (must come before any widget that reads it) ─────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_files" not in st.session_state:
    st.session_state.session_files = []
if "scope_initialized" not in st.session_state:
    # First load — auto-restore scope from whatever is indexed.
    # This prevents silent full-index queries after a page refresh.
    try:
        srcs = requests.get(f"{API_BASE}/sources", timeout=3).json().get("sources", [])
        st.session_state.session_files = list(srcs)
    except Exception:
        pass
    st.session_state.scope_initialized = True

st.markdown("""
<style>
[data-testid="stSidebar"] {
    min-width: 320px !important;
    max-width: 320px !important;
}
[data-testid="stSidebar"] * { font-size: 15px !important; }
[data-testid="stSidebar"] h1 { font-size: 22px !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { font-size: 17px !important; }
[data-testid="stSidebar"] .stButton > button {
    font-size: 15px !important;
    padding: 10px 16px !important;
    height: auto !important;
}
.main * { font-size: 16px !important; }
.main h1 { font-size: 32px !important; }
.main h2 { font-size: 26px !important; }
[data-testid="stChatMessage"] p { font-size: 16px !important; line-height: 1.7 !important; }
textarea[data-testid="stChatInputTextArea"] { font-size: 16px !important; padding: 14px !important; }
[data-testid="stMetricValue"] { font-size: 28px !important; }
.stAlert p { font-size: 15px !important; }
.block-container { padding-left: 3rem !important; padding-right: 3rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
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
            with st.spinner("Indexing documents..."):
                files = [("files", (f.name, f.read(), f.type)) for f in uploaded_files]
                try:
                    response = requests.post(f"{API_BASE}/upload", files=files)
                    results = response.json().get("results", [])
                    for r in results:
                        fname = r["filename"]
                        if r["status"] == "success":
                            st.success(f"✅ **{fname}**  \n{r['chunks_added']} chunks indexed")
                            if fname not in st.session_state.session_files:
                                st.session_state.session_files.append(fname)
                        elif r["status"] == "duplicate":
                            st.info(f"↩ **{fname}**  \nIdentical content already indexed — skipped.")
                            # Still add to session scope — user wants to query this file
                            if fname not in st.session_state.session_files:
                                st.session_state.session_files.append(fname)
                        else:
                            st.error(f"❌ **{fname}**  \n{r.get('reason', 'Error')}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    st.divider()

    # Stats + indexed files
    st.subheader("📊 Index Stats")
    if st.button("🔄 Refresh Stats", use_container_width=True):
        try:
            stats = requests.get(f"{API_BASE}/stats").json()
            col1, col2 = st.columns(2)
            col1.metric("Vectors", stats["total_vectors"])
            col2.metric("Chunks", stats["total_chunks"])
            srcs = requests.get(f"{API_BASE}/sources").json().get("sources", [])
            if srcs:
                st.caption(f"**{len(srcs)} file(s) in index:**")
                for s in srcs:
                    st.caption(f"• {s}")
            else:
                st.caption("No files indexed yet.")
        except Exception as e:
            st.error(f"Stats error: {e}")

    st.divider()

    # Memory
    # Memory
    st.subheader("🧠 Memory")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        try:
            requests.delete(f"{API_BASE}/history")
        except:
            pass
        st.rerun()

    if st.button("👁️ View Memory", use_container_width=True):
        try:
            resp = requests.get(f"{API_BASE}/history").json()
            total = resp.get("total_messages", 0)
            messages = resp.get("messages", [])
            st.caption(f"Total messages in memory: {total}")
            for msg in messages:
                role = "🧑 User" if msg["role"] == "user" else "🤖 Assistant"
                with st.expander(f"{role}: {msg['content'][:50]}..."):
                    st.write(msg["content"])
        except Exception as e:
            st.error(f"Error fetching memory: {e}")

    # ── Active scope panel ─────────────────────────────────────────────────
    st.divider()
    st.subheader("🗂️ Active Scope")
    if st.session_state.session_files:
        st.caption(f"Queries search **only** these {len(st.session_state.session_files)} file(s):")
        for f in st.session_state.session_files:
            st.caption(f"• {f}")
        if st.button("🔄 Reload scope from index", use_container_width=True):
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
        st.warning("⚠️ No scope set — ALL indexed files will be searched.")
        if st.button("📂 Load indexed files into scope", use_container_width=True):
            try:
                srcs = requests.get(f"{API_BASE}/sources", timeout=3).json().get("sources", [])
                if srcs:
                    st.session_state.session_files = list(srcs)
                    st.rerun()
                else:
                    st.info("No files indexed yet — upload some first.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # Backend logs viewer
    st.subheader("📜 Backend Logs")
    if st.button("📋 View Recent Logs", use_container_width=True):
        try:
            logs = requests.get(f"{API_BASE}/logs?n=30", timeout=3).json().get("logs", [])
            if logs:
                # Highlight errors and warnings
                log_text = "\n".join(logs)
                st.code(log_text, language=None)
            else:
                st.info("No logs yet.")
        except Exception as e:
            st.error(f"Could not fetch logs: {e}")

    st.divider()

    # Vector Store
    st.subheader("🗄️ Vector Store")
    st.caption("Remove all indexed documents and start fresh.")
    if st.button("⚠️ Clear Index", use_container_width=True):
        try:
            response = requests.post(f"{API_BASE}/clear")
            result = response.json()
            if result["status"] == "success":
                st.session_state.session_files = []   # reset scope too
                st.success("✅ Index cleared! Upload new documents to start fresh.")
            else:
                st.error("Failed to clear index.")
        except Exception as e:
            st.error(f"Error: {e}")

# ── Main Chat Area ────────────────────────────────────────
st.header("💬 Ask Questions About Your Documents")
st.caption("Upload documents in the sidebar first, then start chatting.")
st.divider()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.session_files:
    st.warning(
        "⚠️ **No session scope active.** Queries will search ALL indexed files. "
        "Upload files and click **Process & Index Files**, or use "
        "**📂 Load indexed files into scope** in the sidebar."
    )
elif not st.session_state.messages:
    st.info("👈 Upload a document in the sidebar, then ask your first question below!")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = ""
        placeholder = st.empty()
        try:
            payload = {"query": prompt}
            if st.session_state.session_files:
                payload["filter_sources"] = st.session_state.session_files
            with requests.post(
                f"{API_BASE}/query",
                json=payload,
                stream=True
            ) as r:
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Error: {e}"
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


 