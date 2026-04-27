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
                        if r["status"] == "success":
                            st.success(f"✅ **{r['filename']}**  \n{r['chunks_added']} chunks indexed")
                        else:
                            st.error(f"❌ **{r['filename']}**  \n{r.get('reason', 'Error')}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    st.divider()

    # Stats
    st.subheader("📊 Index Stats")
    if st.button("🔄 Refresh Stats", use_container_width=True):
        try:
            stats = requests.get(f"{API_BASE}/stats").json()
            col1, col2 = st.columns(2)
            col1.metric("Vectors", stats["total_vectors"])
            col2.metric("Chunks", stats["total_chunks"])
            st.metric("Embedding Dim", stats["embedding_dim"])
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

    st.divider()

    # Vector Store
    st.subheader("🗄️ Vector Store")
    st.caption("Remove all indexed documents and start fresh.")
    if st.button("⚠️ Clear Index", use_container_width=True):
        try:
            response = requests.post(f"{API_BASE}/clear")
            result = response.json()
            if result["status"] == "success":
                st.success("✅ Index cleared! Upload new documents to start fresh.")
            else:
                st.error("Failed to clear index.")
        except Exception as e:
            st.error(f"Error: {e}")

# ── Main Chat Area ────────────────────────────────────────
st.header("💬 Ask Questions About Your Documents")
st.caption("Upload documents in the sidebar first, then start chatting.")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.messages:
    st.info("👈 Upload a document in the sidebar, then ask your first question below!")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = ""
        placeholder = st.empty()
        try:
            with requests.post(
                f"{API_BASE}/query",
                json={"query": prompt},
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


 