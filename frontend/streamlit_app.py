import json
import time
import os
import requests
import streamlit as st
import streamlit.components.v1 as components

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

# ── Fetch backend data ─────────────────────────────────────────────────────────
try:
    backend_online = requests.get(f"{API_BASE}/", timeout=1).status_code == 200
except Exception:
    backend_online = False

try:
    dbg = requests.get(f"{API_BASE}/debug-last", timeout=1).json() if backend_online else {}
except Exception:
    dbg = {}

try:
    raw_logs = requests.get(f"{API_BASE}/logs?n=60", timeout=1).json().get("logs", []) if backend_online else []
except Exception:
    raw_logs = []


# ── Build monitor panel HTML (inner content only) ──────────────────────────────
def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_monitor_html(dbg, logs, online):
    status_col = "#3fb950" if online else "#f85149"
    status_txt = "&#9679; Online" if online else "&#9679; Offline"

    # Log lines
    log_rows = []
    for line in logs:
        e = _esc(line)
        if "[ERROR]" in line:     c = "#f85149"
        elif "[WARNING]" in line: c = "#d29922"
        elif "[DEBUG]" in line:   c = "#6e7681"
        else:                     c = "#3fb950"
        log_rows.append(f'<div style="color:{c};margin:1px 0">{e}</div>')
    log_html = "\n".join(log_rows) if log_rows else \
        '<div style="color:#6e7681">No logs yet — send a query to see activity.</div>'

    # Debug trace
    if dbg:
        mode    = _esc(dbg.get("mode", "—"))
        chunks  = dbg.get("total_chunks", 0)
        n_files = len(dbg.get("files_in_response", []))
        query   = _esc(dbg.get("query", "—"))[:90]
        rw      = _esc(dbg.get("rewritten_query", "—"))[:90]
        is_da   = dbg.get("is_describe_all", False)
        da_col  = "#3fb950" if is_da else "#f85149"
        cpf     = dbg.get("chunks_per_file", {})
        missing = dbg.get("files_missing", [])

        file_rows = "".join(
            f'<div style="color:#3fb950">&#10003; {_esc(f)} &mdash; {c} chunk(s)</div>'
            for f, c in cpf.items()
        ) + "".join(
            f'<div style="color:#f85149">&#10007; MISSING: {_esc(m)}</div>'
            for m in missing
        ) or '<div style="color:#6e7681">No chunks retrieved</div>'

        trace_html = f"""
        <div style="display:flex;gap:6px;margin-bottom:8px">
          <div style="background:#161b22;border-radius:5px;padding:6px 10px;flex:1;text-align:center">
            <div style="color:#6e7681;font-size:10px">MODE</div>
            <div style="color:#e6edf3;font-weight:bold;font-size:13px">{mode}</div>
          </div>
          <div style="background:#161b22;border-radius:5px;padding:6px 10px;flex:1;text-align:center">
            <div style="color:#6e7681;font-size:10px">CHUNKS</div>
            <div style="color:#e6edf3;font-weight:bold;font-size:13px">{chunks}</div>
          </div>
          <div style="background:#161b22;border-radius:5px;padding:6px 10px;flex:1;text-align:center">
            <div style="color:#6e7681;font-size:10px">FILES</div>
            <div style="color:#e6edf3;font-weight:bold;font-size:13px">{n_files}</div>
          </div>
        </div>
        <div style="background:#161b22;border-radius:5px;padding:8px;margin-bottom:8px;font-size:11px">
          <div style="color:#6e7681">QUERY</div>
          <div style="color:#e6edf3;margin:2px 0 6px 0">{query}</div>
          <div style="color:#6e7681">REWRITTEN</div>
          <div style="color:#79c0ff;margin:2px 0 6px 0">{rw}</div>
          <div style="color:#6e7681">DESCRIBE-ALL: <span style="color:{da_col}">{is_da}</span></div>
        </div>
        <div style="background:#161b22;border-radius:5px;padding:8px;margin-bottom:10px;font-size:11px">
          <div style="color:#6e7681;margin-bottom:4px">FILES IN CONTEXT</div>
          {file_rows}
        </div>"""
    else:
        trace_html = '<div style="color:#6e7681;font-size:12px;padding:8px">No query sent yet.</div>'

    return f"""
<div style="color:#e6edf3;font-size:12px;font-weight:600;margin-bottom:6px">
  &#128187; Backend Monitor
  <span style="float:right;color:{status_col};font-size:12px;font-weight:600">{status_txt}</span>
</div>
<hr style="border:none;border-top:1px solid #21262d;margin-bottom:12px">

<div style="color:#e6edf3;font-size:12px;font-weight:600;margin-bottom:6px">
  &#128269; Last Query Trace
</div>
{trace_html}

<div style="color:#e6edf3;font-size:12px;font-weight:600;margin-bottom:6px">
  &#128196; Backend Logs
</div>
<div style="
  background:#010409;
  border:1px solid #21262d;
  border-radius:5px;
  padding:8px;
  font-family:'Courier New',monospace;
  font-size:10.5px;
  height:300px;
  overflow-y:auto;
  line-height:1.55;
">
  {log_html}
</div>
"""


# ── Inject monitor panel via iframe JS → window.parent.document ────────────────
# st.markdown() positions:fixed elements inside Streamlit's own scrollable
# container — they scroll with content and aren't truly viewport-fixed.
# st.components.v1.html() runs in an iframe; from there we can reach
# window.parent.document and inject directly into the real <body>.
def inject_monitor_panel(dbg, logs, online):
    inner_html = build_monitor_html(dbg, logs, online)
    # JSON-encode the HTML string so it safely passes through the JS string
    inner_json = json.dumps(inner_html)

    components.html(f"""
    <script>
    (function() {{
        try {{
        var doc = window.parent.document;

        // ── 1. Layout CSS ──────────────────────────────────────────────────
        var styleId = 'rag-layout-style';
        var style = doc.getElementById(styleId);
        if (!style) {{
            style = doc.createElement('style');
            style.id = styleId;
            doc.head.appendChild(style);
        }}
        style.textContent = [
            // Push main content left so panel doesn't overlap
            '.block-container {{ max-width: 58vw !important; padding-right: 1rem !important; }}',
            // Keep chat input bar in the same 60% zone
            '[data-testid="stBottom"] > div {{ max-width: 58vw !important; }}',
            // Freeze sidebar width
            '[data-testid="stSidebar"] {{ min-width: 300px !important; max-width: 300px !important; }}',
            '[data-testid="stSidebar"] * {{ font-size: 13px !important; }}',
            '[data-testid="stSidebar"] h1 {{ font-size: 19px !important; }}',
            '[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ font-size: 14px !important; }}',
            '[data-testid="stChatMessage"] p {{ font-size: 15px !important; line-height: 1.7 !important; }}'
        ].join('\\n');

        // ── 2. Monitor panel ───────────────────────────────────────────────
        var panelId = 'rag-monitor-panel';
        var panel = doc.getElementById(panelId);
        if (!panel) {{
            panel = doc.createElement('div');
            panel.id = panelId;
            Object.assign(panel.style, {{
                position:   'fixed',
                right:      '0',
                top:        '0',
                width:      '38vw',
                height:     '100vh',
                background: '#0d1117',
                borderLeft: '2px solid #21262d',
                padding:    '14px 16px',
                overflowY:  'auto',
                zIndex:     '9999',
                boxSizing:  'border-box',
                fontFamily: "'Segoe UI', sans-serif"
            }});
            doc.body.appendChild(panel);
        }}

        // Update content every render
        panel.innerHTML = {inner_json};
        }} catch(e) {{
            console.warn('RAG monitor panel error:', e);
        }}
    }})();
    </script>
    """, height=1, scrolling=False)


inject_monitor_panel(dbg, raw_logs, backend_online)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Document QA")
    st.caption("Powered by Gemini · Groq · FAISS")
    st.divider()

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
    st.subheader("📊 Index Stats")
    if st.button("🔄 Refresh Stats", use_container_width=True):
        try:
            s = requests.get(f"{API_BASE}/stats").json()
            c1, c2 = st.columns(2)
            c1.metric("Vectors", s["total_vectors"])
            c2.metric("Chunks",  s["total_chunks"])
            for src in requests.get(f"{API_BASE}/sources").json().get("sources", []):
                st.caption(f"• {src}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
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
    st.subheader("🗄️ Vector Store")
    if st.button("⚠️ Clear Index", use_container_width=True):
        try:
            if requests.post(f"{API_BASE}/clear").json().get("status") == "success":
                st.session_state.session_files = []
                st.success("Index cleared.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.subheader("⚙️ Monitor Settings")
    live = st.toggle("🔄 Live refresh (3 s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = live
    if live:
        st.caption("Monitor panel auto-refreshes every 3 s.")


# ── Main chat area ─────────────────────────────────────────────────────────────
st.header("💬 Ask Questions About Your Documents")
st.caption("Upload documents in the sidebar, then start chatting.")
st.divider()

if not st.session_state.session_files:
    st.warning(
        "⚠️ **No session scope active.** Queries will search ALL indexed files. "
        "Upload files and click **Process & Index Files**, or use "
        "**📂 Load files into scope** in the sidebar."
    )
elif not st.session_state.messages:
    st.info("👈 Upload a document in the sidebar, then ask your first question below!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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
            with requests.post(f"{API_BASE}/query", json=payload, stream=True) as r:
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Error: {e}"
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()  # re-render monitor panel with fresh debug data

# Auto-refresh (triggers full rerun → monitor panel re-fetches logs + debug)
if st.session_state.auto_refresh:
    time.sleep(3)
    st.rerun()
