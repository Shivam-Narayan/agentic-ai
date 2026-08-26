"""Streamlit chat UI for the CTE Knowledge Transfer Assistant — Professional Edition."""

import io
import os
import uuid

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
API_URL         = os.getenv("KT_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 180.0

# ── Sample questions ─────────────────────────────────────────────────────────
SAMPLE_QUESTIONS: dict[str, list[str]] = {
    "💬 General Knowledge": [
        "What is a vector database?",
        "Explain the difference between SQL and NoSQL.",
    ],
    "📄 Company Documents": [
        "What is the Beacon project?",
        "What tech stack is used in the project?",
        "Summarise the KT document",
        "Extract the project name, start date, and owner from the documents",
    ],
    "🗄️ Company Database": [
        "List all tables in the database",
        "What is the status of order #10001?",
        "Show me the top 5 customers by order count as a bar chart",
    ],
    "🌐 Live Web": [
        "What is the current USD to INR exchange rate?",
        "Latest news about LangChain",
    ],
    "🧮 Calculator": [
        "What is 15% of 85000?",
        "Calculate (250 + 300) * 12 / 100",
    ],
    "📊 Charts": [
        "Show monthly sales as a bar chart: Jan=1200, Feb=1500, Mar=1100, Apr=1800",
        "Plot a pie chart: Engineering=40, Sales=25, Marketing=20, HR=15",
    ],
}

# ── Datasource config ─────────────────────────────────────────────────────────
ROUTE_CONFIG: dict[str, dict] = {
    "direct_llm":   {"icon": "💬", "label": "LLM",       "color": "#6366f1"},
    "company_docs": {"icon": "📄", "label": "Documents",  "color": "#0ea5e9"},
    "database":     {"icon": "🗄️", "label": "Database",   "color": "#10b981"},
    "web_search":   {"icon": "🌐", "label": "Web Search", "color": "#f59e0b"},
    "calculation":  {"icon": "🧮", "label": "Calculator", "color": "#8b5cf6"},
    "chart":        {"icon": "📊", "label": "Chart",      "color": "#06b6d4"},
    "multiple":     {"icon": "🔀", "label": "Multi-tool", "color": "#ef4444"},
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataDialogue — KT Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Keep sidebar collapse/expand arrow visible ── */
[data-testid="collapsedControl"],
button[kind="headerNoPadding"],
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button {
    background: #161b27 !important;
    border: 1px solid #1e2536 !important;
    border-radius: 6px !important;
    color: #64748b !important;
}
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="collapsedControl"] button:hover {
    background: #1e2a3a !important;
    color: #94a3b8 !important;
}

/* ── App background ── */
.stApp {
    background: #0f1117;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2536;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #1e2a3a !important;
    border: 1px solid #2d3f55 !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    padding: 6px 10px !important;
    transition: all 0.2s;
    text-align: left !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #243347 !important;
    border-color: #3b82f6 !important;
    color: #e2e8f0 !important;
}

/* ── Main content area ── */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 900px;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 0 1rem 0;
    border-bottom: 1px solid #1e2536;
    margin-bottom: 1.5rem;
}
.app-header-icon {
    font-size: 2rem;
    line-height: 1;
}
.app-header-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.3px;
    margin: 0;
}
.app-header-subtitle {
    font-size: 0.78rem;
    color: #64748b;
    margin: 0;
}
.status-dot {
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
}
[data-testid="stChatMessageContent"] {
    background: #161b27 !important;
    border: 1px solid #1e2536 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    color: #e2e8f0 !important;
}
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3 {
    color: #f1f5f9 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    margin-top: 0.6rem !important;
    margin-bottom: 0.2rem !important;
}
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li {
    color: #cbd5e1 !important;
    font-size: 0.88rem;
    line-height: 1.7;
}
[data-testid="stChatMessageContent"] code {
    background: #0f1117 !important;
    color: #7dd3fc !important;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.82rem;
}
[data-testid="stChatMessageContent"] pre {
    background: #0f1117 !important;
    border: 1px solid #1e2536;
    border-radius: 8px;
    padding: 12px;
}

/* ── User message bubble ── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: #1a2744 !important;
    border-color: #2d4270 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #161b27 !important;
    border: 1px solid #2d3f55 !important;
    border-radius: 12px !important;
    padding: 4px 8px !important;
}
[data-testid="stChatInput"] textarea {
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}

/* ── Tool badge ── */
.tool-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #fff;
    margin-right: 6px;
    letter-spacing: 0.3px;
}

/* ── Citation pill ── */
.citation-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #0f1a2e;
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: #60a5fa;
    margin: 2px 3px 2px 0;
    cursor: default;
    transition: background 0.15s;
}
.citation-pill:hover {
    background: #162444;
}

/* ── Citation block ── */
.citation-block {
    border-top: 1px solid #1e2536;
    margin-top: 10px;
    padding-top: 8px;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 2px;
}
.citation-label {
    font-size: 0.68rem;
    color: #475569;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-right: 6px;
}

/* ── Meta row below answer ── */
.meta-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    flex-wrap: wrap;
}
.tool-name-chip {
    font-size: 0.68rem;
    color: #64748b;
    background: #0f1117;
    border: 1px solid #1e2536;
    border-radius: 4px;
    padding: 1px 6px;
    font-family: 'Courier New', monospace;
}

/* ── Sidebar section headers ── */
.sidebar-section {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #475569 !important;
    margin: 1rem 0 0.4rem 0;
}

/* ── Doc item in sidebar ── */
.doc-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 8px;
    background: #0f1117;
    border: 1px solid #1e2536;
    border-radius: 8px;
    margin-bottom: 4px;
}
.doc-name {
    font-size: 0.78rem;
    color: #94a3b8;
    font-weight: 500;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.doc-size {
    font-size: 0.68rem;
    color: #475569;
}

/* ── Welcome card ── */
.welcome-card {
    background: linear-gradient(135deg, #161b27 0%, #1a2035 100%);
    border: 1px solid #1e2d45;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 1rem;
}
.welcome-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 8px;
}
.welcome-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 12px;
}
.welcome-item {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #0f1117;
    border: 1px solid #1e2536;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.78rem;
    color: #94a3b8;
}

/* ── Primary upload button ── */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #60a5fa, #3b82f6) !important;
    box-shadow: 0 4px 12px rgba(59,130,246,0.3) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0f1117 !important;
    border: 1px solid #1e2536 !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
}
[data-testid="stExpander"] summary {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-top-color: #3b82f6 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2d4270; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fetch_documents() -> list[dict]:
    try:
        r = httpx.get(f"{API_URL.rstrip('/')}/documents", timeout=8)
        r.raise_for_status()
        return r.json().get("documents", [])
    except Exception:
        return []


def _upload_files(uploaded_files) -> dict:
    files_payload = [
        ("files", (f.name, io.BytesIO(f.read()), f.type or "application/octet-stream"))
        for f in uploaded_files
    ]
    r = httpx.post(f"{API_URL.rstrip('/')}/upload", files=files_payload, timeout=300)
    r.raise_for_status()
    return r.json()


def _check_backend() -> bool:
    try:
        r = httpx.get(f"{API_URL.rstrip('/')}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "backend_ok" not in st.session_state:
    st.session_state.backend_ok = _check_backend()


# ── Render assistant message ──────────────────────────────────────────────────
def _render_assistant_message(msg: dict) -> None:
    st.markdown(msg["content"])

    datasource = msg.get("datasource")
    tools_used = msg.get("tools_used") or []
    citations  = msg.get("citations") or []
    chart_data = msg.get("chart_data")

    # Chart
    if chart_data:
        try:
            import plotly.graph_objects as go
            fig = go.Figure(chart_data)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0f1117",
                font_color="#cbd5e1",
            )
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{id(msg)}")
        except Exception as exc:
            st.warning(f"Could not render chart: {exc}")

    # Meta row — badge + tool chips
    if datasource:
        cfg   = ROUTE_CONFIG.get(datasource, {"icon": "🔧", "label": datasource, "color": "#666"})
        badge = (
            f'<span class="tool-badge" style="background:{cfg["color"]}">'
            f'{cfg["icon"]} {cfg["label"]}</span>'
        )
        chips = "".join(
            f'<span class="tool-name-chip">{t}</span>'
            for t in tools_used
        )
        st.markdown(
            f'<div class="meta-row">{badge}{chips}</div>',
            unsafe_allow_html=True,
        )

    # Citations
    if citations:
        pills = "".join(
            f'<span class="citation-pill" title="{c.get("detail","")}">'
            f'📎 {c["source"]}</span>'
            for c in citations
        )
        st.markdown(
            f'<div class="citation-block">'
            f'<span class="citation-label">Sources</span>{pills}</div>',
            unsafe_allow_html=True,
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div style="padding: 8px 0 16px 0;">
        <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.3px;">
            🧠 DataDialogue
        </div>
        <div style="font-size:0.72rem;color:#475569;margin-top:2px;">
            Knowledge Transfer Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Backend status
    status_color = "#22c55e" if st.session_state.backend_ok else "#ef4444"
    status_text  = "Backend connected" if st.session_state.backend_ok else "Backend offline"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:6px;padding:6px 10px;'
        f'background:#0f1117;border:1px solid #1e2536;border-radius:8px;margin-bottom:12px;">'
        f'<div style="width:7px;height:7px;background:{status_color};border-radius:50%;"></div>'
        f'<span style="font-size:0.72rem;color:#64748b;">{status_text}</span>'
        f'<span style="font-size:0.68rem;color:#334155;margin-left:auto;">'
        f'{st.session_state.session_id[:8]}…</span></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Upload
    st.markdown('<div class="sidebar-section">Upload Documents</div>', unsafe_allow_html=True)
    st.caption("PDF · DOCX · XLSX · CSV · TXT")

    uploaded = st.file_uploader(
        label="files",
        type=["pdf", "docx", "doc", "xlsx", "xls", "csv", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("⬆️ Upload & Index", type="primary", use_container_width=True):
            with st.spinner("Indexing…"):
                try:
                    result   = _upload_files(uploaded)
                    saved    = result.get("saved", [])
                    rejected = result.get("rejected", [])
                    if saved:
                        st.success(f"✅ {len(saved)} file(s) indexed")
                    if rejected:
                        st.warning(f"⚠️ {len(rejected)} file(s) skipped")
                    st.session_state.pop("docs_cache", None)
                except httpx.ConnectError:
                    st.error("Backend offline")
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

    st.divider()

    # Documents
    st.markdown('<div class="sidebar-section">Indexed Documents</div>', unsafe_allow_html=True)

    if "docs_cache" not in st.session_state:
        st.session_state.docs_cache = _fetch_documents()

    docs = st.session_state.docs_cache
    if docs:
        FILE_ICONS = {"PDF": "📄", "DOCX": "📝", "DOC": "📝",
                      "XLSX": "📊", "XLS": "📊", "CSV": "📊", "TXT": "📃"}
        for doc in docs:
            icon = FILE_ICONS.get(doc["type"], "📎")
            st.markdown(
                f'<div class="doc-item">'
                f'<span>{icon}</span>'
                f'<span class="doc-name">{doc["name"]}</span>'
                f'<span class="doc-size">{doc["size_kb"]} KB</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No documents yet.")

    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.docs_cache = _fetch_documents()
        st.rerun()

    st.divider()

    # Session controls
    st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            try:
                httpx.delete(f"{API_URL}/sessions/{st.session_state.session_id}/history", timeout=10)
            except Exception:
                pass
            st.rerun()
    with col2:
        if st.button("🆕 New", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages   = []
            st.rerun()

    st.divider()

    # Sample questions
    st.markdown('<div class="sidebar-section">Sample Questions</div>', unsafe_allow_html=True)
    for group, questions in SAMPLE_QUESTIONS.items():
        with st.expander(group, expanded=False):
            for q in questions:
                if st.button(q, key=f"sample_{q}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="app-header">
    <span class="app-header-icon">🧠</span>
    <div>
        <div class="app-header-title">DataDialogue</div>
        <div class="app-header-subtitle">
            <span class="status-dot"></span>
            Ask questions about your documents, database, or the live web
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Welcome card (only when no messages)
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-title">What can I help you with?</div>
        <div style="font-size:0.82rem;color:#64748b;">
            I can search your documents, query the database, run calculations,
            generate charts, or look up live web data.
        </div>
        <div class="welcome-grid">
            <div class="welcome-item">📄 Company documents &amp; reports</div>
            <div class="welcome-item">🗄️ Database queries &amp; analytics</div>
            <div class="welcome-item">🌐 Live web facts &amp; prices</div>
            <div class="welcome-item">📊 Charts &amp; calculations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "👤"):
        if msg["role"] == "assistant":
            _render_assistant_message(msg)
        else:
            st.markdown(msg["content"])

# Input
typed_input = st.chat_input("Ask anything…")
prompt = st.session_state.pop("pending_question", None) or typed_input

if not prompt:
    st.stop()

prompt = prompt.strip()
if not prompt:
    st.stop()

# Show user message
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user", avatar="👤"):
    st.markdown(prompt)

# Call backend
with st.chat_message("assistant", avatar="🧠"):
    with st.spinner(""):
        answer     = ""
        datasource = None
        tools_used = []
        citations  = []
        chart_data = None

        try:
            response = httpx.post(
                f"{API_URL.rstrip('/')}/ask",
                json={"question": prompt, "session_id": st.session_state.session_id},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload    = response.json()
            answer     = payload.get("answer") or "I could not generate an answer."
            datasource = payload.get("datasource")
            tools_used = payload.get("tools_used") or []
            citations  = payload.get("citations") or []
            chart_data = payload.get("chart_data")
            st.session_state.backend_ok = True

        except httpx.ConnectError:
            st.session_state.backend_ok = False
            answer = (
                "⚠️ **Cannot reach the backend.**\n\n"
                f"Make sure FastAPI is running:\n"
                f"```\nuvicorn app:app --reload --port 8000\n```\n"
                f"Expected at: `{API_URL}`"
            )
        except httpx.TimeoutException:
            answer = (
                "⚠️ **Request timed out.**\n\n"
                "The agent is still processing — try asking again."
            )
        except Exception as exc:
            answer = f"⚠️ **Unexpected error:** {exc}"

    assistant_msg = {
        "role":       "assistant",
        "content":    answer,
        "datasource": datasource,
        "tools_used": tools_used,
        "citations":  citations,
        "chart_data": chart_data,
    }
    _render_assistant_message(assistant_msg)

st.session_state.messages.append(assistant_msg)
