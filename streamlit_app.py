"""Streamlit chat UI for the CTE Knowledge Transfer Assistant — Professional Edition."""

import io
import json
import os
import uuid
from pathlib import Path

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
    "direct_llm":   {"icon": "✦",  "label": "Direct answer", "color": "#6366f1", "bg": "#1e1b4b"},
    "company_docs": {"icon": "≡",  "label": "Documents",     "color": "#38bdf8", "bg": "#0c2340"},
    "database":     {"icon": "◈",  "label": "Database",      "color": "#34d399", "bg": "#052e1c"},
    "web_search":   {"icon": "↗",  "label": "Web search",    "color": "#a78bfa", "bg": "#1e1040"},
    "calculation":  {"icon": "∑",  "label": "Calculator",    "color": "#c084fc", "bg": "#1e0b35"},
    "chart":        {"icon": "◎",  "label": "Chart",         "color": "#22d3ee", "bg": "#042f3e"},
    "multiple":     {"icon": "⊕",  "label": "Multi-tool",    "color": "#94a3b8", "bg": "#1e2536"},
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataDialogue — KT Assistant",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium theme CSS ─────────────────────────────────────────────────────────
st.html(Path(__file__).parent / "assets" / "premium.css")

# ── Agent status cards (thinking / tools / indexing) ──────────────────────────
def _agent_status_html(title: str, subtitle: str, steps: list[str]) -> str:
    items = "".join(f"<li>{step}</li>" for step in steps)
    return (
        '<div class="agent-status">'
        '<div class="agent-orb-wrap">'
        '<div class="agent-orb"></div>'
        '<div class="agent-orb-core"></div>'
        '<div class="agent-orb-ring"></div>'
        "</div>"
        "<div>"
        f'<div class="agent-status-title">{title}</div>'
        f'<div class="agent-status-sub">{subtitle}</div>'
        f'<ul class="agent-steps">{items}</ul>'
        '<div class="shimmer-bar"></div>'
        "</div></div>"
    )


THINKING_HTML = _agent_status_html(
    "Agent thinking",
    "Planning the answer and deciding which tools to use",
    ["Thinking", "Calling tools", "Composing answer"],
)

INDEXING_HTML = _agent_status_html(
    "Indexing documents",
    "Parsing files and updating the knowledge base",
    ["Reading files", "Chunking", "Embedding"],
)


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

if "sources_open" not in st.session_state:
    st.session_state.sources_open = False

if "active_sources" not in st.session_state:
    st.session_state.active_sources = []

if "active_tools" not in st.session_state:
    st.session_state.active_tools = []


# ── Render assistant message ──────────────────────────────────────────────────
def _render_assistant_message(msg: dict, msg_index: int = 0) -> None:
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
                plot_bgcolor="rgba(13, 19, 32, 0.65)",
                font_color="#cbd5e1",
            )
            st.plotly_chart(fig, width="stretch", key=f"chart_{id(msg)}")
        except Exception as exc:
            st.warning(f"Could not render chart: {exc}")

    # Meta row — only if there's something to show
    if datasource or citations:
        cfg = ROUTE_CONFIG.get(datasource or "", {
            "icon": "·", "label": datasource or "", "color": "#64748b", "bg": "#1e2536"
        })

        # Build the badge HTML
        badge_html = (
            f'<span class="tool-badge" style="'
            f'color:{cfg["color"]};'
            f'background:{cfg["bg"]};'
            f'border-color:{cfg["color"]}22;">'
            f'<span class="badge-icon">{cfg["icon"]}</span>'
            f'{cfg["label"]}'
            f'</span>'
        )

        # Tool name (first tool, lowercase)
        tool_label = ""
        if tools_used:
            tool_label = f'<span class="tool-name-chip">{tools_used[0]}</span>'

        st.markdown(
            f'<div class="meta-row">{badge_html}{tool_label}</div>',
            unsafe_allow_html=True,
        )

        # Sources button — separate row, right-aligned, minimal
        if citations:
            is_active = (
                st.session_state.sources_open and
                st.session_state.active_sources == citations
            )
            btn_label = f"{'↙' if is_active else '↗'} {len(citations)} source{'s' if len(citations) != 1 else ''}"

            st.markdown('<div class="src-btn-wrap">', unsafe_allow_html=True)
            if st.button(btn_label, key=f"src_{msg_index}_{id(msg)}"):
                if is_active:
                    st.session_state.sources_open   = False
                    st.session_state.active_sources = []
                    st.session_state.active_tools   = []
                else:
                    st.session_state.sources_open   = True
                    st.session_state.active_sources = citations
                    st.session_state.active_tools   = tools_used
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="brand-mark">
        <div class="brand-title">DataDialogue</div>
        <div class="brand-sub">Knowledge transfer assistant</div>
    </div>
    """, unsafe_allow_html=True)

    # Backend status
    dot_cls = "ok" if st.session_state.backend_ok else "bad"
    status_text = "Backend connected" if st.session_state.backend_ok else "Backend offline"
    st.markdown(
        f'<div class="status-pill">'
        f'<div class="status-pill-dot {dot_cls}"></div>'
        f'<span style="font-size:0.72rem;color:#94a3b8;">{status_text}</span>'
        f'<span style="font-size:0.68rem;color:#475569;margin-left:auto;">'
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
        if st.button("Upload and index", type="primary", icon=":material/upload:", width="stretch"):
            index_slot = st.empty()
            index_slot.markdown(INDEXING_HTML, unsafe_allow_html=True)
            try:
                result   = _upload_files(uploaded)
                saved    = result.get("saved", [])
                rejected = result.get("rejected", [])
                index_slot.empty()
                if saved:
                    st.success(f"{len(saved)} file(s) indexed", icon=":material/check_circle:")
                if rejected:
                    st.warning(f"{len(rejected)} file(s) skipped", icon=":material/warning:")
                st.session_state.pop("docs_cache", None)
            except httpx.ConnectError:
                index_slot.empty()
                st.error("Backend offline", icon=":material/error:")
            except Exception as exc:
                index_slot.empty()
                st.error(f"Upload failed: {exc}", icon=":material/error:")

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

    if st.button("Refresh", icon=":material/refresh:", width="stretch"):
        st.session_state.docs_cache = _fetch_documents()
        st.rerun()

    st.divider()

    # Session controls
    st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear", icon=":material/delete:", width="stretch"):
            st.session_state.messages = []
            try:
                httpx.delete(f"{API_URL}/sessions/{st.session_state.session_id}/history", timeout=10)
            except Exception:
                pass
            st.rerun()
    with col2:
        if st.button("New", icon=":material/add:", width="stretch"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages   = []
            st.rerun()

    st.divider()

    # Sample questions
    st.markdown('<div class="sidebar-section">Sample Questions</div>', unsafe_allow_html=True)
    for group, questions in SAMPLE_QUESTIONS.items():
        with st.expander(group, expanded=False):
            for q in questions:
                if st.button(q, key=f"sample_{q}", width="stretch"):
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

# Layout: chat column + optional sources panel
if st.session_state.sources_open:
    chat_col, sources_col = st.columns([2, 1], gap="medium")
else:
    chat_col = st.container()
    sources_col = None

with chat_col:
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
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(
            msg["role"],
            avatar=":material/psychology:" if msg["role"] == "assistant" else ":material/person:",
        ):
            if msg["role"] == "assistant":
                _render_assistant_message(msg, msg_index=i)
            else:
                st.markdown(msg["content"])

# Sources panel
if sources_col is not None:
    with sources_col:
        # Panel header
        hcol1, hcol2 = st.columns([5, 1])
        with hcol1:
            st.markdown(
                '<div style="font-size:0.82rem;font-weight:700;color:#e2e8f0;'
                'padding:4px 0 10px 0;letter-spacing:-0.1px;">Sources</div>',
                unsafe_allow_html=True,
            )
        with hcol2:
            if st.button("✕", key="close_sources"):
                st.session_state.sources_open   = False
                st.session_state.active_sources = []
                st.session_state.active_tools   = []
                st.rerun()

        st.markdown(
            '<div style="height:1px;background:#1e2d45;margin-bottom:14px;"></div>',
            unsafe_allow_html=True,
        )

        # Tools used
        if st.session_state.active_tools:
            st.markdown(
                '<div class="section-label">Tools used</div>',
                unsafe_allow_html=True,
            )
            for t in st.session_state.active_tools:
                cfg = ROUTE_CONFIG.get("web_search" if "web" in t else
                                       "company_docs" if "document" in t or "summarise" in t or "extract" in t else
                                       "database" if "database" in t or "table" in t else
                                       "calculation" if t == "calculate" else
                                       "chart" if t == "generate_chart" else "direct_llm", {})
                color = cfg.get("color", "#64748b")
                st.markdown(
                    f'<div style="display:inline-flex;align-items:center;gap:6px;'
                    f'background:#0d1320;border:1px solid #1a2540;border-radius:6px;'
                    f'padding:4px 10px;margin-bottom:10px;margin-right:4px;">'
                    f'<span style="width:6px;height:6px;background:{color};border-radius:50%;'
                    f'flex-shrink:0;"></span>'
                    f'<span style="font-size:0.72rem;color:#94a3b8;font-family:monospace;">{t}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # References
        st.markdown(
            '<div class="section-label">References</div>',
            unsafe_allow_html=True,
        )

        for c in st.session_state.active_sources:
            source = c.get("source", "")
            detail = c.get("detail", "")
            is_url = source.startswith("http")

            if is_url:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(source)
                    domain = parsed.netloc.replace("www.", "")
                    path   = parsed.path[:40] + "…" if len(parsed.path) > 40 else parsed.path
                except Exception:
                    domain = source[:30]
                    path   = ""

                st.markdown(
                    f'<a href="{source}" target="_blank" style="text-decoration:none;">'
                    f'<div class="source-card">'
                    f'<div class="source-icon">↗</div>'
                    f'<div class="source-content">'
                    f'<div class="source-domain">{domain}</div>'
                    f'<div class="source-detail">{path}</div>'
                    f'</div></div></a>',
                    unsafe_allow_html=True,
                )
            else:
                # File source
                ext = source.rsplit(".", 1)[-1].upper() if "." in source else "FILE"
                icon_map = {"PDF": "PDF", "DOCX": "DOC", "XLSX": "XLS", "CSV": "CSV", "TXT": "TXT"}
                icon_label = icon_map.get(ext, "≡")
                detail_str = detail[:80] if detail else ""

                st.markdown(
                    f'<div class="source-card" style="cursor:default;">'
                    f'<div class="source-icon" style="font-size:0.6rem;font-weight:700;'
                    f'color:#38bdf8;">{icon_label}</div>'
                    f'<div class="source-content">'
                    f'<div class="source-domain" style="color:#94a3b8;">{source}</div>'
                    + (f'<div class="source-detail">{detail_str}</div>' if detail_str else "")
                    + f'</div></div>',
                    unsafe_allow_html=True,
                )

# ── Input ─────────────────────────────────────────────────────────────────────
typed_input = st.chat_input("Ask anything…", submit_mode="disable")
prompt = st.session_state.pop("pending_question", None) or typed_input

if not prompt:
    st.stop()

prompt = prompt.strip()
if not prompt:
    st.stop()

# Show user message
st.session_state.messages.append({"role": "user", "content": prompt})
with chat_col:
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)

# Call backend
with chat_col:
    with st.chat_message("assistant", avatar=":material/psychology:"):
        thinking = st.empty()
        thinking.markdown(THINKING_HTML, unsafe_allow_html=True)

        answer     = ""
        datasource = None
        tools_used = []
        citations  = []
        chart_data = None

        # ── Streaming via SSE /stream endpoint ───────────────────────────────
        # We consume the SSE stream chunk-by-chunk so tokens appear as the
        # LLM generates them, exactly like ChatGPT's typewriter effect.
        answer_slot = st.empty()   # live-updating text area
        tool_slot   = st.empty()   # "using tool …" indicator

        try:
            with httpx.stream(
                "GET",
                f"{API_URL.rstrip('/')}/stream",
                params={"question": prompt, "session_id": st.session_state.session_id},
                timeout=REQUEST_TIMEOUT,
            ) as r:
                r.raise_for_status()
                for raw_line in r.iter_lines():
                    if not raw_line.startswith("data: "):
                        continue
                    event = json.loads(raw_line[6:])
                    etype = event.get("type")

                    if etype == "status":
                        # Still in "thinking" phase — keep the spinner
                        pass

                    elif etype == "token":
                        # First token arrives → clear the spinner immediately
                        if not answer:
                            thinking.empty()
                        answer += event.get("text", "")
                        answer_slot.markdown(answer + "▌")   # blinking cursor

                    elif etype == "tool":
                        tool_name = event.get("name", "tool")
                        tool_slot.markdown(
                            f'<div class="tool-badge" style="color:#a78bfa;'
                            f'background:#1e1040;border-color:#a78bfa22;">'
                            f'<span class="badge-icon">⚙</span> using {tool_name}…</div>',
                            unsafe_allow_html=True,
                        )

                    elif etype == "done":
                        payload    = event.get("payload", {})
                        datasource = payload.get("datasource")
                        tools_used = payload.get("tools_used") or []
                        citations  = payload.get("citations") or []
                        chart_data = payload.get("chart_data")
                        if not answer:
                            # done with no tokens → use generation field as fallback
                            answer = payload.get("generation", "I could not generate an answer.")

                    elif etype == "error":
                        if not answer:
                            answer = f"⚠️ **Agent error:** {event.get('detail', 'Unknown error')}"

            # Remove the blinking cursor and tool indicator now that we're done
            answer_slot.empty()
            tool_slot.empty()
            thinking.empty()
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
        finally:
            thinking.empty()
            answer_slot.empty()
            tool_slot.empty()

        thinking.empty()

        assistant_msg = {
            "role":       "assistant",
            "content":    answer,
            "datasource": datasource,
            "tools_used": tools_used,
            "citations":  citations,
            "chart_data": chart_data,
        }
        msg_index = len(st.session_state.messages)
        _render_assistant_message(assistant_msg, msg_index=msg_index)

st.session_state.messages.append(assistant_msg)
