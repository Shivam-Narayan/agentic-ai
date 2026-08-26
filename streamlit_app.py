"""Streamlit chat UI for the CTE Knowledge Transfer Assistant.

Features:
- Multi-turn conversation memory (session_id per browser tab)
- Source citations shown under each AI answer
- Plotly charts rendered inline when the agent generates one
- Tool badge showing which tool(s) were used
- Sidebar: document upload with auto-indexing, sample questions, session controls
"""

import io
import os
import uuid

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # load KT_API_URL and other vars from .env

# ── Config ──────────────────────────────────────────────────────────────────
API_URL = os.getenv("KT_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 180.0  # seconds — LLM + tool calls can be slow on first run

# ── Sample questions for every tool path ────────────────────────────────────
SAMPLE_QUESTIONS: dict[str, list[str]] = {
    "💬 Direct LLM": [
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

# ── Datasource display config ────────────────────────────────────────────────
ROUTE_CONFIG: dict[str, dict] = {
    "direct_llm":  {"icon": "💬", "label": "LLM answered directly",         "color": "#6c757d"},
    "company_docs":{"icon": "📄", "label": "Company documents",              "color": "#0d6efd"},
    "database":    {"icon": "🗄️", "label": "Company database",               "color": "#198754"},
    "web_search":  {"icon": "🌐", "label": "Live web search",                "color": "#fd7e14"},
    "calculation": {"icon": "🧮", "label": "Calculator",                     "color": "#6f42c1"},
    "chart":       {"icon": "📊", "label": "Chart generated",                "color": "#20c997"},
    "multiple":    {"icon": "🔀", "label": "Multiple tools used",            "color": "#dc3545"},
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _fetch_documents() -> list[dict]:
    """Return the list of indexed documents from the backend, or [] on error."""
    try:
        r = httpx.get(f"{API_URL.rstrip('/')}/documents", timeout=8)
        r.raise_for_status()
        return r.json().get("documents", [])
    except Exception:
        return []


def _upload_files(uploaded_files) -> dict:
    """POST files to /upload and return the response JSON."""
    files_payload = [
        ("files", (f.name, io.BytesIO(f.read()), f.type or "application/octet-stream"))
        for f in uploaded_files
    ]
    r = httpx.post(
        f"{API_URL.rstrip('/')}/upload",
        files=files_payload,
        timeout=300,   # indexing large files can take a while
    )
    r.raise_for_status()
    return r.json()


# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(    page_title="KT Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Normalise heading sizes inside chat bubbles */
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3,
[data-testid="stChatMessageContent"] h4 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.2rem !important;
}
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li {
    font-size: 0.93rem;
    line-height: 1.65;
}
/* Citation pill */
.citation-pill {
    display: inline-block;
    background: #f0f4ff;
    border: 1px solid #c9d8ff;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #3a5bd9;
    margin: 2px 3px 2px 0;
}
/* Tool badge */
.tool-badge {
    display: inline-block;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.76rem;
    font-weight: 600;
    color: #fff;
    margin-right: 4px;
}
/* Thin separator above citations */
.citation-block {
    border-top: 1px solid #e9ecef;
    margin-top: 8px;
    padding-top: 6px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state bootstrap ──────────────────────────────────────────────────
if "session_id" not in st.session_state:
    # Each browser tab gets its own UUID — gives true per-tab conversation memory
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm your **Knowledge Transfer Assistant**. Ask me anything:\n\n"
                "- 📄 Questions about company documents (PDFs, Word, Excel)\n"
                "- 🗄️ Database queries (orders, customers, employees...)\n"
                "- 🌐 Live web facts (news, prices, weather)\n"
                "- 🧮 Calculations and number crunching\n"
                "- 📊 Data visualisation (charts from your data)\n\n"
                "I remember the full conversation — feel free to ask follow-up questions!"
            ),
            "datasource": None,
            "tools_used": [],
            "citations": [],
            "chart_data": None,
        }
    ]


# ── Helper: render a single assistant message ────────────────────────────────
def _render_assistant_message(msg: dict) -> None:
    """Render answer text, tool badge, chart, and citations for one assistant turn."""
    st.markdown(msg["content"])

    datasource  = msg.get("datasource")
    tools_used  = msg.get("tools_used") or []
    citations   = msg.get("citations") or []
    chart_data  = msg.get("chart_data")

    # ── Inline chart ──────────────────────────────────────────────────────
    if chart_data:
        try:
            import plotly.graph_objects as go
            fig = go.Figure(chart_data)
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{id(msg)}")
        except Exception as exc:
            st.warning(f"Could not render chart: {exc}")

    # ── Tool badge + tools list ───────────────────────────────────────────
    if datasource:
        cfg   = ROUTE_CONFIG.get(datasource, {"icon": "🔧", "label": datasource, "color": "#666"})
        color = cfg["color"]
        label = cfg["label"]
        icon  = cfg["icon"]

        badge_html = (
            f'<span class="tool-badge" style="background:{color}">'
            f'{icon} {label}</span>'
        )
        if tools_used:
            tools_str = " · ".join(f"`{t}`" for t in tools_used)
            st.markdown(
                badge_html + f'<span style="font-size:0.78rem;color:#555"> {tools_str}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(badge_html, unsafe_allow_html=True)

    # ── Citations ─────────────────────────────────────────────────────────
    if citations:
        pills_html = "".join(
            f'<span class="citation-pill" title="{c.get("detail","")}">'
            f'📎 {c["source"]}</span>'
            for c in citations
        )
        st.markdown(
            f'<div class="citation-block">'
            f'<span style="font-size:0.76rem;color:#888;font-weight:600">SOURCES </span>'
            f'{pills_html}</div>',
            unsafe_allow_html=True,
        )


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 KT Assistant")
    st.caption(f"Session `{st.session_state.session_id[:8]}…`")
    st.divider()

    # ── Document upload ───────────────────────────────────────────────────
    st.markdown("### 📂 Upload Documents")
    st.caption("PDF, DOCX, XLSX, CSV, TXT — dropped into `data/` and indexed automatically.")

    uploaded = st.file_uploader(
        label="Choose files",
        type=["pdf", "docx", "doc", "xlsx", "xls", "csv", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("⬆️ Upload & Index", type="primary", use_container_width=True):
            with st.spinner("Uploading and building index… this may take a minute."):
                try:
                    result = _upload_files(uploaded)
                    saved    = result.get("saved", [])
                    rejected = result.get("rejected", [])
                    indexed  = result.get("indexed", [])

                    if saved:
                        st.success(
                            f"✅ {len(saved)} file(s) uploaded and indexed:\n"
                            + "\n".join(f"• {f}" for f in saved)
                        )
                    if rejected:
                        st.warning(
                            f"⚠️ {len(rejected)} file(s) skipped (unsupported format):\n"
                            + "\n".join(f"• {f}" for f in rejected)
                        )
                    # Force the document list to refresh
                    st.session_state.pop("docs_cache", None)

                except httpx.ConnectError:
                    st.error("Cannot reach the backend. Is FastAPI running?")
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

    st.divider()

    # ── Indexed documents list ────────────────────────────────────────────
    st.markdown("### 📋 Indexed Documents")

    # Cache the doc list in session state so it doesn't re-fetch on every keystroke
    if "docs_cache" not in st.session_state:
        st.session_state.docs_cache = _fetch_documents()

    docs = st.session_state.docs_cache

    if docs:
        for doc in docs:
            icon = {"PDF": "📄", "DOCX": "📝", "DOC": "📝",
                    "XLSX": "📊", "XLS": "📊", "CSV": "📊", "TXT": "📃"}.get(doc["type"], "📎")
            st.markdown(
                f"{icon} **{doc['name']}** "
                f"<span style='color:#888;font-size:0.78rem'>{doc['size_kb']} KB</span>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No documents indexed yet. Upload files above to get started.")

    if st.button("🔄 Refresh list", use_container_width=True):
        st.session_state.docs_cache = _fetch_documents()
        st.rerun()

    st.divider()

    # ── Session controls ──────────────────────────────────────────────────
    st.markdown("### 💬 Session")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            try:
                httpx.delete(
                    f"{API_URL}/sessions/{st.session_state.session_id}/history",
                    timeout=10,
                )
            except Exception:
                pass
            st.rerun()
    with col2:
        if st.button("🆕 New session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages   = []
            st.rerun()

    st.divider()

    # ── Sample questions ──────────────────────────────────────────────────
    st.markdown("### 💡 Try a sample question")
    for group, questions in SAMPLE_QUESTIONS.items():
        with st.expander(group, expanded=False):
            for q in questions:
                if st.button(q, key=f"sample_{q}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()

    st.divider()
    st.markdown("**Backend**")
    st.code(API_URL, language="text")
    st.caption("Start: `uvicorn app:app --reload --port 8000`")


# ── Main layout ──────────────────────────────────────────────────────────────
st.title("CTE Knowledge Transfer Assistant 🤖")
st.caption(
    "Ask anything about company documents, databases, or the live web. "
    "Charts, calculations, and source citations included."
)

# Render existing conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_assistant_message(msg)
        else:
            st.markdown(msg["content"])

# ── Input handling ───────────────────────────────────────────────────────────
prompt = (
    st.session_state.pop("pending_question", None)
    or st.chat_input("Ask a question…")
)

if not prompt:
    st.stop()

prompt = prompt.strip()
if not prompt:
    st.stop()

# Show user message immediately
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.markdown(prompt)

# ── Call the FastAPI backend ─────────────────────────────────────────────────
with st.chat_message("assistant"):
    with st.spinner("Thinking…"):
        answer     = ""
        datasource = None
        tools_used = []
        citations  = []
        chart_data = None

        try:
            response = httpx.post(
                f"{API_URL.rstrip('/')}/ask",
                json={
                    "question":   prompt,
                    "session_id": st.session_state.session_id,
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload    = response.json()
            answer     = payload.get("answer") or "I could not generate an answer."
            datasource = payload.get("datasource")
            tools_used = payload.get("tools_used") or []
            citations  = payload.get("citations") or []
            chart_data = payload.get("chart_data")

        except httpx.ConnectError:
            answer = (
                "⚠️ **Cannot reach the backend.**\n\n"
                f"Make sure FastAPI is running:\n```\nuvicorn app:app --reload --port 8000\n```\n"
                f"Expected at: `{API_URL}`"
            )
        except httpx.TimeoutException:
            answer = (
                "⚠️ **Request timed out.** "
                "The agent may still be processing — try asking again or "
                "check the backend logs."
            )
        except Exception as exc:
            answer = f"⚠️ **Unexpected error:** {exc}"

    # Build the message dict so _render_assistant_message can use it
    assistant_msg = {
        "role":       "assistant",
        "content":    answer,
        "datasource": datasource,
        "tools_used": tools_used,
        "citations":  citations,
        "chart_data": chart_data,
    }
    _render_assistant_message(assistant_msg)

# Persist to session state so the message is re-rendered on the next rerun
st.session_state.messages.append(assistant_msg)
