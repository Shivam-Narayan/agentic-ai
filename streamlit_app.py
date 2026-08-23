import os

import httpx
import streamlit as st
from pydantic import ValidationError

from src.agent.schemas import QuestionRequest

API_URL = os.getenv("KT_API_URL", "http://192.168.88.6:8000")

SAMPLE_QUESTIONS = {
    "LLM only": [
        "What is Python?",
        "Explain what a REST API is.",
    ],
    "Live tool (web)": [
        "What is the current weather in Bangalore?",
    ],
    "Company documents": [
        "What is Beacon and what does this project do?",
        "What is the tech stack used in this project?",
    ],
    "Company database": [
        "What is the status of order #12345?",
        "What is the status of order #10001?",
    ],
}

ROUTE_LABELS = {
    "direct_llm": "💬 LLM answered directly (no tools)",
    "company_docs": "📄 Tool: search_company_documents",
    "database": "🗄️ Tool: company database (MCP)",
    "web_search": "🌐 Tool: search_web",
    "multiple": "🔀 Multiple tools used",
}

st.set_page_config(
    page_title="CTE Knowledge Transfer Assistant",
    page_icon="🤖",
    layout="centered",
)

# ── Scoped CSS: normalize heading sizes inside chat messages ──────────────
st.markdown("""
<style>
/* Normalize all markdown headings inside Streamlit chat messages to body size */
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3,
[data-testid="stChatMessageContent"] h4,
[data-testid="stChatMessageContent"] h5,
[data-testid="stChatMessageContent"] h6 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.25rem !important;
    line-height: 1.5 !important;
}

[data-testid="stChatMessageContent"] p {
    font-size: 0.95rem;
    line-height: 1.6;
    margin-bottom: 0.4rem;
}

[data-testid="stChatMessageContent"] ul,
[data-testid="stChatMessageContent"] ol {
    font-size: 0.95rem;
    padding-left: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("CTE Knowledge Transfer Assistant 🤖")
st.caption(
    "Ask a question. General knowledge goes to the LLM. Company documents, orders, "
    "and live facts go through tools, then the LLM writes the answer."
)

st.sidebar.markdown("**API**")
st.sidebar.code(API_URL, language="text")
st.sidebar.caption("Start the backend first: `uvicorn app:app --reload --port 8000`")

st.sidebar.markdown("**Try each path**")
for group, questions in SAMPLE_QUESTIONS.items():
    st.sidebar.caption(group)
    for sample in questions:
        if st.sidebar.button(sample, use_container_width=True):
            st.session_state.pending_question = sample
            st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ask anything. Examples:\n"
                "- What is Python? → LLM only\n"
                "- Weather in Bangalore? → web tool\n"
                "- What is Beacon / this project? → company documents tool\n"
                "- Status of order #12345? → company database tool"
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("route"):
            label = ROUTE_LABELS.get(message["route"], f"Routed to: {message['route']}")
            tools = message.get("tools_used", [])
            caption_parts = [label]
            if tools:
                caption_parts.append(f"Tools called: `{'`, `'.join(tools)}`")
            st.caption(" · ".join(caption_parts))

prompt = st.session_state.pop("pending_question", None) or st.chat_input("Type your question...")

if not prompt:
    st.stop()

try:
    QuestionRequest(question=prompt.strip())
except ValidationError:
    st.error("Please enter a question between 1 and 2000 characters.")
    st.stop()

st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.markdown(prompt)

route = None
tools_used = []
with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
        try:
            response = httpx.post(
                f"{API_URL.rstrip('/')}/ask",
                json={"question": prompt.strip()},
                timeout=120.0,
            )
            response.raise_for_status()
            payload = response.json()
            answer = payload.get("answer") or "I could not generate an answer."
            route = payload.get("datasource")
            tools_used = payload.get("tools_used") or []
        except httpx.ConnectError:
            answer = (
                "⚠️ Could not reach FastAPI. Start the backend with "
                "`uvicorn app:app --reload --port 8000` and try again."
            )
        except Exception as exc:
            answer = f"⚠️ Failed to process the question: {exc}"

    st.markdown(answer)
    if route:
        label = ROUTE_LABELS.get(route, f"Routed to: {route}")
        caption_parts = [label]
        if tools_used:
            caption_parts.append(f"Tools called: `{'`, `'.join(tools_used)}`")
        st.caption(" · ".join(caption_parts))

st.session_state.messages.append(
    {"role": "assistant", "content": answer, "route": route, "tools_used": tools_used}
)
