import os

import httpx
import streamlit as st
from pydantic import ValidationError

from src.kt_agent.schemas import QuestionRequest

API_URL = os.getenv("KT_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="CTE Knowledge Transfer Assistant",
    page_icon="🤖",
)

st.title("CTE Knowledge Transfer Assistant 🤖")
st.caption("Ask about the project knowledge base. The Streamlit UI sends your question to FastAPI, which runs LangGraph.")

st.sidebar.markdown("**API**")
st.sidebar.code(API_URL, language="text")
st.sidebar.caption("Start the backend first: `uvicorn app:app --reload --port 8000`")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your personal CTE KT assistant. Which project do you need help with?",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Type your question...")

if not prompt:
    st.stop()

try:
    QuestionRequest(question=prompt.strip())
except ValidationError:
    st.error("Please enter a question between 1 and 2000 characters.")
    st.stop()

st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.write(prompt)

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
            datasource = payload.get("datasource")
        except httpx.ConnectError:
            answer = (
                "Could not reach FastAPI. Start the backend with "
                "`uvicorn app:app --reload --port 8000` and try again."
            )
            datasource = None
        except Exception as exc:
            answer = f"Failed to process the question: {exc}"
            datasource = None

    st.write(answer)
    if datasource:
        st.caption(f"Routed to: {datasource}")

st.session_state.messages.append({"role": "assistant", "content": answer})
