"""FastAPI backend for the KT Agent.

Session memory is stored in-process (dict keyed by session_id).
For multi-worker deployments replace SessionStore with a Redis-backed store.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.config import DATA_DIR, setup_logging
from src.agent.rag import SUPPORTED_EXTENSIONS, _discover_documents, rebuild_index
from src.agent.schemas import Citation, QuestionRequest, QuestionResponse
from src.agent.workflow import aask

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KT Knowledge Transfer Assistant",
    description=(
        "Agentic Q&A over company documents, database, and the web. "
        "Supports conversation memory via session_id."
    ),
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# In-memory session store
# Stores the raw LangChain message objects (HumanMessage / AIMessage) per session.
# Each list element is a dict: {"role": "human"|"ai", "content": str}
# We store plain dicts so they survive JSON serialisation for the /history endpoint,
# and reconstruct LangChain message objects when passing to aask().
# ---------------------------------------------------------------------------

_sessions: dict[str, list[dict[str, str]]] = defaultdict(list)

MAX_HISTORY_TURNS = 20   # keep last 20 exchanges (40 messages) per session


def _get_lc_history(session_id: str) -> list:
    """Return the stored session as LangChain message objects."""
    lc_messages = []
    for msg in _sessions[session_id]:
        if msg["role"] == "human":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages


def _append_to_session(session_id: str, question: str, answer: str) -> None:
    """Append the latest exchange and trim to MAX_HISTORY_TURNS."""
    store = _sessions[session_id]
    store.append({"role": "human", "content": question})
    store.append({"role": "ai",    "content": answer})
    # Keep only the last MAX_HISTORY_TURNS * 2 messages
    if len(store) > MAX_HISTORY_TURNS * 2:
        _sessions[session_id] = store[-(MAX_HISTORY_TURNS * 2):]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["system"])
async def root() -> dict:
    """Root endpoint — confirms the API is running."""
    return {
        "status": "running",
        "message": "KT Knowledge Transfer Assistant API is running successfully! 🚀",
        "docs": "http://localhost:8000/docs",
        "health": "http://localhost:8000/health",
    }


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/ask", response_model=QuestionResponse, tags=["agent"])
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    """Submit a question to the KT agent.

    - **question**: The user's question (1–2000 characters).
    - **session_id**: Optional session identifier for multi-turn conversation memory.
      Use the same session_id across requests to maintain context.
      Defaults to `"default"`.
    """
    session_id = request.session_id or "default"
    question   = request.question.strip()

    logger.info("[session=%s] question: %s", session_id, question)

    try:
        history = _get_lc_history(session_id)
        result  = await aask(question, history=history)
    except Exception as exc:
        logger.exception("[session=%s] agent failed", session_id)
        raise HTTPException(status_code=500, detail="Agent failed to process the request.") from exc

    answer     = result.get("generation", "")
    datasource = result.get("datasource")
    tools_used = result.get("tools_used") or []
    raw_cits   = result.get("citations") or []
    chart_data = result.get("chart_data")

    # Persist this exchange for future turns
    _append_to_session(session_id, question, answer)

    citations = [
        Citation(source=c["source"], detail=c.get("detail", ""))
        for c in raw_cits
    ]

    logger.info(
        "[session=%s] datasource=%s tools=%s citations=%d chart=%s",
        session_id, datasource, tools_used, len(citations), chart_data is not None,
    )

    return QuestionResponse(
        answer=answer,
        datasource=datasource,
        tools_used=tools_used,
        citations=citations,
        chart_data=chart_data,
    )


@app.get("/sessions/{session_id}/history", tags=["memory"])
async def get_session_history(session_id: str) -> dict[str, Any]:
    """Return the conversation history for a session.

    Useful for debugging or pre-populating a UI on page reload.
    """
    history = _sessions.get(session_id, [])
    return {
        "session_id": session_id,
        "turn_count": len(history) // 2,
        "messages":   history,
    }


@app.delete("/sessions/{session_id}/history", tags=["memory"])
async def clear_session_history(session_id: str) -> dict[str, str]:
    """Clear the conversation history for a session (start fresh)."""
    _sessions.pop(session_id, None)
    logger.info("Cleared session: %s", session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/sessions", tags=["memory"])
async def list_sessions() -> dict[str, Any]:
    """List all active sessions and their message counts."""
    return {
        "sessions": [
            {"session_id": sid, "message_count": len(msgs)}
            for sid, msgs in _sessions.items()
        ]
    }


# ---------------------------------------------------------------------------
# Document upload + indexing
# ---------------------------------------------------------------------------

@app.post("/upload", tags=["documents"])
async def upload_documents(files: list[UploadFile] = File(...)) -> JSONResponse:
    """Upload one or more documents to the data/ folder and rebuild the index.

    Supported formats: PDF, DOCX, DOC, XLSX, XLS, CSV, TXT.
    The index is rebuilt automatically — no manual CLI step needed.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    saved: list[str] = []
    rejected: list[str] = []

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            rejected.append(upload.filename)
            continue

        dest = DATA_DIR / upload.filename
        content = await upload.read()
        dest.write_bytes(content)
        saved.append(upload.filename)
        logger.info("Saved uploaded file: %s (%d bytes)", dest, len(content))

    if not saved:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No supported files in upload. "
                f"Rejected: {rejected}. "
                f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    # Rebuild the full index (all files in data/) and clear the cache
    try:
        indexed = rebuild_index()
    except Exception as exc:
        logger.exception("Index rebuild failed after upload")
        raise HTTPException(
            status_code=500,
            detail=f"Files saved but index rebuild failed: {exc}",
        ) from exc

    return JSONResponse({
        "status":    "indexed",
        "saved":     saved,
        "rejected":  rejected,
        "indexed":   indexed,
        "message":   f"{len(saved)} file(s) uploaded and index rebuilt successfully.",
    })


@app.get("/documents", tags=["documents"])
async def list_documents() -> dict[str, Any]:
    """List all documents currently in the data/ folder."""
    docs = _discover_documents()
    return {
        "count":     len(docs),
        "documents": [
            {
                "name":  p.name,
                "size_kb": round(p.stat().st_size / 1024, 1),
                "type":  p.suffix.lower().lstrip(".").upper(),
            }
            for p in sorted(docs, key=lambda p: p.name.lower())
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
