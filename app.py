"""FastAPI backend for the KT Agent.

Conversation memory is handled by LangGraph's SqliteSaver checkpointer.
Each session_id maps to a LangGraph thread — history is persisted in
memory_store/conversations.db and survives server restarts.
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent.config import DATA_DIR, setup_logging
from src.agent.rag import SUPPORTED_EXTENSIONS, _discover_documents, rebuild_index
from src.agent.schemas import Citation, QuestionRequest, QuestionResponse
from src.agent.workflow import aask

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangGraph SQLite checkpointer
#
# Persists the full message history for every session_id to a local SQLite
# database. LangGraph reloads it automatically on the next request with the
# same session_id — no manual history management needed.
# ---------------------------------------------------------------------------

_MEMORY_DIR = Path(__file__).parent / "memory_store"
_MEMORY_DIR.mkdir(exist_ok=True)

# SqliteSaver.from_conn_string returns a context manager.
# We enter it once at module load time and keep it open for the
# lifetime of the server process.
_checkpointer_ctx = SqliteSaver.from_conn_string(str(_MEMORY_DIR / "conversations.db"))
_checkpointer = _checkpointer_ctx.__enter__()

app = FastAPI(
    title="KT Knowledge Transfer Assistant",
    description=(
        "Agentic Q&A over company documents, database, and the web. "
        "Supports conversation memory via session_id."
    ),
    version="2.0.0",
)


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
        result = await aask(
            question,
            session_id=session_id,
            checkpointer=_checkpointer,
        )
    except Exception as exc:
        logger.exception("[session=%s] agent failed", session_id)
        raise HTTPException(status_code=500, detail="Agent failed to process the request.") from exc

    answer     = result.get("generation", "")
    datasource = result.get("datasource")
    tools_used = result.get("tools_used") or []
    raw_cits   = result.get("citations") or []
    chart_data = result.get("chart_data")

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
    """Return the conversation history for a session from the SQLite checkpointer."""
    try:
        # Load the latest checkpoint for this thread
        config = {"configurable": {"thread_id": session_id}}
        state = _checkpointer.get(config)
        if state is None:
            return {"session_id": session_id, "turn_count": 0, "messages": []}

        messages = state.get("channel_values", {}).get("messages", [])
        # Convert LangChain message objects to plain dicts for JSON response
        history = []
        for msg in messages:
            if hasattr(msg, "type"):
                role = "human" if msg.type == "human" else "ai"
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                history.append({"role": role, "content": content})

        # Count only human+ai pairs (exclude system/tool messages)
        turns = sum(1 for m in history if m["role"] == "human")
        return {"session_id": session_id, "turn_count": turns, "messages": history}
    except Exception:
        return {"session_id": session_id, "turn_count": 0, "messages": []}


@app.delete("/sessions/{session_id}/history", tags=["memory"])
async def clear_session_history(session_id: str) -> dict[str, str]:
    """Clear the conversation history for a session by writing an empty checkpoint."""
    try:
        # Write a blank checkpoint to effectively reset the thread
        from langgraph.checkpoint.base import CheckpointMetadata
        config = {"configurable": {"thread_id": session_id}}
        empty_checkpoint = {
            "v": 1,
            "ts": "",
            "id": session_id,
            "channel_values": {"messages": []},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        _checkpointer.put(config, empty_checkpoint, CheckpointMetadata(), {})
        logger.info("Cleared session: %s", session_id)
    except Exception as exc:
        logger.warning("Could not clear session %s: %s", session_id, exc)
    return {"status": "cleared", "session_id": session_id}


@app.get("/sessions", tags=["memory"])
async def list_sessions() -> dict[str, Any]:
    """List all sessions stored in the SQLite checkpointer."""
    try:
        sessions = []
        for config, *_ in _checkpointer.list({}):
            tid = config.get("configurable", {}).get("thread_id", "unknown")
            sessions.append({"session_id": tid})
        return {"sessions": sessions}
    except Exception:
        return {"sessions": []}


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
