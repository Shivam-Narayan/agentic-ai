"""FastAPI backend for the KT Agent.

Conversation memory is handled by LangGraph's AsyncSqliteSaver checkpointer.
Each session_id maps to a LangGraph thread — history is persisted in
memory_store/conversations.db and survives server restarts.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.agent.config import DATA_DIR, setup_logging
from src.agent.rag import SUPPORTED_EXTENSIONS, _discover_documents, rebuild_index
from src.agent.schemas import Citation, QuestionRequest, QuestionResponse
from src.agent.workflow import KnowledgeTransferAgent, aask

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangGraph AsyncSqliteSaver checkpointer
#
# Opened once at startup via FastAPI lifespan, shared across all requests.
# Persists full message history per session_id to conversations.db.
# ---------------------------------------------------------------------------

_MEMORY_DIR = Path(__file__).parent / "memory_store"
_MEMORY_DIR.mkdir(exist_ok=True)

# Module-level reference filled in by the lifespan handler
_checkpointer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the async SQLite checkpointer on startup, close on shutdown."""
    global _checkpointer
    async with AsyncSqliteSaver.from_conn_string(
        str(_MEMORY_DIR / "conversations.db")
    ) as cp:
        _checkpointer = cp
        logger.info("AsyncSqliteSaver opened: %s", _MEMORY_DIR / "conversations.db")
        yield
    _checkpointer = None
    logger.info("AsyncSqliteSaver closed")


app = FastAPI(
    title="KT Knowledge Transfer Assistant",
    description=(
        "Agentic Q&A over company documents, database, and the web. "
        "Supports conversation memory via session_id."
    ),
    version="2.0.0",
    lifespan=lifespan,
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


@app.get("/stream", tags=["agent"])
async def stream_question(question: str, session_id: str = "default"):
    """Stream a question to the KT agent using Server-Sent Events.

    Each SSE event is a JSON object with a `type` field:
    - `{"type": "status", "stage": "thinking"}` — agent started
    - `{"type": "token",  "text": "..."}` — incremental answer token
    - `{"type": "tool",   "name": "..."}` — tool being invoked
    - `{"type": "done",   "payload": {...}}` — final structured result
    - `{"type": "error",  "detail": "..."}` — unrecoverable error
    """
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty.")

    logger.info("[stream][session=%s] question: %s", session_id, question)

    agent = KnowledgeTransferAgent(checkpointer=_checkpointer)

    async def event_generator():
        try:
            async for event in agent.run(question, session_id=session_id):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.exception("[stream][session=%s] agent error", session_id)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind a proxy
        },
    )


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
        config = {"configurable": {"thread_id": session_id}}
        checkpoint_tuple = await _checkpointer.aget_tuple(config)
        if checkpoint_tuple is None:
            return {"session_id": session_id, "turn_count": 0, "messages": []}

        messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
        history = []
        for msg in messages:
            if hasattr(msg, "type") and msg.type in ("human", "ai"):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                history.append({"role": msg.type, "content": content})

        turns = sum(1 for m in history if m["role"] == "human")
        return {"session_id": session_id, "turn_count": turns, "messages": history}
    except Exception as exc:
        logger.warning("Could not get history for %s: %s", session_id, exc)
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
    import sqlite3
    db_path = _MEMORY_DIR / "conversations.db"
    if not db_path.exists():
        return {"sessions": []}
    try:
        con = sqlite3.connect(str(db_path))
        # AsyncSqliteSaver stores thread_id as a direct column in checkpoints
        rows = con.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
        con.close()
        return {"sessions": [{"session_id": row[0]} for row in rows]}
    except Exception as exc:
        logger.warning("Could not list sessions: %s", exc)
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
