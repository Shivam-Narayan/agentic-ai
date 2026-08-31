"""FastAPI backend for the KT Agent.

Conversation memory is handled by LangGraph checkpointers:
  - Default:  AsyncSqliteSaver  → memory_store/conversations.db
  - Postgres: AsyncPostgresSaver → PostgreSQL (USE_POSTGRES_MEMORY=true)

Each session_id maps to a LangGraph thread — history persists across restarts.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.agent.config import DATA_DIR, POSTGRES_URL, USE_PGVECTOR, USE_POSTGRES_MEMORY, setup_logging
from src.agent.rag import SUPPORTED_EXTENSIONS, _discover_documents, rebuild_index
from src.agent.schemas import Citation, QuestionRequest, QuestionResponse
from src.agent.workflow import KnowledgeTransferAgent, aask

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
#
# Keyed on client IP. Limits:
#   /ask    — 20 requests/minute  (blocking — each call is one full LLM round-trip)
#   /stream — 20 requests/minute  (streaming — same cost as /ask)
#   /upload — 10 requests/minute  (heavier — triggers full re-indexing)
#
# Limits can be overridden per-deployment via RATE_LIMIT_ASK,
# RATE_LIMIT_STREAM, RATE_LIMIT_UPLOAD env vars.
# ---------------------------------------------------------------------------

_RATE_ASK    = os.getenv("RATE_LIMIT_ASK",    "20/minute")
_RATE_STREAM = os.getenv("RATE_LIMIT_STREAM", "20/minute")
_RATE_UPLOAD = os.getenv("RATE_LIMIT_UPLOAD", "10/minute")

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

_MEMORY_DIR = Path(__file__).parent / "memory_store"
_MEMORY_DIR.mkdir(exist_ok=True)

# Filled in by the lifespan handler — shared across all requests
_checkpointer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the checkpointer on startup, close on shutdown.

    Uses AsyncPostgresSaver when USE_POSTGRES_MEMORY=true,
    otherwise falls back to AsyncSqliteSaver (default, no extra setup needed).
    """
    global _checkpointer

    if USE_POSTGRES_MEMORY:
        # ── PostgreSQL memory ───────────────────────────────────────────
        # AsyncPostgresSaver requires psycopg3 and langgraph-checkpoint-postgres.
        # cp.setup() creates the checkpoints / checkpoint_writes / checkpoint_blobs
        # tables automatically on first run — no manual SQL needed.
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            # AsyncPostgresSaver expects a plain psycopg3 URL (no +psycopg prefix)
            pg_url = POSTGRES_URL.replace("postgresql+psycopg://", "postgresql://")

            async with AsyncPostgresSaver.from_conn_string(pg_url) as cp:
                await cp.setup()
                _checkpointer = cp
                logger.info("AsyncPostgresSaver opened: %s", pg_url.split("@")[-1])
                yield

        except Exception as exc:
            # If Postgres is unavailable, fall back to SQLite and log a clear warning
            logger.error(
                "AsyncPostgresSaver failed to open (%s). "
                "Falling back to SQLite memory. "
                "Is Docker running? Check POSTGRES_URL in .env.",
                exc,
            )
            async with _open_sqlite_checkpointer() as cp:
                _checkpointer = cp
                yield
    else:
        # ── SQLite memory (default) ─────────────────────────────────────
        async with _open_sqlite_checkpointer() as cp:
            _checkpointer = cp
            yield

    _checkpointer = None


@asynccontextmanager
async def _open_sqlite_checkpointer():
    """Open the AsyncSqliteSaver — extracted so both lifespan branches can use it."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = str(_MEMORY_DIR / "conversations.db")
    async with AsyncSqliteSaver.from_conn_string(db_path) as cp:
        logger.info("AsyncSqliteSaver opened: %s", db_path)
        yield cp


app = FastAPI(
    title="KT Knowledge Transfer Assistant",
    description=(
        "Agentic Q&A over company documents, database, and the web. "
        "Supports conversation memory via session_id."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Attach rate limiter — must be done after app creation
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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
async def health() -> dict[str, Any]:
    """Liveness check — reports active backend."""
    return {
        "status":         "ok",
        "memory_backend": "postgres" if USE_POSTGRES_MEMORY else "sqlite",
        "vector_backend": "pgvector" if USE_PGVECTOR else "json",
    }


@app.get("/stream", tags=["agent"])
@limiter.limit(_RATE_STREAM)
async def stream_question(request: Request, question: str, session_id: str = "default"):
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
@limiter.limit(_RATE_ASK)
async def ask_question(request: Request, body: QuestionRequest) -> QuestionResponse:
    """Submit a question to the KT agent.

    - **question**: The user's question (1–2000 characters).
    - **session_id**: Optional session identifier for multi-turn conversation memory.
      Use the same session_id across requests to maintain context.
      Defaults to `"default"`.
    """
    session_id = body.session_id or "default"
    question   = body.question.strip()

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
    """List all sessions stored in the active checkpointer."""
    if USE_POSTGRES_MEMORY:
        # Query the PostgreSQL checkpoints table
        try:
            import psycopg
            pg_url = POSTGRES_URL.replace("postgresql+psycopg://", "postgresql://")
            async with await psycopg.AsyncConnection.connect(pg_url) as conn:
                rows = await conn.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
                )
                results = await rows.fetchall()
            return {"sessions": [{"session_id": row[0]} for row in results]}
        except Exception as exc:
            logger.warning("Could not list Postgres sessions: %s", exc)
            return {"sessions": []}
    else:
        # Query the SQLite checkpoints table
        import sqlite3
        db_path = _MEMORY_DIR / "conversations.db"
        if not db_path.exists():
            return {"sessions": []}
        try:
            con = sqlite3.connect(str(db_path))
            rows = con.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            ).fetchall()
            con.close()
            return {"sessions": [{"session_id": row[0]} for row in rows]}
        except Exception as exc:
            logger.warning("Could not list SQLite sessions: %s", exc)
            return {"sessions": []}


# ---------------------------------------------------------------------------
# Document upload + indexing
# ---------------------------------------------------------------------------

@app.post("/upload", tags=["documents"])
@limiter.limit(_RATE_UPLOAD)
async def upload_documents(request: Request, files: list[UploadFile] = File(...)) -> JSONResponse:
    """Upload one or more documents to the data/ folder and rebuild the index.

    Supported formats: PDF, DOCX, DOC, XLSX, XLS, CSV, TXT.
    Duplicate detection:
      - Same filename + identical content → rejected as duplicate
      - Same filename + different content → rejected, user must rename
    The index is rebuilt automatically — no manual CLI step needed.
    """
    import hashlib

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    saved:      list[str] = []
    rejected:   list[str] = []
    duplicates: list[str] = []   # exact content match
    conflicts:  list[str] = []   # same name, different content

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            rejected.append(upload.filename)
            continue

        content = await upload.read()
        dest    = DATA_DIR / upload.filename

        if dest.exists():
            existing_hash = hashlib.md5(dest.read_bytes()).hexdigest()
            incoming_hash = hashlib.md5(content).hexdigest()

            if existing_hash == incoming_hash:
                # Byte-for-byte identical — definite duplicate
                logger.info(
                    "Skipping duplicate file (identical content): %s", upload.filename
                )
                duplicates.append(upload.filename)
                continue
            else:
                # Same filename, different content — require explicit rename
                logger.warning(
                    "File conflict (same name, different content): %s", upload.filename
                )
                conflicts.append(upload.filename)
                continue

        dest.write_bytes(content)
        saved.append(upload.filename)
        logger.info("Saved uploaded file: %s (%d bytes)", dest, len(content))

    # Nothing new was saved
    if not saved:
        detail_parts: list[str] = []
        if duplicates:
            detail_parts.append(
                f"Already indexed (identical content): {', '.join(duplicates)}"
            )
        if conflicts:
            detail_parts.append(
                f"Name conflict — rename before uploading: {', '.join(conflicts)}"
            )
        if rejected:
            detail_parts.append(
                f"Unsupported format: {', '.join(rejected)}"
            )
        raise HTTPException(
            status_code=400,
            detail=" | ".join(detail_parts) or "No supported files provided.",
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
        "status":     "indexed",
        "saved":      saved,
        "duplicates": duplicates,
        "conflicts":  conflicts,
        "rejected":   rejected,
        "indexed":    indexed,
        "message":    (
            f"{len(saved)} file(s) uploaded and indexed."
            + (f" {len(duplicates)} duplicate(s) skipped." if duplicates else "")
            + (f" {len(conflicts)} conflict(s) need renaming." if conflicts else "")
        ),
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
