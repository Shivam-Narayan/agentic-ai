"""FastAPI backend for the KT Agent.

Conversation memory is handled by LangGraph checkpointers:
  - Default:  AsyncSqliteSaver  → memory_store/conversations.db
  - Postgres: AsyncPostgresSaver → PostgreSQL (USE_POSTGRES_MEMORY=true)

Each session_id maps to a LangGraph thread — history persists across restarts.
"""

import asyncio
import json
import logging
import os
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import re

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# AsyncPostgresSaver is imported lazily inside lifespan() so deployments
# that don't use Postgres don't require langgraph-checkpoint-postgres installed.

from src.core.config import DATA_DIR, STORAGE_DIR, POSTGRES_URL, USE_PGVECTOR, USE_POSTGRES_MEMORY, setup_logging
from src.retrieval.rag import SUPPORTED_EXTENSIONS, discover_documents, add_documents_to_index, rebuild_index, get_embed_model
from src.agent.schemas import Citation, QuestionRequest, QuestionResponse, UsageMetrics
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

# Maximum upload size per file (50 MB) — prevents memory exhaustion.
_MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

# session_id validation — alphanumeric, hyphens, underscores, max 64 chars.
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_session_id(session_id: str) -> str:
    """Validate session_id format and raise HTTP 400 if invalid.

    Prevents path traversal, SQL metacharacters, and oversized keys
    from reaching the LangGraph checkpointer as thread_id.

    Args:
        session_id: Raw session_id string from the request.

    Returns:
        The validated session_id unchanged.

    Raises:
        HTTPException: 400 if the session_id fails validation.
    """
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid session_id. Use 1–64 characters: "
                "letters, digits, hyphens, or underscores only."
            ),
        )
    return session_id

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

_MEMORY_DIR = STORAGE_DIR / "memory_store"
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

    # Pre-warm the embedding model to avoid cold-start latency on first query.
    logger.info("Pre-warming embedding model...")
    await asyncio.to_thread(get_embed_model().get_text_embedding, "warmup")
    logger.info("Embedding model pre-warmed.")

    if USE_POSTGRES_MEMORY:
        # Lazy import — only required when USE_POSTGRES_MEMORY=true so
        # deployments without langgraph-checkpoint-postgres still start cleanly.
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: PLC0415
        # ── PostgreSQL memory ───────────────────────────────────────────
        # AsyncPostgresSaver requires psycopg3 and langgraph-checkpoint-postgres.
        # cp.setup() creates the checkpoints / checkpoint_writes / checkpoint_blobs
        # tables automatically on first run — no manual SQL needed.
        try:

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
async def root(request: Request) -> dict[str, str]:
    """Root endpoint — confirms the API is running."""
    base = str(request.base_url).rstrip("/")
    return {
        "status":  "running",
        "message": "KT Knowledge Transfer Assistant API is running successfully! 🚀",
        "docs":    f"{base}/docs",
        "health":  f"{base}/health",
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
    """Stream a question to the KT agent using Server-Sent Events."""
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty.")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="question must be 2000 characters or fewer.")

    session_id = _validate_session_id(session_id)
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
    session_id = _validate_session_id(body.session_id or "default")
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
    raw_usage  = result.get("usage") or {}
    prompt_ver = result.get("prompt_version")

    citations = [
        Citation(source=c["source"], detail=c.get("detail", ""))
        for c in raw_cits
    ]

    usage = UsageMetrics(
        prompt_tokens=raw_usage.get("prompt_tokens", 0),
        completion_tokens=raw_usage.get("completion_tokens", 0),
        total_tokens=raw_usage.get("total_tokens", 0),
        cost_usd=raw_usage.get("cost_usd", 0.0),
        model=raw_usage.get("model", ""),
        latency_ms=raw_usage.get("latency_ms", 0),
        llm_calls=raw_usage.get("llm_calls", 0),
    ) if raw_usage else None

    logger.info(
        "[session=%s] datasource=%s tools=%s citations=%d chart=%s "
        "tokens=%d cost=$%.6f latency=%dms",
        session_id, datasource, tools_used, len(citations),
        chart_data is not None,
        raw_usage.get("total_tokens", 0),
        raw_usage.get("cost_usd", 0.0),
        raw_usage.get("latency_ms", 0),
    )

    return QuestionResponse(
        answer=answer,
        datasource=datasource,
        tools_used=tools_used,
        citations=citations,
        chart_data=chart_data,
        usage=usage,
        prompt_version=prompt_ver,
    )


@app.get("/sessions/{session_id}/history", tags=["memory"])
async def get_session_history(session_id: str) -> dict[str, Any]:
    """Return the conversation history for a session from the checkpointer."""
    session_id = _validate_session_id(session_id)

    if _checkpointer is None:
        return {"session_id": session_id, "turn_count": 0, "messages": []}

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
    session_id = _validate_session_id(session_id)
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
        if _checkpointer is not None:
            if hasattr(_checkpointer, "aput"):
                await _checkpointer.aput(config, empty_checkpoint, CheckpointMetadata(), {})
            elif hasattr(_checkpointer, "put"):
                await asyncio.to_thread(_checkpointer.put, config, empty_checkpoint, CheckpointMetadata(), {})
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
        # Query the SQLite checkpoints table (offloaded to thread to avoid blocking event loop)
        import sqlite3
        db_path = _MEMORY_DIR / "conversations.db"
        if not db_path.exists():
            return {"sessions": []}
        try:
            def _query_sqlite():
                con = sqlite3.connect(str(db_path))
                try:
                    rows = con.execute(
                        "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
                    ).fetchall()
                    return [{"session_id": row[0]} for row in rows]
                finally:
                    con.close()

            sessions = await asyncio.to_thread(_query_sqlite)
            return {"sessions": sessions}
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

        if len(content) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {upload.filename}",
            )

        dest = DATA_DIR / upload.filename

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

    # Incrementally index newly uploaded files into the vector store
    saved_paths = [DATA_DIR / f for f in saved]
    try:
        indexed = await asyncio.to_thread(add_documents_to_index, saved_paths)
    except Exception as exc:
        logger.exception("Incremental indexing failed after upload")
        raise HTTPException(
            status_code=500,
            detail=f"Files saved but index update failed: {exc}",
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
    docs = discover_documents()  # public function — no leading underscore
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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
