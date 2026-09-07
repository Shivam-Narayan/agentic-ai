# -*- coding: utf-8 -*-
"""
migrate_to_pgvector.py — One-time migration script.

Reads all documents from data/, re-embeds them, and inserts the vectors
into the PostgreSQL pgvector table. Also sets up the LangGraph checkpoint
tables for PostgreSQL memory.

Run ONCE after docker compose up -d and before setting USE_PGVECTOR=true.

Usage:
    python migrate_to_pgvector.py

What it does:
    1. Verify Docker / Postgres is reachable
    2. Create pgvector extension + document_embeddings table (idempotent)
    3. Re-embed all documents in data/ and insert into pgvector
    4. Set up LangGraph checkpoint tables (AsyncPostgresSaver.setup())
    5. Run a smoke-test query to confirm retrieval works
    6. Print final instructions to flip the feature flags
"""

import asyncio
import sys
import time
from pathlib import Path

# Windows fix: psycopg3 async requires SelectorEventLoop, not ProactorEventLoop.
# Must be set BEFORE any async code runs.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── Make sure we can import from src/ ───────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    width = 60
    print("\n" + "─" * width)
    print(f"  {text}")
    print("─" * width)


def _ok(text: str) -> None:
    print(f"  ✓  {text}")


def _warn(text: str) -> None:
    print(f"  ⚠  {text}")


def _fail(text: str) -> None:
    print(f"  ✗  {text}")


# ---------------------------------------------------------------------------
# Step 1 — verify Postgres is reachable
# ---------------------------------------------------------------------------

def step_check_postgres(pg_url: str) -> bool:
    _banner("Step 1 — Checking PostgreSQL connection")
    try:
        import psycopg
        # Use a plain psycopg3 URL (no +psycopg driver prefix)
        plain_url = pg_url.replace("postgresql+psycopg://", "postgresql://")
        conn = psycopg.connect(plain_url, connect_timeout=5)
        conn.close()
        _ok(f"Connected to {plain_url.split('@')[-1]}")
        return True
    except Exception as exc:
        _fail(f"Cannot connect to PostgreSQL: {exc}")
        print()
        print("  Make sure Docker is running:")
        print("      docker compose up -d")
        print()
        print("  Then check POSTGRES_URL in .env matches your setup.")
        return False


# ---------------------------------------------------------------------------
# Step 2 — create extension + table (idempotent)
# ---------------------------------------------------------------------------

def step_init_schema(pg_url: str) -> bool:
    _banner("Step 2 — Initialising pgvector schema")
    try:
        import psycopg
        plain_url = pg_url.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(plain_url) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id        BIGSERIAL PRIMARY KEY,
                    text      TEXT        NOT NULL,
                    metadata  JSONB       NOT NULL DEFAULT '{}',
                    embedding vector(384)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_embeddings_ivfflat
                    ON document_embeddings
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
            """)
            conn.commit()
        _ok("pgvector extension enabled")
        _ok("document_embeddings table ready")
        _ok("IVFFlat index ready")
        return True
    except Exception as exc:
        _fail(f"Schema init failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Step 3 — embed documents and insert into pgvector
# ---------------------------------------------------------------------------

def step_migrate_documents(pg_url: str) -> bool:
    _banner("Step 3 — Migrating documents to pgvector")

    from dotenv import load_dotenv
    load_dotenv()

    # Temporarily force pgvector mode for this run only
    import os
    os.environ["USE_PGVECTOR"] = "true"

    try:
        from src.agent.rag import (
            _discover_documents,
            _get_file_extractors,
            _get_pg_vector_store,
            _embed_model,
            configure_llama_index,
            DATA_DIR,
        )
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
        from llama_index.core.node_parser import SentenceSplitter

        configure_llama_index()

        discovered = _discover_documents()
        if not discovered:
            _warn(f"No documents found in {DATA_DIR}")
            _warn("Add PDF/DOCX/XLSX/CSV/TXT files to data/ and re-run.")
            return True  # not a failure — user may add files later

        print(f"\n  Found {len(discovered)} file(s):")
        for p in discovered:
            print(f"    • {p.name}")

        print(f"\n  Embedding {len(discovered)} file(s) — this may take a few minutes...")
        t0 = time.time()

        existing_files = [p for p in discovered if p.exists()]
        loader = SimpleDirectoryReader(
            input_files=[str(p) for p in existing_files],
            file_extractor=_get_file_extractors(),
        )
        documents = loader.load_data()
        print(f"  Loaded {len(documents)} document node(s)")

        text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        Settings.text_splitter = text_splitter

        vector_store    = _get_pg_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[text_splitter],
            show_progress=True,
        )

        elapsed = time.time() - t0
        _ok(f"Inserted vectors into pgvector ({elapsed:.1f}s)")
        return True

    except Exception as exc:
        _fail(f"Document migration failed: {exc}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Reset env var so it doesn't affect anything after this script
        os.environ.pop("USE_PGVECTOR", None)


# ---------------------------------------------------------------------------
# Step 4 — set up LangGraph checkpoint tables
# ---------------------------------------------------------------------------

async def _setup_langgraph_tables(pg_url: str) -> bool:
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        plain_url = pg_url.replace("postgresql+psycopg://", "postgresql://")
        async with AsyncPostgresSaver.from_conn_string(plain_url) as cp:
            await cp.setup()
        _ok("LangGraph checkpoint tables created (checkpoints, checkpoint_writes, checkpoint_blobs)")
        return True
    except Exception as exc:
        _fail(f"LangGraph table setup failed: {exc}")
        return False


def step_setup_langgraph(pg_url: str) -> bool:
    _banner("Step 4 — Setting up LangGraph checkpoint tables")
    return asyncio.run(_setup_langgraph_tables(pg_url))


# ---------------------------------------------------------------------------
# Step 5 — smoke-test: retrieve a document from pgvector
# ---------------------------------------------------------------------------

def step_smoke_test() -> bool:
    _banner("Step 5 — Smoke test: retrieve from pgvector")
    import os
    os.environ["USE_PGVECTOR"] = "true"
    try:
        # Clear the lru_cache so get_vector_index() uses pgvector
        from src.agent.rag import get_vector_index, retrieve_documents
        get_vector_index.cache_clear()

        docs = retrieve_documents("company project knowledge")
        if docs:
            _ok(f"Retrieved {len(docs)} chunk(s) from pgvector")
            print(f"\n  Sample (first 120 chars):")
            print(f"    {docs[0].page_content[:120].strip()}...")
        else:
            _warn("No chunks returned — index may be empty (add documents to data/ and re-run step 3)")

        # Reset cache + env
        get_vector_index.cache_clear()
        return True
    except Exception as exc:
        _fail(f"Smoke test failed: {exc}")
        return False
    finally:
        os.environ.pop("USE_PGVECTOR", None)
        try:
            from src.agent.rag import get_vector_index
            get_vector_index.cache_clear()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 62)
    print("   DataDialogue -- pgvector Migration Script")
    print("=" * 62)

    from dotenv import load_dotenv
    load_dotenv()
    import os
    pg_url = os.getenv(
        "POSTGRES_URL",
        "postgresql+psycopg://postgres:password@localhost:5432/datadialogue",
    )
    print(f"\n  POSTGRES_URL: {pg_url.split('@')[-1]}")

    steps = [
        ("Postgres connection",      lambda: step_check_postgres(pg_url)),
        ("Schema initialisation",    lambda: step_init_schema(pg_url)),
        ("Document migration",       lambda: step_migrate_documents(pg_url)),
        ("LangGraph table setup",    lambda: step_setup_langgraph(pg_url)),
        ("Smoke test",               lambda: step_smoke_test()),
    ]

    passed = 0
    for name, fn in steps:
        ok = fn()
        if ok:
            passed += 1
        else:
            print(f"\n  Migration stopped at: {name}")
            print("  Fix the error above and re-run.")
            sys.exit(1)

    # ── Success ──────────────────────────────────────────────────────────
    _banner(f"Migration complete — {passed}/{len(steps)} steps passed")
    print()
    print("  Now flip the feature flags in .env:")
    print()
    print("      USE_PGVECTOR=true")
    print("      USE_POSTGRES_MEMORY=true")
    print()
    print("  Then restart FastAPI:")
    print()
    print("      uvicorn app:app --reload --port 8000")
    print()
    print("  Verify backends at: http://localhost:8000/health")
    print()
    print('  Expected response:')
    print('      {"status":"ok","memory_backend":"postgres","vector_backend":"pgvector"}')
    print()


if __name__ == "__main__":
    main()
