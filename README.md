# DataDialogue — Knowledge Transfer Assistant

A conversational AI agent that answers questions about your company files, database, and the live web — with real-time streaming responses.

Built with **LangGraph · LlamaIndex · FastAPI · Streamlit**

---

## What it does

Ask questions in plain English. The agent picks the right tool automatically:

| Question | What happens |
|---|---|
| "Summarise the annual report" | Reads your uploaded PDF/DOCX |
| "What is the status of order #12345?" | Queries the company database |
| "What is 15% of 85000?" | Runs a safe calculator |
| "Show monthly sales as a bar chart" | Generates an interactive Plotly chart |
| "What is the USD to INR rate today?" | Searches the live web |

Answers stream word-by-word in real time — no waiting for the full response.

---

## Quick Start

### 1. Install dependencies

```bash
cd DataDialogue
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Set up `.env`

Copy the template and provide your API keys:

```bash
cp .env.example .env
```

```env
# LLM — pick at least one
GOOGLE_API_KEY=your_google_api_key_here   # Google Gemini (default)
GROQ_API_KEY=your_groq_api_key_here       # Groq (optional)
COHERE_API_KEY=your_cohere_key_here       # Cohere (optional)

# Web search — all optional, cascades automatically: Tavily → Serper → DuckDuckGo
TAVILY_API_KEY=your_key_here      # tavily.com — 1,000 free/month
SERPER_API_KEY=your_key_here      # serper.dev — 2,500 free, no card

KT_API_URL=http://localhost:8000

# PostgreSQL — only needed if using pgvector or Postgres memory (optional)
POSTGRES_URL=postgresql+psycopg://postgres:password@localhost:5432/datadialogue
USE_PGVECTOR=false
USE_POSTGRES_MEMORY=false
```

> No web search key? DuckDuckGo is used automatically — no signup needed.
> Leave `USE_PGVECTOR` and `USE_POSTGRES_MEMORY` as `false` to use the default SQLite + JSON setup.

### 3. Index your documents

Drop files into `data/` (PDF, DOCX, XLSX, CSV, TXT), then build the index:

```bash
python -m src.agent.rag
```

> **Incremental Uploads**: You can also upload new documents directly via the Streamlit UI or `POST /upload`. The engine uses incremental indexing (`add_documents_to_index`), adding only the new documents without rebuilding the entire store.

### 4. Run

Two terminals:

```bash
# Terminal 1: FastAPI Backend
uvicorn app:app --reload --port 8000

# Terminal 2: Streamlit Frontend
python -m streamlit run streamlit_app.py
```

Open **http://localhost:8501**

---

## Running Tests

Run the automated test suite with pytest:

```bash
pytest tests/
```

The test suite covers:
- **`tests/test_tools.py`**: SQL injection guards, DDL blocking, AST math evaluation, and schema caching TTL.
- **`tests/test_workflow.py`**: State deduplication, tool tracking, and response serialization compatibility.
- **`tests/test_api.py`**: FastAPI routes (`/health`, `/ask` validation, `/sessions`, `/upload`).

---

## PostgreSQL + pgvector (optional upgrade)

Replaces the default SQLite memory store and JSON vector store with a single PostgreSQL instance. Recommended when you have 50+ documents or multiple concurrent users.

### Why upgrade?

| | Default (SQLite + JSON) | PostgreSQL + pgvector |
|---|---|---|
| Documents | Works up to ~20 files | Scales to 100s of files |
| Concurrent users | Single user | Multiple users |
| Retrieval quality | Top 2 chunks | Top 8 chunks |
| Setup | Zero | Docker required |

### Setup

**1. Start Docker:**
```bash
docker compose up -d
```

**2. Run the migration script (once):**
```bash
python migrate_to_pgvector.py
```

This embeds all your documents and inserts them into pgvector, and creates the LangGraph checkpoint tables. Takes a few minutes depending on document count.

**3. Flip the flags in `.env`:**
```env
USE_PGVECTOR=true
USE_POSTGRES_MEMORY=true
```

**4. Restart FastAPI:**
```bash
uvicorn app:app --reload --port 8000
```

**5. Verify:**
```bash
curl http://localhost:8000/health
# {"status":"ok","memory_backend":"postgres","vector_backend":"pgvector"}
```

### Rollback

Set both flags back to `false` and restart. SQLite + JSON still work — no data is lost.

---

## Telegram Bot

```bash
# Add to .env
TELEGRAM_BOT_TOKEN=your_token_here

# Run (FastAPI must be running first)
python telegram_bot.py
```

Each user gets their own conversation memory automatically.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/stream` | Real-time SSE streaming response (`question`, `session_id`) |
| `POST` | `/ask` | Blocking JSON response (used by Telegram and external integrations) |
| `GET` | `/health` | Liveness check + active memory and vector backend info |
| `POST` | `/upload` | Non-blocking incremental document upload & indexing |
| `GET` | `/documents` | List all indexed files in `data/` |
| `GET` | `/sessions` | List active sessions from checkpointer |
| `GET` | `/sessions/{id}/history` | Retrieve conversation turn history for a session |
| `DELETE` | `/sessions/{id}/history` | Asynchronously clear conversation memory for a session |

Interactive docs: **http://localhost:8000/docs**

---

## Troubleshooting

**Web search not working** — Tavily quota hit. Add `SERPER_API_KEY` to `.env` and restart.

**DOCX returns empty results** — Run `pip install llama-index-readers-file`

**`ModuleNotFoundError: ddgs`** — Run `pip install ddgs==9.16.0` inside your `.venv`

**Backend offline error in Streamlit** — Start FastAPI first (Terminal 1).

**Stale index after adding files** — Re-run `python -m src.agent.rag` or use the Upload button in the sidebar.

**pgvector migration fails** — Make sure Docker is running (`docker compose up -d`) and `POSTGRES_URL` in `.env` matches your container.

**FastAPI falls back to SQLite even with `USE_POSTGRES_MEMORY=true`** — Postgres is unreachable. Check `docker ps` and confirm the container is healthy.

---

## Docs

- [Architecture overview](docs/ARCHITECTURE.md)
- [Component design](docs/SYSTEM_DESIGN.md)

