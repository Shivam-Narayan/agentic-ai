# KT Agent — LangGraph + LlamaIndex + FastAPI

> An enterprise-grade **Knowledge Transfer Assistant** built on a modern agentic architecture.  
> The agent dynamically decides whether to answer from its own knowledge, search company documents, query a live database, or look up the web — all in a single ReAct loop.

---

## What it does

- Answers general questions directly via the LLM (no tools needed)
- Searches **company documents** (PDFs, DOCX) via a local vector store
- Queries a **company database** (SQLite today, Postgres-ready) for live structured data
- Searches the **live web** via Tavily for real-time facts (weather, news, etc.)
- Streams the result back through a FastAPI endpoint to a Streamlit chat UI

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| UI | Streamlit | Chat frontend |
| API | FastAPI | HTTP `/ask` endpoint |
| Orchestration | LangGraph | ReAct agent loop (Agent ↔ Tools) |
| LLM | Groq / Gemini / Cohere | Dynamic LLM factory |
| RAG | LlamaIndex + HuggingFace embeddings | Document ingestion & retrieval |
| DB Tools | Python sqlite3 (MCP-compatible) | Structured data queries |
| Web Search | Tavily | Real-time web search tool |

---

## Project Structure

```text
kt-agent-using-langgraph/
├── src/
│   └── agent/
│       ├── __init__.py          # Exports: KnowledgeTransferAgent, aask, ask
│       ├── config.py            # Env vars, paths, logging setup
│       ├── chains.py            # Dynamic LLM factory (Groq/Gemini/Cohere)
│       ├── rag.py               # LlamaIndex: index build + document retrieval
│       ├── tools.py             # LangChain @tool: search_company_documents, search_web
│       ├── mcp_client.py        # DB tools: list_tables, describe_table, query_database
│       ├── workflow.py          # LangGraph ReAct loop (core agent)
│       ├── schemas.py           # Pydantic models: QuestionRequest, QuestionResponse
│       └── ingest_drive.py      # (Optional) Google Drive document ingestion
├── data/
│   ├── KT_document_from_a_real_client_project.docx
│   ├── Knowledge_Transfer_Agent_Design_Plans.pdf
│   └── company.db               # SQLite database (orders, employees, etc.)
├── docs/
│   ├── ARCHITECTURE.md
│   └── SYSTEM_DESIGN.md
├── indexing_data/               # Persisted LlamaIndex vector store
├── tests/
├── app.py                       # FastAPI entry point
├── streamlit_app.py             # Streamlit chat UI
├── requirements.txt
├── .env                         # API keys (never commit this)
└── README.md
```

---

## Quickstart for New Developers

### 1. Prerequisites

- Python 3.11+
- At least **one LLM API key** (Groq recommended — free tier available)
- A **Tavily API key** (free tier available at [tavily.com](https://tavily.com))

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
# Pick at least ONE of the following LLM providers:
GROQ_API_KEY=your_groq_api_key_here
# GOOGLE_API_KEY=your_google_api_key_here
# COHERE_API_KEY=your_cohere_api_key_here

# Required for the web search tool:
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: force a specific provider when multiple keys are set
# LLM_PROVIDER=groq   # Options: groq, google, cohere

# Optional: Google Drive folder for automatic document ingestion
# GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here
```

> **Priority order:** If multiple keys are set, the system picks `groq` → `google` → `cohere` unless `LLM_PROVIDER` is set.

### 4. Build the vector index (run once)

This reads the documents in `data/` and creates the persisted vector store in `indexing_data/`:

```bash
python -m src.agent.rag
```

### 5. Start the backend

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

> ⚠️ **Windows note:** Use `--host 0.0.0.0` (not `127.0.0.1`) and access via your machine's LAN IP (e.g., `http://192.168.x.x:8000`) to avoid Windows socket inheritance issues with old zombie processes. Set `KT_API_URL=http://192.168.x.x:8000` in your `.env` if needed.

### 6. Start the frontend

In a second terminal:

```bash
python -m streamlit run streamlit_app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

---

## How the Agent Works

The agent follows the **ReAct (Reason + Act)** pattern:

```
User question
    │
    ▼
┌─────────────┐
│  Agent Node │◄────────────────────┐
│  (LLM)      │                     │
└──────┬──────┘                     │
       │                            │
  Tool needed?                      │
  ┌────┴────┐                       │
  │  YES    │  NO                   │
  ▼         ▼                       │
┌──────┐  [END]                     │
│Tools │                            │
│ Node │────── tool results ────────┘
└──────┘
```

**The 4 tool paths:**

| Question type | Tool called | Example |
|---|---|---|
| General knowledge | *(none — LLM answers directly)* | "What is Python?" |
| Company documents | `search_company_documents` | "What is the Beacon project?" |
| Company database | `query_company_database` | "Status of order #12345?" |
| Live web | `search_web` | "Current weather in Bangalore?" |

---

## Adding New Capabilities

### Add a new tool (e.g., Jira ticket lookup)

1. Open `src/agent/tools.py`
2. Define a new `@tool` function with a clear docstring (the LLM reads this to decide when to use it)
3. Add it to the `local_tools` list in `src/agent/workflow.py`

```python
# tools.py
@tool
def get_jira_ticket(ticket_id: str) -> str:
    """Look up the details of a Jira ticket by its ID (e.g. 'PROJ-123')."""
    # ... your implementation
    return f"Ticket {ticket_id}: ..."

# workflow.py  — add it here:
local_tools = [search_company_documents, search_web, get_jira_ticket]
```

### Add a new document source

Place the file in `data/` then rebuild the index:

```bash
python -m src.agent.rag
```

### Switch LLM provider

Change `LLM_PROVIDER` in `.env`:

```env
LLM_PROVIDER=google   # or groq, cohere
```

---

## API Reference

### `POST /ask`

```json
// Request
{ "question": "What is the status of order #12345?" }

// Response
{
  "answer": "Order #12345 is currently shipped and expected to arrive on 2026-08-25.",
  "datasource": "database",
  "tools_used": ["query_company_database"]
}
```

`datasource` values: `direct_llm` · `company_docs` · `database` · `web_search` · `multiple`

### `GET /health`

```json
{ "status": "ok" }
```

Interactive API docs: **http://localhost:8000/docs**

---

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture and LangGraph flow
- [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) — Component design and multi-LLM strategy
