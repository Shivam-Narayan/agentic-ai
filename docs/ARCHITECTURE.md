# Architecture Overview

## What this system is

The KT Agent is an **Enterprise Knowledge Transfer Assistant**. It is a conversational AI that answers questions about your company by searching internal documents, querying a structured database, or looking up the live web — all from multiple interfaces including a web UI and Telegram.

The key design principle: **there is no hard-coded routing**. The LLM itself reads the available tools and decides at runtime which one(s) to use. Adding a new data source means writing one Python function — nothing else changes.

---

## High-Level System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          ACCESS CHANNELS                                 │
│                                                                          │
│  ┌──────────────────────┐   ┌─────────────────┐   ┌──────────────────┐  │
│  │   Streamlit Chat UI  │   │  Telegram Bot   │   │  OpenClaw        │  │
│  │  (streamlit_app.py)  │   │ (telegram_bot.py│   │  Webhook         │  │
│  │  http://localhost    │   │  @shivam_llm_bot│   │  (any channel)   │  │
│  │  :8501               │   │                 │   │                  │  │
│  └──────────┬───────────┘   └────────┬────────┘   └────────┬─────────┘  │
└─────────────│────────────────────────│────────────────────│─────────────┘
              │ POST /ask              │ POST /ask           │ POST /openclaw
              │ session_id=<uuid>      │ session_id=         │ /webhook
              │                        │ telegram_<user_id>  │ session_id=
              │                        │                     │ <oc_session>
              └────────────────────────┴─────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND (app.py)                         │
│                          http://localhost:8000                           │
│                                                                          │
│  POST /ask                →  aask(question, session_id, checkpointer)   │
│  GET  /health                                                            │
│  POST /upload             →  rebuild_index()                            │
│  GET  /documents                                                         │
│  GET  /sessions/{id}/history                                             │
│  DELETE /sessions/{id}/history                                           │
│  GET  /openclaw/health    →  OpenClaw health check                      │
│  POST /openclaw/webhook   →  aask() via OpenClaw session                │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH AGENT LOOP (workflow.py)                    │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  AgentState = { messages: [SystemMessage, ...history,            │  │
│   │                             HumanMessage, ToolMessage, ...] }    │  │
│   │                                                                  │  │
│   │   START ──► Agent Node ──► (has tool calls?)                     │  │
│   │               ▲                │                                 │  │
│   │               │      YES ──►  Tool Node                          │  │
│   │               └───────────────┘                                  │  │
│   │                       NO ──► END                                 │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│   Deduplication guard in agent_node:                                     │
│   • blocks search_company_documents being called > once                  │
│   • prevents repeated identical tool+query pairs                        │
│   • re-invokes LLM without tools if response content is empty           │
│                                                                          │
│   Local tools (tools.py):                                                │
│   ┌──────────────────────┐  ┌──────────────────────────────┐           │
│   │ search_company_docs  │  │ summarise_document           │           │
│   │ (LlamaIndex RAG)     │  │ extract_structured_data      │           │
│   └──────────────────────┘  └──────────────────────────────┘           │
│   ┌──────────────────────┐  ┌──────────────────────────────┐           │
│   │ search_web           │  │ calculate                    │           │
│   │ (Tavily API)         │  │ generate_chart (Plotly)      │           │
│   └──────────────────────┘  └──────────────────────────────┘           │
│                                                                          │
│   MCP tools (mcp_client.py):                                             │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │ list_database_tables  describe_database_table                    │  │
│   │ query_company_database                                           │  │
│   └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
┌─────────────────┐  ┌──────────────┐  ┌───────────────┐
│  VECTOR STORE   │  │  SQLITE DB   │  │  TAVILY API   │
│  indexing_data/ │  │  data/*.db   │  │  (live web)   │
│  (LlamaIndex)   │  │  (sqlite3)   │  └───────────────┘
└─────────────────┘  └──────────────┘
        ▲
        │ indexed from
┌───────────────────┐          ┌────────────────────────┐
│   data/ folder    │          │  memory_store/         │
│  *.pdf *.docx     │          │  conversations.db      │
│  *.xlsx *.csv     │          │  (AsyncSqliteSaver)    │
│  *.txt            │          │  per session_id thread │
└───────────────────┘          └────────────────────────┘
```

---

## Channel Architecture

The agent supports three independent access channels. Each channel maps to its own session namespace so conversation memory never leaks between channels.

### Channel 1 — Streamlit Web UI

```
Browser → streamlit_app.py
        → POST /ask {question, session_id=<uuid>}
        → aask(question, session_id, checkpointer)
        → LangGraph agent
        → QuestionResponse (answer + datasource badge + citations + chart)
        → render in chat UI
```

- Session ID is a UUID generated once per browser tab
- Charts render inline as interactive Plotly figures
- Datasource badges (📄 🌐 🧮 📊 🗄️) shown under each answer
- Citation pills show exact filename, URL, or SQL

### Channel 2 — Telegram Bot (Direct)

```
Telegram user → @shivam_llm_bot
             → python-telegram-bot polling
             → telegram_bot.py handle_message()
             → POST /ask {question, session_id=telegram_<user_id>}
             → aask(question, session_id, checkpointer)
             → LangGraph agent
             → reply with answer + citations + tool emoji
```

- Each Telegram user ID gets its own session — memory is per-user
- All 9 tools work — RAG, web search, calculator, database, charts
- Citation sources appended to reply text
- Running: `python telegram_bot.py` (FastAPI must be running first)

### Channel 3 — OpenClaw Webhook (Multi-channel gateway)

```
WhatsApp / Discord / Slack
        → OpenClaw Gateway (port 18789)
        → POST /openclaw/webhook {channel, user_id, session_id, message}
        → openclaw_webhook() in app.py
        → aask(question, session_id=oc_session, checkpointer)
        → LangGraph agent
        → OpenClawWebhookResponse (response + tools_used + citations)
        → OpenClaw sends reply back to originating channel
```

- OpenClaw's session_id is used directly as the LangGraph thread_id
- Supports any channel OpenClaw connects to (Telegram, WhatsApp, Discord, Slack)
- Health check: `GET /openclaw/health`

---

## Conversation Memory

Memory is now **persisted to disk** via `AsyncSqliteSaver` — sessions survive server restarts.

```
Request arrives with session_id
        │
        ▼
AsyncSqliteSaver.aget_tuple(thread_id=session_id)
        │
        ▼
LangGraph loads full message history from memory_store/conversations.db
        │
        ▼
Agent runs with full history context
        │
        ▼
AsyncSqliteSaver automatically saves updated state after each run
```

Previously memory was in-memory (`_sessions` dict) and lost on server restart. The checkpointer replaces this entirely — no manual session management needed.

---

## The ReAct Agent Loop

The entire agent is a two-node LangGraph `StateGraph`. There are no other nodes — no classifier, no pre-router, no if/else logic.

```
User question (from any channel)
     │
     ▼
FastAPI endpoint (POST /ask or POST /openclaw/webhook)
     │
     ▼
aask(question, session_id, checkpointer)   ← workflow.py
     │
     ▼
build_graph(all_tools, checkpointer)
     │
     ▼
┌────────────────────────────────────────────────────┐
│               LangGraph StateGraph                 │
│                                                    │
│  START → Agent Node → tool_calls present?          │
│              ▲               │                     │
│              │         YES → Tool Node             │
│              └───────────────┘                     │
│                        NO → END                    │
└────────────────────────────────────────────────────┘
     │
     ▼
parse_result()   → answer, datasource, tools_used, citations, chart_data
     │
     ▼
Response → channel (Streamlit / Telegram / OpenClaw)
```

### Why two nodes?

The LLM is not just the "brain" — it is also the router. It receives all tool schemas (auto-generated from their docstrings) alongside the user's question, the system prompt, and the full conversation history. It decides:

- **No tools needed** → emits a final answer text → graph goes to END
- **Tool needed** → emits a `tool_calls` list → graph executes the tool, appends results to history, loops back

The loop continues until the LLM stops calling tools and produces a final answer. Most questions resolve in 1–2 loops. The `recursion_limit=8` cap prevents runaway loops.

---

## The System Prompt

Every agent invocation calls `_build_system_prompt()` which stamps the **live server date and time** into the prompt before injecting it as a `SystemMessage`. This ensures date/day-of-week questions are always accurate regardless of the LLM's training data cutoff.

The prompt also enforces:

- **Direct answers** — general knowledge questions (concepts, definitions) must be answered without calling any tools
- **No redundant tool calls** — `search_company_documents` must be called at most once per question
- **Citations** — always cite source filenames, SQL queries, or URLs
- **Accuracy** — use the `calculate` tool for all arithmetic

This is backed up at the code level by the **deduplication guard** — the prompt alone is not reliable enough.

---

## Deduplication Guard

The LLM was observed calling `search_company_documents` 4+ times per question with slightly different queries. Each call consumes a Groq API request, burning through the free-tier rate limit in seconds.

The guard runs inside `agent_node` after every LLM response:

```
LLM response has tool_calls?
        │
       YES
        │
        ▼
Count prior search_company_documents calls in message history
        │
        ├── count >= 1 → BLOCK: strip tool_calls, force direct answer
        │
        └── exact same tool+query already ran → BLOCK: strip tool_calls
                │
                ▼
        Response content is empty?
                │
               YES → re-invoke LLM without tools to get a real answer
```

---

## The 9 Tools

| Tool | File | Triggers when... |
|---|---|---|
| `search_company_documents` | `tools.py` | Question is about internal company knowledge |
| `summarise_document` | `tools.py` | User asks for an overview or summary of a specific file |
| `extract_structured_data` | `tools.py` | User wants specific fields/values pulled from documents |
| `search_web` | `tools.py` | Question needs real-time or external information |
| `calculate` | `tools.py` | Any arithmetic, percentages, or numeric computation |
| `generate_chart` | `tools.py` | User asks for a chart or results are better visualised |
| `list_database_tables` | `mcp_client.py` | LLM needs to discover what tables exist in the DB |
| `describe_database_table` | `mcp_client.py` | LLM needs column names before writing a query |
| `query_company_database` | `mcp_client.py` | Question requires structured data from the database |

---

## The 7 Answer Paths

Every API response includes a `datasource` field that all channels use for display:

```
datasource = "direct_llm"     → LLM answered from training data / live date prompt
datasource = "company_docs"   → search_company_documents / summarise_document / extract_structured_data
datasource = "database"       → query_company_database was called
datasource = "web_search"     → search_web was called
datasource = "calculation"    → calculate was called
datasource = "chart"          → generate_chart was called
datasource = "multiple"       → more than one tool category was used
```

---

## Data Stores

### Vector Store (document search)

```
data/                          ← put your files here
  ├── report.pdf
  ├── handbook.docx            ← parsed by DocxReader (llama-index-readers-file)
  ├── catalog.xlsx
  └── notes.txt
       │
       │  python -m src.agent.rag  OR  POST /upload
       ▼
indexing_data/                 ← auto-generated, do not edit
  ├── default__vector_store.json
  ├── docstore.json
  └── ...
```

### Conversation Memory (per-session history)

```
memory_store/
  └── conversations.db         ← SQLite, managed by AsyncSqliteSaver
       │
       │  keyed by thread_id = session_id
       │  persists across server restarts
       │  shared across all channels (Streamlit, Telegram, OpenClaw)
```

### Company Database (structured queries)

```
data/company.db                ← SQLite, read-only via SELECT
```

---

## File Responsibilities

| File | Layer | What it does |
|---|---|---|
| `streamlit_app.py` | UI | Chat UI — badges, citations, Plotly charts, file upload, session controls |
| `telegram_bot.py` | Channel | Telegram bot — polls for messages, calls `/ask`, replies with answer + citations |
| `app.py` | API | FastAPI — `/ask`, `/health`, `/upload`, `/documents`, `/sessions/*`, `/openclaw/*` |
| `workflow.py` | Agent | LangGraph graph, system prompt with live date, dedup guard, citations, parse_result |
| `chains.py` | LLM | Factory: Groq / Gemini / Cohere; provides TavilySearch tool |
| `tools.py` | Tools | 6 local tools: search, summarise, extract, web search, calculate, chart |
| `rag.py` | RAG | File discovery, DocxReader, LlamaIndex vector store build/load/retrieve |
| `mcp_client.py` | DB | 3 database tools behind MCP-compatible asynccontextmanager |
| `schemas.py` | Models | QuestionRequest/Response + OpenClawWebhookRequest/Response/HealthResponse |
| `config.py` | Config | DATA_DIR, INDEX_DIR paths; env key validation |
| `ingest_drive.py` | Ingestion | Optional: pulls files from Google Drive into data/ |

---

## Multi-LLM Strategy

```
.env keys present         →  LLM selected
──────────────────────────────────────────────────────────────
GROQ_API_KEY              →  Groq  (openai/gpt-oss-20b)   ← default
GOOGLE_API_KEY            →  Gemini (gemini-1.5-flash)
COHERE_API_KEY            →  Cohere (command-r-plus)
LLM_PROVIDER=google       →  forces Google regardless of other keys
```

Priority order: **Groq → Google → Cohere**. All providers use LangChain's `BaseChatModel` — `workflow.py` never references a specific provider.

---

## Technology Stack

| Technology | Role |
|---|---|
| **LangGraph** | Stateful ReAct agent loop (two-node StateGraph) |
| **LangGraph AsyncSqliteSaver** | Persistent conversation memory per session_id |
| **LangChain** | `@tool` decorator, `ToolNode`, `BaseChatModel` interface |
| **LlamaIndex** | Document ingestion, chunking, HuggingFace embeddings, vector store |
| **llama-index-readers-file** | `DocxReader` for proper Word document text extraction |
| **FastAPI** | Async HTTP API — `/ask`, `/upload`, `/openclaw/webhook`, session endpoints |
| **Streamlit** | Web chat UI with badges, citations, Plotly charts, document upload |
| **python-telegram-bot** | Telegram channel — polls Telegram and calls FastAPI `/ask` |
| **Groq** | Default LLM — `openai/gpt-oss-20b` |
| **Google Gemini** | Alternative LLM — `gemini-1.5-flash` |
| **Cohere** | Alternative LLM — `command-r-plus` |
| **Tavily** | Real-time web search via `langchain-tavily` |
| **HuggingFace** | `BAAI/bge-small-en-v1.5` local embedding model |
| **Plotly** | Interactive chart generation |
| **SQLite** | Company database + conversation memory store |
| **OpenClaw** | Optional multi-channel gateway (WhatsApp, Discord, Slack) |

---

## Extending the System

### Add a new tool

```python
# 1. Define in src/agent/tools.py
@tool
def get_jira_ticket(ticket_id: str) -> str:
    """Look up a Jira ticket by its ID (e.g. 'PROJ-123').
    Use this when the user asks about a specific issue or bug report."""
    ...

# 2. Register in src/agent/workflow.py
LOCAL_TOOLS = [search_company_documents, search_web, ..., get_jira_ticket]
```

No routing changes. The LLM starts using it automatically across all channels.

### Add a new channel

1. Create a new file (e.g. `discord_bot.py`)
2. Call `POST /ask` with a unique `session_id` prefix (e.g. `discord_<user_id>`)
3. Memory is automatically maintained per session via the checkpointer

### Add a new document type

Add the extension to `SUPPORTED_EXTENSIONS` in `src/agent/rag.py` and register its reader:

```python
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt", ".md"}
```

### Replace SQLite with Postgres

Replace `mcp_server_context()` in `mcp_client.py` with a real MCP server connection. `workflow.py` does not need to change.
