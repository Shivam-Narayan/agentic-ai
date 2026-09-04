# System Design

## Overview

This document covers the internal design of each component in the KT Agent. It is intended for developers who want to understand how the system works, modify it, or extend it.

For a higher-level view of the overall architecture and data flow, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Component Map

```
.env
 └── config.py  ─────────────────────────────────────────────────────────┐
                                                                         │
chains.py  (LLM factory + web search)                                   │
 ├── get_llm()             → ChatGroq / ChatGoogleGenerativeAI / ChatCohere
 ├── _make_serper_tool()   → GoogleSerperRun (if SERPER_API_KEY set)     │
 ├── _make_ddg_tool()      → DuckDuckGoSearchRun (always available)      │
 ├── _FallbackSearchTool   → BaseTool wrapper: tries primary → fallbacks │
 └── get_web_search_tool() → Tavily (primary) → Serper → DuckDuckGo     │
                                                                         │
rag.py  (document layer)                                                │
 ├── _discover_documents()   → scans data/ for supported files          │
 ├── _get_file_extractors()  → registers DocxReader for .docx/.doc      │
 ├── build_index()           → ingests, embeds, persists                │
 ├── get_vector_index()      → loads from indexing_data/ or pgvector    │
 ├── rebuild_index()         → rebuilds from scratch + clears lru_cache │
 └── retrieve_documents()   → returns List[Document] (top_k=8)         │
                                                                         │
tools.py  (LangChain tools — 6 local tools)                             │
 ├── search_company_documents  → rag.retrieve_documents()               │
 ├── summarise_document        → reads full file via LlamaIndex         │
 ├── extract_structured_data   → RAG + field template                   │
 ├── search_web                → TavilySearch, handles dict response    │
 ├── calculate                 → safe AST evaluator                     │
 └── generate_chart            → Plotly JSON figure                     │
                                                                         │
mcp_client.py  (database tools — 3 MCP tools)                          │
 ├── list_database_tables    → discover tables + row counts             │
 ├── describe_database_table → column schemas + types                   │
 └── query_company_database  → SELECT (or write with ALLOW_DB_WRITES)   │
                                                                         │
workflow.py  (LangGraph agent)  ◄── uses all of the above              │
 ├── _current_system_prompt  (ContextVar — thread-safe prompt injection)│
 ├── AgentState                                                         │
 ├── _get_previous_tool_calls() (dedup helper)                          │
 ├── _first_search_had_results() (smart dedup — only blocks if found)   │
 ├── _compile_graph(tools, checkpointer)  ← bound_llm created once     │
 ├── _get_compiled_graph()    ← dict cache, bounded size                │
 ├── _prepare_run()           ← DRY setup for aask + streaming          │
 ├── agent_node()             ← standalone, testable, uses ContextVar   │
 ├── KnowledgeTransferAgent   ← SSE streaming interface                 │
 └── aask(question, session_id, checkpointer)  ← blocking interface    │
                                                                         │
schemas.py  (Pydantic models)                                           │
 ├── QuestionRequest / QuestionResponse / Citation   (core API)        │
 ├── OpenClawWebhookRequest / OpenClawWebhookResponse  (OpenClaw)      │
 └── OpenClawHealthResponse                           (OpenClaw)        │
                                                                         │
app.py  (FastAPI)                                                        │
 ├── GET  /stream           →  KnowledgeTransferAgent.run() SSE stream  │
 ├── POST /ask              →  aask(question, session_id, checkpointer) │
 ├── GET  /health                                                        │
 ├── POST /upload            →  rebuild_index()                         │
 ├── GET  /documents                                                     │
 ├── GET  /sessions/{id}/history                                         │
 ├── DELETE /sessions/{id}/history                                       │
 ├── GET  /openclaw/health   →  OpenClawHealthResponse                  │
 └── POST /openclaw/webhook  →  aask() via OpenClaw session_id          │
                                                                         │
telegram_bot.py  (Telegram channel)                                     │
 ├── handle_message()        → POST /ask {session_id=telegram_<user_id>}│
 ├── start()                 → /start command handler                   │
 └── help_command()          → /help command handler                    │
                                                                         │
streamlit_app.py  (Web UI)                                              │
 ├── st.chat_input (always called — prevents disappear bug)             │
 ├── pending_question (sample question click flow)                      │
 ├── _render_assistant_message()  (badges, charts, citations)           │
 └── sidebar: upload, indexed docs, session controls, samples          │
```

---

## `config.py` — Environment and Paths

**Purpose:** Single source of truth for all file paths and environment validation.

```python
ROOT_DIR  = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = ROOT_DIR / "data"          # where user puts their files
INDEX_DIR = ROOT_DIR / "indexing_data" # LlamaIndex persists the vector store here
```

**Environment validation (`require_runtime_keys()`):**

Called at startup. Raises a hard error only if no LLM key is present — web search keys are optional:

- **Required (at least one):** `GROQ_API_KEY`, `GOOGLE_API_KEY`, `COHERE_API_KEY`
- **Optional (web search):** `TAVILY_API_KEY`, `SERPER_API_KEY`
  - If neither is set, logs a warning and falls back to DuckDuckGo automatically
  - No crash — the app starts and works without any web search key

Optional environment vars:
- `LLM_PROVIDER` — force a specific provider (`groq`, `google`, `cohere`)
- `KT_API_URL` — Streamlit uses this to reach FastAPI (default: `http://localhost:8000`)
- `OPENCLAW_WEBHOOK_LOGGING` — enable verbose logging for webhook requests

---

## `chains.py` — LLM Factory + Web Search Fallback Chain

**Purpose:** Selects and instantiates the LLM and web search tool based on available API keys.

**LLM selection logic:**

```python
@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "").lower()

    if provider == "groq" or (not provider and GROQ_API_KEY):
        return ChatGroq(model="openai/gpt-oss-120b", temperature=0, streaming=True)

    if provider == "google" or (not provider and GOOGLE_API_KEY):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, streaming=True)

    if provider == "cohere" or (not provider and COHERE_API_KEY):
        return ChatCohere(model="command-r-plus", temperature=0, streaming=True)
```

Priority: **Groq → Google → Cohere**. `@lru_cache` means the client is created once per process. All three providers have `streaming=True` for token-level streaming.

**Web search fallback chain:**

`get_web_search_tool()` returns a `_FallbackSearchTool` that wraps multiple providers and cascades at *query time* — not just at init time. This is critical because Tavily initialises successfully even when its quota is exhausted; the error only surfaces when a query is made.

```python
class _FallbackSearchTool(BaseTool):
    def _run(self, query: str) -> str:
        for tool in [primary] + fallbacks:
            try:
                result = tool.invoke(query)
                # Tavily returns {"error": ...} dict on quota errors instead of raising
                if isinstance(result, dict) and "error" in result:
                    raise RuntimeError(result["error"])
                return result
            except Exception as exc:
                logger.warning("Search tool '%s' failed: %s — trying next", tool.name, exc)
        raise RuntimeError("All web search providers failed")
```

Provider selection priority:
1. **Tavily** — `TAVILY_API_KEY` set — AI-optimised results, best quality
2. **Serper** — `SERPER_API_KEY` set — real Google results, 2,500 free credits
3. **DuckDuckGo** — no key needed — always available, appended unconditionally

If only Tavily is configured → `_FallbackSearchTool(primary=Tavily, fallbacks=[DuckDuckGo])`
If Tavily + Serper configured → `_FallbackSearchTool(primary=Tavily, fallbacks=[Serper, DuckDuckGo])`
If no keys → `DuckDuckGoSearchRun` returned directly (no wrapper overhead)

---

## `rag.py` — Document Ingestion and Retrieval

**Purpose:** Manages the full lifecycle of document knowledge — discovery, parsing, indexing, retrieval.

### Supported file formats

```python
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt"}
```

### DOCX parsing fix

LlamaIndex's default `SimpleDirectoryReader` falls back to a raw binary reader for `.docx` if `llama-index-readers-file` is not installed. The fix explicitly registers `DocxReader` for Word files:

```python
def _get_file_extractors() -> dict:
    from llama_index.readers.file import DocxReader
    return {".docx": DocxReader(), ".doc": DocxReader()}
```

### Index cache invalidation

`rebuild_index()` calls `get_vector_index.cache_clear()` after every rebuild so the next query loads the fresh index without requiring a server restart. Files uploaded via `POST /upload` also trigger a full rebuild automatically.

---

## `tools.py` — The 6 Local Tools

### 1. `search_company_documents`
Calls `retrieve_documents(query)` → top 4 chunks from vector store. Each chunk prefixed with `[Source: filename]` for citation extraction. Text capped at 700 chars per chunk.

### 2. `summarise_document`
Loads full file text (capped at 6000 chars) with `[Full text of <filename>]` prefix. LLM synthesises summary in the next reasoning step.

### 3. `extract_structured_data`
Retrieves relevant chunks then returns context + JSON field template. LLM fills in the values.

### 4. `search_web`
Calls the `_FallbackSearchTool` which tries Tavily → Serper → DuckDuckGo in order. Handles Tavily's dict-format quota error (`{"error": ValueError(...)}`) by treating it as a failure and cascading. Results formatted as `[Source: url]\ncontent`.

### 5. `calculate`
Uses Python's `ast` module — safe arithmetic without `eval()`. Only numeric constants and arithmetic operators allowed. Returns `"expression = result"` format.

### 6. `generate_chart`
Builds a Plotly `go.Figure` and returns `CHART_JSON::{figure_json}`. The `_extract_chart()` function in `workflow.py` detects this prefix and extracts the JSON for Streamlit to render inline.

---

## `mcp_client.py` — Database Tools

Three LangChain tools wrapped in an MCP-compatible `asynccontextmanager`:

```python
list_database_tables()               # discover available tables with row counts
describe_database_table(table_name)  # get column names, types, constraints
query_company_database(sql_query)    # run queries (SELECT or write if ALLOW_DB_WRITES=true)
```

**Safety & Validation:**
- `_sanitise_identifier()` — strict regex `^\w+$` on table/column names before any PRAGMA or schema query — blocks SQL injection.
- `_is_blocked_statement()` — disallows DDL (`DROP`, `TRUNCATE`, `ALTER`, `CREATE`) regardless of `ALLOW_DB_WRITES`.
- Read-only by default — `INSERT`, `UPDATE`, `DELETE` require `ALLOW_DB_WRITES=true` in `.env`.
- Both SQLite and PostgreSQL backends supported — branches on `USE_PGVECTOR` flag.

**MCP compatibility:** The entire database layer can be replaced with a real MCP server by swapping only `mcp_server_context()`. `workflow.py` does not change.

---

## `workflow.py` — The LangGraph Agent

**Purpose:** Assembles all tools, runs the ReAct loop, and parses the final result.

### Key improvements over original design

- **Thread-safe prompts** — `_current_system_prompt` is a `ContextVar`. Each async Task gets its own value so concurrent requests never corrupt each other's system prompt.
- **Graph cached** — `_get_compiled_graph()` uses a bounded dict cache keyed on `(sorted tool names, checkpointer id)`. Graph is compiled once per tool-set, not on every request.
- **LLM bound once** — `bound_llm = get_llm().bind_tools(...)` runs in `_compile_graph()` at compile time, not on every loop iteration.
- **Timeout enforced** — `_astream_complete()` wraps the LLM stream in `asyncio.wait_for(timeout=30)`.
- **Smart dedup** — `_first_search_had_results()` only blocks a second search if the first one returned content. If retrieval returned empty, the LLM can retry with a different query.

### Public interface

```python
# Blocking — called by POST /ask and Telegram
async def aask(question, session_id, checkpointer, history) -> dict

# Streaming — called by GET /stream
class KnowledgeTransferAgent:
    async def run(question, session_id, history) -> AsyncIterator[dict]
    # yields: status | token | tool | done | error events
```

### Streaming: `KnowledgeTransferAgent`

`KnowledgeTransferAgent` wraps the same LangGraph graph as `aask()` but exposes it as an `AsyncIterator[dict]` suitable for SSE:

```python
class KnowledgeTransferAgent:
    async def run(self, question: str, session_id: str = "default") -> AsyncIterator[dict]:
        yield {"type": "status", "stage": "thinking"}
        async for mode, chunk in graph.astream(..., stream_mode=["messages", "values"]):
            if mode == "messages":
                # filter to agent node, extract text chunks
                yield {"type": "token", "text": chunk_text}
            elif mode == "values":
                # detect new tool calls
                yield {"type": "tool", "name": tool_name}
        yield {"type": "done", "payload": parse_result(final_values)}
```

FastAPI's `GET /stream` wraps this in a `StreamingResponse` with `media_type="text/event-stream"`. The Streamlit UI consumes it via `httpx.stream()`.

`aask()` still exists and is used by Telegram and `POST /ask` — it calls `graph.ainvoke()` (blocking, single response).

```python
for msg in reversed(messages):
    if msg.type == "ai" and msg.content and str(msg.content).strip():
        answer = str(msg.content).strip()
        break
```

Walks backwards through messages to find the last non-empty AI response — handles the case where the dedup guard produced an empty AI message.

---

## `schemas.py` — Pydantic Models

### Core API schemas

```python
class QuestionRequest(BaseModel):
    question:   str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default")

class Citation(BaseModel):
    source: str   # filename, table name, or URL
    detail: str   # SQL query, expression, or empty

class QuestionResponse(BaseModel):
    answer:     str
    datasource: str | None = None
    tools_used: list[str]
    citations:  list[Citation]
    chart_data: dict | None   # Plotly figure JSON or null
```

### OpenClaw integration schemas (added)

```python
class OpenClawWebhookRequest(BaseModel):
    channel:    str        # "telegram", "whatsapp", "discord", etc.
    user_id:    str        # unique user identifier from the channel
    session_id: str        # OpenClaw session key — used as LangGraph thread_id
    message:    str        # the user's question (max 4000 chars)
    timestamp:  str        # ISO 8601
    metadata:   dict | None  # optional channel-specific data

class OpenClawWebhookResponse(BaseModel):
    response:   str        # agent's answer text
    success:    bool       # whether processing succeeded
    tools_used: list[str]  # tool names called
    citations:  list[Citation]
    datasource: str | None
    error:      str | None # set when success=False

class OpenClawHealthResponse(BaseModel):
    status:      str   # "ok"
    agent_ready: bool  # True when checkpointer is initialised
    version:     str   # "1.0.0"
```

---

## `app.py` — FastAPI Backend

### Persistent memory via lifespan

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSqliteSaver.from_conn_string(
        str(_MEMORY_DIR / "conversations.db")
    ) as cp:
        _checkpointer = cp   # shared across all requests
        yield
```

The `AsyncSqliteSaver` is opened once at startup and shared by all endpoints. It persists full message history per `session_id` to `memory_store/conversations.db` — sessions survive server restarts.

### Endpoints

```
GET  /stream
  ← query params: question, session_id
  → text/event-stream (SSE)
  → events: status | token | tool | done | error

POST /ask
  ← QuestionRequest(question, session_id)
  → QuestionResponse(answer, datasource, tools_used, citations, chart_data)

GET  /health
  → {"status": "ok"}

POST /upload
  ← multipart files
  → saves to data/, calls rebuild_index(), returns saved/rejected/indexed

GET  /documents
  → list of files in data/ with name, size_kb, type

GET  /sessions/{session_id}/history
  → {session_id, turn_count, messages}

DELETE /sessions/{session_id}/history
  → writes empty checkpoint to reset the thread

GET  /sessions
  → lists all session IDs from SQLite checkpoints table

GET  /openclaw/health
  → OpenClawHealthResponse(status, agent_ready, version)

POST /openclaw/webhook
  ← OpenClawWebhookRequest(channel, user_id, session_id, message, timestamp)
  → OpenClawWebhookResponse(response, success, tools_used, citations, datasource)
```

### OpenClaw webhook handler

```python
@app.post("/openclaw/webhook")
async def openclaw_webhook(request: OpenClawWebhookRequest):
    result = await aask(
        question=request.message,
        session_id=request.session_id,   # OpenClaw manages the session key
        checkpointer=_checkpointer,
    )
    return OpenClawWebhookResponse(
        response=result["generation"],
        success=True,
        tools_used=result["tools_used"],
        citations=[Citation(source=c["source"], detail=c.get("detail", ""))
                   for c in result["citations"]],
        datasource=result["datasource"],
    )
```

OpenClaw's `session_id` is passed directly as the LangGraph `thread_id` — conversation memory works automatically per OpenClaw session.

---

## `telegram_bot.py` — Telegram Channel

**Purpose:** A `python-telegram-bot` polling bot that bridges Telegram messages to the DataDialogue FastAPI backend. Runs as a separate process alongside FastAPI.

### Design

```
Telegram API (polling)
      ↓
Application.run_polling()
      ↓
handle_message(update, context)
      ↓
httpx.AsyncClient.post(
    "http://localhost:8000/ask",
    json={
        "question": update.message.text,
        "session_id": f"telegram_{user.id}"
    }
)
      ↓
QuestionResponse
      ↓
update.message.reply_text(answer + citations + tool_emoji)
```

### Session namespacing

Each Telegram user ID gets its own session: `telegram_<user_id>`. This ensures:
- Memory is per-user, not per-bot
- Sessions don't collide with web UI sessions (which use UUIDs)
- Sessions persist across bot restarts via the checkpointer

### Reply formatting

- Citations are appended as `📎 *Sources:* filename1, filename2`
- Tool emojis (📄 🌐 🧮 📊 🗄️) are appended to indicate data source
- Markdown parsing is enabled for bold text

### Commands

| Command | Handler | Description |
|---------|---------|-------------|
| `/start` | `start()` | Welcome message with example questions |
| `/help` | `help_command()` | Full capability list |
| Any text | `handle_message()` | Forward to DataDialogue agent |

### Running

```bash
# FastAPI must be running first
python telegram_bot.py
```

The bot uses long-polling — no webhook URL or public server required. Suitable for local development and self-hosted setups.

---

## `streamlit_app.py` — Web Chat Frontend

### Key design decisions

**`st.chat_input` always rendered:** Previously, when a sample question was clicked, `pending_question` short-circuited the `or` expression and `st.chat_input(...)` was never called. Streamlit removed the widget from the DOM on subsequent renders. Fixed:

```python
# Always call chat_input first — Streamlit hides it if skipped even once
typed_input = st.chat_input("Ask a question…")
prompt = st.session_state.pop("pending_question", None) or typed_input
```

**Session ID:** Each browser tab gets a UUID. Sent with every `/ask` so conversations are isolated per tab. Memory persists across page refreshes via the checkpointer.

**Timeout:** 180 seconds for API calls — LLM + tool calls can be slow on first run with cold index.

### Datasource display config

```python
ROUTE_CONFIG = {
    "direct_llm":   {"icon": "💬", "color": "#6c757d"},
    "company_docs": {"icon": "📄", "color": "#0d6efd"},
    "database":     {"icon": "🗄️", "color": "#198754"},
    "web_search":   {"icon": "🌐", "color": "#fd7e14"},
    "calculation":  {"icon": "🧮", "color": "#6f42c1"},
    "chart":        {"icon": "📊", "color": "#20c997"},
    "multiple":     {"icon": "🔀", "color": "#dc3545"},
}
```

---

## Data Flow: End to End

Complete trace of "What are Shivam's technical skills?" sent via **Telegram**:

```
User sends Telegram message: "What are Shivam's technical skills?"
│
├─ telegram_bot.py: handle_message()
│   session_id = "telegram_8341015221"
│   httpx.post("http://localhost:8000/ask", json={question, session_id})
│
├─ FastAPI POST /ask:
│   aask(question, session_id="telegram_8341015221", checkpointer=cp)
│
├─ AsyncSqliteSaver: loads prior message history for thread_id
│   (empty on first message, populated on follow-ups)
│
├─ workflow.py aask():
│   all_tools = 6 local + 3 MCP = 9 total
│   graph = build_graph(all_tools, checkpointer=cp)
│   initial_state = {messages: [HumanMessage("What are Shivam's skills?")]}
│
├─ Agent Node:
│   LLM reads 9 tool schemas + system prompt with live date
│   Decides: call search_company_documents("Shivam technical skills")
│   Dedup guard: first call, allowed through
│
├─ Tool Node:
│   search_company_documents runs
│   LlamaIndex cosine search → top 4 chunks from resume DOCX
│   "[Source: Shivam_Narayan_Resume_Revised.docx]\nTECHNICAL SKILLS..."
│
├─ Agent Node (second pass):
│   Dedup guard: search_company_documents count = 1, blocks further calls
│   LLM generates final answer from resume chunks
│   → END
│
├─ AsyncSqliteSaver: saves updated state (human + tool + ai messages)
│
├─ parse_result():
│   answer = last non-empty AI message
│   tools_used = ["search_company_documents"]
│   datasource = "company_docs"
│   citations = [{"source": "Shivam_Narayan_Resume_Revised.docx"}]
│
├─ FastAPI: returns QuestionResponse
│
└─ telegram_bot.py: reply_text(answer + "\n📎 Sources: resume.docx\n📄")
```

Same flow applies for Streamlit (renders as badge + citation pill + markdown) and OpenClaw webhook (returns JSON to OpenClaw which sends to the originating channel).

---

## Known Limitations and Recommended Solutions

| Limitation | Current state | Recommended solution |
|---|---|---|
| Vector store is file-based | JSON files in `indexing_data/` | Replace with Postgres + pgvector |
| Company database is SQLite | `data/company.db` | Connect a real MCP Postgres server |
| Telegram bot requires FastAPI running | Direct HTTP call to localhost | Add retry/backoff logic; deploy both on same server |
| No authentication on `/ask` or `/openclaw/webhook` | Endpoints are open | Add API key header middleware or OAuth |
| Charts not sent as images via Telegram | Chart JSON returned but not rendered | Use `plotly.io.to_image()` to export PNG and send via `send_photo` |
| OpenClaw requires exact model availability | Groq model names change; caused 401 errors during setup | Use `GROQ_API_KEY` directly in DataDialogue rather than relying on OpenClaw's LLM |
| Groq free tier rate limit | 30 RPM | Dedup guard + `parallel_tool_calls=False` keeps usage low; upgrade to paid tier for heavy use |
| Tavily free tier quota | 1,000 searches/month | Add `SERPER_API_KEY` — system cascades automatically; DuckDuckGo always available as final fallback |

---

## Evaluation

The project ships with an LLM-as-judge evaluation script at `tests/evaluate.py`.

```bash
python tests/evaluate.py                          # all questions
python tests/evaluate.py --datasource company_docs  # filter by type
python tests/evaluate.py --verbose                  # per-question detail
python tests/evaluate.py --output tests/results.json
```

Test questions live in `tests/eval_questions.json`. Fill in `ground_truth` values for your actual documents before running.

Metrics measured — all scored 0.0–1.0:

| Metric | What low score means | Fix |
|---|---|---|
| `faithfulness` | LLM hallucinating | Tighten system prompt in `prompt.py` |
| `answer_relevancy` | LLM going off-topic | Reduce recursion limit in `workflow.py` |
| `context_precision` | Wrong chunks retrieved | Reduce `chunk_size` in `rag.py` |
| `context_recall` | Missing relevant chunks | Increase `similarity_top_k` in `rag.py` |

FastAPI does not need to be running — the script calls `aask()` directly.

The `tests/test_agents.py` file contains smoke tests for embedding model initialisation and data directory validation, runnable with `pytest tests/`.
