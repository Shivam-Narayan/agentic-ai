# System Design

## Overview

This document covers the internal design of each component in the KT Agent. It is intended for developers who want to understand how the system works, modify it, or extend it.

For a higher-level view of the overall architecture and data flow, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Component Map

```
.env
 └── config.py  ──────────────────────────────────────────────────┐
                                                                   │
chains.py  (LLM factory)                                          │
 ├── get_llm()          → ChatGroq / ChatGemini / ChatCohere      │
 └── get_web_search_tool() → TavilySearchResults                  │
                                                                   │
rag.py  (document layer)                                          │
 ├── _discover_documents()  → scans data/ for supported files     │
 ├── build_index()           → ingests, embeds, persists           │
 ├── get_vector_index()      → loads from indexing_data/           │
 └── retrieve_documents()   → returns List[Document]              │
                                                                   │
tools.py  (LangChain tools — 6 local tools)                       │
 ├── search_company_documents  → calls rag.retrieve_documents()   │
 ├── summarise_document        → reads full file via LlamaIndex   │
 ├── extract_structured_data   → RAG + field template             │
 ├── search_web                → calls chains.get_web_search_tool()│
 ├── calculate                 → safe AST evaluator               │
 └── generate_chart            → Plotly JSON figure               │
                                                                   │
mcp_client.py  (database tools — 3 MCP tools)                    │
 ├── list_database_tables                                         │
 ├── describe_database_table                                       │
 └── query_company_database    (SELECT only)                      │
                                                                   │
workflow.py  (LangGraph agent)  ◄── uses all of the above        │
 ├── SYSTEM_PROMPT                                                │
 ├── AgentState                                                   │
 ├── build_graph(tools)                                           │
 ├── agent_node()          (injects SystemMessage each turn)      │
 ├── should_continue()                                            │
 ├── _extract_citations()                                         │
 ├── _extract_chart()                                             │
 ├── parse_result()                                               │
 ├── LOCAL_TOOLS = [6 tools]                                      │
 └── aask(question, history)  ← FastAPI calls this               │
                                                                   │
app.py  (FastAPI)                                                  │
 ├── POST /ask          →  aask(question, history)                │
 ├── GET  /health                                                 │
 ├── POST /upload        →  rebuild_index()                       │
 ├── GET  /documents                                              │
 ├── GET  /sessions/{id}/history                                  │
 └── DELETE /sessions/{id}/history                               │
                                                                   │
streamlit_app.py  (UI)                                            │
 ├── st.chat_input  →  httpx.post(/ask, session_id)              │
 ├── _render_assistant_message()  (badges, charts, citations)     │
 └── sidebar: upload, indexed docs, session controls, samples    │
```

---

## `config.py` — Environment and Paths

**Purpose:** Single source of truth for all file paths and environment validation.

```python
ROOT_DIR  = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = ROOT_DIR / "data"          # where user puts their files
INDEX_DIR = ROOT_DIR / "indexing_data" # where LlamaIndex persists the vector store
```

**Environment validation (`require_runtime_keys()`):**

Called at startup to fail fast if required keys are missing:
- At least one of: `GROQ_API_KEY`, `GOOGLE_API_KEY`, `COHERE_API_KEY`
- `TAVILY_API_KEY` for the web search tool

Optional:
- `LLM_PROVIDER` — force a specific provider (`groq`, `google`, `cohere`)
- `GOOGLE_DRIVE_FOLDER_ID` — enables Google Drive ingestion
- `KT_API_URL` — Streamlit uses this to reach FastAPI (default: `http://localhost:8000`)

---

## `chains.py` — Dynamic LLM Factory

**Purpose:** Selects and instantiates the LLM and web search tool based on available API keys. The rest of the codebase never references a specific LLM provider.

**LLM selection logic:**

```python
@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "").lower()

    if provider == "groq" or (not provider and GROQ_API_KEY):
        return ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    if provider == "google" or (not provider and GOOGLE_API_KEY):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    if provider == "cohere" or (not provider and COHERE_API_KEY):
        return ChatCohere(model="command-r-plus", temperature=0)
```

Priority: **Groq → Google → Cohere**. `@lru_cache` means the client is created once and reused.

---

## `rag.py` — Document Ingestion and Retrieval

**Purpose:** Manages the full lifecycle of document-based knowledge.

### Supported file formats

```python
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt"}
```

### Auto-discovery

`_discover_documents()` scans `data/` automatically — no code changes needed when adding files.

### Index building

```bash
python -m src.agent.rag
```

1. Discovers files in `data/`
2. Parses each file into text
3. Splits into 512-token chunks (50 token overlap)
4. Embeds with `BAAI/bge-small-en-v1.5` (local, no API key)
5. Persists to `indexing_data/`

### Index loading

```python
@lru_cache(maxsize=1)
def get_vector_index():
    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    return load_index_from_storage(storage_context)
```

Loaded from disk once per server process, then kept in memory.

### Document retrieval

Embeds the query with the same model and finds top-k most similar chunks by cosine similarity. Results are converted from LlamaIndex `NodeWithScore` to LangChain `Document` objects.

---

## `tools.py` — The 6 Local Tools

### 1. `search_company_documents`

Calls `retrieve_documents(query)` to get top 4 chunks from the vector store. Each chunk is prefixed with `[Source: filename]` so the citation extractor can pick it up. Text is capped at 700 characters per chunk.

### 2. `summarise_document`

Accepts a filename, loads the full file text via `SimpleDirectoryReader`, caps at 6000 characters, and returns it with a `[Full text of <filename>]` prefix. The LLM then synthesises the summary in its next reasoning step.

### 3. `extract_structured_data`

Retrieves relevant document chunks, then returns the context alongside a JSON field template (e.g. `{"project_name": "<extracted value or null>"}`). The LLM fills in the template values in its response.

### 4. `search_web`

Calls `TavilySearchResults(k=3)` and formats results with `[Source: url]` prefixes for citation extraction.

### 5. `calculate`

Uses Python's `ast` module to safely evaluate arithmetic expressions without `eval()`:

```python
_SAFE_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub,
                   ast.Mult: operator.mul, ast.Div: operator.truediv, ...}
```

Only numeric constants and the defined operators are allowed — no function calls, no variable access. Returns `"expression = result"` format.

### 6. `generate_chart`

Accepts `data_json` (list of dicts or dict of lists), `chart_type` (`bar`, `line`, `pie`, `scatter`), and a `title`. Builds a Plotly `go.Figure` and returns `CHART_JSON::{figure_json}`. The `parse_result()` function detects this prefix and extracts the JSON for the UI to render.

---

## `mcp_client.py` — Database Tools

**Purpose:** Three LangChain tools wrapped in an MCP-compatible `asynccontextmanager`.

### The three tools

```python
list_database_tables()              # discover available tables
describe_database_table(table_name) # get column names and types
query_company_database(sql_query)   # run a SELECT query
```

### Safety: SELECT-only enforcement

```python
if not sql_query.strip().upper().startswith("SELECT"):
    return "Error: Only SELECT queries are allowed."
```

### MCP-compatible interface

```python
@asynccontextmanager
async def mcp_server_context() -> AsyncGenerator[List[BaseTool], None]:
    yield [list_database_tables, describe_database_table, query_company_database]
```

The entire database layer can be replaced with a real MCP server by swapping only this function. `workflow.py` does not change.

### Typical agent interaction

For "How many open orders are there?", the LLM will:
1. Call `list_database_tables()` → sees available tables
2. Call `describe_database_table("orders")` → sees column names
3. Call `query_company_database("SELECT COUNT(*) FROM orders WHERE status = 'open'")` → gets count
4. Generate a natural language answer with the SQL shown in a code block

---

## `workflow.py` — The LangGraph Agent

**Purpose:** The heart of the system. Assembles all tools, runs the ReAct loop, and parses the final result.

### System Prompt

Injected as a `SystemMessage` at the start of every agent turn. Enforces:
- Always cite sources (filenames, SQL queries, URLs)
- Use `calculate` for all arithmetic — never compute in the LLM head
- Proactively offer charts for tabular/numeric results
- Use `extract_structured_data` for field extraction tasks
- Use `summarise_document` for overview/summary requests
- Leverage conversation history for follow-up questions

### State definition

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

Holds the full message history: `SystemMessage` → `HumanMessage` (prior turns) → `AIMessage` (with tool_calls) → `ToolMessage` (results) → `AIMessage` (answer). The `add_messages` annotation appends rather than replaces.

### Agent node

```python
def agent_node(state: AgentState) -> dict:
    llm = get_llm().bind_tools(dynamic_tools)
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages_with_system)
    return {"messages": [response]}
```

The system prompt is prepended on **every turn**, not just the first, so it applies to all follow-up questions.

### Routing (`should_continue`)

```python
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else END
```

### Entry point (`aask`)

```python
LOCAL_TOOLS = [
    search_company_documents, search_web,
    summarise_document, extract_structured_data,
    calculate, generate_chart,
]

async def aask(question: str, history: list | None = None) -> dict:
    prior_messages = history or []
    async with mcp_server_context() as mcp_tools:
        all_tools = LOCAL_TOOLS + mcp_tools   # 9 tools total
        graph = build_graph(all_tools)
        result = await graph.ainvoke({
            "messages": prior_messages + [HumanMessage(content=question)]
        })
    return parse_result(result)
```

A new graph is built for each request to avoid state leaking between requests.

### Citation extraction (`_extract_citations`)

Walks all `ToolMessage` objects in the final state:

| Tool | Citation extracted |
|---|---|
| `search_company_documents`, `search_web` | `[Source: value]` regex matches |
| `query_company_database` | SQL from the preceding `AIMessage.tool_calls` |
| `calculate` | Expression and result from content |
| `summarise_document` | Filename from `[Full text of <file>]` prefix |
| `extract_structured_data` | `[Source: value]` regex matches |
| `list_database_tables`, `describe_database_table` | `"schema lookup via <tool_name>"` |

### Chart extraction (`_extract_chart`)

Looks for a `ToolMessage` from `generate_chart` whose content starts with `CHART_JSON::` and parses the Plotly figure JSON.

### Result parsing (`parse_result`)

Maps tool names to datasource labels:

| Tool name | `datasource` |
|---|---|
| `search_company_documents`, `summarise_document`, `extract_structured_data` | `company_docs` |
| `search_web` | `web_search` |
| `query_company_database`, `list_database_tables`, `describe_database_table` | `database` |
| `calculate` | `calculation` |
| `generate_chart` | `chart` |
| *(no tools)* | `direct_llm` |
| *(multiple categories)* | `multiple` |

---

## `app.py` — FastAPI Backend

**Purpose:** HTTP wrapper around `aask()` with in-memory session management.

### Endpoints

```
POST /ask
  ← QuestionRequest(question: str, session_id: str = "default")
  → QuestionResponse(answer, datasource, tools_used, citations, chart_data)

GET  /health
  → {"status": "ok"}

GET  /docs
  → Swagger UI

POST /upload
  ← multipart files
  → rebuilds index, returns saved/rejected/indexed lists

GET  /documents
  → list of files in data/ with name, size_kb, type

GET  /sessions/{session_id}/history
  → {session_id, turn_count, messages}

DELETE /sessions/{session_id}/history
  → clears session
```

### Session memory

```python
_sessions: dict[str, list[dict]] = defaultdict(list)
MAX_HISTORY_TURNS = 20
```

Each session stores plain dicts `{"role": "human"|"ai", "content": str}`. On each `/ask` request:
1. `_get_lc_history(session_id)` reconstructs LangChain message objects
2. Passed to `aask(question, history=history)`
3. `_append_to_session()` stores the new exchange, trimming to 40 messages

---

## `streamlit_app.py` — Chat Frontend

**Purpose:** Chat UI with rich output rendering.

### Key features

- **Session ID:** Each browser tab gets a UUID (`st.session_state.session_id`). Sent with every `/ask` request so FastAPI maintains per-tab memory.
- **Message rendering (`_render_assistant_message`):**
  - Markdown answer text
  - Inline Plotly chart if `chart_data` is present
  - Coloured tool badge showing datasource and tool names used
  - Citation pills showing source filenames/URLs with tooltip for detail
- **Document upload:** Sidebar file uploader → `POST /upload` → auto-index. Document list fetched from `GET /documents` and cached in session state.
- **Session controls:** "Clear chat" calls `DELETE /sessions/{id}/history`. "New session" generates a fresh UUID.
- **Sample questions:** Organised by tool path — LLM, Documents, Database, Web, Calculator, Charts.
- **Timeout:** 180 seconds (LLM + tool calls can be slow on first run due to model loading).

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

## `schemas.py` — Pydantic Models

```python
class QuestionRequest(BaseModel):
    question:   str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default")

class Citation(BaseModel):
    source: str   # filename, table name, or URL
    detail: str   # page number, SQL query, search snippet

class QuestionResponse(BaseModel):
    answer:     str
    datasource: str | None = None
    tools_used: list[str]
    citations:  list[Citation]
    chart_data: dict | None     # Plotly figure JSON or null
```

---

## Data Flow: End to End

Complete trace of a document question:

```
User types: "What does the KT document say about the database schema?"
│
├─ Streamlit: appends to st.session_state.messages
│             calls httpx.post("/ask",
│                              json={"question": "...", "session_id": "abc123"},
│                              timeout=180)
│
├─ FastAPI (app.py):
│   validates QuestionRequest
│   history = _get_lc_history("abc123")  → prior HumanMessage/AIMessage list
│   calls await aask("...", history=history)
│
├─ workflow.py aask():
│   prior_messages + [HumanMessage("...")]
│   async with mcp_server_context() → 3 DB tools
│   all_tools = 9 tools total
│   graph = build_graph(all_tools)
│   result = await graph.ainvoke({"messages": [...]})
│
├─ Agent Node (LLM):
│   receives [SystemMessage, ...history, HumanMessage] + 9 tool schemas
│   decides: call search_company_documents("database schema")
│   returns AIMessage(tool_calls=[{name: "search_company_documents", ...}])
│
├─ should_continue() → "tools"
│
├─ Tool Node:
│   executes search_company_documents("database schema")
│   → retrieve_documents() → cosine similarity → top 4 chunks
│   → each chunk prefixed with [Source: filename]
│   appends ToolMessage(content="[Source: kt_doc.pdf]\n...", name="search_company_documents")
│
├─ Agent Node (LLM):
│   reads full history including ToolMessage
│   generates answer citing "kt_doc.pdf"
│   no tool_calls → END
│
├─ parse_result():
│   tools_used = ["search_company_documents"]
│   datasource = "company_docs"
│   _extract_citations() → [{"source": "kt_doc.pdf", "detail": ""}]
│   _extract_chart() → None
│
├─ _append_to_session("abc123", question, answer)
│
├─ FastAPI returns QuestionResponse JSON
│
└─ Streamlit:
    renders answer markdown
    renders tool badge "📄 Company documents · `search_company_documents`"
    renders citation pill "📎 kt_doc.pdf"
    appends to st.session_state.messages
```

---

## Known Limitations and Recommended Solutions

| Limitation | Current state | Recommended solution |
|---|---|---|
| Vector store is file-based | JSON files in `indexing_data/` | Replace with Postgres + pgvector |
| Database is SQLite | `data/company.db` | Connect a real MCP Postgres server |
| Session state is in-memory | `_sessions` dict in `app.py` | Redis-backed session store for multi-worker |
| No authentication on `/ask` | Endpoint is open | Add API key header middleware or OAuth |
| No streaming responses | Full answer returned at once | Implement SSE via `graph.astream()` (`KnowledgeTransferAgent.run()` is ready) |
| Google Drive sync is manual | Run `ingest_drive.py` by hand | Add a scheduled job (cron / Celery beat) |
| Index rebuild requires restart | `@lru_cache` holds stale index | Add cache invalidation on index rebuild |
| Chart data not persisted | `chart_data` lost on page reload | Store Plotly JSON in session state |
