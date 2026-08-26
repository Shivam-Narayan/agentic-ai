# System Design

## Overview

This document covers the internal design of each component in the KT Agent. It is intended for developers who want to understand how the system works, modify it, or extend it.

For a higher-level view of the overall architecture and data flow, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Component Map

```
.env
 └── config.py  ─────────────────────────────────────────────────────┐
                                                                      │
chains.py  (LLM factory)                                             │
 ├── get_llm()             → ChatGroq(openai/gpt-oss-20b) / Gemini / Cohere
 └── get_web_search_tool() → TavilySearch (langchain-tavily)         │
                                                                      │
rag.py  (document layer)                                             │
 ├── _discover_documents()   → scans data/ for supported files       │
 ├── _get_file_extractors()  → registers DocxReader for .docx/.doc   │
 ├── build_index()           → ingests, embeds, persists             │
 ├── get_vector_index()      → loads from indexing_data/             │
 ├── rebuild_index()         → rebuilds + clears lru_cache           │
 └── retrieve_documents()   → returns List[Document]                 │
                                                                      │
tools.py  (LangChain tools — 6 local tools)                          │
 ├── search_company_documents  → rag.retrieve_documents()            │
 ├── summarise_document        → reads full file via LlamaIndex      │
 ├── extract_structured_data   → RAG + field template                │
 ├── search_web                → TavilySearch, handles dict response │
 ├── calculate                 → safe AST evaluator                  │
 └── generate_chart            → Plotly JSON figure                  │
                                                                      │
mcp_client.py  (database tools — 3 MCP tools)                       │
 ├── list_database_tables                                            │
 ├── describe_database_table                                          │
 └── query_company_database  (SELECT only)                           │
                                                                      │
workflow.py  (LangGraph agent)  ◄── uses all of the above           │
 ├── _build_system_prompt()   (live date/time injection)             │
 ├── AgentState                                                      │
 ├── _get_previous_tool_calls() (dedup helper)                       │
 ├── build_graph(tools)                                              │
 │    └── agent_node()       (dedup guard + parallel_tool_calls=False)
 ├── should_continue()                                               │
 ├── _extract_citations()                                            │
 ├── _extract_chart()                                                │
 ├── parse_result()          (walks back for last non-empty answer)  │
 ├── LOCAL_TOOLS = [6 tools]                                         │
 └── aask(question, history)  ← FastAPI calls this                  │
                                                                      │
app.py  (FastAPI)                                                     │
 ├── POST /ask          →  aask(question, history)                   │
 ├── GET  /health                                                    │
 ├── POST /upload        →  rebuild_index()                          │
 ├── GET  /documents                                                 │
 ├── GET  /sessions/{id}/history                                     │
 └── DELETE /sessions/{id}/history                                   │
                                                                      │
streamlit_app.py  (UI)                                               │
 ├── st.chat_input (always called — prevents disappear bug)          │
 ├── pending_question (sample question click flow)                   │
 ├── _render_assistant_message()  (badges, charts, citations)        │
 └── sidebar: upload, indexed docs, session controls, samples       │
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
- `KT_API_URL` — Streamlit uses this to reach FastAPI (default: `http://localhost:8000`)

---

## `chains.py` — Dynamic LLM Factory

**Purpose:** Selects and instantiates the LLM and web search tool based on available API keys.

**LLM selection logic:**

```python
@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "").lower()

    if provider == "groq" or (not provider and GROQ_API_KEY):
        return ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    if provider == "google" or (not provider and GOOGLE_API_KEY):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    if provider == "cohere" or (not provider and COHERE_API_KEY):
        return ChatCohere(model="command-r-plus", temperature=0)
```

Priority: **Groq → Google → Cohere**. `@lru_cache` means the client is created once per process.

**Web search tool:**

```python
@lru_cache(maxsize=1)
def get_web_search_tool():
    from langchain_tavily import TavilySearch
    return TavilySearch(max_results=5)
```

Uses the newer `langchain-tavily` package (`TavilySearch`) rather than the deprecated `TavilySearchResults` from `langchain-community`. Returns a dict with a `results` list (not a plain list).

---

## `rag.py` — Document Ingestion and Retrieval

**Purpose:** Manages the full lifecycle of document-based knowledge — discovery, parsing, indexing, and retrieval.

### Supported file formats

```python
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt"}
```

### DOCX parsing fix

LlamaIndex's default `SimpleDirectoryReader` falls back to a raw binary reader for `.docx` files if `llama-index-readers-file` is not installed. This caused Word documents to be indexed as binary garbage, returning no readable results.

The fix: `_get_file_extractors()` explicitly registers `DocxReader` (backed by `docx2txt`) for `.docx` and `.doc` files:

```python
def _get_file_extractors() -> dict:
    from llama_index.readers.file import DocxReader
    return {".docx": DocxReader(), ".doc": DocxReader()}
```

This is passed to `SimpleDirectoryReader(file_extractor=_get_file_extractors())`.

### Index building

```bash
python -m src.agent.rag
```

1. Discovers files in `data/` with `_discover_documents()`
2. Parses each file using the correct reader (DocxReader for .docx, pypdf for .pdf, etc.)
3. Splits into 512-token chunks (50 token overlap) with `SentenceSplitter`
4. Embeds with `BAAI/bge-small-en-v1.5` (runs locally, no API key)
5. Persists to `indexing_data/`

### Index loading

```python
@lru_cache(maxsize=1)
def get_vector_index():
    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    return load_index_from_storage(storage_context)
```

Loaded from disk once per server process, then cached. `rebuild_index()` calls `get_vector_index.cache_clear()` after every rebuild so the next query loads the fresh index — no server restart required.

### Document retrieval

Embeds the query with the same `BAAI/bge-small-en-v1.5` model and finds the top-k most similar chunks by cosine similarity. Results are converted from LlamaIndex `NodeWithScore` to LangChain `Document` objects for use by the tools.

---

## `tools.py` — The 6 Local Tools

### 1. `search_company_documents`

Calls `retrieve_documents(query)` to get top 4 chunks from the vector store. Each chunk is prefixed with `[Source: filename]` so the citation extractor can pick it up. Text is capped at 700 characters per chunk.

### 2. `summarise_document`

Accepts a filename (case-insensitive match against `data/`), loads the full file text via `SimpleDirectoryReader` with `_get_file_extractors()`, caps at 6000 characters, and returns it with a `[Full text of <filename>]` prefix. The LLM synthesises the summary in its next reasoning step.

### 3. `extract_structured_data`

Retrieves relevant document chunks, then returns the context alongside a JSON field template (e.g. `{"project_name": "<extracted value or null>"}`). The LLM fills in the template values in its response.

### 4. `search_web`

Calls `TavilySearch` and handles both response formats:
- `TavilySearch` returns `{"results": [...], "answer": "..."}` — a dict
- Legacy `TavilySearchResults` returned a plain list

Results are formatted as `[Source: url]\ncontent` strings for citation extraction. The tool docstring tells the LLM not to quote dates from snippets — only quote the numerical value.

### 5. `calculate`

Uses Python's `ast` module to safely evaluate arithmetic without `eval()`:

```python
_SAFE_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub,
                   ast.Mult: operator.mul, ast.Div: operator.truediv, ...}
```

Only numeric constants and the defined operators are allowed — no function calls, no variable access. Returns `"expression = result"` format.

### 6. `generate_chart`

Accepts `data_json` (list of dicts or dict of lists), `chart_type` (`bar`, `line`, `pie`, `scatter`), and a `title`. Builds a Plotly `go.Figure` and returns `CHART_JSON::{figure_json}`. The `_extract_chart()` function detects this prefix and extracts the JSON for the Streamlit UI to render inline.

---

## `mcp_client.py` — Database Tools

**Purpose:** Three LangChain tools wrapped in an MCP-compatible `asynccontextmanager`.

### The three tools

```python
list_database_tables()               # discover available tables
describe_database_table(table_name)  # get column names and types
query_company_database(sql_query)    # run a SELECT query
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

The entire database layer can be replaced with a real MCP server by swapping only this function. `workflow.py` does not need to change.

---

## `workflow.py` — The LangGraph Agent

**Purpose:** The heart of the system. Assembles all tools, runs the ReAct loop, and parses the final result.

### Dynamic System Prompt (`_build_system_prompt`)

Previously a static constant (`SYSTEM_PROMPT = """..."""`). Now a function that stamps the live server date and time on every call:

```python
def _build_system_prompt() -> str:
    now = datetime.now()
    date_str = now.strftime("%A, %d %B %Y")   # "Wednesday, 26 August 2026"
    time_str = now.strftime("%H:%M")
    return f"...CURRENT DATE AND TIME: {date_str}, {time_str}..."
```

This fixes the problem where the LLM would answer "What day is today?" using its training data (which had the wrong day of week).

The prompt also contains explicit rules:
- General knowledge questions must be answered directly — no tools
- `search_company_documents` must be called at most once per question
- Web search snippets: quote the value, not the date

### State definition

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

The `add_messages` annotation means new messages are appended to the list, not replacing it. The full conversation thread — system message, history, human message, tool calls, tool results, final answer — lives in this single list.

### Agent node and deduplication guard

```python
def agent_node(state: AgentState) -> dict:
    # Bind tools with parallel_tool_calls=False — forces one tool per step
    llm = get_llm().bind_tools(dynamic_tools, parallel_tool_calls=False)

    # Fresh system prompt with current date prepended to full history
    messages_with_system = [SystemMessage(content=_build_system_prompt())] + state["messages"]
    response = llm.invoke(messages_with_system)

    # Deduplication guard
    if response.tool_calls:
        already_used = _get_previous_tool_calls(state["messages"])
        search_count = sum(1 for k in already_used if k.startswith("search_company_documents::"))

        for tc in response.tool_calls:
            name = tc["name"]
            # Block: search_company_documents already ran once
            if name == "search_company_documents" and search_count >= 1:
                response.tool_calls = []   # strip all tool calls
                if not response.content.strip():
                    # Re-invoke without tools so LLM writes a real answer
                    response = get_llm().invoke(messages_with_system + [force_answer_message])
                break
```

`parallel_tool_calls=False` is set at the API level — the model physically cannot request multiple tools in one response. The dedup guard is the second layer preventing sequential repeated calls.

### Routing (`should_continue`)

```python
def should_continue(state: AgentState) -> str:
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END
```

### `parse_result` — walks backwards for the answer

```python
for msg in reversed(messages):
    if msg.type == "ai" and msg.content and str(msg.content).strip():
        answer = str(msg.content).strip()
        break
```

Previously used `messages[-1].content` which broke when the dedup guard produced an empty AI message. Walking backwards finds the last non-empty AI response correctly.

### Entry point (`aask`)

```python
LOCAL_TOOLS = [
    search_company_documents, search_web,
    summarise_document, extract_structured_data,
    calculate, generate_chart,
]

async def aask(question: str, history: list | None = None) -> dict:
    async with mcp_server_context() as mcp_tools:
        all_tools = LOCAL_TOOLS + mcp_tools   # 9 tools total
        graph = build_graph(all_tools)
        result = await graph.ainvoke(
            {"messages": (history or []) + [HumanMessage(content=question)]},
            config={"recursion_limit": 8},  # caps the agent loop
        )
    return parse_result(result)
```

A new graph is built for each request to avoid state leaking between requests. `recursion_limit=8` means at most 8 agent↔tool loops — enough for complex multi-step queries, tight enough to stop runaway loops.

---

## `app.py` — FastAPI Backend

### Endpoints

```
POST /ask
  ← QuestionRequest(question: str, session_id: str = "default")
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
  → clears session
```

### Session memory

```python
_sessions: dict[str, list[dict]] = defaultdict(list)
MAX_HISTORY_TURNS = 20
```

Each session stores plain dicts `{"role": "human"|"ai", "content": str}`. On each `/ask`:
1. `_get_lc_history(session_id)` → reconstructs `HumanMessage` / `AIMessage` objects
2. Passed to `aask(question, history=history)`
3. `_append_to_session()` stores the new exchange, trimming to 40 messages

---

## `streamlit_app.py` — Chat Frontend

### Key features

- **Session ID:** Each browser tab gets a UUID. Sent with every `/ask` so FastAPI maintains per-tab memory.
- **`st.chat_input` always rendered:** Previously, when a sample question was clicked, `pending_question` short-circuited the `or` expression and `st.chat_input(...)` was never called. Streamlit removed the widget from the DOM. Fixed by always calling `st.chat_input` first, then checking `pending_question`:

  ```python
  # Always call chat_input — Streamlit hides it if skipped even once
  typed_input = st.chat_input("Ask a question…")
  prompt = st.session_state.pop("pending_question", None) or typed_input
  ```

- **Document upload:** Sidebar file uploader → `POST /upload` → index rebuilds automatically. Document list cached in session state, refreshed on demand.
- **Timeout:** 180 seconds for the API call (LLM + tool calls can be slow on first run).

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
    detail: str   # SQL query, expression, or empty

class QuestionResponse(BaseModel):
    answer:     str
    datasource: str | None = None
    tools_used: list[str]
    citations:  list[Citation]
    chart_data: dict | None   # Plotly figure JSON or null
```

---

## Data Flow: End to End

Complete trace of "What are Shivam's technical skills?":

```
User types: "What are Shivam's technical skills?"
│
├─ Streamlit: httpx.post("/ask", json={"question": "...", "session_id": "abc"})
│
├─ FastAPI: validates request, gets lc_history, calls await aask(...)
│
├─ workflow.py aask():
│   all_tools = 6 local + 3 MCP = 9 total
│   graph = build_graph(all_tools)
│   state = {messages: [HumanMessage("...")]}
│
├─ Agent Node:
│   llm bound with parallel_tool_calls=False
│   system prompt injected with live date/time
│   LLM decides: call search_company_documents("Shivam technical skills")
│   dedup guard: first call, allowed through
│
├─ Tool Node:
│   search_company_documents runs
│   retrieve_documents() → LlamaIndex cosine search
│   DocxReader-indexed chunks returned with readable text
│   "[Source: Shivam_Narayan_Resume_Revised.docx]\nTECHNICAL SKILLS..."
│
├─ Agent Node (second pass):
│   reads ToolMessage with resume content
│   dedup guard: search_company_documents count is now 1
│   LLM generates answer from resume text, no more tool calls
│   → END
│
├─ parse_result():
│   walk backwards → last non-empty AI message = answer text
│   tools_used = ["search_company_documents"]
│   datasource = "company_docs"
│   citations = [{"source": "Shivam_Narayan_Resume_Revised.docx"}]
│
├─ FastAPI: _append_to_session, return QuestionResponse
│
└─ Streamlit: renders answer + "📄 Company documents" badge + citation pill
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
| Chart data not persisted | `chart_data` lost on page reload | Store Plotly JSON in session state |
| Groq free tier rate limit | 30 RPM | Dedup guard + parallel_tool_calls=False keeps usage low; upgrade to paid tier for heavy use |
