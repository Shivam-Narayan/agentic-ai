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
tools.py  (LangChain tools)                                       │
 ├── search_company_documents  → calls rag.retrieve_documents()   │
 └── search_web                → calls chains.get_web_search_tool()│
                                                                   │
mcp_client.py  (database tools)                                   │
 ├── list_database_tables                                         │
 ├── describe_database_table                                       │
 └── query_company_database    (SELECT only)                      │
                                                                   │
workflow.py  (LangGraph agent)  ◄── uses all of the above        │
 ├── AgentState                                                   │
 ├── build_graph(tools)                                           │
 ├── agent_node()                                                 │
 ├── should_continue()                                            │
 ├── parse_result()                                               │
 └── aask(question)  ← FastAPI calls this                        │
                                                                   │
app.py  (FastAPI)                                                  │
 └── POST /ask  →  aask()  →  QuestionResponse                   │
                                                                   │
streamlit_app.py  (UI)                                            │
 └── st.chat_input  →  httpx.post(/ask)  →  render answer        │
```

---

## `config.py` — Environment and Paths

**Purpose:** Single source of truth for all file paths and environment validation. Every other module imports paths from here rather than computing them independently.

```python
ROOT_DIR  = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = ROOT_DIR / "data"          # where user puts their files
INDEX_DIR = ROOT_DIR / "indexing_data" # where LlamaIndex persists the vector store
```

**Environment validation (`require_runtime_keys()`):**

Called at startup to fail fast if required keys are missing:
- At least one of: `GROQ_API_KEY`, `GOOGLE_API_KEY`, `COHERE_API_KEY`
- `TAVILY_API_KEY` for the web search tool

Optional keys:
- `LLM_PROVIDER` — force a specific provider (`groq`, `google`, `cohere`)
- `GOOGLE_DRIVE_FOLDER_ID` — enables Google Drive ingestion
- `KT_API_URL` — Streamlit uses this to reach FastAPI (default: `http://localhost:8000`)

---

## `chains.py` — Dynamic LLM Factory

**Purpose:** Selects and instantiates the LLM and web search tool based on available API keys. The rest of the codebase never references a specific LLM provider — it only calls `get_llm()`.

**LLM selection logic:**

```python
@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "").lower()

    if provider == "groq" or (not provider and GROQ_API_KEY):
        return ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY)

    if provider == "google" or (not provider and GOOGLE_API_KEY):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

    if provider == "cohere" or (not provider and COHERE_API_KEY):
        return ChatCohere(model="command-r-plus", cohere_api_key=COHERE_API_KEY)

    raise ValueError("No LLM API key found in .env")
```

Priority when multiple keys are present: **Groq → Google → Cohere**

**Why `@lru_cache`?** The LLM client is an expensive object (authenticates on creation, sets up HTTP connections). Caching it means it is created once when the first request arrives and reused for every subsequent request.

**Web search tool:**

```python
@lru_cache(maxsize=1)
def get_web_search_tool() -> TavilySearchResults:
    return TavilySearchResults(max_results=3, tavily_api_key=TAVILY_API_KEY)
```

Returns top 3 results per query. Also cached.

---

## `rag.py` — Document Ingestion and Retrieval

**Purpose:** Manages the full lifecycle of document-based knowledge — from raw files in `data/` to searchable vector chunks at query time.

### Supported file formats

```python
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt"}
```

| Format | Parsed by |
|---|---|
| `.pdf` | LlamaIndex built-in (pypdf) |
| `.docx`, `.doc` | LlamaIndex built-in (docx2txt) |
| `.xlsx`, `.xls` | LlamaIndex + openpyxl |
| `.csv` | LlamaIndex built-in (pandas) |
| `.txt` | LlamaIndex built-in |

### Auto-discovery

```python
def _discover_documents(data_dir: Path = DATA_DIR) -> List[Path]:
    return [
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
```

When `build_index()` is called without explicit file paths (which is the normal case), it calls `_discover_documents()` to automatically find every supported file in `data/`. No code changes are needed when adding or removing files — just re-run the indexer.

### Index building (`build_index`)

Run once (or whenever files change):

```bash
python -m src.agent.rag
```

What happens:
1. `_discover_documents()` scans `data/` and returns all supported file paths
2. `SimpleDirectoryReader` loads and parses each file into text
3. `SentenceSplitter(chunk_size=512, chunk_overlap=50)` splits text into overlapping chunks
4. `HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")` converts each chunk to a vector
5. `VectorStoreIndex` stores all chunks and their vectors in memory
6. `index.storage_context.persist(persist_dir=INDEX_DIR)` writes everything to `indexing_data/`

**Embedding model choice:** `BAAI/bge-small-en-v1.5` is a small (33M parameter), high-quality English embedding model that runs entirely locally. No API key, no network call, no cost. The model is downloaded from HuggingFace on first use and cached locally.

**Chunk settings explained:**
- `chunk_size=512` — each chunk is at most 512 tokens (~380 words). Large enough to contain a full thought, small enough for precise retrieval.
- `chunk_overlap=50` — consecutive chunks share 50 tokens so sentences at chunk boundaries are not lost.

### Index loading (`get_vector_index`)

```python
@lru_cache(maxsize=1)
def get_vector_index():
    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    return load_index_from_storage(storage_context)
```

`@lru_cache` means the index is loaded from disk exactly once per server process, then kept in memory. Every call to `retrieve_documents()` reuses the same in-memory index.

### Document retrieval (`retrieve_documents`)

```python
def retrieve_documents(question: str) -> List[Document]:
    nodes = get_vector_index().as_retriever().retrieve(question)
    return [Document(page_content=node.node.text) for node in nodes]
```

The retriever embeds the question using the same `bge-small-en-v1.5` model and finds the top-k most similar chunks by cosine similarity. The results are converted from LlamaIndex `NodeWithScore` objects to LangChain `Document` objects (the format the `@tool` functions return to the agent).

---

## `tools.py` — Local Agent Tools

**Purpose:** Wraps `rag.py` and Tavily into LangChain `@tool` functions that the LangGraph agent can call.

```python
@tool
def search_company_documents(query: str) -> str:
    """Search the indexed company documents (PDFs, Word docs, Excel files, CSVs).
    Use this for any question about internal company knowledge, projects, reports,
    policies, or data stored in uploaded files."""
    docs = retrieve_documents(query)
    if not docs:
        return "No relevant documents found."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


@tool
def search_web(query: str) -> str:
    """Search the live web for real-time information.
    Use this for current events, prices, weather, or anything that changes over time."""
    results = get_web_search_tool().invoke(query)
    return str(results)
```

**Why docstrings matter:** The docstring is literally sent to the LLM as the tool description. It is the only thing the LLM reads to decide when to call this tool. A vague docstring leads to wrong tool selection. Keep docstrings precise and use concrete examples of when to use each tool.

---

## `mcp_client.py` — Database Tools

**Purpose:** Exposes the company SQLite database as three LangChain tools wrapped in an `asynccontextmanager` that mirrors the Model Context Protocol (MCP) server interface.

### The three tools

```python
@tool
def list_database_tables() -> str:
    """List all tables in the company database.
    Call this first when the user asks a database question, to discover what data exists."""

@tool
def describe_database_table(table_name: str) -> str:
    """Get the column names and types for a database table.
    Call this before writing a SQL query to understand the table schema."""

@tool
def query_company_database(sql_query: str) -> str:
    """Run a read-only SQL SELECT query against the company database.
    Only SELECT statements are allowed. Use list_database_tables and
    describe_database_table first to understand the schema."""
```

### Safety: SELECT-only enforcement

```python
if not sql_query.strip().upper().startswith("SELECT"):
    return "Error: Only SELECT queries are allowed."
```

This check runs before every query execution. The tool description also signals this constraint to the LLM, reducing the chance of the model even attempting a write.

### MCP-compatible interface

```python
@asynccontextmanager
async def mcp_server_context() -> AsyncGenerator[List[BaseTool], None]:
    yield [list_database_tables, describe_database_table, query_company_database]
```

The `asynccontextmanager` pattern mirrors how a real MCP server connection works — you open a context, get a list of tools, use them, then the context closes. This design means the entire database layer can be replaced with a real MCP server (e.g., connecting to a Postgres database) by swapping only this function. `workflow.py` does not need to change at all.

### Typical agent interaction with the database

For a question like "How many open orders are there?", the LLM will:
1. Call `list_database_tables()` → sees `["orders", "customers", "products"]`
2. Call `describe_database_table("orders")` → sees `[("id", "INTEGER"), ("status", "TEXT"), ...]`
3. Call `query_company_database("SELECT COUNT(*) FROM orders WHERE status = 'open'")` → gets the count
4. Generate: "There are 47 open orders."

This three-step pattern (discover → inspect → query) mirrors how a human analyst would approach an unknown database.

---

## `workflow.py` — The LangGraph Agent

**Purpose:** The heart of the system. Defines the LangGraph `StateGraph`, assembles all tools, runs the ReAct loop, and parses the final result.

### State definition

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

`messages` holds the full conversation history: `HumanMessage` → `AIMessage` (with tool_calls) → `ToolMessage` (with tool results) → `AIMessage` (final answer). The `add_messages` annotation means each node's output is **appended** to the list rather than replacing it, so nothing is lost between iterations.

### Graph construction (`build_graph`)

```python
def build_graph(dynamic_tools: list):
    tool_node = ToolNode(dynamic_tools)          # LangGraph built-in

    def agent_node(state: AgentState):
        llm_with_tools = get_llm().bind_tools(dynamic_tools)
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}          # appended to state

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "agent")           # always loop back
    return builder.compile()
```

**`ToolNode`** is LangGraph's built-in prebuilt node. Given an `AIMessage` with `tool_calls`, it:
1. Looks up each called tool by name in the registered tool list
2. Executes the tool function with the provided arguments
3. Creates a `ToolMessage` with the result for each call
4. Appends all `ToolMessage` objects to state

Multiple parallel tool calls are handled automatically by `ToolNode`.

### Routing (`should_continue`)

```python
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END
```

Simple binary decision: if the last LLM output contains `tool_calls`, run tools. Otherwise, the LLM is done and the graph ends.

### Entry point (`aask`)

```python
async def aask(question: str) -> dict:
    local_tools = [search_company_documents, search_web]

    async with mcp_server_context() as mcp_tools:
        all_tools = local_tools + mcp_tools        # 5 tools total
        graph = build_graph(all_tools)
        result = await graph.ainvoke({
            "messages": [HumanMessage(content=question)]
        })

    return parse_result(result)
```

A new graph is built for each request. This is intentional — it ensures tools are always bound fresh and avoids any state leaking between requests.

### Result parsing (`parse_result`)

```python
def parse_result(result: dict) -> dict:
    messages = result["messages"]
    tool_names = [m.name for m in messages if m.type == "tool"]
    # maps tool names to datasource labels
    # sets datasource = "multiple" if more than one unique source was used
```

Inspects the final message list for `ToolMessage` objects. Maps known tool names to their datasource label:

| Tool name | `datasource` |
|---|---|
| `search_company_documents` | `company_docs` |
| `search_web` | `web_search` |
| `query_company_database` | `database` |
| `list_database_tables` | `database` |
| `describe_database_table` | `database` |
| *(no tools called)* | `direct_llm` |
| *(multiple different sources)* | `multiple` |

---

## `app.py` — FastAPI Backend

**Purpose:** Thin HTTP wrapper around `aask()`. Handles request validation and error responses.

```
POST /ask
  ← QuestionRequest(question: str)    # 1–2000 characters, validated by Pydantic
  → QuestionResponse(
        answer: str,
        datasource: str,
        tools_used: List[str]
    )

GET /health
  → {"status": "ok"}

GET /docs
  → Swagger UI (auto-generated by FastAPI)
```

The endpoint is fully async — it calls `await aask(question)` and does not block the event loop while the LangGraph agent is running.

---

## `streamlit_app.py` — Chat Frontend

**Purpose:** A chat UI built with Streamlit that talks to the FastAPI backend over HTTP.

Key implementation details:

- **Session state:** `st.session_state.messages` holds the full chat history as a list of `{"role": ..., "content": ..., "datasource": ...}` dicts. History persists across interactions within a session.
- **Rendering:** Each message is rendered with `st.chat_message(role)`. AI messages include a `st.caption()` showing which tool was used, colour-coded by datasource.
- **API call:** Uses `httpx` with a 120-second timeout (LangGraph + LLM calls can take time on first run due to model loading).
- **Sidebar:** Pre-built sample questions for each of the 4 answer paths — useful for demos and testing.
- **API URL:** Reads `KT_API_URL` from the environment. Defaults to `http://localhost:8000`. Change this if FastAPI is running on a different host or port.

---

## `schemas.py` — Pydantic Models

```python
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

class QuestionResponse(BaseModel):
    answer: str
    datasource: str      # direct_llm | company_docs | database | web_search | multiple
    tools_used: List[str]
```

These models are shared between FastAPI (for HTTP validation) and the internal `parse_result()` function.

---

## `ingest_drive.py` — Google Drive Ingestion (Optional)

**Purpose:** Pulls documents from a Google Drive folder into the vector store. Useful for teams that store their knowledge base in Drive.

**How it works:**
1. Uses `llama_index.readers.google.GoogleDriveReader` (requires OAuth `credentials.json`)
2. Downloads all documents from the folder specified by `GOOGLE_DRIVE_FOLDER_ID` in `.env`
3. Calls `build_index(extra_documents=drive_docs)` to merge Drive docs with local `data/` files

**To use:**
1. Set `GOOGLE_DRIVE_FOLDER_ID=your_folder_id` in `.env`
2. Place your OAuth `credentials.json` in the project root
3. Run: `python -m src.agent.ingest_drive`

A browser window will open for OAuth authorisation on first run.

---

## Data Flow: End to End

Here is the complete trace of a document question through every layer of the system:

```
User types: "What does the KT document say about the database schema?"
│
├─ Streamlit: appends to st.session_state.messages
│             calls httpx.post("http://localhost:8000/ask",
│                              json={"question": "..."}, timeout=120)
│
├─ FastAPI (app.py):
│   receives QuestionRequest
│   calls await aask("What does the KT document say about the database schema?")
│
├─ workflow.py aask():
│   local_tools = [search_company_documents, search_web]
│   opens mcp_server_context() → [list_database_tables,
│                                  describe_database_table,
│                                  query_company_database]
│   all_tools = 5 tools total
│   graph = build_graph(all_tools)
│   result = await graph.ainvoke({"messages": [HumanMessage(...)]})
│
├─ Agent Node (LLM):
│   receives HumanMessage + 5 tool schemas
│   decides: call search_company_documents("database schema")
│   returns AIMessage(tool_calls=[{name: "search_company_documents",
│                                   args: {"query": "database schema"}}])
│
├─ should_continue(): tool_calls present → route to "tools"
│
├─ Tool Node:
│   executes search_company_documents("database schema")
│   → calls retrieve_documents("database schema")
│   → calls get_vector_index().as_retriever().retrieve(...)
│   → LlamaIndex embeds query → cosine similarity search → top 4 chunks
│   → returns formatted text chunks
│   appends ToolMessage(content="...chunks...", name="search_company_documents")
│
├─ should_continue(): back to Agent Node
│
├─ Agent Node (LLM):
│   reads full message history including ToolMessage
│   generates final answer synthesised from the chunks
│   returns AIMessage(content="The KT document describes the database schema as...")
│   no tool_calls → should_continue() → END
│
├─ parse_result():
│   finds ToolMessage with name="search_company_documents"
│   returns {answer: "...", datasource: "company_docs",
│             tools_used: ["search_company_documents"]}
│
├─ FastAPI: returns QuestionResponse JSON
│
└─ Streamlit:
    appends answer to st.session_state.messages
    renders with st.chat_message("assistant")
    shows caption: "📄 Tool: search_company_documents"
```

---

## Known Limitations and Recommended Solutions

| Limitation | Current state | Recommended solution |
|---|---|---|
| Vector store is file-based | JSON files in `indexing_data/` | Replace with Postgres + pgvector |
| Database is SQLite | `data/company.db` | Connect a real MCP Postgres server |
| No authentication on `/ask` | Endpoint is open | Add API key header middleware or OAuth |
| Single-user session state | Streamlit `st.session_state` is per-tab | Add Redis-backed session management |
| No streaming responses | Full answer returned at once | Implement SSE via `graph.astream()` |
| Google Drive sync is manual | Run `ingest_drive.py` by hand | Add a scheduled job (cron / Celery beat) |
| Index rebuild requires restart | `@lru_cache` holds stale index | Add cache invalidation on index rebuild |
| No conversation memory | Each request starts fresh | Pass previous messages in `AgentState` |
