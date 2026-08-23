# System Design

## Overview

This system is an agentic Knowledge Transfer Assistant. Its job is to answer questions by intelligently routing them to the right data source — company documents, a structured database, the live web, or the LLM's own training knowledge — without any hard-coded routing logic.

---

## Component Design

### `config.py` — Environment & Paths

Centralises all configuration in one place:

```python
DATA_DIR   = Path(__file__).parent.parent.parent / "data"
INDEX_DIR  = Path(__file__).parent.parent.parent / "indexing_data"
```

Validates on startup that:
- At least one LLM API key is present (`GROQ_API_KEY`, `GOOGLE_API_KEY`, or `COHERE_API_KEY`)
- `TAVILY_API_KEY` is present for web search

---

### `chains.py` — Dynamic LLM Factory

Selects and initialises the LLM based on `.env` contents:

```python
def get_llm():
    if provider == "groq" or GROQ_API_KEY:
        return ChatGroq(model="openai/gpt-oss-120b")
    if provider == "google" or GOOGLE_API_KEY:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    if provider == "cohere" or COHERE_API_KEY:
        return ChatCohere(model="command-r-plus")
```

Priority: **Groq → Google → Cohere** (override with `LLM_PROVIDER=google` in `.env`).

The returned model implements LangChain's `BaseChatModel` interface so the rest of the codebase is vendor-agnostic.

> **Current model in use:** `openai/gpt-oss-120b` via Groq API — confirmed to support tool calling.

---

### `rag.py` — Document Ingestion & Retrieval

**Index building** (run once):
```bash
python -m src.agent.rag
```
- Reads documents from `data/` (PDF, DOCX)
- Splits with `SentenceSplitter(chunk_size=512, chunk_overlap=50)`
- Embeds with `BAAI/bge-small-en-v1.5` (local, no API key needed)
- Persists the index to `indexing_data/`

**Retrieval at query time:**
```python
def retrieve_documents(question: str) -> List[Document]:
    return get_vector_index().as_retriever().retrieve(question)
```

The vector index is loaded once via `@lru_cache` and reused for all requests.

> **Note:** To add new documents, place them in `data/` and re-run `python -m src.agent.rag`.

---

### `tools.py` — Local Agent Tools

Defines two tools for the LangGraph agent:

| Tool | Trigger | Implementation |
|---|---|---|
| `search_company_documents` | Questions about company/project | Calls `retrieve_documents()` from `rag.py` |
| `search_web` | Real-time facts, news, weather | Calls Tavily via `get_web_search_tool()` |

The `@tool` docstring is critical — it is literally sent to the LLM to help it decide when to use the tool:

```python
@tool
def search_company_documents(query: str) -> str:
    """Search indexed company / project documents (vector store)."""
```

---

### `mcp_client.py` — Database Tools (MCP-Compatible)

Provides three tools for querying the company SQLite database.

The `mcp_server_context()` is an `asynccontextmanager` that yields a list of LangChain tools. This mirrors the interface of a real **MCP (Model Context Protocol)** server, making it easy to replace the SQLite implementation with a real MCP server (e.g., connecting to Postgres) without changing `workflow.py`.

```python
@asynccontextmanager
async def mcp_server_context() -> AsyncGenerator[List[BaseTool], None]:
    yield [list_database_tables, describe_database_table, query_company_database]
```

| Tool | Purpose |
|---|---|
| `list_database_tables` | Shows the agent what tables exist |
| `describe_database_table` | Shows column names and types for a table |
| `query_company_database` | Runs a read-only SELECT query |

> **Safety:** Only `SELECT` queries are allowed. Any other SQL is rejected before execution.

---

### `workflow.py` — The LangGraph Agent

This is the heart of the system. It builds a `StateGraph` with two nodes:

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full message history

def build_graph(dynamic_tools: list):
    tool_node = ToolNode(dynamic_tools)

    def agent_node(state):
        llm = get_llm().bind_tools(dynamic_tools)
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()
```

**`should_continue`:** Routes to `tools` if the LLM emitted `tool_calls`, otherwise to `END`.

**`parse_result`:** Inspects the final message history to determine which tools were called and sets the `datasource` field accordingly.

**`aask(question)`:** The async entry point called by FastAPI:
```python
async def aask(question: str) -> dict:
    local_tools = [search_company_documents, search_web]
    async with mcp_server_context() as mcp_tools:
        all_tools = local_tools + mcp_tools
        graph = build_graph(all_tools)
        result = await graph.ainvoke({"messages": [HumanMessage(content=question)]})
        return parse_result(result)
```

---

### `app.py` — FastAPI Backend

Thin wrapper that exposes the agent over HTTP:

```
POST /ask   →  QuestionRequest(question) → aask() → QuestionResponse(answer, datasource, tools_used)
GET  /health → {"status": "ok"}
GET  /docs   → Swagger UI
```

---

### `streamlit_app.py` — Chat Frontend

- Renders chat history with `st.chat_message`
- Injects CSS to normalise markdown heading sizes inside chat bubbles
- Shows a caption after each AI message indicating which tool was used (`datasource`)
- Sidebar buttons for sample questions across all 4 answer paths
- Connects to FastAPI via `KT_API_URL` env var (defaults to `http://192.168.88.6:8000`)

---

## Multi-LLM Strategy

The system uses LangChain's `BaseChatModel` interface to remain vendor-agnostic. All three providers are tested and working with tool-calling:

| Provider | Model | Notes |
|---|---|---|
| Groq | `openai/gpt-oss-120b` | **Default.** Fast, free tier, confirmed tool calling ✅ |
| Google | `gemini-1.5-flash` | Requires `GOOGLE_API_KEY` |
| Cohere | `command-r-plus` | Requires `COHERE_API_KEY`, strong RAG performance |

**To switch:**
```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your_key_here
```

---

## Data Flow: End-to-End

```
User types "Status of order #12345?"
    │
    ▼
Streamlit POST → FastAPI /ask
    │
    ▼
aask("Status of order #12345?")
    │
    ├─ Opens mcp_server_context() → gets 3 DB tools
    ├─ Combines with local tools → 5 tools total
    ├─ Builds LangGraph
    │
    ▼
Agent Node: LLM reads question + 5 tool schemas
    → Decides to call query_company_database("SELECT * FROM orders WHERE id=12345")
    │
    ▼
Tool Node: Executes query_company_database → returns CSV row
    │
    ▼
Agent Node: LLM reads DB result → generates natural language answer
    │
    ▼
parse_result() → datasource="database", tools_used=["query_company_database"]
    │
    ▼
FastAPI returns QuestionResponse
    │
    ▼
Streamlit renders answer + "Tool: query_company_database" caption
```

---

## Known Limitations & Future Work

| Limitation | Recommended Solution |
|---|---|
| Vector store is file-based | Replace with Postgres + pgvector |
| Database is SQLite | Connect a real MCP server to Postgres/MySQL |
| No authentication on `/ask` | Add API key middleware or OAuth |
| Single-user session state | Add Redis-backed session management |
| No streaming responses | Implement SSE with `graph.astream()` |
| Google Drive sync is manual | Add a scheduled cron job calling `ingest_drive.py` |
