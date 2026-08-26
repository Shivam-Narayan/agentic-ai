# Architecture Overview

## What this system is

The KT Agent is an **Enterprise Knowledge Transfer Assistant**. It is a conversational AI that answers questions about your company by searching internal documents, querying a structured database, or looking up the live web — all from a single chat interface.

The key design principle: **there is no hard-coded routing**. The LLM itself reads the available tools and decides at runtime which one(s) to use. Adding a new data source means writing one Python function — nothing else changes.

---

## High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER'S BROWSER                           │
│                    http://localhost:8501                         │
│                  ┌──────────────────────┐                       │
│                  │   Streamlit Chat UI  │                       │
│                  │   (streamlit_app.py) │                       │
│                  └──────────┬───────────┘                       │
└─────────────────────────────│───────────────────────────────────┘
                              │  HTTP POST /ask  (question + session_id)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND                          │
│                    http://localhost:8000                         │
│                       (app.py)                                  │
│   POST /ask  ──►  aask(question, history)                       │
│   GET  /health                                                  │
│   POST /upload                                                  │
│   GET  /documents                                               │
│   GET  /sessions/{id}/history                                   │
│   DELETE /sessions/{id}/history                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH AGENT LOOP                         │
│                      (workflow.py)                              │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  AgentState = { messages: [SystemMessage, ...history,   │   │
│   │                             HumanMessage, ...] }        │   │
│   │                                                         │   │
│   │   START ──► Agent Node ──► (has tool calls?)            │   │
│   │               ▲                │                        │   │
│   │               │      YES ──►  Tool Node                 │   │
│   │               └───────────────┘                        │   │
│   │                       NO ──► END                        │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Local tools (tools.py):                                       │
│   ┌──────────────────────┐  ┌──────────────────────────────┐   │
│   │ search_company_docs  │  │ summarise_document           │   │
│   │ (LlamaIndex RAG)     │  │ extract_structured_data      │   │
│   └──────────────────────┘  └──────────────────────────────┘   │
│   ┌──────────────────────┐  ┌──────────────────────────────┐   │
│   │ search_web           │  │ calculate                    │   │
│   │ (Tavily API)         │  │ generate_chart (Plotly)      │   │
│   └──────────────────────┘  └──────────────────────────────┘   │
│                                                                 │
│   MCP tools (mcp_client.py):                                    │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ list_database_tables  describe_database_table            │  │
│   │ query_company_database                                   │  │
│   └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────┘
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
┌───────────────────┐
│   data/ folder    │
│  *.pdf *.docx     │
│  *.xlsx *.csv     │
│  *.txt            │
└───────────────────┘
```

---

## The ReAct Agent Loop

The entire agent is a two-node LangGraph `StateGraph`. There are no other nodes — no classifier, no pre-router, no if/else logic.

```mermaid
flowchart TD
    U[User via Streamlit] --> API["FastAPI\nPOST /ask"]
    API --> AASK["aask(question, history)\nworkflow.py"]

    AASK --> BUILD["build_graph(all_tools)\nCreates fresh graph per request"]
    BUILD --> START

    subgraph Graph ["LangGraph StateGraph"]
        START([START]) --> AGENT["Agent Node\nSystemMessage + history + LLM + bound tools"]
        AGENT -- "tool_calls present" --> TOOLS["Tool Node\nExecutes Python function"]
        TOOLS -- "ToolMessage appended to state" --> AGENT
        AGENT -- "no tool_calls → final answer" --> ENDNODE([END])
    end

    TOOLS -.->|calls| RAG["search_company_documents\nrag.py → LlamaIndex vector store"]
    TOOLS -.->|calls| SUMM["summarise_document\nrag.py → full document text"]
    TOOLS -.->|calls| EXTRACT["extract_structured_data\nrag.py → field extraction"]
    TOOLS -.->|calls| DB["query_company_database\nlist_database_tables\ndescribe_database_table\nmcp_client.py → SQLite"]
    TOOLS -.->|calls| WEB["search_web\nchains.py → Tavily API"]
    TOOLS -.->|calls| CALC["calculate\nSafe AST evaluator"]
    TOOLS -.->|calls| CHART["generate_chart\nPlotly JSON figure"]

    ENDNODE --> PARSE["parse_result()\nExtracts datasource, tools_used\ncitations, chart_data"]
    PARSE --> RESP["QuestionResponse\nreturned to Streamlit"]
```

### Why two nodes?

The LLM is not just the "brain" — it is also the router. It receives all tool schemas (auto-generated from their docstrings) alongside the user's question plus a system prompt and the full conversation history. It decides:

- **No tools needed** → emits a final answer text → graph goes to END
- **One or more tools needed** → emits a `tool_calls` list → graph executes the tools, appends results to message history, and loops back to the agent

This loop continues until the LLM stops calling tools and produces a final answer. Most questions resolve in 1–2 loops.

---

## The System Prompt

Every agent invocation injects a `SystemMessage` as the first message. It defines the agent's persona and enforces key behaviours:

- **Citations** — always cite source filenames, SQL queries, or URLs
- **Accuracy** — use the `calculate` tool for all arithmetic (never compute in the LLM head)
- **Charts** — proactively offer charts when presenting tabular or numeric results
- **Extraction** — use `extract_structured_data` for pulling specific fields from documents
- **Summaries** — use `summarise_document` for overview/summary requests
- **Memory** — use conversation history to handle follow-up questions

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

The LLM learns what each tool does from its **docstring**. Well-written docstrings are critical to correct routing.

---

## The 6 Answer Paths

Every API response includes a `datasource` field that tells you how the answer was produced:

```
datasource = "direct_llm"     → LLM answered from training data, no tools used
datasource = "company_docs"   → search_company_documents / summarise_document / extract_structured_data
datasource = "database"       → query_company_database was called
datasource = "web_search"     → search_web was called
datasource = "calculation"    → calculate was called
datasource = "chart"          → generate_chart was called
datasource = "multiple"       → more than one tool category was used
```

### Path 1 — Direct LLM
```
User: "What is a vector database?"
LLM:  Answers directly — no tool_calls emitted
→ datasource: "direct_llm"
```

### Path 2 — Company Documents (RAG)
```
User: "What does the KT document say about deployment?"
LLM:  Calls search_company_documents("deployment process")
      Tool → LlamaIndex vector store → top 4 chunks with [Source: filename] markers
LLM:  Synthesises answer, cites filename
→ datasource: "company_docs"
```

### Path 3 — Database
```
User: "What is the status of order #12345?"
LLM:  Calls list_database_tables() → sees "orders" table
      Calls describe_database_table("orders") → sees column names
      Calls query_company_database("SELECT * FROM orders WHERE id = 12345")
LLM:  Formats row as natural language, shows SQL in code block
→ datasource: "database"
```

### Path 4 — Web Search
```
User: "What is the current USD to INR exchange rate?"
LLM:  Calls search_web("USD to INR exchange rate today")
      → Tavily returns results with URLs
LLM:  Extracts rate, cites URL
→ datasource: "web_search"
```

### Path 5 — Calculation
```
User: "What is 15% of 85000?"
LLM:  Calls calculate("85000 * 0.15") → "85000 * 0.15 = 12750"
LLM:  Returns the result
→ datasource: "calculation"
```

### Path 6 — Chart
```
User: "Show monthly sales as a bar chart: Jan=1200, Feb=1500"
LLM:  Calls generate_chart(data_json='[...]', chart_type='bar', title='Monthly Sales')
      → returns CHART_JSON::{plotly figure JSON}
Streamlit: renders Plotly chart inline in the chat
→ datasource: "chart"
```

---

## Conversation Memory

The agent supports multi-turn conversation via `session_id`. FastAPI stores message history per session in `_sessions` (an in-memory dict). On each request:

1. FastAPI calls `_get_lc_history(session_id)` to reconstruct prior `HumanMessage` / `AIMessage` objects
2. `aask(question, history=history)` prepends the history before the new question in `AgentState`
3. The agent node prepends `SystemMessage` to the full history on every turn
4. After the answer, FastAPI calls `_append_to_session()` to store the new exchange

Session history is capped at 20 turns (40 messages) to stay within context limits.

---

## Citations

Every response includes a `citations` list. The `_extract_citations()` function walks all `ToolMessage` objects in the final state and extracts:

| Tool | Citation source |
|---|---|
| `search_company_documents` | `[Source: filename]` markers in tool output |
| `search_web` | `[Source: url]` markers in tool output |
| `query_company_database` | The SQL query used, shown as `detail` |
| `calculate` | The expression and result |
| `summarise_document` | The filename extracted from `[Full text of ...]` prefix |
| `extract_structured_data` | `[Source: filename]` markers |
| `list_database_tables` / `describe_database_table` | `"schema lookup via <tool_name>"` |

Citations are rendered in the Streamlit UI as clickable pills under each assistant message.

---

## Data Stores

### Vector Store (for documents)

```
data/                          ← you put your files here
  ├── report.pdf
  ├── handbook.docx
  ├── catalog.xlsx
  ├── sales.csv
  └── notes.txt
       │
       │  python -m src.agent.rag  (run once to index)
       ▼
indexing_data/                 ← auto-generated, do not edit
  ├── default__vector_store.json
  ├── docstore.json
  ├── index_store.json
  ├── graph_store.json
  └── image__vector_store.json
```

- **Supported formats:** PDF, DOCX, DOC, XLSX, XLS, CSV, TXT
- **Embedding model:** `BAAI/bge-small-en-v1.5` — runs locally, no API key needed
- **Chunk size:** 512 tokens, 50 token overlap
- **Implementation:** LlamaIndex `VectorStoreIndex` with file-based persistence
- **Loading:** `@lru_cache` — index is loaded from disk once per server process

### Company Database

- **Location:** `data/company.db`
- **Technology:** SQLite (swappable with Postgres via real MCP server)
- **Access:** Read-only — `query_company_database` rejects any SQL that is not a `SELECT`
- **Interface:** MCP-compatible `asynccontextmanager` in `mcp_client.py`

---

## File Responsibilities

| File | Layer | What it does |
|---|---|---|
| `streamlit_app.py` | UI | Chat interface, session state, datasource badges, citation pills, Plotly chart rendering, document upload, sidebar sample questions |
| `app.py` | API | FastAPI — `/ask`, `/health`, `/upload`, `/documents`, `/sessions/*`, in-memory session store |
| `workflow.py` | Agent | Builds the LangGraph, system prompt, agent node, tool node, routing, citation/chart extraction, result parsing |
| `chains.py` | LLM | Factory: auto-selects Groq / Gemini / Cohere; provides Tavily tool |
| `tools.py` | Tools | 6 local tools: `search_company_documents`, `summarise_document`, `extract_structured_data`, `search_web`, `calculate`, `generate_chart` |
| `rag.py` | RAG | Auto-discovers files in `data/`, builds/loads LlamaIndex vector store |
| `mcp_client.py` | DB | Three database tools behind an MCP-compatible `asynccontextmanager` |
| `config.py` | Config | `DATA_DIR`, `INDEX_DIR` paths; env key validation |
| `schemas.py` | Models | `QuestionRequest` (with `session_id`), `QuestionResponse` (with `citations`, `chart_data`) |
| `ingest_drive.py` | Ingestion | Optional: pulls documents from Google Drive into the vector store |

---

## Multi-LLM Strategy

The system is **vendor-agnostic at the agent level**. All LLM providers implement LangChain's `BaseChatModel` so `workflow.py` never references a specific provider.

```
.env keys present         →  LLM selected
──────────────────────────────────────────
GROQ_API_KEY              →  Groq  (openai/gpt-oss-120b)   ← default
GOOGLE_API_KEY            →  Gemini (gemini-1.5-flash)
COHERE_API_KEY            →  Cohere (command-r-plus)
LLM_PROVIDER=google       →  forces Google regardless of other keys
```

Priority order when multiple keys are present: **Groq → Google → Cohere**

---

## Technology Stack

| Technology | Role |
|---|---|
| **LangGraph** | Stateful ReAct agent loop (two-node StateGraph) |
| **LangChain** | `@tool` decorator, `ToolNode`, `BaseChatModel` interface |
| **LlamaIndex** | Document ingestion, chunking, HuggingFace embeddings, persisted vector store |
| **FastAPI** | Async HTTP API with Pydantic request/response validation; in-memory session store |
| **Streamlit** | Chat UI with session memory, datasource badges, citation pills, Plotly charts, document upload |
| **Groq** | Default LLM — `openai/gpt-oss-120b`, fast free-tier, tool-calling support |
| **Google Gemini** | Alternative LLM — `gemini-1.5-flash` |
| **Cohere** | Alternative LLM — `command-r-plus`, strong at RAG tasks |
| **Tavily** | Real-time web search API |
| **HuggingFace** | `BAAI/bge-small-en-v1.5` local embedding model (no API key needed) |
| **Plotly** | Interactive chart generation and rendering |
| **SQLite** | Company database — swappable to Postgres via MCP |
| **openpyxl** | Excel file parsing (`.xlsx`, `.xls`) |

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

No routing changes. The LLM starts using it automatically.

### Add a new document type

Add the extension to `SUPPORTED_EXTENSIONS` in `src/agent/rag.py`:

```python
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt", ".md"}
```

### Replace SQLite with Postgres

Replace `mcp_server_context()` in `mcp_client.py` with a real MCP server connection. `workflow.py` does not need to change.
