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
│   Deduplication guard in agent_node:                            │
│   • blocks search_company_documents being called > once         │
│   • prevents repeated identical tool+query pairs               │
│   • re-invokes LLM without tools if response content is empty  │
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
│  *.pdf *.docx     │  ← DocxReader (llama-index-readers-file)
│  *.xlsx *.csv     │
│  *.txt            │
└───────────────────┘
```

---

## The ReAct Agent Loop

The entire agent is a two-node LangGraph `StateGraph`. There are no other nodes — no classifier, no pre-router, no if/else logic.

```
User question
     │
     ▼
FastAPI POST /ask
     │
     ▼
aask(question, history)   ← workflow.py
     │
     ▼
build_graph(all_tools)    ← fresh graph per request
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
QuestionResponse → Streamlit
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
- **Web search caveats** — quote the value, not the date shown in snippets

This is backed up at the code level by the **deduplication guard** (see below) — the prompt alone is not reliable enough.

---

## Deduplication Guard

The LLM was observed calling `search_company_documents` 4+ times per question with slightly different queries ("Shivam skills", "Shivam profile", "Shivam", ...). Each call consumes a Groq API request, burning through the 30 RPM free-tier limit in seconds.

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

This is a code-level guarantee — the model cannot bypass it regardless of what the prompt says.

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

## The 7 Answer Paths

Every API response includes a `datasource` field:

```
datasource = "direct_llm"     → LLM answered from training data / live date prompt
datasource = "company_docs"   → search_company_documents / summarise_document / extract_structured_data
datasource = "database"       → query_company_database was called
datasource = "web_search"     → search_web was called
datasource = "calculation"    → calculate was called
datasource = "chart"          → generate_chart was called
datasource = "multiple"       → more than one tool category was used
```

### Path 1 — Direct LLM (including date/time)
```
User: "What is a vector database?"  OR  "What day is today?"
LLM:  Answers directly — system prompt contains live date/time
→ datasource: "direct_llm"
```

### Path 2 — Company Documents (RAG)
```
User: "What are Shivam's technical skills?"
LLM:  Calls search_company_documents("Shivam technical skills")
      Tool → LlamaIndex vector store (indexed with DocxReader) → top 4 chunks
LLM:  Synthesises answer, cites filename
→ datasource: "company_docs"
```

### Path 3 — Database
```
User: "What is the status of order #12345?"
LLM:  Calls list_database_tables() → calls describe_database_table("orders")
      Calls query_company_database("SELECT * FROM orders WHERE id = 12345")
LLM:  Formats result, shows SQL in code block
→ datasource: "database"
```

### Path 4 — Web Search
```
User: "What is the current USD to INR exchange rate?"
LLM:  Calls search_web("USD to INR exchange rate today")
      → TavilySearch returns structured results with URLs
LLM:  Extracts rate value, cites source URL
→ datasource: "web_search"
```

### Path 5 — Calculation
```
User: "What is 15% of 85000?"
LLM:  Calls calculate("85000 * 0.15") → "85000 * 0.15 = 12750"
→ datasource: "calculation"
```

### Path 6 — Chart
```
User: "Show monthly sales as a bar chart: Jan=1200, Feb=1500"
LLM:  Calls generate_chart(data_json='[...]', chart_type='bar', title='Monthly Sales')
Streamlit: renders Plotly chart inline in the chat
→ datasource: "chart"
```

---

## Conversation Memory

The agent supports multi-turn conversation via `session_id`. FastAPI stores message history per session in `_sessions` (in-memory dict). On each request:

1. FastAPI reconstructs prior `HumanMessage` / `AIMessage` objects
2. `aask(question, history=history)` prepends history before the new question
3. `_build_system_prompt()` is called fresh — date is always current
4. After the answer, FastAPI stores the new exchange

Session history is capped at 20 turns (40 messages) to stay within context limits.

---

## Citations

Every response includes a `citations` list. `_extract_citations()` walks all `ToolMessage` objects in the final state:

| Tool | Citation source |
|---|---|
| `search_company_documents` | `[Source: filename]` markers in tool output |
| `search_web` | `[Source: url]` markers in tool output |
| `query_company_database` | The SQL query used, shown as `detail` |
| `calculate` | The expression and result |
| `summarise_document` | The filename from `[Full text of ...]` prefix |
| `extract_structured_data` | `[Source: filename]` markers |
| `list_database_tables` / `describe_database_table` | `"schema lookup via <tool_name>"` |

---

## Data Stores

### Vector Store (for documents)

```
data/                          ← put your files here
  ├── report.pdf
  ├── handbook.docx            ← parsed by DocxReader (requires llama-index-readers-file)
  ├── catalog.xlsx
  └── notes.txt
       │
       │  python -m src.agent.rag  OR  POST /upload from Streamlit
       ▼
indexing_data/                 ← auto-generated, do not edit
  ├── default__vector_store.json
  ├── docstore.json
  ├── index_store.json
  └── ...
```

- **Supported formats:** PDF, DOCX, DOC, XLSX, XLS, CSV, TXT
- **DOCX parsing:** `DocxReader` from `llama-index-readers-file` (backed by `docx2txt`)
- **Embedding model:** `BAAI/bge-small-en-v1.5` — runs locally, no API key needed
- **Chunk size:** 512 tokens, 50 token overlap
- **Cache invalidation:** `get_vector_index.cache_clear()` is called after every index rebuild, so new files are searchable immediately on the next query

### Company Database

- **Location:** `data/company.db`
- **Technology:** SQLite (swappable with Postgres via real MCP server)
- **Access:** Read-only — `query_company_database` rejects any SQL that is not a `SELECT`

---

## File Responsibilities

| File | Layer | What it does |
|---|---|---|
| `streamlit_app.py` | UI | Chat interface, session state, datasource badges, citation pills, Plotly chart rendering, document upload, sidebar sample questions |
| `app.py` | API | FastAPI — `/ask`, `/health`, `/upload`, `/documents`, `/sessions/*`, in-memory session store |
| `workflow.py` | Agent | Builds the LangGraph, dynamic system prompt with live date, deduplication guard, agent node, tool node, routing, citation/chart extraction, result parsing |
| `chains.py` | LLM | Factory: auto-selects Groq / Gemini / Cohere; provides `TavilySearch` web tool |
| `tools.py` | Tools | 6 local tools: `search_company_documents`, `summarise_document`, `extract_structured_data`, `search_web`, `calculate`, `generate_chart` |
| `rag.py` | RAG | Auto-discovers files in `data/`, registers `DocxReader`, builds/loads LlamaIndex vector store |
| `mcp_client.py` | DB | Three database tools behind an MCP-compatible `asynccontextmanager` |
| `config.py` | Config | `DATA_DIR`, `INDEX_DIR` paths; env key validation |
| `schemas.py` | Models | `QuestionRequest` (with `session_id`), `QuestionResponse` (with `citations`, `chart_data`) |
| `ingest_drive.py` | Ingestion | Optional: pulls documents from Google Drive into the vector store |

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

> **Note on Groq rate limits:** The free tier allows 30 RPM. The deduplication guard and `parallel_tool_calls=False` keep most questions within 2–3 API calls.

---

## Technology Stack

| Technology | Role |
|---|---|
| **LangGraph** | Stateful ReAct agent loop (two-node StateGraph) |
| **LangChain** | `@tool` decorator, `ToolNode`, `BaseChatModel` interface |
| **LlamaIndex** | Document ingestion, chunking, HuggingFace embeddings, persisted vector store |
| **llama-index-readers-file** | `DocxReader` for proper Word document text extraction |
| **FastAPI** | Async HTTP API with Pydantic request/response validation; in-memory session store |
| **Streamlit** | Chat UI with session memory, datasource badges, citation pills, Plotly charts, document upload |
| **Groq** | Default LLM — `openai/gpt-oss-20b`, fast free-tier, tool-calling support |
| **Google Gemini** | Alternative LLM — `gemini-1.5-flash` |
| **Cohere** | Alternative LLM — `command-r-plus` |
| **langchain-tavily** | Real-time web search (`TavilySearch`) |
| **HuggingFace** | `BAAI/bge-small-en-v1.5` local embedding model |
| **Plotly** | Interactive chart generation and rendering |
| **SQLite** | Company database — swappable to Postgres via MCP |
| **docx2txt** | Underlying DOCX text extractor used by DocxReader |

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

Add the extension to `SUPPORTED_EXTENSIONS` in `src/agent/rag.py` and register its reader in `_get_file_extractors()`:

```python
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt", ".md"}
```

### Replace SQLite with Postgres

Replace `mcp_server_context()` in `mcp_client.py` with a real MCP server connection. `workflow.py` does not need to change.
