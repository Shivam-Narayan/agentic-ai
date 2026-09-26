# Architecture Overview

## What this system is

DataDialogue is an **Enterprise Knowledge Transfer Assistant** — a conversational AI agent that answers questions about your company by searching internal documents, querying a structured database, or looking up the live web, with real-time streaming responses across multiple channels.

The key design principle: **there is no hard-coded routing**. The LLM reads the available tool schemas and decides at runtime which tool(s) to use. Adding a new data source means writing one Python function — nothing else changes.

---

## High-Level System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          ACCESS CHANNELS                                 │
│                                                                          │
│  ┌──────────────────────┐   ┌──────────────────────────────────────────┐ │
│  │   Streamlit Chat UI  │   │  OpenClaw Webhook                        │ │
│  │  (streamlit_ui.py)   │   │  (any channel: WhatsApp, Discord, Slack) │ │
│  │  http://localhost    │   │                                          │ │
│  │  :8501               │   │                                          │ │
│  └──────────┬───────────┘   └────────────────────┬─────────────────────┘ │
└─────────────│────────────────────────────────────│────────────────────────┘
              │ GET /stream (SSE)                  │ POST /openclaw/webhook
              │ session_id=<uuid>                  │ session_id=<oc_session>
              │ (token-by-token)                   │
              └────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND (api.py)                         │
│                          http://localhost:8000                           │
│                                                                          │
│  Security: _validate_session_id() — regex on all session_id inputs      │
│  Safety:   _MAX_UPLOAD_BYTES = 50MB upload limit                        │
│  Memory:   Lazy AsyncPostgresSaver import in lifespan()                 │
│                                                                          │
│  GET  /stream             →  KnowledgeTransferAgent.run() SSE stream    │
│  POST /ask                →  aask(question, session_id, checkpointer)   │
│  GET  /health             →  liveness & active backend status           │
│  POST /upload             →  add_documents_to_index() (incremental)     │
│  GET  /documents          →  list indexed files                         │
│  GET  /sessions           →  list active sessions                       │
│  GET  /sessions/{id}/hist →  session turn history                       │
│  DELETE /sessions/{id}/…  →  reset conversation memory                  │
│  GET  /openclaw/health    →  OpenClaw health check                      │
│  POST /openclaw/webhook   →  aask() via OpenClaw session                │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│               LANGGRAPH PLANNER–EXECUTOR–REFLECTION LOOP                 │
│                          (workflow.py)                                   │
│                                                                          │
│   START ──► [complexity_router]                                          │
│               │ complex              │ simple                           │
│               ▼                      │                                  │
│          [planner node]              │                                  │
│          numbered plan               │                                  │
│               └──────────► [agent node] ◄──────── system prompt        │
│                                 │   ▲            + live date/time      │
│                         tool    │   │                                   │
│                         calls?  │   │                                   │
│                           │  yes│   │ loop                              │
│                           ▼    ─┘   │                                   │
│                       [tool node] ──┘                                   │
│                           │  no tool calls                              │
│                           ▼                                              │
│                    [reflection node]                                     │
│                    PASS → answer returned                                │
│                    IMPROVED → rewrite + return                          │
│                                                                          │
│  Guards:  dedup (same tool+args blocked in same turn)                   │
│           parallel_tool_calls=False (one tool at a time)                │
│           asyncio.wait_for() — 90s LLM, 60s tool timeouts              │
│           UsageTracker — tokens + cost per request                      │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
              ┌───────────────┼──────────────────┐
              ▼               ▼                  ▼
┌─────────────────────┐  ┌──────────────┐  ┌───────────────────────────┐
│  RETRIEVAL PIPELINE │  │  SQLITE DB   │  │  WEB SEARCH               │
│  (rag.py)           │  │  .storage/   │  │  Tavily → Serper → DDG    │
│                     │  │  data/*.db   │  └───────────────────────────┘
│  Stage 1: Hybrid    │  └──────────────┘
│  ┌───────────────┐  │
│  │ Semantic      │  │
│  │ (BAAI/bge)    │  │
│  └───────┬───────┘  │
│  ┌───────┴───────┐  │
│  │ BM25 keyword  │  │
│  └───────┬───────┘  │
│          │ RRF      │
│  Stage 2: Reranking │
│  ┌───────────────┐  │
│  │ FlashRank     │  │
│  │ cross-encoder │  │
│  └───────────────┘  │
│  top-8 chunks       │
└─────────────────────┘
        ▲
        │ indexed from (chunk_size=1024)
┌─────────────────────┐       ┌────────────────────────┐
│ .storage/data/      │       │ .storage/memory_store/ │
│  *.pdf *.docx       │       │  conversations.db      │
│  *.xlsx *.csv *.txt │       │  (AsyncSqliteSaver)    │
└─────────────────────┘       └────────────────────────┘
```

---

## Retrieval Pipeline (3-Stage)

The document retrieval system uses a 3-stage pipeline controlled by feature flags in `.env`:

```
User question
      │
      ├── Stage 1a: Semantic retrieval (always on)
      │   BAAI/bge-small-en-v1.5 embeddings, cosine similarity → top-8 chunks
      │
      ├── Stage 1b: BM25 keyword retrieval (USE_HYBRID_SEARCH=true)
      │   Term frequency matching → top-8 chunks
      │
      │   Both lists fused via Reciprocal Rank Fusion → top-8 candidates
      │
      └── Stage 2: FlashRank cross-encoder reranking (USE_RERANKER=true)
          Jointly encodes (question, chunk) pairs → final top-8 in precision order
```

| Flag | Default | Effect |
|---|---|---|
| `USE_HYBRID_SEARCH=true` | true | Adds BM25 + RRF on top of semantic |
| `USE_RERANKER=true` | true | FlashRank cross-encoder after retrieval |

**Why this matters:**
- Semantic alone misses exact IDs, codes, numbers
- BM25 alone misses conceptual/paraphrase queries
- Cross-encoder reranking places the most relevant chunk at position #1, directly improving LLM faithfulness scores

---

## Channel Architecture

Each channel maps to its own session namespace — conversation memory never leaks between channels.

### Channel 1 — Streamlit Web UI

```
Browser → streamlit_ui.py
        → GET /stream?question=...&session_id=<uuid>  (SSE)
        → KnowledgeTransferAgent.run() async generator
        → token events streamed word-by-word
        → done event carries datasource + citations + chart_data
```

- UUID session per browser tab — persists across page refreshes
- Live token streaming with `▌` blinking cursor
- Tool-use indicator during tool calls
- Interactive Plotly charts inline on `done` event
- Datasource badges: 📄 🌐 🧮 📊 🗄️

### Channel 2 — OpenClaw Webhook

```
WhatsApp / Discord / Slack
        → OpenClaw Gateway
        → POST /openclaw/webhook {channel, user_id, session_id, message}
        → aask() → LangGraph agent
        → OpenClawWebhookResponse → back to originating channel
```

---

## Real-Time Streaming

`KnowledgeTransferAgent` in `workflow.py` is an async generator yielding SSE events:

```
GET /stream
  yield {"type": "status",     "stage": "thinking"}
  yield {"type": "status",     "stage": "planning"}   ← complex questions only
  yield {"type": "plan",       "steps": [...]}
  yield {"type": "tool",       "name": "search_company_documents"}
  yield {"type": "token",      "text": "The annual report..."}   ← per word/chunk
  yield {"type": "reflection", "status": "pass"|"improved"}
  yield {"type": "done",       "payload": {datasource, citations, usage_metrics, ...}}
```

---

## LangSmith Observability

Every agent run is traced automatically when `LANGCHAIN_TRACING_V2=true`:

```
Request
  │
  ├── LangSmith trace created (run_id)
  │    ├── planner_node invocation + tokens
  │    ├── agent_node invocations + tokens
  │    ├── tool calls (names, args, results)
  │    ├── reflection_node + tokens
  │    └── total cost, latency, model
  │
  └── View at https://smith.langchain.com/ → project: DataDialogue
```

**Required `.env` vars:**
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_PROJECT=DataDialogue
```

---

## Cost & Token Tracking

Every request accumulates token usage through `UsageTracker` in `telemetry.py`:

```
Agent run
  ├── planner_node: tracker.record(response)
  ├── agent_node:   tracker.record(response)  ← per LLM call
  └── reflection_node: tracker.record(response)

tracker.to_metrics() → UsageMetrics {
    prompt_tokens, completion_tokens, total_tokens,
    cost_usd, model, latency_ms, llm_calls
}
```

Cost calculated per-model using `_COST_TABLE` in `telemetry.py` (Groq, Gemini, OpenAI, Azure, Cohere). Fallback price for unknown models.

---

## Web Search Fallback Chain

```
search_web("query")
        │
        ▼ try TavilySearch (TAVILY_API_KEY set)
        │   ✓ result → done
        │   ✗ quota / {"error": ...} dict →
        ▼ try GoogleSerperRun (SERPER_API_KEY set)
        │   ✓ result → done
        │   ✗ error →
        ▼ DuckDuckGoSearchRun
            ✓ always available (no key needed)
```

---

## Conversation Memory

```
Request → AsyncSqliteSaver.aget_tuple(thread_id=session_id)
        → LangGraph loads full history
        → Agent runs with context
        → State automatically saved after each run
        → .storage/memory_store/conversations.db
```

Survives server restarts. Shared across all channels via same `session_id`.

---

## The 9 Tools

| Tool | Source | Triggers when… |
|---|---|---|
| `search_company_documents` | `tools.py` | Question is about internal documents |
| `summarise_document` | `tools.py` | User asks for a file overview or summary |
| `extract_structured_data` | `tools.py` | User wants specific fields pulled from docs |
| `search_web` | `tools.py` | Question needs real-time or external info |
| `calculate` | `tools.py` | Any arithmetic, percentages, computation |
| `generate_chart` | `tools.py` | User asks for a chart or visualisation |
| `list_database_tables` | `mcp_tools.py` | LLM needs to discover available tables |
| `describe_database_table` | `mcp_tools.py` | LLM needs column names before querying |
| `query_company_database` | `mcp_tools.py` | Question requires structured DB data |

---

## The 7 Answer Paths

```
datasource = "direct_llm"    → LLM answered from training data / live date
datasource = "company_docs"  → search_company_documents / summarise / extract
datasource = "database"      → query_company_database called
datasource = "web_search"    → search_web called
datasource = "calculation"   → calculate called
datasource = "chart"         → generate_chart called
datasource = "multiple"      → more than one tool category used
```

---

## Evaluation & Monitoring

### RAGAS Evaluation

```bash
python tests/evaluate.py                         # all 18 questions
python tests/evaluate.py --save-baseline         # save as baseline
python tests/evaluate.py --compare eval_baseline.json  # regression report
python tests/evaluate.py --datasource company_docs     # filter by type
```

Metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
Pass threshold: `0.70`

Each run reports per-question cost, tokens, latency, and model used.

### Regression Comparison

Baseline saved to `tests/eval_baseline.json`. Future runs auto-compare:
- 📈 Metric improved > 1%
- 📉 Metric regressed > 1%
- 💰/💚 Cost increased/decreased

### Monitoring Thresholds (`tests/monitoring_config.json`)

| Category | Warning | Critical |
|---|---|---|
| RAGAS metrics | < 0.75 | < 0.65 |
| P95 latency | > 15s | > 30s |
| Daily cost | > $10 | > $25 |
| Avg tokens/request | > 8,000 | > 12,000 |

See `docs/MONITORING_GUIDE.md` for full alerting playbook.

---

## Data Stores

```
.storage/
  ├── data/                    ← user documents (PDF, DOCX, XLSX, CSV, TXT)
  ├── indexing_data/           ← LlamaIndex JSON vector store (auto-generated)
  └── memory_store/
        └── conversations.db   ← SQLite, AsyncSqliteSaver, per session_id
```

Optional Postgres backends (set `USE_PGVECTOR=true`, `USE_POSTGRES_MEMORY=true`):
- `document_embeddings` table — pgvector, 384-dim, IVFFlat index
- LangGraph checkpoint tables — created by `AsyncPostgresSaver.setup()`

---

## Technology Stack

| Technology | Role |
|---|---|
| **LangGraph** | Stateful Planner→Executor→Reflection graph |
| **LangGraph AsyncSqliteSaver** | Persistent conversation memory per session |
| **LangChain** | `@tool` decorator, `ToolNode`, `BaseChatModel` |
| **LlamaIndex** | Document ingestion, chunking (1024 tokens), embeddings, vector store |
| **BAAI/bge-small-en-v1.5** | Local HuggingFace embedding model, 384-dim |
| **BM25Retriever** | Keyword retrieval (llama-index-retrievers-bm25) |
| **QueryFusionRetriever** | Reciprocal Rank Fusion of semantic + BM25 |
| **FlashRankRerank** | Local cross-encoder reranking (no API key, no GPU) |
| **FastAPI** | Async HTTP API with SSE streaming |
| **Streamlit** | Web chat UI — live token streaming, Plotly, upload |
| **Groq** | Default LLM — `openai/gpt-oss-120b`, streaming |
| **Google Gemini** | Alternative LLM — `gemini-1.5-flash`, streaming |
| **Cohere** | Alternative LLM — `command-r-plus`, streaming |
| **Tavily / Serper / DuckDuckGo** | Web search fallback chain |
| **Plotly** | Interactive chart generation |
| **RAGAS** | LLM-as-judge evaluation metrics |
| **LangSmith** | Distributed tracing, cost tracking, dashboards |
| **FlashRank** | Local cross-encoder model (ms-marco-MiniLM-L-12-v2) |
| **SQLite** | Company DB + conversation memory store |
| **PostgreSQL + pgvector** | Optional production backend |

---

## Security Controls

| Control | Location | What it prevents |
|---|---|---|
| `_validate_session_id()` regex | `api.py` | Path traversal, injection via session IDs |
| `_MAX_UPLOAD_BYTES = 50MB` | `api.py` | Memory exhaustion from large uploads |
| `_sanitise_identifier()` ASCII regex | `mcp_tools.py` | Unicode-based SQL injection |
| `_is_blocked_statement()` | `mcp_tools.py` | DDL execution (DROP, TRUNCATE, ALTER) |
| `ALLOW_DB_WRITES=false` default | `mcp_tools.py` | Write queries require explicit opt-in |
| Lazy `AsyncPostgresSaver` import | `api.py` | Prevents crash for non-Postgres users |
| `_checkpointer is None` guard | `api.py` | Prevents runtime crash before init |

---

## Extending the System

### Add a new tool

```python
# 1. Define in src/tools/tools.py
@tool
def get_jira_ticket(ticket_id: str) -> str:
    """Look up a Jira ticket by ID (e.g. 'PROJ-123')."""
    ...

# 2. Add to LOCAL_TOOLS list in src/agent/workflow.py
```

No routing changes needed — the LLM picks it up automatically.

### Add a new channel

1. Create `src/apps/your_channel.py`
2. Call `POST /ask` with prefix `your_channel_<user_id>` as session_id
3. Memory persists automatically per session

### Switch to Postgres backend

```env
USE_PGVECTOR=true
USE_POSTGRES_MEMORY=true
POSTGRES_URL=postgresql+psycopg://user:pass@host:5432/datadialogue
```

Run `python migrate_to_pgvector.py` once to migrate existing vectors.
