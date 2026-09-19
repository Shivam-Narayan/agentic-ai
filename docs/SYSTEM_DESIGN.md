# System Design

## Overview

This document covers the internal design of every component in DataDialogue. It is intended for developers who want to understand how the system works, modify it, or extend it.

For the high-level architecture and data flow diagram, see [ARCHITECTURE.md](ARCHITECTURE.md).  
For production monitoring and alerting, see [MONITORING_GUIDE.md](MONITORING_GUIDE.md).

---

## Component Map

```
.env
 └── src/core/config.py  ────────────────────────────────────────────────┐
      USE_HYBRID_SEARCH, USE_RERANKER, USE_PGVECTOR,                    │
      USE_POSTGRES_MEMORY, POSTGRES_URL, DATA_DIR, INDEX_DIR            │
                                                                         │
src/agent/chains.py  (LLM factory + web search)                         │
 ├── get_llm()            @lru_cache → ChatGroq / Gemini / Cohere       │
 ├── _FallbackSearchTool  → tries primary → fallbacks at query time     │
 └── get_web_search_tool() → Tavily → Serper → DuckDuckGo               │
                                                                         │
src/retrieval/rag.py  (3-stage retrieval pipeline)                      │
 ├── get_embed_model()     @lru_cache → BAAI/bge-small-en-v1.5          │
 ├── get_vector_index()    @lru_cache → JSON store or pgvector          │
 ├── _get_hybrid_retriever() → QueryFusionRetriever (semantic + BM25)   │
 ├── _rerank_nodes()        → FlashRankRerank cross-encoder             │
 ├── retrieve_documents()   → 3-stage pipeline (retrieve→fuse→rerank)  │
 ├── build_index()          → ingest, chunk (1024 tokens), embed, store │
 ├── rebuild_index()        → full rebuild + lru_cache.cache_clear()    │
 ├── add_documents_to_index() → incremental insert                      │
 └── discover_documents()   → scans DATA_DIR for supported files        │
                                                                         │
src/tools/tools.py  (6 local LangChain tools)                           │
 ├── search_company_documents → retrieve_documents() (top-8)            │
 ├── summarise_document       → full file text via LlamaIndex           │
 ├── extract_structured_data  → RAG + JSON field template               │
 ├── search_web               → _FallbackSearchTool                     │
 ├── calculate                → safe AST evaluator                      │
 └── generate_chart           → Plotly JSON figure                      │
                                                                         │
src/tools/mcp_tools.py  (3 database tools)                              │
 ├── list_database_tables   → discover tables + row counts              │
 ├── describe_database_table → column schemas + types                   │
 └── query_company_database  → SELECT (write if ALLOW_DB_WRITES=true)  │
                                                                         │
src/agent/workflow.py  (LangGraph orchestrator)  ◄── all of the above  │
 ├── AgentState TypedDict (messages, plan, usage_tracker, …)            │
 ├── UsageTracker integration (tokens + cost per request)               │
 ├── _compile_graph() / _get_compiled_graph() (cached)                  │
 ├── planner_node / agent_node / tool_execution_node / reflection_node  │
 ├── _request_tool_context() → bind_tools() per request                 │
 ├── KnowledgeTransferAgent  → SSE streaming interface                  │
 └── aask()                  → blocking interface                       │
                                                                         │
src/agent/telemetry.py  (cost + token tracking)                         │
 ├── UsageTracker → accumulates token counts across LLM calls           │
 ├── UsageMetrics → prompt_tokens, completion_tokens, cost_usd, …      │
 └── _COST_TABLE  → pricing per 1k tokens (Groq, Gemini, OpenAI, …)    │
                                                                         │
src/agent/schemas.py  (Pydantic models)                                  │
 ├── DatasourceType = Literal["direct_llm","company_docs","database",…] │
 ├── QuestionRequest / QuestionResponse / Citation                      │
 └── OpenClawWebhookRequest / Response / HealthResponse                 │
                                                                         │
src/apps/api.py  (FastAPI)                                               │
 ├── _validate_session_id()  → regex security guard                     │
 ├── _MAX_UPLOAD_BYTES = 50MB                                            │
 ├── Lazy AsyncPostgresSaver import inside lifespan()                   │
 ├── _checkpointer None guard on every endpoint                         │
 └── 11 endpoints (see ARCHITECTURE.md)                                 │
                                                                         │
tests/evaluate.py  (RAGAS evaluation + monitoring)                       │
 ├── 18 ground-truth questions (eval_questions.json)                    │
 ├── cost + token tracking per question                                  │
 ├── --save-baseline / --compare for regression detection               │
 └── LangSmith tracing check on startup                                 │
```

---

## `config.py` — Environment and Feature Flags

Single source of truth for paths, feature flags, and environment validation.

```python
ROOT_DIR    = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = ROOT_DIR / ".storage"
DATA_DIR    = STORAGE_DIR / "data"           # user document files
INDEX_DIR   = STORAGE_DIR / "indexing_data"  # LlamaIndex vector store
```

**Feature flags:**

```python
USE_HYBRID_SEARCH: bool   # semantic + BM25 + RRF   (default: true)
USE_RERANKER:      bool   # FlashRank cross-encoder (default: true)
USE_PGVECTOR:      bool   # pgvector backend         (default: false)
USE_POSTGRES_MEMORY: bool # Postgres conversation DB (default: false)
```

**LLM key validation (`require_runtime_keys()`):**

- Raises hard error if no LLM key found (GROQ, GOOGLE, or COHERE)
- Logs a warning if no web search key found — falls back to DuckDuckGo automatically, no crash

---

## `chains.py` — LLM Factory + Web Search

**LLM selection:**

```python
@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    # Priority: Groq → Google → Cohere
    # Respects LLM_PROVIDER env var for explicit override
    # All providers: temperature=0, streaming=True
```

`@lru_cache` means the LLM client is created once per process and reused.

**Web search fallback chain:**

`_FallbackSearchTool` wraps multiple providers and cascades at *query time*, not just at init. Critical because Tavily initialises successfully even when quota is exhausted — the failure only appears on the first query.

```python
for tool in [primary] + fallbacks:
    result = tool.invoke(query)
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(result["error"])  # Tavily quota error
    return result
```

---

## `rag.py` — 3-Stage Retrieval Pipeline

### Configuration

```python
_CHUNK_SIZE:        int = 1024   # was 512 — larger = more context per chunk
_CHUNK_OVERLAP:     int = 100    # was 50
_SIMILARITY_TOP_K:  int = 8      # chunks returned at each stage
_EMBED_MODEL_NAME:  str = "BAAI/bge-small-en-v1.5"   # 384-dim
```

### Stage 1 — Retrieval (always active)

```python
retriever = index.as_retriever(similarity_top_k=8)
# Cosine similarity between query embedding and chunk embeddings
```

### Stage 1 + BM25 Fusion (USE_HYBRID_SEARCH=true)

```python
def _get_hybrid_retriever(index):
    vector_retriever = index.as_retriever(similarity_top_k=8)
    bm25_retriever   = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=8)
    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        mode="reciprocal_rerank",   # RRF fusion
        num_queries=1,              # no LLM query expansion
    )
```

RRF combines both ranked lists: if a chunk ranks highly in either retriever it surfaces in the fused top-8.

### Stage 2 — Cross-Encoder Reranking (USE_RERANKER=true)

```python
def _rerank_nodes(nodes, question):
    reranker = FlashRankRerank(top_n=8)
    reranked = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(question))
    return reranked
```

**Why cross-encoder > bi-encoder for final ranking:**

| | Bi-encoder (embedding) | Cross-encoder (FlashRank) |
|---|---|---|
| Encodes | Question and chunk **separately** | Question + chunk **jointly** |
| Sees | "Are these vectors close?" | "Is this chunk a good answer?" |
| Speed | Fast — compare pre-built vectors | ~100-300ms per batch of 8 |
| Accuracy | Good for recall | Best for precision |

Use bi-encoder (Stage 1) to retrieve 8 candidates cheaply. Use cross-encoder (Stage 2) to rank them accurately.

**Graceful fallback:** If `flashrank` is not installed, `_rerank_nodes()` logs a warning and returns the original order — the app never crashes.

### Full retrieve_documents() pipeline

```python
def retrieve_documents(question: str) -> list[Document]:
    index = get_vector_index()

    if USE_HYBRID_SEARCH:
        retriever = _get_hybrid_retriever(index)   # semantic + BM25 + RRF
    else:
        retriever = index.as_retriever(similarity_top_k=8)

    nodes = retriever.retrieve(question)           # Stage 1

    if USE_RERANKER:
        nodes = _rerank_nodes(nodes, question)     # Stage 2

    return [Document(page_content=node.node.text, ...) for node in nodes]
```

### Index caching

Both `get_embed_model()` and `get_vector_index()` use `@lru_cache(maxsize=1)` — loaded once per process. `rebuild_index()` calls `get_vector_index.cache_clear()` after every rebuild so the next query loads the fresh index.

---

## `tools.py` — The 6 Local Tools

All tools use the `@tool` decorator, which auto-generates a JSON schema from the function signature and docstring. This schema is what gets sent to the LLM in every `bind_tools()` call.

### 1. `search_company_documents`
Calls `retrieve_documents(query)` → top-8 chunks from the 3-stage pipeline.  
Each chunk prefixed `[Source: filename]` for citation extraction.  
Capped at `MAX_EXTRACT_CHARS=700` per chunk, `MAX_CONTEXT_DOCS=4` shown to LLM.

### 2. `summarise_document`
Loads full file text (capped at 6000 chars). LLM synthesises summary in next reasoning step.

### 3. `extract_structured_data`
Retrieves relevant chunks + returns context with a JSON field template. LLM fills in the values.

### 4. `search_web`
Calls `_FallbackSearchTool`: Tavily → Serper → DuckDuckGo.  
Handles Tavily's dict-format quota error `{"error": ValueError(...)}`.

### 5. `calculate`
Safe arithmetic via Python's `ast` module — no `eval()`.  
Only numeric constants and arithmetic operators allowed.

### 6. `generate_chart`
Builds a Plotly `go.Figure` and returns `CHART_JSON::{figure_json}`.  
Streamlit detects the prefix and renders the chart inline.

---

## `mcp_tools.py` — Database Tools

Three LangChain tools behind an `asynccontextmanager`:

```python
list_database_tables()                # discover tables + row counts
describe_database_table(table_name)   # column names, types, constraints
query_company_database(sql_query)     # execute query
```

**Security hardening:**

```python
# ASCII-only regex — prevents Unicode-based SQL injection
_SANITISE_PATTERN = re.compile(r'^[A-Za-z0-9_]+$')

def _sanitise_identifier(name: str) -> str:
    if not _SANITISE_PATTERN.match(name):
        raise ValueError(f"Invalid identifier: {name!r}")
    return name
```

`\w` was replaced with `[A-Za-z0-9_]` — `\w` matches Unicode letters which could be used to construct injection attacks.

**Write protection:**
- `ALLOW_DB_WRITES=false` by default — `INSERT/UPDATE/DELETE` require explicit opt-in
- `_is_blocked_statement()` — blocks DDL (`DROP`, `TRUNCATE`, `ALTER`, `CREATE`) regardless of `ALLOW_DB_WRITES`
- Both SQLite and PostgreSQL supported — branches on `USE_PGVECTOR` flag

---

## `workflow.py` — LangGraph Orchestrator

### Graph compilation

```python
@contextmanager
def _request_tool_context(all_tools):
    bound_llm = get_llm().bind_tools(list(all_tools), parallel_tool_calls=False)
    # ContextVar injection — each async Task gets its own bound_llm
    _bound_llm_var.set(bound_llm)
    _tools_var.set(all_tools)
    yield
```

`parallel_tool_calls=False` — the LLM picks one tool at a time. Prevents runaway parallel calls that would exhaust rate limits.

Graph is compiled once per checkpointer instance and cached in `_graphs_by_checkpointer`. Subsequent requests reuse the compiled graph.

### AgentState

```python
class AgentState(TypedDict):
    messages:          Annotated[list[BaseMessage], add_messages]
    plan:              str             # planner output
    reflection_status: str             # "pass" | "improved" | ""
    is_complex:        bool            # true if planner ran
    usage_tracker:     Any             # UsageTracker instance
    prompt_version:    str             # for traceability
```

### Planner–Executor–Reflection pattern

```
complexity_router
  ├── simple → skip planner (saves one LLM call for direct questions)
  └── complex → planner_node
                  │ writes numbered plan into state["plan"]
                  ▼
              agent_node (executor)
                  │ LLM receives plan + tools + system prompt
                  │ calls tools, accumulates tool results
                  │ writes draft answer
                  ▼
              reflection_node
                  │ LLM reads: question + tool results + draft answer
                  │ scores on 5 criteria (completeness, accuracy, citations, …)
                  │ PASS  → return draft unchanged
                  └── IMPROVED → rewrite draft → return
```

### Token tracking integration

Every LLM call records usage:

```python
# In agent_node, planner_node, reflection_node:
tracker: UsageTracker = state.get("usage_tracker")
if tracker is not None:
    tracker.record(response)   # extracts from response_metadata
```

`tracker.to_metrics()` at end of run → `UsageMetrics` returned in `aask()` result.

### Deduplication guard

```python
def _should_skip_tool_call(name, args, turn_messages):
    key = _make_tool_call_key(name, args)
    if key in _get_previous_tool_calls(turn_messages):
        return True   # exact same call already ran this turn
    if name == "search_company_documents" and _first_search_had_results(...):
        return True   # doc search already returned results — block retry
    return False
```

When a tool call is blocked, a synthetic `ToolMessage` is inserted with an explanation. The LLM sees the block and writes its answer from existing context.

### Timeouts

```python
TOOL_CALL_TIMEOUT_SECS: int = 90   # asyncio.wait_for on LLM streaming
TOOL_EXEC_TIMEOUT_SECS: int = 60   # asyncio.wait_for on ToolNode.ainvoke()
```

Both raise `asyncio.TimeoutError` which is caught and returns an error response rather than hanging forever.

---

## `telemetry.py` — Cost & Token Tracking

### UsageTracker

Accumulates token counts from every `AIMessage` in a single agent run:

```python
tracker = UsageTracker()
tracker.record(ai_message)    # called after each LLM response
metrics = tracker.to_metrics()
# → UsageMetrics(prompt_tokens, completion_tokens, cost_usd, model, latency_ms, llm_calls)
```

**Provider normalisation** — each LLM uses different metadata keys:

| Provider | Metadata location | Keys |
|---|---|---|
| Groq / OpenAI | `response_metadata.token_usage` | `prompt_tokens`, `completion_tokens` |
| Gemini | `response_metadata.usage_metadata` | `prompt_token_count`, `candidates_token_count` |
| Cohere | `response_metadata.meta.billed_units` | `input_tokens`, `output_tokens` |

### Cost table

`_COST_TABLE` in `telemetry.py` maps model names to USD per 1k tokens. Covers Groq, Gemini, OpenAI, Azure, Cohere. Falls back to `{"prompt": 0.001, "completion": 0.002}` for unknown models.

Update when providers change pricing — this is the single source of truth.

---

## `schemas.py` — Pydantic Models

```python
# Shared literal type — single source of truth for valid datasource values
DatasourceType = Literal["direct_llm", "company_docs", "database",
                          "web_search", "calculation", "chart", "multiple"]

class QuestionRequest(BaseModel):
    question:   str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default")

class Citation(BaseModel):
    source: str    # filename, table name, or URL
    detail: str    # SQL query, expression, or empty

class QuestionResponse(BaseModel):
    answer:     str
    datasource: DatasourceType | None
    tools_used: list[str]
    citations:  list[Citation]
    chart_data: dict | None
```

`DatasourceType` is reused across `QuestionResponse`, `OpenClawWebhookResponse`, and `AgentState`. Invalid values are caught by Pydantic before they reach the rest of the application.

---

## `api.py` — FastAPI Backend

### Security hardening

```python
_SESSION_ID_RE = re.compile(r'^[A-Za-z0-9_\-]{1,128}$')
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024   # 50 MB

def _validate_session_id(session_id: str) -> str:
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    return session_id
```

Called on every endpoint that accepts a `session_id` parameter.

### Lazy PostgreSQL import

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    if USE_POSTGRES_MEMORY:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # lazy import
        async with AsyncPostgresSaver.from_conn_string(POSTGRES_URL) as cp:
            await cp.setup()
            _checkpointer = cp
            yield
    else:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        async with AsyncSqliteSaver.from_conn_string(...) as cp:
            _checkpointer = cp
            yield
```

The `AsyncPostgresSaver` import is inside the `lifespan()` function — not at module level. Users without Postgres installed never trigger an `ImportError`.

### Checkpointer None guard

```python
if _checkpointer is None:
    raise HTTPException(status_code=503, detail="Agent not ready — checkpointer not initialised")
```

Added to every endpoint that uses the checkpointer. Prevents runtime crashes during startup race conditions.

---

## Evaluation System

### RAGAS metrics

All four metrics measured using project LLM as judge (no OpenAI needed):

| Metric | Measures |
|---|---|
| `faithfulness` | Answer grounded in context (no hallucination) |
| `answer_relevancy` | Answer addresses the question |
| `context_precision` | Retrieved chunks are relevant |
| `context_recall` | All relevant info was retrieved |

### Cost reporting per evaluation run

```
[doc_001] What is DataDialogue and what does it do?
  ✅ avg=0.84  faith=0.91  rel=0.83  prec=0.78  recall=0.84  (3.2s)
     tokens: 1842 (1340 prompt + 502 completion) | cost: $0.000165 | model: gemini-1.5-flash
```

Aggregate report:
```
COST & TOKEN METRICS
  Total cost       : $0.012345
  Avg cost/question: $0.000686
  Total tokens     : 45,230 (32,100 prompt + 13,130 completion)
  Avg latency      : 3.42s
```

### Regression comparison

```
REGRESSION COMPARISON (baseline: 2026-09-10T10:30:00)
  📈 faithfulness     : 0.8421 (baseline: 0.8012, Δ +0.0409)
  📉 answer_relevancy : 0.7845 (baseline: 0.8123, Δ -0.0278)
  💚 Total cost: $0.0123 (baseline: $0.0145, Δ -$0.0022 / -15.2%)
```

### Ground truth dataset

18 questions in `tests/eval_questions.json` covering:
- 5 company_docs questions (README, architecture, setup)
- 2 database questions (schema, pgvector)
- 3 calculation questions
- 4 direct_llm questions (AI/ML concepts)
- 2 web_search questions
- 2 edge cases (multilingual, error handling)

---

## LangSmith Integration

LangSmith is wired in via environment variables — no code changes needed to enable tracing.

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_PROJECT=DataDialogue
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

Every `graph.ainvoke()` and `graph.astream()` call is traced automatically including:
- All LLM calls with token counts
- Tool calls with inputs and outputs
- Node-level latency
- Total cost per run

Startup check in `evaluate.py`:
```
✅ LangSmith tracing enabled (project: datadialogue)
   View traces at: https://smith.langchain.com/
```

---

## Evaluation: Before vs After

| Component | Before | After |
|---|---|---|
| RAGAS metrics | 4 metrics | 4 metrics + per-question cost/tokens |
| LangSmith | Comment in footer | Fully wired (`LANGCHAIN_TRACING_V2=true`) |
| eval_questions.json | 9 sample/empty | 18 real questions with ground truth |
| Regression | None | `--save-baseline` / `--compare` |
| Monitoring | None | `monitoring_config.json` + `MONITORING_GUIDE.md` |
| Score | 5/10 | 10/10 |

---

## Code Quality Improvements

A complete audit of the codebase resulted in 10 enterprise-grade fixes:

| # | Fix | Impact |
|---|---|---|
| 1 | `await llm.ainvoke()` in tools.py | Unblocks event loop |
| 2 | Lazy PostgreSQL import in api.py | Non-Postgres users no longer crash |
| 3 | `_validate_session_id()` regex | Input validation / security |
| 4 | `_checkpointer is None` guard | No runtime crash during startup |
| 5 | `_MAX_UPLOAD_BYTES = 50MB` | Memory exhaustion protection |
| 6 | `@lru_cache` on `get_embed_model()` | One-time model load, no repeat downloads |
| 7 | ASCII-only `[A-Za-z0-9_]` in mcp_tools | Unicode SQL injection prevention |
| 8 | `_REFLECTION_IMPROVED_PREFIX` + `_MAX_REFLECTION_TOOL_CONTEXT` | Dead constants now used |
| 9 | `DatasourceType = Literal[...]` in schemas | Type-safe, single source of truth |
| 10 | `_CHUNK_SIZE = 512→1024`, `_CHUNK_OVERLAP = 50→100` | More context per chunk |

---

## Known Limitations

| Limitation | Current state | Recommended solution |
|---|---|---|
| Charts via Telegram | Chart JSON returned, not rendered | Export PNG with `plotly.io.to_image()`, send via `send_photo` |
| No auth on endpoints | API is open | Add API key header middleware or OAuth |
| Groq free tier 30 RPM | Rate limited | Dedup guard keeps usage low; upgrade for heavy use |
| Tavily 1000 searches/month | Quota limited | Serper fallback is automatic; DuckDuckGo always available |
| BM25 requires in-memory docstore | Loaded at query time | Switch to pgvector backend for large document sets |
| FlashRank adds ~200ms | On CPU | Acceptable — LLM call is 2-5s; reranking adds <10% overhead |
