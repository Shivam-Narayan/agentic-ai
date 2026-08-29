# KT Agent — Knowledge Transfer Assistant

> A conversational AI agent that answers questions about your company by searching internal documents, querying a database, running calculations, generating charts, or looking up the live web — all from a single chat interface with full conversation memory.

Built with **LangGraph + LlamaIndex + FastAPI + Streamlit**.

Supports **two access channels** — web browser (Streamlit) and **Telegram** — both powered by the same agent backend.

---

## What does it do?

You ask a question in plain English. The agent figures out the best way to answer it:

| Your question | How the agent answers |
|---|---|
| "What is Python?" | Answers directly from its own knowledge |
| "What does our KT document say about deployment?" | Searches your company PDFs and DOCX files |
| "Summarise the annual report" | Reads the full document and summarises it |
| "Extract the project name and budget from the documents" | Pulls specific fields from uploaded files |
| "What is in the Q3 revenue Excel sheet?" | Searches your uploaded Excel / CSV files |
| "What is the status of order #12345?" | Queries the company SQLite database |
| "What is 15% of 85000?" | Uses a safe calculator — never guesses numbers |
| "Show monthly sales as a bar chart" | Generates an interactive Plotly chart inline |
| "What is the weather in Bangalore today?" | Searches the live web via Tavily |
| "What day is today?" | Answers from the live server clock — always accurate |

No manual routing. No dropdowns. The LLM reads the available tools and decides which one(s) to use.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Chat UI | Streamlit |
| Telegram channel | `python-telegram-bot` v21.6 |
| API backend | FastAPI |
| Agent loop | LangGraph (ReAct pattern) |
| Conversation memory | LangGraph `AsyncSqliteSaver` (persisted to disk) |
| LLM | Groq (default) / Google Gemini / Cohere |
| Document search | LlamaIndex + HuggingFace embeddings |
| DOCX parsing | `llama-index-readers-file` (DocxReader) |
| Charts | Plotly (rendered inline in chat) |
| Database queries | Python `sqlite3` (MCP-compatible interface) |
| Web search | Tavily API (`langchain-tavily`) |

---

## Project Structure

```
DataDialogue/
│
├── app.py                        # FastAPI backend — all API endpoints
├── streamlit_app.py              # Streamlit web chat UI
├── telegram_bot.py               # Telegram bot — direct channel
├── requirements.txt
├── .env                          # Your API keys (never commit this)
│
├── data/                         # PUT YOUR COMPANY FILES HERE
│   ├── company.db                # SQLite database
│   ├── your_report.pdf
│   ├── quarterly_data.xlsx
│   └── project_notes.docx
│
├── indexing_data/                # Auto-generated — LlamaIndex vector store (do not edit)
│
├── memory_store/                 # Auto-generated — conversation history (do not edit)
│   └── conversations.db          # Persists across server restarts
│
├── src/
│   └── agent/
│       ├── config.py             # Paths and environment setup
│       ├── chains.py             # LLM factory (Groq / Gemini / Cohere)
│       ├── rag.py                # Document ingestion, DocxReader, vector search
│       ├── tools.py              # 6 local tools: search, summarise, extract, web, calc, chart
│       ├── mcp_client.py         # 3 database tools (MCP-compatible)
│       ├── workflow.py           # LangGraph agent loop, dedup guard, citations
│       ├── schemas.py            # Pydantic models for API requests/responses

│
├── docs/
│   ├── ARCHITECTURE.md           # System architecture and data flow
│   └── SYSTEM_DESIGN.md          # Component design details
│
└── tests/
    └── test_agents.py
```

---

## Quick Start — New Developer Setup

Follow these steps in order. All commands run from the `DataDialogue/` folder.

### Step 1 — Prerequisites

- Python **3.11** or newer
- At least one LLM API key — Groq is recommended (free tier, fast)
- A Tavily API key for web search (free tier)

Get your free keys:

| Service | URL |
|---|---|
| Groq (LLM) | https://console.groq.com |
| Tavily (web search) | https://tavily.com |
| Google Gemini (alternative LLM) | https://aistudio.google.com |

---

### Step 2 — Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt after activation.

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> `llama-index-readers-file` is required for proper DOCX text extraction. Without it, Word documents are indexed as binary garbage and return no results.

---

### Step 4 — Create your `.env` file

Create `.env` in the project root (same folder as `app.py`):

```env
# ── LLM Provider (pick at least ONE) ──────────────────────────────────────
GROQ_API_KEY=your_groq_api_key_here

# Uncomment to use Google Gemini instead of Groq:
# GOOGLE_API_KEY=your_google_api_key_here

# Uncomment to use Cohere instead:
# COHERE_API_KEY=your_cohere_api_key_here

# ── Web Search (required) ──────────────────────────────────────────────────
TAVILY_API_KEY=your_tavily_api_key_here

# ── API URL — must match the port FastAPI runs on ──────────────────────────
KT_API_URL=http://localhost:8000

# ── Optional ──────────────────────────────────────────────────────────────
# Force a specific LLM when multiple keys are present:
# LLM_PROVIDER=groq    # Options: groq, google, cohere


```

> If you have multiple LLM keys, the system auto-selects: **Groq → Google → Cohere**. Set `LLM_PROVIDER` to override.

---

### Step 5 — Add your company files

Place your files inside the `data/` folder:

| Format | Extension |
|---|---|
| PDF | `.pdf` |
| Word document | `.docx`, `.doc` |
| Excel spreadsheet | `.xlsx`, `.xls` |
| CSV | `.csv` |
| Plain text | `.txt` |

---

### Step 6 — Build the document index

```bash
python -m src.agent.rag
```

> Re-run whenever you add, update, or remove files in `data/`.
> You can also upload files from the Streamlit sidebar — the index rebuilds automatically.

---

### Step 7 — Start the FastAPI backend

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Verify: http://localhost:8000/health → `{"status": "ok"}`

Interactive API docs: http://localhost:8000/docs

---

### Step 8 — Start the Streamlit web UI

Open a **second terminal** with the venv active:

```bash
python -m streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser.

> FastAPI (port 8000) and Streamlit (port 8501) must both be running at the same time.

---

## Telegram Channel Setup

The agent also works through Telegram via `telegram_bot.py`. Every Telegram message is forwarded to your DataDialogue agent and the reply is sent back.

### Step 1 — Create a Telegram bot

1. Open Telegram → search for **@BotFather**
2. Send: `/newbot`
3. Choose a name and a username ending in `bot`
4. Copy the **bot token** (looks like `123456789:ABCdefGHI...`)

### Step 2 — Set your bot token

Open `telegram_bot.py` and update line 14:

```python
TELEGRAM_BOT_TOKEN = "your_bot_token_here"
```

### Step 3 — Start the Telegram bot

Make sure FastAPI is running first, then open a **third terminal**:

```bash
python telegram_bot.py
```

You should see:
```
Application started
HTTP Request: POST .../getUpdates "HTTP/1.1 200 OK"
```

### Step 4 — Test in Telegram

1. Open Telegram → search for your bot by username
2. Send `/start`
3. Ask anything — "What day is today?", "What is 15% of 85000?"

### How it works

```
User message in Telegram
        ↓
telegram_bot.py (long-polling)
        ↓
POST http://localhost:8000/ask
     { "question": "...", "session_id": "telegram_<user_id>" }
        ↓
FastAPI → LangGraph agent → tools → answer
        ↓
Reply sent back to user in Telegram
```

- Each Telegram user gets their own conversation memory (`telegram_<user_id>`)
- Memory persists across bot restarts via `memory_store/conversations.db`
- All 9 tools work: RAG, web search, calculator, database, charts

---

## Running Everything — Summary

Three terminals, all with `(.venv)` active:

```
Terminal 1:  python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Terminal 2:  python -m streamlit run streamlit_app.py
Terminal 3:  python telegram_bot.py
```

| Service | URL / Access |
|---|---|
| FastAPI backend | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Streamlit web UI | http://localhost:8501 |
| Telegram bot | @your_bot_username in Telegram |

---

## Example Questions

**Documents:**
- "Summarise the annual report"
- "What are Shivam's technical skills?"
- "Extract the project name, start date, and owner from the documents"

**Database:**
- "List all tables in the database"
- "What is the status of order #12345?"

**Calculations:**
- "What is 15% of 85000?"
- "Calculate (250 + 300) * 12 / 100"

**Charts:**
- "Show monthly sales as a bar chart: Jan=1200, Feb=1500, Mar=1100"
- "Plot a pie chart: Engineering=40, Sales=25, Marketing=20, HR=15"

**Web search:**
- "What is the current USD to INR exchange rate?"
- "Latest news about LangChain"

**Follow-up questions (memory):**
- Ask anything → follow up with "Can you show that as a chart?"
- The agent remembers the full conversation thread

---

## Uploading Documents via the Web UI

From the Streamlit sidebar:

1. Click **Upload Documents** and select your files
2. Click **⬆️ Upload & Index**
3. Files are saved to `data/` and the index rebuilds automatically
4. Ask questions about the new content immediately

---

## Adding New Capabilities

### Add a new tool

```python
# 1. src/agent/tools.py — define the tool
@tool
def get_jira_ticket(ticket_id: str) -> str:
    """Look up a Jira ticket by ID (e.g. 'PROJ-123').
    Use this when the user asks about a specific ticket or bug report."""
    return f"Ticket {ticket_id}: ..."

# 2. src/agent/workflow.py — register it
LOCAL_TOOLS = [search_company_documents, search_web, ..., get_jira_ticket]
```

No routing changes needed — the LLM starts using the new tool automatically across all channels.

### Switch LLM provider

```env
# .env
LLM_PROVIDER=google
GOOGLE_API_KEY=your_key_here
```

Restart FastAPI.

### Add a new channel (e.g. Discord)

Create `discord_bot.py` that:
1. Receives messages from Discord
2. Calls `POST /ask` with `session_id=f"discord_{user_id}"`
3. Sends `answer` back to Discord

Memory works automatically — the checkpointer handles persistence per `session_id`.

---

## API Reference

### `POST /ask`

```json
// Request
{ "question": "What does the KT document say about deployment?", "session_id": "my-session" }

// Response
{
  "answer": "The KT document describes...",
  "datasource": "company_docs",
  "tools_used": ["search_company_documents"],
  "citations": [{ "source": "kt_document.pdf", "detail": "" }],
  "chart_data": null
}
```

**`datasource` values:**

| Value | Meaning |
|---|---|
| `direct_llm` | LLM answered from its own training data |
| `company_docs` | Answer from your uploaded documents |
| `database` | Answer from the company database |
| `web_search` | Answer from a live web search |
| `calculation` | Calculator tool was used |
| `chart` | A chart was generated |
| `multiple` | More than one tool category was used |

### All endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ask` | Submit a question to the agent |
| `GET` | `/health` | Liveness check |
| `POST` | `/upload` | Upload documents and rebuild the index |
| `GET` | `/documents` | List all indexed documents |
| `GET` | `/sessions/{id}/history` | Conversation history for a session |
| `DELETE` | `/sessions/{id}/history` | Clear a session's history |
| `GET` | `/sessions` | List all active session IDs |

---

## Troubleshooting

**"No supported documents found in data/"**
Make sure files are inside `data/` with a supported extension (`.pdf`, `.docx`, `.xlsx`, `.csv`, `.txt`).

**DOCX files return empty or garbage results**
Run `pip install llama-index-readers-file` then rebuild: `python -m src.agent.rag`.

**Agent hits rate limit (HTTP 429)**
Wait 1 minute (Groq free tier: 30 RPM). The deduplication guard in `workflow.py` prevents most repeat calls.

**"ValueError: No LLM API key found"**
Check `.env` has at least one of `GROQ_API_KEY`, `GOOGLE_API_KEY`, or `COHERE_API_KEY`.

**"ModuleNotFoundError: No module named 'openpyxl'"**
Run: `pip install openpyxl==3.1.5`

**Index is stale after adding new files**
Run `python -m src.agent.rag` or use the Upload button in Streamlit.

**"Cannot reach the backend. Is FastAPI running?"**
1. Start FastAPI: `python -m uvicorn app:app --host 0.0.0.0 --port 8000`
2. Check `KT_API_URL=http://localhost:8000` is in `.env`

**Telegram bot: "Cannot reach the DataDialogue backend"**
Start Terminal 1 (FastAPI) before Terminal 3 (telegram_bot.py).

**Charts not rendering in Streamlit**
Run: `pip install plotly`

---

## Documentation

| File | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, channel diagram, data flow |
| [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) | Component internals, design decisions |
