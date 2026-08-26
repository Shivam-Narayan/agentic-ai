# KT Agent — Knowledge Transfer Assistant

> A conversational AI agent that answers questions about your company by searching internal documents, querying a database, running calculations, generating charts, or looking up the live web — all from a single chat interface with full conversation memory.

Built with **LangGraph + LlamaIndex + FastAPI + Streamlit**.

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

No manual routing. No dropdowns. The LLM reads the available tools and decides which one(s) to use.

---

## How it looks

```
You:   What does the KT document say about the system architecture?

Agent: The KT document describes a three-layer architecture consisting of...
       [📄 Company documents · search_company_documents]
       [📎 kt_document.pdf]
```

The response always shows:
- A **tool badge** indicating which data source was used
- **Citation pills** showing the exact filename, URL, or SQL query
- **Inline charts** when the agent generates a visualisation

---

## Tech Stack

| Layer | Technology |
|---|---|
| Chat UI | Streamlit |
| API backend | FastAPI |
| Agent loop | LangGraph (ReAct pattern) |
| LLM | Groq (default) / Google Gemini / Cohere |
| Document search | LlamaIndex + HuggingFace embeddings |
| Charts | Plotly (rendered inline in chat) |
| Database queries | Python sqlite3 (MCP-compatible interface) |
| Web search | Tavily API |

---

## Project Structure

```
kt-agent-using-langgraph/
│
├── app.py                        # FastAPI backend
├── streamlit_app.py              # Streamlit chat UI
├── requirements.txt
├── .env                          # Your API keys (never commit this)
│
├── data/                         # PUT YOUR COMPANY FILES HERE
│   ├── company.db                # SQLite database
│   ├── your_report.pdf           # PDFs go here
│   ├── quarterly_data.xlsx       # Excel files go here
│   └── project_notes.docx        # Word docs go here
│
├── indexing_data/                # Auto-generated — vector store (do not edit)
│
├── src/
│   └── agent/
│       ├── config.py             # Paths and environment setup
│       ├── chains.py             # LLM factory (Groq / Gemini / Cohere)
│       ├── rag.py                # Document ingestion and search
│       ├── tools.py              # 6 local tools (search, summarise, extract, web, calc, chart)
│       ├── mcp_client.py         # Database query tools (MCP-compatible)
│       ├── workflow.py           # LangGraph agent — core logic, citations, chart parsing
│       ├── schemas.py            # API request/response models
│       └── ingest_drive.py       # Optional: sync from Google Drive
│
├── docs/
│   ├── ARCHITECTURE.md           # System architecture and data flow
│   └── SYSTEM_DESIGN.md          # Component design details
│
└── tests/
    └── test_agents.py
```

---

## Setup Guide (Step by Step)

### Step 1 — Prerequisites

- Python 3.11 or newer
- At least one LLM API key (Groq is recommended — free tier available)
- A Tavily API key (free tier available at [tavily.com](https://tavily.com))

Get your free API keys:
- Groq: https://console.groq.com
- Tavily: https://tavily.com
- Google Gemini: https://aistudio.google.com (alternative to Groq)

---

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3 — Create your `.env` file

Create a file called `.env` in the project root (same folder as `app.py`):

```env
# ── LLM Provider (pick at least ONE) ──────────────────────────────────────
GROQ_API_KEY=your_groq_api_key_here

# Uncomment to use Google Gemini instead:
# GOOGLE_API_KEY=your_google_api_key_here

# Uncomment to use Cohere instead:
# COHERE_API_KEY=your_cohere_api_key_here

# ── Web Search (required) ──────────────────────────────────────────────────
TAVILY_API_KEY=your_tavily_api_key_here

# ── Optional overrides ────────────────────────────────────────────────────
# Force a specific LLM even when multiple keys are present:
# LLM_PROVIDER=groq          # Options: groq, google, cohere

# API URL used by Streamlit to talk to FastAPI:
# KT_API_URL=http://localhost:8000
```

> If you have multiple LLM keys, the system auto-selects in this order: **Groq → Google → Cohere**. Set `LLM_PROVIDER` to override.

---

### Step 4 — Add your company data

Place your files inside the `data/` folder. Supported formats:

| Format | Extension |
|---|---|
| PDF | `.pdf` |
| Word document | `.docx`, `.doc` |
| Excel spreadsheet | `.xlsx`, `.xls` |
| CSV | `.csv` |
| Plain text | `.txt` |

No code changes needed. The system auto-discovers all supported files.

---

### Step 5 — Build the document index

Run this **once** to scan your `data/` folder and build the search index:

```bash
python -m src.agent.rag
```

> Re-run this command whenever you add, remove, or update files in `data/`.  
> Alternatively, upload files directly from the Streamlit sidebar — the index rebuilds automatically.

---

### Step 6 — Start the backend API

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Verify it is running: http://localhost:8000/health should return `{"status": "ok"}`

The interactive API docs are at: http://localhost:8000/docs

---

### Step 7 — Start the chat UI

Open a second terminal and run:

```bash
python -m streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser.

---

## Using the Agent

Once running, type any question in the chat box. Some examples:

**Document questions:**
- "Summarise the annual report"
- "What does the employee handbook say about leave policy?"
- "Extract the project name, start date, and owner from the documents"

**Database questions:**
- "List all tables in the database"
- "What is the status of order #12345?"
- "Show the top 5 customers by order count as a bar chart"

**Calculations:**
- "What is 15% of 85000?"
- "Calculate (250 + 300) * 12 / 100"

**Charts:**
- "Show monthly sales as a bar chart: Jan=1200, Feb=1500, Mar=1100"
- "Plot a pie chart: Engineering=40, Sales=25, Marketing=20, HR=15"

**Web search:**
- "What is the current USD to INR exchange rate?"
- "Latest news about LangChain"

**Follow-up questions (conversation memory):**
- "What was the total from that last calculation?"
- "Can you show that as a chart instead?"

---

## Uploading Documents via the UI

You don't need to manually copy files to `data/` and re-run the indexer. From the Streamlit sidebar:

1. Click **Upload Documents** and select your files
2. Click **Upload & Index**
3. The backend saves the files to `data/` and rebuilds the index automatically
4. The document list refreshes to show the newly indexed files

---

## Adding New Capabilities (for developers)

### Add a new tool (e.g., Jira lookup)

1. Open `src/agent/tools.py` and add a new `@tool` function:

```python
@tool
def get_jira_ticket(ticket_id: str) -> str:
    """Look up a Jira ticket by its ID (e.g. 'PROJ-123').
    Use this when the user asks about a specific ticket, issue, or bug report."""
    # your implementation here
    return f"Ticket {ticket_id}: ..."
```

2. Register it in `src/agent/workflow.py`:

```python
LOCAL_TOOLS = [search_company_documents, search_web, ..., get_jira_ticket]
```

That's it. No routing changes needed — the LLM will start using the new tool automatically.

### Switch LLM provider

Update `.env`:

```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your_key_here
```

Restart the server.

---

## API Reference

### `POST /ask`

Send a question, get an answer with full metadata.

**Request:**
```json
{
  "question": "What does the KT document say about deployment?",
  "session_id": "my-session-123"
}
```

**Response:**
```json
{
  "answer": "The KT document describes a three-stage deployment process...",
  "datasource": "company_docs",
  "tools_used": ["search_company_documents"],
  "citations": [
    { "source": "kt_document.pdf", "detail": "" }
  ],
  "chart_data": null
}
```

**`datasource` values:**

| Value | Meaning |
|---|---|
| `direct_llm` | LLM answered from its own training data |
| `company_docs` | Answer came from your uploaded documents |
| `database` | Answer came from the company database |
| `web_search` | Answer came from a live web search |
| `calculation` | Answer used the calculator tool |
| `chart` | A chart was generated |
| `multiple` | More than one tool category was used |

### `POST /upload`

Upload one or more documents. Files are saved to `data/` and the index is rebuilt automatically.

### `GET /documents`

List all documents currently indexed.

### `GET /health`

```json
{ "status": "ok" }
```

### `GET /sessions/{session_id}/history`

Returns the conversation history for a session.

### `DELETE /sessions/{session_id}/history`

Clears the conversation history for a session (start fresh).

---

## Troubleshooting

**"No supported documents found in data/"**  
Make sure your files are inside `data/` and have a supported extension (`.pdf`, `.docx`, `.xlsx`, `.csv`, `.txt`).

**"ModuleNotFoundError: No module named 'openpyxl'"**  
Run: `pip install openpyxl==3.1.5`

**"ValueError: No LLM API key found"**  
Check that your `.env` file exists and has at least one of `GROQ_API_KEY`, `GOOGLE_API_KEY`, or `COHERE_API_KEY` set.

**Index is stale after adding new files**  
Re-run `python -m src.agent.rag` to rebuild the index, or use the Upload button in the Streamlit sidebar.

**Streamlit cannot connect to the API**  
Make sure the FastAPI server is running and that `KT_API_URL` in `.env` matches the address (default: `http://localhost:8000`).

**Charts not rendering**  
Run: `pip install plotly`

---

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — How the system is designed and how data flows through it
- [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) — Detailed component breakdown and design decisions
