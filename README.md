# DataDialogue — Knowledge Transfer Assistant

A conversational AI agent that answers questions about your company files, database, and the live web — with real-time streaming responses.

Built with **LangGraph · LlamaIndex · FastAPI · Streamlit**

---

## What it does

Ask questions in plain English. The agent picks the right tool automatically:

| Question | What happens |
|---|---|
| "Summarise the annual report" | Reads your uploaded PDF/DOCX |
| "What is the status of order #12345?" | Queries the company database |
| "What is 15% of 85000?" | Runs a safe calculator |
| "Show monthly sales as a bar chart" | Generates an interactive Plotly chart |
| "What is the USD to INR rate today?" | Searches the live web |

Answers stream word-by-word in real time — no waiting for the full response.

---

## Quick Start

### 1. Install dependencies

```bash
cd DataDialogue
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Set up `.env`

```env
# LLM — pick at least one
GROQ_API_KEY=your_key_here

# Web search — all optional, cascades automatically: Tavily → Serper → DuckDuckGo
TAVILY_API_KEY=your_key_here      # tavily.com — 1,000 free/month
SERPER_API_KEY=your_key_here      # serper.dev — 2,500 free, no card

KT_API_URL=http://localhost:8000
```

> No web search key? DuckDuckGo is used automatically — no signup needed.

### 3. Index your documents

Drop files into `data/` (PDF, DOCX, XLSX, CSV, TXT), then:

```bash
python -m src.agent.rag
```

### 4. Run

Two terminals:

```bash
# Terminal 1
uvicorn app:app --reload --port 8000

# Terminal 2
streamlit run streamlit_app.py
```

Open **http://localhost:8501**

---

## Telegram Bot

```bash
# Add to .env
TELEGRAM_BOT_TOKEN=your_token_here

# Run (FastAPI must be running first)
python telegram_bot.py
```

Each user gets their own conversation memory automatically.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/stream` | Real-time SSE streaming response |
| `POST` | `/ask` | Blocking response (used by Telegram) |
| `GET` | `/health` | Liveness check |
| `POST` | `/upload` | Upload + index documents |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/sessions/{id}/history` | Clear conversation memory |

Interactive docs: **http://localhost:8000/docs**

---

## Troubleshooting

**Web search not working** — Tavily quota hit. Add `SERPER_API_KEY` to `.env` and restart.

**DOCX returns empty results** — Run `pip install llama-index-readers-file`

**`ModuleNotFoundError: ddgs`** — Run `pip install ddgs==9.16.0` inside your `.venv`

**Backend offline error in Streamlit** — Start FastAPI first (Terminal 1).

**Stale index after adding files** — Re-run `python -m src.agent.rag` or use the Upload button in the sidebar.

---

## Docs

- [Architecture overview](docs/ARCHITECTURE.md)
- [Component design](docs/SYSTEM_DESIGN.md)
