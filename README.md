# KT Agent using LangGraph

This repository contains a knowledge-transfer assistant with a layered stack:

```text
User (Streamlit)
   ↓
FastAPI
   ↓
LangGraph orchestration
   ↓
LangChain (LLM + tools + prompts)
   ↓
LlamaIndex (data / RAG)
   ↓
Vector store (indexing_data; Postgres/pgvector can replace this)
```

## What the system does

- Ingests and indexes project-related documents
- Retrieves relevant chunks from a persisted vector store
- Routes questions between vector retrieval and web search
- Generates grounded answers using an LLM
- Validates answer quality with LangGraph grader nodes enforcing structured Pydantic outputs
- Exposes FastAPI `/ask` and a Streamlit chat UI that calls it
- Silently recovers from transient LLM/Search API errors using Tenacity retry logic

## How it works

1. The user asks a question.
2. The workflow decides whether to answer from local indexed documents or use web search.
3. If local knowledge is used, the system retrieves relevant document chunks from the vector index.
4. The LLM generates an answer using the retrieved context.
5. The system grades the answer for grounding and usefulness before returning it.

## Why these technologies

- **FastAPI**: HTTP API and input validation.
- **LangGraph**: graph-based orchestration (classify, retrieve, search, generate).
- **LangChain**: LLM, prompts, tools, and structured output.
- **LlamaIndex**: document ingestion and RAG retrieval.
- **Vector store**: persisted embeddings in `indexing_data/`.

## Project structure

```text
kt-agent-using-langgraph/
├── src/                    
│   └── kt_agent/
│       ├── __init__.py
│       ├── config.py       # Environment and logging
│       ├── chains.py       # LangChain (LLM, tools, prompts)
│       ├── rag.py          # LlamaIndex (index + retrieve)
│       ├── schemas.py      # API request/response models
│       └── workflow.py     # LangGraph orchestration
├── data/
├── indexing_data/          # Vector database files
├── tests/
├── app.py                  # FastAPI
├── streamlit_app.py        # User UI (calls FastAPI)
├── requirements.txt
└── README.md
```

## Architecture summary

The solution follows this layered architecture:

1. **User** (`streamlit_app.py`) — chat UI
2. **FastAPI** (`app.py`) — `/ask` validation and HTTP
3. **LangGraph** (`workflow.py`) — orchestration
4. **LangChain** (`chains.py`) — LLM, tools, prompts
5. **LlamaIndex** (`rag.py`) — RAG
6. **Vector store** (`indexing_data/`) — persisted embeddings

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and set the required environment variables:

   ```env
   GOOGLE_API_KEY=your_google_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

4. Create the index (Run this once before starting the app):

   ```bash
   python -m src.kt_agent.rag
   ```

5. Start FastAPI, then Streamlit:

   ```bash
   uvicorn app:app --reload --port 8000
   streamlit run streamlit_app.py
   ```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md)
