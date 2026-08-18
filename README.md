# KT Agent using LangGraph

This repository contains a production-ready knowledge-transfer assistant built with LangGraph, LlamaIndex, LangChain, and Streamlit. The system is designed to answer questions about a domain-specific knowledge base using retrieval-augmented generation (RAG), and it can fall back to web search when the question is outside the indexed scope.

## What the system does

- Ingests and indexes project-related documents
- Retrieves relevant chunks from a persisted vector store
- Routes questions between vector retrieval and web search
- Generates grounded answers using an LLM
- Validates answer quality with LangGraph grader nodes enforcing structured Pydantic outputs
- Exposes the experience through a Streamlit UI
- Silently recovers from transient LLM/Search API errors using Tenacity retry logic

## How it works

1. The user asks a question.
2. The workflow decides whether to answer from local indexed documents or use web search.
3. If local knowledge is used, the system retrieves relevant document chunks from the vector index.
4. The LLM generates an answer using the retrieved context.
5. The system grades the answer for grounding and usefulness before returning it.

## Why these technologies

- **LangGraph**: defines a graph-based decision workflow with nodes and conditionals.
- **LangChain**: manages prompts, model calls, and structured output parsing.
- **LlamaIndex**: builds and loads an embedding-based search index from documents.

## Project structure

```text
kt-agent-using-langgraph/
├── src/                    
│   └── kt_agent/
│       ├── __init__.py
│       ├── config.py       # Configuration and environment setups
│       ├── indexing.py     # LlamaIndex ingestion logic
│       ├── prompts.py      # LLM Prompts with Pydantic structured models
│       └── workflow.py     # LangGraph StateGraph logic with Retry resilience
├── data/                   # Raw documents (e.g., docx, pdf files)
├── indexing_data/          # Persisted Vector DB
├── tests/                  
│   └── test_agents.py      # Script for testing individual components
├── app.py                  # CLI Entry point for the LangGraph agent
├── streamlit_app.py        # Streamlit-based UI for document Q&A
├── requirements.txt
└── README.md
```

## Architecture summary

The solution follows a layered architecture:

1. **Ingestion layer** (`src/kt_agent/indexing.py`)
   - loads source documents from `data/`
   - splits them into chunks
   - creates embeddings and stores them in `indexing_data/`

2. **Reasoning layer** (`src/kt_agent/workflow.py`)
   - routes the question
   - retrieves context from the vector index or web search
   - generates and evaluates the answer

3. **Presentation layer** (`app.py`, `streamlit_app.py`)
   - provides a CLI or UI for document upload and question answering

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
   python src/kt_agent/indexing.py
   ```

5. Run the workflow or UI:

   ```bash
   python app.py
   ```

   or

   ```bash
   streamlit run streamlit_app.py
   ```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md)
