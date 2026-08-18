# System Design

## Overview

This system is a retrieval-augmented generation (RAG) assistant for knowledge transfer. It is designed to answer questions about project-specific documents while also falling back to web search when the question is outside the indexed domain.

## What a beginner should know

The project builds a small AI pipeline that does three main things:
1. Turn documents into searchable knowledge.
2. Decide whether a question can be answered from that knowledge or needs web search.
3. Generate a grounded answer and check whether the answer is good.

The main components are:
- LangGraph: defines the workflow steps and decision paths.
- LangChain: manages prompts, chains, and model output parsing.
- LlamaIndex: builds and queries the vector search index.

## Why these components are used

### What is LangGraph?
LangGraph is a graph-based orchestrator for AI workflows. In this project, it lets us define nodes (functions) and conditional edges (decisions) so the system can run a flexible reasoning process rather than one fixed script.

The workflow is not just a single call to the model. It is a set of steps that can route, retry, and validate results.

### What is LangChain?
LangChain provides the building blocks for prompts, parsers, and model calls. It is used here to:
- define prompt templates
- connect prompts to the LLM
- parse JSON outputs from grading prompts
- build the RAG generation chain

### What is LlamaIndex?
LlamaIndex turns documents into an embedding-based vector index. The index stores semantic vectors so the system can retrieve the most relevant text chunks for a query.

This project uses LlamaIndex to load an existing index from disk and make it available to the retriever.

## Core project components

### config.py
Sets up the runtime environment and external tools:
- loads necessary environment variables
- configures Hugging Face embeddings
- configures Gemini as the LLM
- loads the persisted vector index
- creates the web search tool

### prompts.py
Holds reusable prompt templates for:
- retrieval grading
- answer generation
- hallucination grading
- answer usefulness grading
- question routing

### workflow.py
Defines the LangGraph workflow and the nodes that implement each step.

This is the most important file for understanding how the project runs.

### app.py
A small entry point that creates the agent and runs it on a question.

### streamlit_app.py
The interactive UI for asking questions. It is simplified to call the same workflow and display answers.

## How the LangGraph workflow works

LangGraph lets us define:
- nodes: individual processing steps such as retrieval, grading, generation, and search
- a state dictionary: the shared data passed between nodes
- conditional edges: decisions that choose the next node based on current state

The workflow in this project follows this path:

1. `route_question`: decide whether to use the vector store or web search.
2. `retrieve`: fetch documents from the persisted index when vector search is chosen.
3. `grade_documents`: inspect each retrieved document and keep only relevant ones.
4. `decide_to_generate`: if a relevant document exists, continue to generation; otherwise use web search.
5. `web_search`: call the external search tool and append results to documents.
6. `generate`: build an answer using the retrieved context.
7. `grade_generation_v_documents_and_question`: validate the answer with two graders.
8. `rewrite_query` (Conditional): if the answer is not useful, rewrite the user's question before looping back to web search to prevent infinite loops.

The graph can loop back if the answer is not grounded or not useful, up to a strict `MAX_RETRIES` limit to ensure graceful failure.

## How hallucinator-grader and answer-grader work

This project uses two separate grading steps after the model generates an answer.

### Hallucination grader
The hallucination grader asks:
- "Is this answer supported by the documents?"

It receives:
- the retrieved document text
- the model-generated answer

If the answer is not grounded, the workflow will retry or use web search.

### Answer grader
The answer grader asks:
- "Does this answer actually resolve the user question?"

It receives:
- the generated answer
- the original user question

If the answer is not useful, the workflow can fall back to a different path.

## What the nodes create

Each workflow node builds or updates the shared state:
- `retrieve` produces `documents`
- `grade_documents` filters `documents` and may set `web_search`
- `web_search` appends search results to `documents`
- `generate` produces `generation`
- `grade_generation_v_documents_and_question` returns a final quality decision

This lets the system keep a clear trace of how the answer was built.

## What this project is teaching you

If you want to create a similar system from scratch, the main lessons are:
- separate knowledge ingestion from query-time reasoning
- use embeddings and a vector index for semantic search
- keep the question-answering step grounded in retrieved text
- build explicit decision logic with a workflow engine
- validate generated answers with additional grading prompts

## Practical building blocks from scratch

1. prepare documents and build an index with LlamaIndex
2. create prompt templates with LangChain
3. wrap the model calls in a workflow using LangGraph
4. add tools for web search and fallback behavior
5. validate outputs with grader prompts
6. expose the system through a simple UI like Streamlit

## Integration points

### External services
- Google Gemini: used for LLM reasoning and grading prompts
- Hugging Face embeddings: used for semantic retrieval
- Tavily search: used as a fallback when vector search is not enough

### Local files
- `indexing_data`: stored vector index and metadata

## Deployment notes

The project is now fully structured for production use:
- environment configuration is centralized via `config.py` and `.env` files.
- `src/kt_agent` serves as a clean Python package for the workflow logic.
- Tenacity error handling prevents transient API issues from crashing the application.
- Self-RAG infinite loops are strictly mitigated with a query-rewriting and retry-limiting node design.

## Summary

This project demonstrates a production-grade RAG assistant. It connects document search, routing, query rewriting, answer generation, and strict Pydantic grading inside a fault-tolerant graph workflow.
