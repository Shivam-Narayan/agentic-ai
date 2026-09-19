-- init.sql
-- Runs ONCE automatically when the Postgres container starts for the first time.
-- (Docker mounts this into /docker-entrypoint-initdb.d/ — see docker-compose.yml)
--
-- Purpose: enable pgvector extension so LlamaIndex and the migration script work.
-- LlamaIndex creates its own table (data_document_embeddings) at runtime.
-- LangGraph creates its own checkpoint tables via AsyncPostgresSaver.setup().
-- We only need to ensure the vector extension is available before they run.

-- Enable pgvector — MUST run before any vector column or index is created
CREATE EXTENSION IF NOT EXISTS vector;
