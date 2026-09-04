-- init.sql
-- Runs once automatically when the Postgres container starts for the first time.
-- LangGraph's AsyncPostgresSaver creates its own checkpoint tables on first run
-- via cp.setup() — we only need to create the pgvector extension and embeddings table.

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Document embeddings table
-- embed_dim=384 matches BAAI/bge-small-en-v1.5 (the project's embedding model)
CREATE TABLE IF NOT EXISTS document_embeddings (
    id          BIGSERIAL PRIMARY KEY,
    text        TEXT        NOT NULL,
    metadata    JSONB       NOT NULL DEFAULT '{}',
    embedding   vector(384)
);

-- IVFFlat index for fast approximate nearest-neighbour search
-- lists=100 is a good default for up to ~1M vectors; increase for larger collections
CREATE INDEX IF NOT EXISTS idx_document_embeddings_ivfflat
    ON document_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Exact index for smaller collections (used as fallback when IVFFlat isn't probed)
CREATE INDEX IF NOT EXISTS idx_document_embeddings_metadata
    ON document_embeddings
    USING GIN (metadata);
