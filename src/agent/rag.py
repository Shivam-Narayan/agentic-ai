"""LlamaIndex data / RAG layer — JSON file store (default) or pgvector backend.

Backend selection is controlled by the USE_PGVECTOR flag in .env:

    USE_PGVECTOR=false  →  JSON files in indexing_data/  (default, works out of the box)
    USE_PGVECTOR=true   →  PostgreSQL + pgvector          (run migrate_to_pgvector.py first)

All public functions (build_index, rebuild_index, retrieve_documents) have the
same signature regardless of backend — nothing above this layer changes.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .config import DATA_DIR, INDEX_DIR, POSTGRES_URL, USE_PGVECTOR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported file extensions — add more here if needed
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt"}

# Number of chunks to retrieve per query.
# Bumped from the LlamaIndex default of 2 → 8 so 100-doc collections
# surface relevant content that may be spread across many files.
_SIMILARITY_TOP_K = 8

# Embedding model dimensions — must match what pgvector table was created with.
# BAAI/bge-small-en-v1.5 produces 384-dimensional vectors.
_EMBED_DIM = 384

# Module-level embedding model — loaded once, reused for all queries
_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# ---------------------------------------------------------------------------
# LlamaIndex global config
# ---------------------------------------------------------------------------

def configure_llama_index() -> None:
    """Set HuggingFace embeddings and disable LlamaIndex's own LLM."""
    Settings.embed_model = _embed_model
    Settings.llm = None


# ---------------------------------------------------------------------------
# pgvector store factory
# ---------------------------------------------------------------------------

def _parse_postgres_url(url: str) -> dict:
    """Parse POSTGRES_URL into individual components for PGVectorStore.from_params().

    PGVectorStore requires host/port/user/password/database as separate args —
    it does not accept a full connection string.

    Handles both formats:
      postgresql+psycopg://user:password@host:port/database
      postgresql://user:password@host:port/database
    """
    from urllib.parse import urlparse
    # Strip driver prefix so urlparse handles it correctly
    clean = url.replace("postgresql+psycopg://", "postgresql://")
    parsed = urlparse(clean)
    return {
        "host":     parsed.hostname or "localhost",
        "port":     parsed.port    or 5432,
        "user":     parsed.username or "postgres",
        "password": parsed.password or "password",
        "database": (parsed.path or "/datadialogue").lstrip("/"),
    }


def _get_pg_vector_store():
    """Build a PGVectorStore connected to the datadialogue database.

    Only imported when USE_PGVECTOR=true so the package is not required
    for the default JSON-backed setup.
    """
    from llama_index.vector_stores.postgres import PGVectorStore

    parts = _parse_postgres_url(POSTGRES_URL)
    return PGVectorStore.from_params(
        host=parts["host"],
        port=parts["port"],
        user=parts["user"],
        password=parts["password"],
        database=parts["database"],
        table_name="document_embeddings",
        embed_dim=_EMBED_DIM,
    )


# ---------------------------------------------------------------------------
# Index load (cached per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_vector_index():
    """Load and cache the vector index from the configured backend.

    JSON backend:    reads from indexing_data/ (default)
    pgvector backend: connects to PostgreSQL (USE_PGVECTOR=true)
    """
    configure_llama_index()

    if USE_PGVECTOR:
        logger.info("Loading vector index from pgvector (PostgreSQL)")
        vector_store = _get_pg_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # load_index_from_storage is not used for pgvector —
        # we construct the index directly from the existing store
        return VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )

    # JSON file backend (default)
    logger.info("Loading vector index from JSON store: %s", INDEX_DIR)
    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    return load_index_from_storage(storage_context)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_documents(question: str) -> List[Document]:
    """Retrieve the top-k most relevant chunks for a question.

    Returns LangChain Document objects so tools.py doesn't need to change.
    similarity_top_k=8 (up from default 2) improves answer quality when
    the collection has many documents.
    """
    nodes = get_vector_index().as_retriever(
        similarity_top_k=_SIMILARITY_TOP_K
    ).retrieve(question)
    return [
        Document(
            page_content=node.node.text,
            metadata=node.node.metadata or {},
        )
        for node in nodes
    ]


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------

def build_index(
    document_paths: List[Path] | None = None,
    extra_documents: list | None = None,
) -> None:
    """Build (or rebuild) the vector index from documents in data/.

    Works for both backends:
    - JSON: persists to indexing_data/
    - pgvector: inserts vectors into PostgreSQL document_embeddings table
    """
    configure_llama_index()

    document_paths = document_paths or _discover_documents()
    existing_files = [p for p in document_paths if p.exists()]

    documents = []
    if existing_files:
        loader = SimpleDirectoryReader(
            input_files=[str(p) for p in existing_files],
            file_extractor=_get_file_extractors(),
        )
        documents.extend(loader.load_data())

    if extra_documents:
        documents.extend(extra_documents)

    if not documents:
        raise ValueError(
            f"No source documents found. "
            f"Place PDF, DOCX, XLSX, CSV, or TXT files in: {DATA_DIR}"
        )

    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    Settings.text_splitter = text_splitter

    if USE_PGVECTOR:
        logger.info("Building index → pgvector (PostgreSQL)")
        vector_store    = _get_pg_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[text_splitter],
        )
        logger.info("pgvector index built — %d source file(s)", len(existing_files))
    else:
        logger.info("Building index → JSON store: %s", INDEX_DIR)
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[text_splitter],
        )
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(INDEX_DIR))
        logger.info("JSON index persisted to %s", INDEX_DIR)


# ---------------------------------------------------------------------------
# Rebuild (clears cache so next query loads the fresh index)
# ---------------------------------------------------------------------------

def rebuild_index() -> list[str]:
    """Rebuild the vector index from scratch and clear the lru_cache.

    Returns the list of filenames that were indexed.
    """
    discovered = _discover_documents()
    if not discovered:
        raise ValueError(f"No supported documents found in {DATA_DIR}")

    build_index(document_paths=discovered)

    # Invalidate the cached index so the next retrieve_documents() call
    # loads the freshly built index rather than the stale one.
    get_vector_index.cache_clear()
    logger.info("lru_cache cleared — fresh index will be loaded on next query")

    return [p.name for p in discovered]


# ---------------------------------------------------------------------------
# File discovery + extractors
# ---------------------------------------------------------------------------

def _discover_documents(data_dir: Path = DATA_DIR) -> List[Path]:
    """Return all supported files in the data directory."""
    found = [
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if found:
        logger.info(
            "Discovered %d document(s) in %s: %s",
            len(found), data_dir, [p.name for p in found],
        )
    else:
        logger.warning("No supported documents found in %s", data_dir)
    return found


def _get_file_extractors() -> dict:
    """Register explicit file extractors so LlamaIndex uses the right parser per type."""
    extractors: dict = {}
    try:
        from llama_index.readers.file import DocxReader
        extractors[".docx"] = DocxReader()
        extractors[".doc"]  = DocxReader()
    except ImportError:
        pass  # fall back to LlamaIndex default reader
    return extractors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    backend = "pgvector (PostgreSQL)" if USE_PGVECTOR else "JSON file store"
    print(f"Backend : {backend}")
    print(f"Scanning: {DATA_DIR}")

    discovered = _discover_documents()
    if not discovered:
        print(f"No supported files found in {DATA_DIR}.")
        print("Add PDF, DOCX, XLSX, CSV, or TXT files and re-run.")
        return

    print(f"Found {len(discovered)} file(s): {[p.name for p in discovered]}")
    print("Building index...")
    build_index(document_paths=discovered)
    print("Done.")


if __name__ == "__main__":
    main()
