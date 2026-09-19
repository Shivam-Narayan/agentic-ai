"""LlamaIndex data / RAG layer — JSON file store (default) or pgvector backend.

Backend selection is controlled by the USE_PGVECTOR flag in .env:

    USE_PGVECTOR=false  →  JSON files in indexing_data/  (default, works out of the box)
    USE_PGVECTOR=true   →  PostgreSQL + pgvector          (run migrate_to_pgvector.py first)

All public functions (build_index, rebuild_index, retrieve_documents) have the
same signature regardless of backend — nothing above this layer changes.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter

from src.core.config import DATA_DIR, INDEX_DIR, POSTGRES_URL, USE_HYBRID_SEARCH, USE_PGVECTOR, USE_RERANKER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported file extensions — add more here if needed.
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt"}

# Number of chunks to retrieve per query.
_SIMILARITY_TOP_K: int = 8

# Embedding model dimensions — must match what pgvector table was created with.
# BAAI/bge-small-en-v1.5 produces 384-dimensional vectors.
_EMBED_DIM: int = 384

# Chunking parameters — defined once here to avoid divergence between
# build_index() and add_documents_to_index().
# 1024 tokens gives enough context per chunk for multi-sentence answers
# while staying well within embedding model input limits.
_CHUNK_SIZE:    int = 1024
_CHUNK_OVERLAP: int = 100

# Embedding model name — single source of truth used by get_embed_model() and api.py.
_EMBED_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# Lazy embedding model — loaded on first call, not at import time.
# Importing rag.py no longer triggers a model download.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embed_model():
    """Return the cached HuggingFace embedding model (loaded on first call).

    Uses lru_cache so the model is initialised once per process and reused
    for all subsequent calls. This avoids the expensive cold-start penalty
    on every import of this module.

    Returns:
        HuggingFaceEmbedding instance backed by BAAI/bge-small-en-v1.5.
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: PLC0415
    logger.info("Loading embedding model: %s", _EMBED_MODEL_NAME)
    return HuggingFaceEmbedding(model_name=_EMBED_MODEL_NAME)


# ---------------------------------------------------------------------------
# LlamaIndex global config
# ---------------------------------------------------------------------------

def configure_llama_index() -> None:
    """Set HuggingFace embeddings and disable LlamaIndex's own LLM."""
    Settings.embed_model = get_embed_model()
    Settings.llm = None


# ---------------------------------------------------------------------------
# pgvector store factory
# ---------------------------------------------------------------------------

def _parse_postgres_url(url: str) -> dict[str, Any]:
    """Parse POSTGRES_URL into individual components for PGVectorStore.from_params().

    PGVectorStore requires host/port/user/password/database as separate args —
    it does not accept a full connection string.

    Handles both formats:
      postgresql+psycopg://user:password@host:port/database
      postgresql://user:password@host:port/database

    Args:
        url: Full PostgreSQL connection string.

    Returns:
        Dict with keys: host, port, user, password, database.
    """
    from urllib.parse import urlparse  # noqa: PLC0415
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

    Returns:
        PGVectorStore instance ready for indexing or querying.
    """
    from llama_index.vector_stores.postgres import PGVectorStore  # noqa: PLC0415

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
def get_vector_index() -> VectorStoreIndex:
    """Load and cache the vector index from the configured backend.

    JSON backend:     reads from indexing_data/ (default).
    pgvector backend: connects to PostgreSQL (USE_PGVECTOR=true).

    Returns:
        VectorStoreIndex ready for querying.

    Raises:
        RuntimeError: if the JSON index files are missing. Run build_index() first.
    """
    configure_llama_index()

    if USE_PGVECTOR:
        logger.info("Loading vector index from pgvector (PostgreSQL)")
        vector_store = _get_pg_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )

    logger.info("Loading vector index from JSON store: %s", INDEX_DIR)
    if not (INDEX_DIR / "docstore.json").exists():
        raise RuntimeError(
            f"Vector index not found at {INDEX_DIR}. "
            "Run `python -m src.retrieval.rag` to build the index first."
        )
    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    return load_index_from_storage(storage_context)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _get_hybrid_retriever(index: VectorStoreIndex):
    """Build a hybrid retriever combining semantic + BM25 keyword search.

    Uses Reciprocal Rank Fusion (RRF) to merge ranked lists from both
    retrievers. Strictly better than semantic-only for queries containing
    exact names, codes, or numeric identifiers.

    Args:
        index: Loaded VectorStoreIndex.

    Returns:
        QueryFusionRetriever combining vector + BM25 results.
    """
    from llama_index.core.retrievers import QueryFusionRetriever  # noqa: PLC0415
    from llama_index.retrievers.bm25 import BM25Retriever          # noqa: PLC0415

    vector_retriever = index.as_retriever(similarity_top_k=_SIMILARITY_TOP_K)
    bm25_retriever   = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=_SIMILARITY_TOP_K,
    )
    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=_SIMILARITY_TOP_K,
        num_queries=1,        # 1 = no LLM query expansion, just fuse the two lists
        mode="reciprocal_rerank",
        use_async=False,
    )


def _rerank_nodes(nodes: list, question: str) -> list:
    """Rerank retrieved nodes using FlashRank cross-encoder model.

    A cross-encoder scores every (question, chunk) pair jointly — far more
    accurate than embedding cosine similarity which encodes them separately.
    FlashRank runs locally with no API key and no GPU required.

    Falls back silently to the original order if the package is missing,
    so the app never crashes when USE_RERANKER=true but flashrank is not
    installed yet.

    Args:
        nodes:    Retrieved NodeWithScore list from the retriever.
        question: Original user question used as the reranker query.

    Returns:
        Reranked list — same nodes, better order, trimmed to _SIMILARITY_TOP_K.
    """
    try:
        from llama_index.postprocessor.flashrank_rerank import FlashRankRerank  # noqa: PLC0415
        from llama_index.core.schema import QueryBundle                          # noqa: PLC0415
        reranker  = FlashRankRerank(top_n=_SIMILARITY_TOP_K)
        reranked  = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(question))
        logger.debug("Reranker: FlashRank applied, %d → %d nodes", len(nodes), len(reranked))
        return reranked
    except ImportError:
        logger.warning(
            "FlashRank not installed — skipping reranking. "
            "Run: pip install llama-index-postprocessor-flashrank-rerank flashrank"
        )
        return nodes
    except Exception as exc:
        logger.warning("Reranking failed (%s) — using original order", exc)
        return nodes


def retrieve_documents(question: str) -> list[Document]:
    """Retrieve the top-k most relevant chunks for a question.

    Pipeline (each stage controlled by its feature flag in .env):
      1. Retrieval  — semantic only              (default)
                    — hybrid: semantic+BM25+RRF  (USE_HYBRID_SEARCH=true)
      2. Reranking  — FlashRank cross-encoder    (USE_RERANKER=true)
                      Re-scores every (question, chunk) pair jointly for
                      higher precision than cosine similarity alone.

    Args:
        question: The user's question or search query.

    Returns:
        List of LangChain Document objects with page_content and metadata.
    """
    index = get_vector_index()

    if USE_HYBRID_SEARCH:
        logger.debug("Retrieval mode: hybrid (semantic + BM25 + RRF)")
        retriever = _get_hybrid_retriever(index)
    else:
        logger.debug("Retrieval mode: semantic only")
        retriever = index.as_retriever(similarity_top_k=_SIMILARITY_TOP_K)

    nodes = retriever.retrieve(question)

    if USE_RERANKER:
        logger.debug("Reranking: FlashRank cross-encoder")
        nodes = _rerank_nodes(nodes, question)

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

def _make_text_splitter() -> SentenceSplitter:
    """Return a SentenceSplitter using the module-level chunk constants.

    Centralised so build_index() and add_documents_to_index() always use
    identical chunking parameters — no risk of divergence.

    Returns:
        Configured SentenceSplitter instance.
    """
    return SentenceSplitter(chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)


def build_index(
    document_paths: list[Path] | None = None,
    extra_documents: list | None = None,
) -> None:
    """Build (or rebuild) the vector index from documents in data/.

    Works for both backends:
    - JSON: persists to indexing_data/
    - pgvector: inserts vectors into PostgreSQL document_embeddings table

    Args:
        document_paths:  Explicit list of file paths to index. Defaults to
                         all supported files discovered in DATA_DIR.
        extra_documents: Additional pre-loaded LlamaIndex Document objects
                         to include alongside the files on disk.

    Raises:
        ValueError: if no source documents are found.
    """
    configure_llama_index()

    document_paths = document_paths or discover_documents()
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

    text_splitter = _make_text_splitter()
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

    Returns:
        List of filenames that were indexed.

    Raises:
        ValueError: if no supported documents are found in DATA_DIR.
    """
    discovered = discover_documents()
    if not discovered:
        raise ValueError(f"No supported documents found in {DATA_DIR}")

    build_index(document_paths=discovered)

    # Invalidate the cached index so the next retrieve_documents() call
    # loads the freshly built index rather than the stale one.
    get_vector_index.cache_clear()
    logger.info("lru_cache cleared — fresh index will be loaded on next query")

    return [p.name for p in discovered]


def add_documents_to_index(document_paths: list[Path]) -> list[str]:
    """Incrementally add new documents to the existing index without a full rebuild.

    Falls back to a full rebuild if the index store doesn't exist yet or if
    incremental insertion fails.

    Args:
        document_paths: List of file paths to index.

    Returns:
        List of filenames added to the index.
    """
    existing_files = [p for p in document_paths if p.exists()]
    if not existing_files:
        return []

    # If JSON store directory doesn't exist yet, do full build
    if not USE_PGVECTOR and not (INDEX_DIR / "docstore.json").exists():
        build_index()
        get_vector_index.cache_clear()
        return [p.name for p in existing_files]

    configure_llama_index()
    loader = SimpleDirectoryReader(
        input_files=[str(p) for p in existing_files],
        file_extractor=_get_file_extractors(),
    )
    documents = loader.load_data()
    if not documents:
        return []

    text_splitter = _make_text_splitter()
    Settings.text_splitter = text_splitter

    try:
        index = get_vector_index()
        for doc in documents:
            index.insert(doc)

        if not USE_PGVECTOR:
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            index.storage_context.persist(persist_dir=str(INDEX_DIR))
            logger.info(
                "Persisted updated JSON index with %d new file(s)", len(existing_files)
            )
        else:
            logger.info(
                "Inserted %d document(s) into pgvector index", len(existing_files)
            )

        get_vector_index.cache_clear()
        return [p.name for p in existing_files]

    except Exception as exc:
        logger.warning(
            "Incremental indexing failed (%s). Falling back to rebuild_index().", exc
        )
        try:
            rebuild_index()
        except Exception as rebuild_exc:
            logger.error("rebuild_index() also failed: %s", rebuild_exc)
            raise
        return [p.name for p in existing_files]


# ---------------------------------------------------------------------------
# File discovery + extractors
# ---------------------------------------------------------------------------

def discover_documents(data_dir: Path = DATA_DIR) -> list[Path]:
    """Return all supported files in the data directory.

    Public alias for the former private _discover_documents(). Exposed so
    api.py and other callers don't need to import a private symbol.

    Args:
        data_dir: Directory to scan. Defaults to the configured DATA_DIR.

    Returns:
        List of Path objects for every supported file found.
    """
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


# Keep the old private name as an alias so any remaining internal callers
# don't break until they're updated.
_discover_documents = discover_documents


def _get_file_extractors() -> dict[str, Any]:
    """Register explicit file extractors so LlamaIndex uses the right parser per type.

    Returns:
        Dict mapping file extension → reader instance.
        Empty dict if llama-index-readers-file is not installed (graceful degradation).
    """
    extractors: dict[str, Any] = {}
    try:
        from llama_index.readers.file import DocxReader  # noqa: PLC0415
        extractors[".docx"] = DocxReader()
        extractors[".doc"]  = DocxReader()
    except ImportError:
        logger.debug(
            "llama-index-readers-file not installed — DOCX will use default reader. "
            "Run: pip install llama-index-readers-file"
        )
    return extractors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Build the vector index from files in DATA_DIR (CLI entry point)."""
    backend = "pgvector (PostgreSQL)" if USE_PGVECTOR else "JSON file store"
    logger.info("Backend : %s", backend)
    logger.info("Scanning: %s", DATA_DIR)

    discovered = discover_documents()
    if not discovered:
        logger.error(
            "No supported files found in %s. "
            "Add PDF, DOCX, XLSX, CSV, or TXT files and re-run.",
            DATA_DIR,
        )
        return

    logger.info("Found %d file(s): %s", len(discovered), [p.name for p in discovered])
    logger.info("Building index…")
    build_index(document_paths=discovered)
    logger.info("Done.")


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s | %(message)s")
    main()
