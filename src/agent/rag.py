"""LlamaIndex data / RAG layer and persisted vector store."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .config import DATA_DIR, INDEX_DIR

logger = logging.getLogger(__name__)

# Supported file extensions — add more here if needed
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt"}

# Module-level embedding model — loaded once, reused for all queries
_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def configure_llama_index() -> None:
    """Set HuggingFace embeddings and disable LLM for retrieval."""
    Settings.embed_model = _embed_model
    Settings.llm = None


@lru_cache(maxsize=1)
def get_vector_index():
    """Load the persisted vector database (file store today; Postgres/pgvector can replace this)."""
    configure_llama_index()
    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    index = load_index_from_storage(storage_context)
    logger.info("Loaded vector store from %s", INDEX_DIR)
    return index


def retrieve_documents(question: str) -> List[Document]:
    nodes = get_vector_index().as_retriever().retrieve(question)
    return [Document(page_content=node.node.text) for node in nodes]


def rebuild_index() -> list[str]:
    """Rebuild the vector index from scratch and clear the lru_cache so the
    next retrieve_documents() call loads the fresh index.

    Returns the list of filenames that were indexed.
    """
    discovered = _discover_documents()
    if not discovered:
        raise ValueError(f"No supported documents found in {DATA_DIR}")

    build_index(document_paths=discovered)

    # Invalidate the cached index so the next query loads the new one
    get_vector_index.cache_clear()
    logger.info("lru_cache cleared — fresh index will be loaded on next query")

    return [p.name for p in discovered]


def _discover_documents(data_dir: Path = DATA_DIR) -> List[Path]:
    """Discover all supported files in the data directory."""
    found = [
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if found:
        logger.info("Discovered %d document(s) in %s: %s", len(found), data_dir, [p.name for p in found])
    else:
        logger.warning("No supported documents found in %s", data_dir)
    return found


def _get_file_extractors() -> dict:
    """Return explicit file extractors so LlamaIndex uses the correct parser per type."""
    extractors = {}

    # DOCX -- use DocxReader (backed by docx2txt) for clean text extraction
    try:
        from llama_index.readers.file import DocxReader
        extractors[".docx"] = DocxReader()
        extractors[".doc"] = DocxReader()
    except ImportError:
        pass  # fall back to LlamaIndex default

    # PDF -- pypdf is already installed; LlamaIndex uses it by default
    # CSV / TXT -- LlamaIndex handles these natively, no override needed

    return extractors


def build_index(document_paths: List[Path] | None = None, extra_documents: list | None = None) -> None:
    configure_llama_index()

    # Auto-discover all supported files from data/ when no explicit list is given
    document_paths = document_paths or _discover_documents()

    existing_files = [path for path in document_paths if path.exists()]

    documents = []
    if existing_files:
        loader = SimpleDirectoryReader(
            input_files=[str(path) for path in existing_files],
            file_extractor=_get_file_extractors(),
        )
        documents.extend(loader.load_data())

    if extra_documents:
        documents.extend(extra_documents)

    if not documents:
        raise ValueError(
            f"No source documents were found. "
            f"Place PDF, DOCX, XLSX, CSV, or TXT files in: {DATA_DIR}"
        )

    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    Settings.text_splitter = text_splitter

    index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter])
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    logger.info("Index persisted to %s", INDEX_DIR)


def main() -> None:
    print(f"Scanning {DATA_DIR} for documents...")
    discovered = _discover_documents()
    if not discovered:
        print(f"No supported files found in {DATA_DIR}. Add PDF, DOCX, XLSX, CSV, or TXT files and re-run.")
        return
    print(f"Found {len(discovered)} file(s): {[p.name for p in discovered]}")
    print("Building index...")
    build_index(document_paths=discovered)
    print(f"Done. Vector store saved to: {INDEX_DIR}")


if __name__ == "__main__":
    main()
