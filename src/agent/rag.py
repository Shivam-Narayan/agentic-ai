"""LlamaIndex data / RAG layer and persisted vector store."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

from .config import DATA_DIR, INDEX_DIR, require_runtime_keys

logger = logging.getLogger(__name__)

SOURCE_DOCUMENTS = [
    DATA_DIR / "KT_document_from_a_real_client_project.docx",
    DATA_DIR / "Knowledge_Transfer_Agent_Design_Plans.pdf",
]


def configure_llama_index() -> None:
    require_runtime_keys()
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Gemini(model_name="models/gemini-pro-latest", temperature=0.0)


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


def build_index(document_paths: List[Path] | None = None) -> None:
    configure_llama_index()
    document_paths = document_paths or SOURCE_DOCUMENTS

    existing_files = [path for path in document_paths if path.exists()]
    if not existing_files:
        raise FileNotFoundError("No source documents were found for indexing.")

    loader = SimpleDirectoryReader(input_files=[str(path) for path in existing_files])
    documents = loader.load_data()
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    Settings.text_splitter = text_splitter

    index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter])
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    logger.info("Index persisted to %s", INDEX_DIR)


def main() -> None:
    print("Building the document index...")
    build_index()
    print(f"Indexing complete. Vector store: {INDEX_DIR}")


if __name__ == "__main__":
    main()
