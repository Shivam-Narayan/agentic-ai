import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = ROOT_DIR / "indexing_data"
DATA_DIR = ROOT_DIR / "data"
SOURCE_DOCUMENTS = [
    DATA_DIR / "KT_document_from_a_real_client_project.docx",
    DATA_DIR / "Knowledge_Transfer_Agent_Design_Plans.pdf"
]


def require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Please set the {name} environment variable")
    return value


def configure_runtime() -> None:
    load_dotenv()
    require_env_var("GOOGLE_API_KEY")
    require_env_var("TAVILY_API_KEY")

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Gemini(model_name="models/gemini-pro-latest", temperature=0.0)


def build_index(document_paths: List[Path]) -> None:
    configure_runtime()

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
    print(f"Index persisted to {INDEX_DIR}")


def main() -> None:
    print("Building the document index...")
    build_index(SOURCE_DOCUMENTS)
    print("Indexing complete.")


if __name__ == "__main__":
    main()

