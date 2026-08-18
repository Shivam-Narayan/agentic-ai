import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = ROOT_DIR / "indexing_data"

def require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Please set the {name} environment variable")
    return value


@lru_cache(maxsize=1)
def configure_runtime() -> None:
    require_env_var("GOOGLE_API_KEY")
    require_env_var("TAVILY_API_KEY")

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Gemini(model_name="models/gemini-pro-latest", temperature=0.0)


def get_llm() -> ChatGoogleGenerativeAI:
    configure_runtime()
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )


def get_retriever():
    configure_runtime()
    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    index = load_index_from_storage(storage_context)
    return index.as_retriever()


def get_web_search_tool() -> TavilySearchResults:
    configure_runtime()
    return TavilySearchResults(k=3)
