import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR  = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = ROOT_DIR / "indexing_data"
DATA_DIR  = ROOT_DIR / "data"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PostgreSQL / pgvector config
# ---------------------------------------------------------------------------

# Full psycopg3 connection string — used by both rag.py (pgvector) and
# app.py (AsyncPostgresSaver). Defaults to local Docker setup.
POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL",
    "postgresql+psycopg://postgres:password@localhost:5432/datadialogue",
)

# Feature flags — flip to true after running migrate_to_pgvector.py
USE_PGVECTOR:         bool = os.getenv("USE_PGVECTOR",         "false").lower() == "true"
USE_POSTGRES_MEMORY:  bool = os.getenv("USE_POSTGRES_MEMORY",  "false").lower() == "true"


def setup_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not configured")
    return value


def require_runtime_keys() -> None:
    # LLM key — at least one must be present
    if not any([
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("GROQ_API_KEY"),
        os.getenv("COHERE_API_KEY")
    ]):
        raise RuntimeError("At least one of GOOGLE_API_KEY, GROQ_API_KEY, or COHERE_API_KEY must be configured in .env")

    # Web search key — warn if none present, but don't crash.
    # get_web_search_tool() will fall back to DuckDuckGo (no key needed).
    if not any([
        os.getenv("TAVILY_API_KEY"),
        os.getenv("SERPER_API_KEY"),
    ]):
        logger.warning(
            "No web search API key found (TAVILY_API_KEY or SERPER_API_KEY). "
            "Falling back to DuckDuckGo — live search results may be slower or rate-limited."
        )
