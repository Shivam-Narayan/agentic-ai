import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = ROOT_DIR / "indexing_data"
DATA_DIR = ROOT_DIR / "data"

logger = logging.getLogger(__name__)


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
    require_env_var("GOOGLE_API_KEY")
    require_env_var("TAVILY_API_KEY")
