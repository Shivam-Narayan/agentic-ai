"""LangChain layer: LLM instance and web search tool with fallback chain."""

import logging
import os
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from .config import require_runtime_keys

logger = logging.getLogger(__name__)


def _make_serper_tool():
    """Build a Serper (Google) search tool. Returns None if unavailable."""
    try:
        from langchain_community.utilities import GoogleSerperAPIWrapper
        from langchain_community.tools import GoogleSerperRun
        logger.info("Web search: using Serper (Google)")
        return GoogleSerperRun(api_wrapper=GoogleSerperAPIWrapper(k=5))
    except Exception as exc:
        logger.warning("Serper init failed (%s)", exc)
        return None


def _make_ddg_tool():
    """Build a DuckDuckGo search tool. Always available — no key needed."""
    from langchain_community.tools import DuckDuckGoSearchRun
    logger.info("Web search: using DuckDuckGo (no key required)")
    return DuckDuckGoSearchRun()


class _FallbackSearchTool(BaseTool):
    """Wraps Tavily and falls back to Serper → DuckDuckGo on quota/runtime errors.

    This is needed because Tavily initialises fine even when the monthly quota is
    exhausted — the error only surfaces on the actual search call.
    """

    name: str = "web_search"
    description: str = (
        "Search the live web for current information, news, prices, and real-time data. "
        "Input should be a search query string."
    )
    # Tools stored as Any to avoid Pydantic field issues with arbitrary objects
    _primary: Any = None
    _fallbacks: list = []

    def __init__(self, primary, fallbacks: list):
        super().__init__()
        object.__setattr__(self, "_primary", primary)
        object.__setattr__(self, "_fallbacks", fallbacks)

    def _run(self, query: str) -> str:
        candidates = [self._primary] + self._fallbacks
        last_exc = None
        for tool in candidates:
            if tool is None:
                continue
            tool_name = getattr(tool, "name", type(tool).__name__)
            try:
                result = tool.invoke(query)
                # Tavily returns {"error": ...} instead of raising on quota errors
                if isinstance(result, dict) and "error" in result:
                    raise RuntimeError(result["error"])
                return result
            except Exception as exc:
                logger.warning("Search tool '%s' failed: %s — trying next fallback", tool_name, exc)
                last_exc = exc
        raise RuntimeError(f"All web search providers failed. Last error: {last_exc}")

    async def _arun(self, query: str) -> str:
        # Run sync version in the default executor — all three providers are sync-only
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, query)


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    require_runtime_keys()
    
    # Priority 1: Explicitly requested provider
    provider = os.getenv("LLM_PROVIDER", "").lower()
    
    # Priority 2: Use whatever key is available
    if provider == "groq" or (not provider and os.getenv("GROQ_API_KEY")):
        from langchain_groq import ChatGroq
        logger.info("Initializing Groq LLM (openai/gpt-oss-120b)")
        return ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            max_retries=5,
            streaming=True,
        )
        
    if provider == "google" or (not provider and os.getenv("GOOGLE_API_KEY")):
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info("Initializing Google Gemini LLM (gemini-1.5-flash)")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, streaming=True)
        
    if provider == "cohere" or (not provider and os.getenv("COHERE_API_KEY")):
        from langchain_cohere import ChatCohere
        logger.info("Initializing Cohere LLM (command-r-plus)")
        return ChatCohere(model="command-r-plus", temperature=0, streaming=True)
        
    raise RuntimeError("No valid LLM configuration found. Set LLM_PROVIDER or configure an API key.")


@lru_cache(maxsize=1)
def get_web_search_tool():
    """Return a web search tool with automatic runtime fallback.

    Priority order (first available wins as primary):
      1. Tavily     — TAVILY_API_KEY set  (AI-optimised, best for RAG)
      2. Serper     — SERPER_API_KEY set  (real Google results, 2 500 free credits)
      3. DuckDuckGo — no key needed       (always-available free fallback)

    Even if Tavily is selected as primary, quota/runtime errors at query time
    automatically cascade to Serper then DuckDuckGo via _FallbackSearchTool.
    """
    primary = None
    fallbacks = []

    # --- Primary: Tavily ---
    if os.getenv("TAVILY_API_KEY"):
        try:
            from langchain_tavily import TavilySearch
            primary = TavilySearch(max_results=5)
            logger.info("Web search primary: Tavily")
        except Exception as exc:
            logger.warning("Tavily init failed (%s)", exc)

    # --- Fallback 1: Serper ---
    if os.getenv("SERPER_API_KEY"):
        serper = _make_serper_tool()
        if serper:
            if primary is None:
                primary = serper
                logger.info("Web search primary: Serper (Tavily unavailable)")
            else:
                fallbacks.append(serper)
                logger.info("Web search fallback #1: Serper")

    # --- Fallback 2: DuckDuckGo (always appended) ---
    ddg = _make_ddg_tool()
    if primary is None:
        primary = ddg
        logger.info("Web search primary: DuckDuckGo (no keys configured)")
    else:
        fallbacks.append(ddg)
        logger.info("Web search fallback #%d: DuckDuckGo", len(fallbacks))

    # If there are no fallbacks, return the primary directly (no wrapper overhead)
    if not fallbacks:
        return primary

    return _FallbackSearchTool(primary=primary, fallbacks=fallbacks)
