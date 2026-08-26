"""LangChain layer: LLM instance."""

import logging
import os
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from .config import require_runtime_keys

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    require_runtime_keys()
    
    # Priority 1: Explicitly requested provider
    provider = os.getenv("LLM_PROVIDER", "").lower()
    
    # Priority 2: Use whatever key is available
    if provider == "groq" or (not provider and os.getenv("GROQ_API_KEY")):
        from langchain_groq import ChatGroq
        logger.info("Initializing Groq LLM (openai/gpt-oss-20b)")
        return ChatGroq(model="openai/gpt-oss-20b", temperature=0)
        
    if provider == "google" or (not provider and os.getenv("GOOGLE_API_KEY")):
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info("Initializing Google Gemini LLM (gemini-1.5-flash)")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
    if provider == "cohere" or (not provider and os.getenv("COHERE_API_KEY")):
        from langchain_cohere import ChatCohere
        logger.info("Initializing Cohere LLM (command-r-plus)")
        return ChatCohere(model="command-r-plus", temperature=0)
        
    raise RuntimeError("No valid LLM configuration found. Set LLM_PROVIDER or configure an API key.")


@lru_cache(maxsize=1)
def get_web_search_tool():
    require_runtime_keys()
    from langchain_tavily import TavilySearch
    return TavilySearch(max_results=5)
