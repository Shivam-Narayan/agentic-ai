"""Tools the LLM uses for company docs and live web data.

Database queries are now handled dynamically by the MCP server (SQLite/Postgres).
The MCP server auto-discovers the database schema and provides query tools to the LLM.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool

from .rag import retrieve_documents

logger = logging.getLogger(__name__)


@tool
def search_company_documents(query: str) -> str:
    """Search indexed company / project documents (vector store)."""
    logger.info("Tool search_company_documents: %s", query)
    documents: List[Document] = retrieve_documents(query)
    if not documents:
        return "No matching company documents were found."

    snippets = []
    for doc in documents[:4]:
        text = " ".join(doc.page_content.split())
        if len(text) > 700:
            text = text[:700].rsplit(" ", 1)[0] + "..."
        snippets.append(text)
    return "\n\n".join(snippets)


@tool
def search_web(query: str) -> str:
    """Search the live web for facts the LLM does not know (weather, news, current events)."""
    logger.info("Tool search_web: %s", query)
    from .chains import get_web_search_tool

    docs = get_web_search_tool().invoke({"query": query})
    if docs and isinstance(docs, list):
        if isinstance(docs[0], dict):
            return "\n".join(d.get("content", str(d)) for d in docs)
        return "\n".join(str(d) for d in docs)
    return str(docs)
