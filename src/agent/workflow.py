"""LangGraph: agentic tool-calling architecture with conversation memory and source citations."""

import json
import logging
import re
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .chains import get_llm
from .mcp_client import mcp_server_context
from .tools import (
    calculate,
    extract_structured_data,
    generate_chart,
    search_company_documents,
    search_web,
    summarise_document,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — injected at the start of every agent invocation.
# Drives citation behaviour and sets the agent's persona.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the CTE Knowledge Transfer Assistant — an expert at helping
team members find information about company projects, documents, databases, and data.

RULES YOU MUST ALWAYS FOLLOW:
1. CITATIONS — Always cite your sources:
   - For document answers: mention the exact filename (e.g. "According to handbook.docx...")
   - For database answers: show the SQL query you used in a code block
   - For web answers: include the URL
   - For calculations: show the expression and result
2. ACCURACY — Never guess numbers. Use the `calculate` tool for all arithmetic.
3. CHARTS — When presenting tabular or numeric results, proactively offer to generate
   a chart using the `generate_chart` tool. If the user asks for a chart, always use it.
4. EXTRACTION — When asked to pull specific fields or facts from a document, use
   the `extract_structured_data` tool rather than quoting manually.
5. SUMMARIES — When asked for an overview or summary of a whole document, use the
   `summarise_document` tool rather than relying only on vector search.
6. MEMORY — You have access to the full conversation history. Use it to answer
   follow-up questions without asking the user to repeat themselves.
"""


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else END


def build_graph(dynamic_tools: list):
    tool_node = ToolNode(dynamic_tools)

    def agent_node(state: AgentState) -> dict:
        llm = get_llm().bind_tools(dynamic_tools)
        # Always prepend the system prompt as the first message so it applies
        # to every turn including follow-up questions in a multi-turn session.
        messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages_with_system)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile()


# ---------------------------------------------------------------------------
# Result parsing — extracts datasource, citations, and chart data
# ---------------------------------------------------------------------------

# Maps tool name → datasource label
_TOOL_DATASOURCE = {
    "search_company_documents": "company_docs",
    "summarise_document":        "company_docs",
    "extract_structured_data":   "company_docs",
    "search_web":                "web_search",
    "calculate":                 "calculation",
    "generate_chart":            "chart",
    # MCP database tools (both old and new naming conventions)
    "query_company_database":    "database",
    "query-database":            "database",
    "read-query":                "database",
    "list_database_tables":      "database",
    "list-tables":               "database",
    "describe_database_table":   "database",
}


def _extract_citations(messages: list) -> list[dict]:
    """
    Walk all ToolMessages and build a list of citation dicts.
    Citations are sourced from:
      - [Source: <value>] prefixes emitted by search_company_documents / search_web
      - SQL queries extracted from query_company_database arguments
      - calculate expressions
      - summarise_document / extract_structured_data filenames
    """
    citations: list[dict] = []
    seen: set[str] = set()

    def _add(source: str, detail: str = "") -> None:
        key = f"{source}::{detail}"
        if key not in seen:
            seen.add(key)
            citations.append({"source": source, "detail": detail})

    for msg in messages:
        if msg.type != "tool":
            continue

        name = msg.name or ""
        content = msg.content or ""

        if name in ("search_company_documents", "search_web"):
            # Extract [Source: <value>] markers from the tool output
            for match in re.finditer(r"\[Source:\s*(.+?)\]", content):
                _add(match.group(1).strip())

        elif name in ("query_company_database", "query-database", "read-query"):
            # The tool was called with a SQL argument — extract from the AIMessage above
            for ai_msg in messages:
                if ai_msg.type == "ai" and getattr(ai_msg, "tool_calls", None):
                    for tc in ai_msg.tool_calls:
                        if tc.get("name") == name:
                            sql = (
                                tc.get("args", {}).get("sql_query")
                                or tc.get("args", {}).get("query")
                                or ""
                            )
                            if sql:
                                _add("company database", sql.strip())

        elif name == "calculate":
            # Content is "expression = result"
            _add("calculator", content.strip())

        elif name == "summarise_document":
            # Content starts with "[Full text of <filename>"
            match = re.match(r"\[Full text of (.+?)\]", content)
            if match:
                _add(match.group(1).strip())
            else:
                _add("company documents")

        elif name == "extract_structured_data":
            # Pull any [Source: ...] markers that rag.py embedded
            for match in re.finditer(r"\[Source:\s*(.+?)\]", content):
                _add(match.group(1).strip())

        elif name in ("list_database_tables", "list-tables",
                      "describe_database_table"):
            _add("company database", f"schema lookup via {name}")

    return citations


def _extract_chart(messages: list) -> dict | None:
    """Return the Plotly figure JSON dict if any ToolMessage contains a chart."""
    for msg in messages:
        if msg.type == "tool" and msg.name == "generate_chart":
            content = msg.content or ""
            if content.startswith("CHART_JSON::"):
                try:
                    return json.loads(content[len("CHART_JSON::"):])
                except json.JSONDecodeError:
                    logger.warning("Could not parse chart JSON from generate_chart output")
    return None


def parse_result(result: dict) -> dict:
    messages = result["messages"]
    answer = messages[-1].content

    tools_used: list[str] = []
    datasources: set[str] = set()

    for msg in messages:
        if msg.type == "tool":
            name = msg.name or ""
            tools_used.append(name)
            ds = _TOOL_DATASOURCE.get(name)
            if ds:
                datasources.add(ds)

    if len(datasources) > 1:
        datasource = "multiple"
    elif datasources:
        datasource = next(iter(datasources))
    else:
        datasource = "direct_llm"

    citations = _extract_citations(messages)
    chart_data = _extract_chart(messages)

    return {
        "generation":  answer,
        "datasource":  datasource,
        "tools_used":  tools_used,
        "citations":   citations,
        "chart_data":  chart_data,
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

LOCAL_TOOLS = [
    search_company_documents,
    search_web,
    summarise_document,
    extract_structured_data,
    calculate,
    generate_chart,
]


async def aask(question: str, history: list | None = None) -> dict:
    """Async entry point called by FastAPI.

    Args:
        question: The user's current question.
        history:  Optional list of prior LangChain message objects for
                  conversation memory. Pass [] or None for a fresh session.
    """
    prior_messages = history or []

    async with mcp_server_context() as mcp_tools:
        all_tools = LOCAL_TOOLS + mcp_tools
        graph = build_graph(all_tools)
        initial_state = {
            "messages": prior_messages + [HumanMessage(content=question)]
        }
        result = await graph.ainvoke(initial_state)
        return parse_result(result)


def ask(question: str, history: list | None = None) -> dict:
    """Synchronous wrapper around aask — for scripts and tests."""
    import asyncio
    return asyncio.run(aask(question, history))


class KnowledgeTransferAgent:
    """Streaming interface for the agent (useful for SSE / future streaming endpoint)."""

    def __init__(self) -> None:
        pass

    async def run(self, question: str, history: list | None = None):
        prior_messages = history or []
        async with mcp_server_context() as mcp_tools:
            all_tools = LOCAL_TOOLS + mcp_tools
            graph = build_graph(all_tools)
            async for step in graph.astream(
                {"messages": prior_messages + [HumanMessage(content=question)]}
            ):
                yield step
