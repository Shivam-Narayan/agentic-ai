"""
LangGraph agentic workflow for the CTE Knowledge Transfer Assistant.

Architecture overview:
  User question
       |
       v
  [ agent_node ]  <--------------------+
       |                               |
       | tool_calls present?           |
      yes                              |
       |                               |
       v                               |
  [ tool_node ]  (executes the tool)   |
       |                               |
       +-------------------------------+  (loop back to agent with tool result)
       |
       | no tool_calls (final answer ready)
       v
  [ END ]  -> parse_result() -> FastAPI response

Key design decisions:
  - parallel_tool_calls=False  : forces the LLM to call one tool at a time
  - Deduplication guard        : code-level block on repeated search_company_documents calls
  - Dynamic system prompt      : stamped with live date/time on every request
  - recursion_limit=8          : caps the agent loop to prevent runaway API usage
"""

import json
import logging
import re
from datetime import datetime
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
# System prompt
#
# Built fresh on every agent invocation so the date/time is always accurate.
# Contains three key sections:
#   1. When to answer directly vs. use a tool  (prevents unnecessary tool calls)
#   2. Hard constraints on tool usage           (prevents the loop/rate-limit problem)
#   3. Output and citation rules
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    """Return the system prompt with the current server date/time embedded."""
    now = datetime.now()
    date_str = now.strftime("%A, %d %B %Y")  # e.g. "Wednesday, 26 August 2026"
    time_str = now.strftime("%H:%M")

    return f"""You are the CTE Knowledge Transfer Assistant -- an expert at helping
team members find information about company projects, documents, databases, and data.

CURRENT DATE AND TIME: {date_str}, {time_str} (server local time)
Always use the date and day above when answering questions about today's date or day.
Never guess or infer the day of the week from your training data.

WHEN TO USE TOOLS vs ANSWER DIRECTLY:

Answer DIRECTLY from your own knowledge (NO tools needed) when the question is about:
- General technology concepts: "What is a vector database?", "Explain RAG", "What is Python?"
- Programming, software engineering, or AI/ML concepts
- Definitions, explanations, how-things-work questions
- Today's date or day of the week (use the CURRENT DATE AND TIME above)
- Anything that doesn't reference a specific internal document, person, or company data

Use tools ONLY when the question refers to:
- A specific internal document, file, or uploaded content ("Shivam's resume", "the KT doc")
- Company-specific data, projects, or people
- A live web fact (prices, news, current events)
- A calculation or chart request

CRITICAL CONSTRAINTS:

1. NO REDUNDANT TOOL CALLS -- NEVER call the same tool more than once per question.
   Call a search tool ONCE, get the result, then write your final answer. Do not
   re-search with different wording.

2. SINGLE TOOL PER STEP -- Call exactly ONE tool per reasoning step.

3. STOP AFTER ONE SEARCH -- After receiving tool results, your next message must be
   your final answer. Never call another search tool after getting results.

TOOL SELECTION GUIDE (only when a tool is actually needed):
- Internal document/person/file question -> search_company_documents ONCE
- "Summarise [filename]" -> summarise_document ONCE
- "Extract [fields] from [doc]" -> extract_structured_data ONCE
- Live web fact -> search_web ONCE
- Math calculation -> calculate ONCE
- Chart/graph request -> generate_chart ONCE

RULES:
- Always cite sources for document/database/web answers (filename, SQL, URL)
- For web search results: quote the VALUE from the source (price, rate, number) and cite
  the URL. Do NOT repeat dates shown inside snippets -- just say "as of the latest data"
  unless the source explicitly states today's date
- Never guess numbers -- use calculate tool for arithmetic
- Use conversation history for follow-up questions without re-calling tools
"""


# ---------------------------------------------------------------------------
# Agent state
#
# LangGraph passes this TypedDict between every node in the graph.
# `add_messages` is a reducer that appends new messages to the list
# instead of replacing it, which gives us the full conversation history.
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Router: decides whether to run another tool or stop
#
# Called after every agent_node run. If the LLM produced tool_calls, we
# route to the tool node. If it produced a plain text response, we're done.
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    """Return 'tools' if the last AI message requested a tool call, else END."""
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else END


# ---------------------------------------------------------------------------
# Deduplication helper
#
# Scans all previous AI messages to build a set of "toolname::query" keys
# for every tool call that has already been executed in this conversation.
# Used by the dedup guard inside agent_node.
# ---------------------------------------------------------------------------

def _get_previous_tool_calls(messages: list) -> set[str]:
    """
    Return a set of 'toolname::query' keys for every tool call already made.

    Covers the primary argument of each tool:
      - search_company_documents / search_web -> 'query'
      - summarise_document                    -> 'document_name'
      - calculate                             -> 'expression'
    """
    used = set()
    for msg in messages:
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})
                # Pick the most identifying argument for this tool type
                query = (
                    args.get("query")
                    or args.get("document_name")
                    or args.get("expression")
                    or ""
                )
                used.add(f"{name}::{query.lower().strip()}")
    return used


# ---------------------------------------------------------------------------
# Graph construction
#
# Builds a compiled LangGraph StateGraph with two nodes:
#   - "agent"  : runs the LLM, optionally producing tool_calls
#   - "tools"  : executes whichever tool the LLM requested
#
# The graph loops  agent -> tools -> agent  until the LLM stops calling tools.
# ---------------------------------------------------------------------------

def build_graph(dynamic_tools: list):
    """
    Compile and return the LangGraph agent graph.

    Args:
        dynamic_tools: Combined list of local tools + MCP database tools.
                       MCP tools are discovered at runtime per request.
    """
    # ToolNode wraps all available tools and dispatches by tool name.
    tool_node = ToolNode(dynamic_tools)

    def agent_node(state: AgentState) -> dict:
        """
        Core LLM node. Steps:
          1. Bind all tools to the LLM (with parallel_tool_calls=False so
             the model calls one tool at a time, reducing API usage).
          2. Inject the system prompt (with live date) at the front of the
             message list.
          3. Invoke the LLM.
          4. Run the deduplication guard to block repeated search calls.
          5. If the guard blocks and the response is empty, re-invoke the
             LLM without tools to force a direct answer.
        """
        # Bind tools and enforce single tool per step at the API level
        llm = get_llm().bind_tools(dynamic_tools, parallel_tool_calls=False)

        # Prepend the system prompt to the full conversation history
        messages_with_system = [SystemMessage(content=_build_system_prompt())] + state["messages"]

        # First LLM call — may produce tool_calls or a direct answer
        response = llm.invoke(messages_with_system)

        # ── Deduplication guard ──────────────────────────────────────────
        # Problem: LLMs sometimes call search_company_documents multiple
        # times with slightly different queries ("Shivam skills", "Shivam",
        # "Shivam profile"...). Each call burns a Groq API request, hitting
        # the 30 RPM rate limit within seconds.
        #
        # Fix: track which tool+query combos have already been used this
        # turn and block any repeats at the Python level — prompt alone is
        # not reliable enough to stop this behaviour.
        if getattr(response, "tool_calls", None):

            # Count how many times search_company_documents has already run
            already_used = _get_previous_tool_calls(state["messages"])
            search_call_count = sum(
                1 for k in already_used
                if k.startswith("search_company_documents::")
            )

            blocked = False
            filtered_calls = []

            for tc in response.tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})
                query = (
                    args.get("query")
                    or args.get("document_name")
                    or args.get("expression")
                    or ""
                )
                key = f"{name}::{query.lower().strip()}"

                # Block condition 1: exact same tool+query already ran
                # Block condition 2: search_company_documents called > once
                #   (regardless of query variation — one search is enough)
                is_redundant_search = (
                    name == "search_company_documents" and search_call_count >= 1
                )

                if key in already_used or is_redundant_search:
                    logger.warning(
                        "Dedup guard: blocked redundant tool call %s(%s). "
                        "Forcing direct answer.",
                        name, query
                    )
                    blocked = True
                    break  # stop processing further tool calls

                filtered_calls.append(tc)

            if blocked:
                # Remove all pending tool calls from the response
                response.tool_calls = []

                if not response.content or not str(response.content).strip():
                    # The model put its "thinking" inside the tool call and
                    # left content empty. Re-invoke without tools so it is
                    # forced to write a real text answer instead.
                    logger.info("Dedup guard: re-invoking LLM for direct answer")
                    bare_llm = get_llm()  # no tools bound this time
                    direct_messages = messages_with_system + [
                        SystemMessage(
                            content=(
                                "You have already searched the documents. "
                                "Now write your final answer directly to the user "
                                "based on the search results in the conversation above. "
                                "Do NOT call any more tools."
                            )
                        )
                    ]
                    response = bare_llm.invoke(direct_messages)

            elif filtered_calls != response.tool_calls:
                # Some calls were filtered but not all — update the list
                response.tool_calls = filtered_calls

        return {"messages": [response]}

    # ── Wire up the graph ────────────────────────────────────────────────
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    # Always start at the agent node
    builder.add_edge(START, "agent")

    # After agent: go to tools if there are tool_calls, otherwise finish
    builder.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )

    # After tool execution: always go back to the agent for the next step
    builder.add_edge("tools", "agent")

    # checkpointer=None means no persistent memory between separate requests
    # (session memory is handled at the FastAPI layer via _sessions dict)
    return builder.compile(checkpointer=None)


# ---------------------------------------------------------------------------
# Result parsing
#
# After the graph finishes, walk the message list to extract:
#   - The final answer text (last non-empty AI message)
#   - Which tools were used (from ToolMessages)
#   - Which datasource category those tools map to
#   - Citations (file names, URLs, SQL queries)
#   - Chart data (Plotly JSON emitted by generate_chart)
# ---------------------------------------------------------------------------

# Maps each tool name to a UI datasource label shown in the Streamlit frontend
_TOOL_DATASOURCE = {
    "search_company_documents": "company_docs",
    "summarise_document":        "company_docs",
    "extract_structured_data":   "company_docs",
    "search_web":                "web_search",
    "calculate":                 "calculation",
    "generate_chart":            "chart",
    # MCP database tools — support both naming conventions (hyphen and underscore)
    "query_company_database":    "database",
    "query-database":            "database",
    "read-query":                "database",
    "list_database_tables":      "database",
    "list-tables":               "database",
    "describe_database_table":   "database",
}


def _extract_citations(messages: list) -> list[dict]:
    """
    Walk all ToolMessages and build a deduplicated list of citation dicts.

    Each citation is {"source": str, "detail": str}.  Sources are extracted:
      - search_company_documents / search_web: from [Source: <value>] markers
        that the tools embed in their output strings
      - query_company_database: SQL query pulled from the preceding AI message's
        tool_call args (so we show the exact query that ran)
      - calculate: the expression + result string
      - summarise_document: the filename parsed from the "[Full text of ...]" prefix
      - extract_structured_data: same [Source: ...] pattern as search tools
      - schema tools (list_tables, describe_table): labelled as "schema lookup"
    """
    citations: list[dict] = []
    seen: set[str] = set()

    def _add(source: str, detail: str = "") -> None:
        """Add a citation only if we haven't seen this source+detail pair before."""
        key = f"{source}::{detail}"
        if key not in seen:
            seen.add(key)
            citations.append({"source": source, "detail": detail})

    for msg in messages:
        # Only ToolMessages carry citation-relevant information
        if msg.type != "tool":
            continue

        name = msg.name or ""
        content = msg.content or ""

        if name in ("search_company_documents", "search_web"):
            # Tools embed "[Source: filename_or_url]" at the start of each snippet
            for match in re.finditer(r"\[Source:\s*(.+?)\]", content):
                _add(match.group(1).strip())

        elif name in ("query_company_database", "query-database", "read-query"):
            # The SQL is in the AI message's tool_call args, not in the tool output
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
            # Tool returns "expression = result" — cite the full expression
            _add("calculator", content.strip())

        elif name == "summarise_document":
            # Tool wraps output with "[Full text of <filename>]"
            match = re.match(r"\[Full text of (.+?)\]", content)
            if match:
                _add(match.group(1).strip())
            else:
                _add("company documents")

        elif name == "extract_structured_data":
            # Same [Source: ...] markers as search tools
            for match in re.finditer(r"\[Source:\s*(.+?)\]", content):
                _add(match.group(1).strip())

        elif name in ("list_database_tables", "list-tables", "describe_database_table"):
            _add("company database", f"schema lookup via {name}")

    return citations


def _extract_chart(messages: list) -> dict | None:
    """
    Return the Plotly figure as a JSON dict if generate_chart was used.

    generate_chart prefixes its output with "CHART_JSON::" followed by the
    serialised Plotly figure so parse_result can detect and forward it to
    the Streamlit frontend for inline rendering.
    """
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
    """
    Convert the raw LangGraph output into a clean response dict for FastAPI.

    Returns:
        {
            "generation":  str   — the final answer text
            "datasource":  str   — which data source was used (for UI badge)
            "tools_used":  list  — names of every tool that ran
            "citations":   list  — list of {"source", "detail"} dicts
            "chart_data":  dict  — Plotly figure JSON, or None
        }
    """
    messages = result["messages"]

    # Walk backwards through messages to find the last non-empty AI response.
    # We can't just take messages[-1] because the dedup guard might have
    # produced an empty AI message before the re-invoked direct answer.
    answer = ""
    for msg in reversed(messages):
        if msg.type == "ai" and msg.content and str(msg.content).strip():
            answer = str(msg.content).strip()
            break

    if not answer:
        answer = "I was unable to generate an answer. Please try rephrasing your question."

    # Collect tool names and map them to datasource categories
    tools_used: list[str] = []
    datasources: set[str] = set()

    for msg in messages:
        if msg.type == "tool":
            name = msg.name or ""
            tools_used.append(name)
            ds = _TOOL_DATASOURCE.get(name)
            if ds:
                datasources.add(ds)

    # If multiple datasource types were used, label it "multiple"
    if len(datasources) > 1:
        datasource = "multiple"
    elif datasources:
        datasource = next(iter(datasources))
    else:
        # No tools ran — the LLM answered from its own knowledge
        datasource = "direct_llm"

    citations  = _extract_citations(messages)
    chart_data = _extract_chart(messages)

    return {
        "generation":  answer,
        "datasource":  datasource,
        "tools_used":  tools_used,
        "citations":   citations,
        "chart_data":  chart_data,
    }


# ---------------------------------------------------------------------------
# Static tool list
#
# These tools are always available regardless of database configuration.
# MCP tools (database query tools) are discovered dynamically per request
# inside aask() via mcp_server_context().
# ---------------------------------------------------------------------------

LOCAL_TOOLS = [
    search_company_documents,  # vector search over indexed PDFs / DOCX / XLSX
    search_web,                # live Tavily web search
    summarise_document,        # read and summarise a full file by name
    extract_structured_data,   # pull specific fields from document context
    calculate,                 # safe arithmetic expression evaluator
    generate_chart,            # Plotly chart generator, returns JSON for UI
]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def aask(question: str, history: list | None = None) -> dict:
    """
    Primary async entry point — called by the FastAPI /ask endpoint.

    Flow:
      1. Prepend conversation history so the agent has multi-turn memory.
      2. Discover MCP tools (SQLite/Postgres database tools) for this request.
      3. Combine local tools + MCP tools and build the graph.
      4. Run the graph (agent <-> tools loop) until a final answer is produced.
      5. Parse and return a structured response dict.

    Args:
        question: The user's current message.
        history:  Prior LangChain message objects (HumanMessage / AIMessage)
                  from the session store. Pass None or [] for a fresh session.
    """
    prior_messages = history or []

    async with mcp_server_context() as mcp_tools:
        # Combine static local tools with dynamic database tools from MCP
        all_tools = LOCAL_TOOLS + mcp_tools
        graph = build_graph(all_tools)

        initial_state = {
            "messages": prior_messages + [HumanMessage(content=question)]
        }

        # recursion_limit=8 caps the agent loop to at most 8 steps
        # (prevents runaway loops from burning through rate limits)
        result = await graph.ainvoke(
            initial_state,
            config={"recursion_limit": 8},
        )
        return parse_result(result)


def ask(question: str, history: list | None = None) -> dict:
    """
    Synchronous wrapper around aask — used by CLI scripts and unit tests.

    Not suitable for production use inside an async web server (blocks the
    event loop). Use aask() directly in FastAPI route handlers.
    """
    import asyncio
    return asyncio.run(aask(question, history))


class KnowledgeTransferAgent:
    """
    Streaming interface for the agent.

    Yields each graph step as it completes, which enables Server-Sent Events
    (SSE) for real-time streaming responses in the frontend. Not yet wired
    into the FastAPI app but ready to use.
    """

    def __init__(self) -> None:
        pass

    async def run(self, question: str, history: list | None = None):
        """Stream graph steps for the given question."""
        prior_messages = history or []
        async with mcp_server_context() as mcp_tools:
            all_tools = LOCAL_TOOLS + mcp_tools
            graph = build_graph(all_tools)
            async for step in graph.astream(
                {"messages": prior_messages + [HumanMessage(content=question)]}
            ):
                yield step
