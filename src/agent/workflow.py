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

import logging
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .chains import get_llm
from .mcp_client import mcp_server_context
from .prompt import _build_system_prompt
from .parser import parse_result
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
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Router: decides whether to run another tool or stop
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    """Return 'tools' if the last AI message requested a tool call, else END."""
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else END


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------

def _get_previous_tool_calls(messages: list) -> set[str]:
    """
    Return a set of 'toolname::query' keys for every tool call already made.
    """
    used = set()
    for msg in messages:
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})
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
# ---------------------------------------------------------------------------

def build_graph(dynamic_tools: list, checkpointer=None):
    """Compile and return the LangGraph agent graph."""
    tool_node = ToolNode(dynamic_tools)

    def agent_node(state: AgentState) -> dict:
        """Core LLM node with deduplication guard."""
        llm = get_llm().bind_tools(dynamic_tools, parallel_tool_calls=False)
        messages_with_system = [SystemMessage(content=_build_system_prompt())] + state["messages"]
        response = llm.invoke(messages_with_system)

        # ── Deduplication guard ──────────────────────────────────────────
        if getattr(response, "tool_calls", None):
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

                is_redundant_search = (
                    name == "search_company_documents" and search_call_count >= 1
                )

                if key in already_used or is_redundant_search:
                    logger.warning(
                        "Dedup guard: blocked redundant tool call %s(%s). Forcing direct answer.",
                        name, query
                    )
                    blocked = True
                    break

                filtered_calls.append(tc)

            if blocked:
                response.tool_calls = []
                if not response.content or not str(response.content).strip():
                    logger.info("Dedup guard: re-invoking LLM for direct answer")
                    bare_llm = get_llm()
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
                response.tool_calls = filtered_calls

        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Static tool list
# ---------------------------------------------------------------------------

LOCAL_TOOLS = [
    search_company_documents,  
    search_web,                
    summarise_document,        
    extract_structured_data,   
    calculate,                 
    generate_chart,            
]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def aask(question: str, session_id: str = "default",
               checkpointer=None, history: list | None = None) -> dict:
    """Primary async entry point — called by the FastAPI /ask endpoint."""
    async with mcp_server_context() as mcp_tools:
        all_tools = LOCAL_TOOLS + mcp_tools
        graph = build_graph(all_tools, checkpointer=checkpointer)

        if checkpointer is not None:
            initial_state = {"messages": [HumanMessage(content=question)]}
        else:
            prior_messages = history or []
            initial_state = {"messages": prior_messages + [HumanMessage(content=question)]}

        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 8,
        }

        result = await graph.ainvoke(initial_state, config=config)
        return parse_result(result)


def ask(question: str, session_id: str = "default",
        checkpointer=None, history: list | None = None) -> dict:
    """Synchronous wrapper around aask — used by CLI scripts and unit tests."""
    import asyncio
    return asyncio.run(aask(question, session_id=session_id,
                            checkpointer=checkpointer, history=history))


class KnowledgeTransferAgent:
    """Streaming interface for the agent."""
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
