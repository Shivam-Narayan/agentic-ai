"""Graph compilation, caching, and routing logic."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal
from weakref import WeakKeyDictionary

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from .chains import get_llm
from .nodes import agent_node, planner_node, reflection_node, run_tools_node
from .prompt import build_system_prompt
from .state import (
    AGENT_RECURSION_LIMIT,
    COMPLEX_KEYWORDS,
    MIN_WORDS_FOR_PLANNER,
    AgentState,
    _get_current_turn_messages,
)

logger = logging.getLogger(__name__)

# Per-request injection. Not stored on the compiled graph or checkpointer config.
_bound_llm_var: ContextVar[Any] = ContextVar("kt_bound_llm")
_tools_var: ContextVar[tuple[BaseTool, ...]] = ContextVar("kt_tools")

_graph_lock = threading.Lock()
_graph_without_checkpointer: Any = None
_graphs_by_checkpointer: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


# ---------------------------------------------------------------------------
# Complexity router  (Pattern 3 / Pattern 4 gate)
# ---------------------------------------------------------------------------

def _classify_complexity(question: str) -> bool:
    """Return True if the question is complex enough to benefit from planning."""
    q_lower = question.lower()
    word_count = len(question.split())
    if word_count < MIN_WORDS_FOR_PLANNER:
        return False
    return any(kw in q_lower for kw in COMPLEX_KEYWORDS)


def complexity_router(state: AgentState) -> Literal["planner", "agent"]:
    """Route to planner for multi-step questions; skip straight to agent otherwise."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    question = str(last_human.content) if last_human else ""
    if _classify_complexity(question):
        logger.info("complexity_router → planner (question: %.80s…)", question)
        return "planner"
    logger.info("complexity_router → agent (simple path)")
    return "agent"


# ---------------------------------------------------------------------------
# Router: after agent node — tool loop vs reflection
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> Literal["tools", "reflection", "__end__"]:
    """Go to tools if the LLM requested a tool call; reflect only on complex answers."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    # Skip reflection for simple direct answers — no tools were used and the
    # planner didn't run, so the LLM answered from its own knowledge.
    if not state.get("is_complex"):
        # Check if any tools were actually used in this turn
        turn_msgs = _get_current_turn_messages(list(state["messages"]))
        has_tool_use = any(m.type == "tool" for m in turn_msgs)
        if not has_tool_use:
            return "__end__"
    return "reflection"


# ---------------------------------------------------------------------------
# Runtime wrappers (resolve ContextVars for the compiled graph)
# ---------------------------------------------------------------------------

async def _agent_node_runtime(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    try:
        bound_llm = _bound_llm_var.get()
    except LookupError as exc:
        raise RuntimeError(
            "Agent graph invoked without a bound LLM. Use aask(), ask(), or "
            "KnowledgeTransferAgent.run() so tools are installed for the request."
        ) from exc
    return await agent_node(state, config, bound_llm=bound_llm)


async def _tools_node_runtime(
    state: AgentState, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    try:
        tools = _tools_var.get()
    except LookupError as exc:
        raise RuntimeError(
            "Agent graph invoked without tools. Use aask(), ask(), or "
            "KnowledgeTransferAgent.run()."
        ) from exc
    return await run_tools_node(state, config, tools)


# ---------------------------------------------------------------------------
# Graph compilation — once per checkpointer
# ---------------------------------------------------------------------------

def _compile_graph(checkpointer: Any = None) -> Any:
    """Build the planner → executor → reflection graph."""
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("planner", planner_node)
    builder.add_node("agent", _agent_node_runtime)
    builder.add_node("tools", _tools_node_runtime)
    builder.add_node("reflection", reflection_node)

    # Entry: classify question complexity first
    builder.add_conditional_edges(
        START,
        complexity_router,
        {"planner": "planner", "agent": "agent"},
    )

    # After planner → always go to agent (executor)
    builder.add_edge("planner", "agent")

    # After agent → call tools, reflect on the draft, or end directly
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "reflection": "reflection", "__end__": END},
    )

    # Tools loop back to agent
    builder.add_edge("tools", "agent")

    # Reflection always ends the graph
    builder.add_edge("reflection", END)

    return builder.compile(checkpointer=checkpointer)


def _get_compiled_graph(checkpointer: Any = None) -> Any:
    """Return a process-cached compiled graph for this checkpointer."""
    global _graph_without_checkpointer
    with _graph_lock:
        if checkpointer is None:
            if _graph_without_checkpointer is None:
                _graph_without_checkpointer = _compile_graph(None)
            return _graph_without_checkpointer
        graph = _graphs_by_checkpointer.get(checkpointer)
        if graph is None:
            graph = _compile_graph(checkpointer)
            try:
                _graphs_by_checkpointer[checkpointer] = graph
            except TypeError:
                logger.warning(
                    "Checkpointer is not weak-referenceable; compiling without cache"
                )
        return graph


@contextmanager
def _request_tool_context(all_tools: tuple[BaseTool, ...]):
    """Bind tools + LLM for one invoke without storing them on the graph."""
    bound_llm = get_llm().bind_tools(list(all_tools), parallel_tool_calls=False)
    t_tools = _tools_var.set(all_tools)
    t_llm = _bound_llm_var.set(bound_llm)
    try:
        yield
    finally:
        _bound_llm_var.reset(t_llm)
        _tools_var.reset(t_tools)


# ---------------------------------------------------------------------------
# Shared run-preparation helper
# ---------------------------------------------------------------------------

def _prepare_run(
    question: str,
    session_id: str,
    checkpointer: Any,
    history: list[BaseMessage] | None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Build initial state and config. Caller must enter ``_request_tool_context``."""
    graph = _get_compiled_graph(checkpointer)

    if checkpointer is not None:
        initial_state: dict[str, Any] = {
            "messages": [HumanMessage(content=question)],
            "plan": [],
            "reflection_status": "",
            "is_complex": False,
        }
    else:
        prior_messages = list(history or [])
        initial_state = {
            "messages": prior_messages + [HumanMessage(content=question)],
            "plan": [],
            "reflection_status": "",
            "is_complex": False,
        }

    config: dict[str, Any] = {
        "configurable": {
            "thread_id": session_id,
            "system_prompt": build_system_prompt(),
        },
        "recursion_limit": AGENT_RECURSION_LIMIT,
    }
    return graph, initial_state, config


def build_graph(_dynamic_tools: list | None = None, checkpointer: Any = None) -> Any:
    """Return the compiled graph for ``checkpointer``.

    ``_dynamic_tools`` is accepted for call-site compatibility; tools are
    injected per request via ``_request_tool_context``, not compiled in.
    """
    return _get_compiled_graph(checkpointer)
