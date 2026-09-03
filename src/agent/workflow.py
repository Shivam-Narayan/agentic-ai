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
  - parallel_tool_calls=False    forces the LLM to call one tool at a time
  - Deduplication guard          code-level block on repeated/empty search calls
  - Dynamic system prompt        stamped with live date/time once per request
  - _AGENT_RECURSION_LIMIT       caps the agent loop to prevent runaway API usage
  - _get_compiled_graph()        dict cache keyed on tool names + checkpointer id
  - _TOOL_CALL_TIMEOUT_SECS      enforced via asyncio.wait_for on every LLM call
  - Thread-safe prompt injection via per-run contextvars (not shared mutable list)
"""

import asyncio
import logging
import uuid
import warnings
from contextvars import ContextVar
from typing import Annotated, Any, AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .chains import get_llm
from .mcp_client import mcp_server_context
from .parser import parse_result, serialize_parse_result
from .prompt import _build_system_prompt
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
# Module-level constants
# ---------------------------------------------------------------------------

# Maximum tool-call loops before the graph forces a final answer.
_AGENT_RECURSION_LIMIT: int = 8

# Fix #1: per-LLM-call timeout — now actually enforced via asyncio.wait_for().
# Prevents a slow web search or rate-limited LLM from hanging the agent loop.
_TOOL_CALL_TIMEOUT_SECS: int = 30

# Fix #2: bounded graph cache — evict oldest entry when this size is exceeded.
_GRAPH_CACHE_MAX_SIZE: int = 16

# ---------------------------------------------------------------------------
# Static tool list — tuple prevents accidental mutation
# ---------------------------------------------------------------------------

LOCAL_TOOLS: tuple = (
    search_company_documents,
    search_web,
    summarise_document,
    extract_structured_data,
    calculate,
    generate_chart,
)

# ---------------------------------------------------------------------------
# Fix #3: thread-safe prompt injection via ContextVar
#
# Previously a shared mutable list (_prompt_holder) was attached to the cached
# graph object. Under concurrent async requests both coroutines would write to
# the same list and race — the last writer would win, corrupting the prompt for
# the first request.
#
# ContextVar is coroutine-safe: each asyncio Task gets its own copy of the
# value so concurrent requests never interfere.
# ---------------------------------------------------------------------------

_current_system_prompt: ContextVar[str] = ContextVar(
    "_current_system_prompt", default=""
)

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
# Streaming helpers
# ---------------------------------------------------------------------------

def _chunk_text(content: Any) -> str:
    """Normalise a streaming chunk's content into a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" or "text" in block:
                    parts.append(str(block.get("text") or ""))
            elif hasattr(block, "text"):
                parts.append(str(getattr(block, "text") or ""))
        return "".join(parts)
    return str(content)


def _has_tool_call_chunks(token: Any) -> bool:
    """Return True if the token contains tool-call delta chunks (not answer text)."""
    return bool(getattr(token, "tool_call_chunks", None))


# ---------------------------------------------------------------------------
# LLM streaming assembler
# Fix #1: wrapped in asyncio.wait_for() to enforce _TOOL_CALL_TIMEOUT_SECS
# ---------------------------------------------------------------------------

async def _astream_complete(llm, messages: list) -> AIMessage:
    """Stream the LLM and assemble chunks into a single AIMessage.

    Enforces _TOOL_CALL_TIMEOUT_SECS via asyncio.wait_for so a slow or
    rate-limited LLM cannot hang the agent loop indefinitely.

    Raises:
        asyncio.TimeoutError: if the LLM takes longer than _TOOL_CALL_TIMEOUT_SECS.
        RuntimeError:         if the LLM returns no content at all.
    """
    async def _stream() -> AIMessage:
        assembled = None
        async for chunk in llm.astream(messages):
            assembled = chunk if assembled is None else assembled + chunk

        if assembled is None:
            raise RuntimeError(
                "LLM returned no content. This usually means a network error, "
                "rate-limit, or the model timed out. Check your API key and quota."
            )

        if isinstance(assembled, AIMessage) and not type(assembled).__name__.endswith("Chunk"):
            return assembled

        return AIMessage(
            content=assembled.content,
            tool_calls=list(getattr(assembled, "tool_calls", None) or []),
            additional_kwargs=dict(getattr(assembled, "additional_kwargs", None) or {}),
            response_metadata=dict(getattr(assembled, "response_metadata", None) or {}),
            id=getattr(assembled, "id", None),
        )

    return await asyncio.wait_for(_stream(), timeout=_TOOL_CALL_TIMEOUT_SECS)


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def _get_previous_tool_calls(messages: list) -> set[str]:
    """Return 'toolname::query' keys for every tool call already made."""
    used: set[str] = set()
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


def _first_search_had_results(messages: list) -> bool:
    """Return True if search_company_documents already ran AND returned content.

    Only block a second search call if the first one actually found something.
    If it returned empty results the LLM should be allowed to retry with a
    different query rather than being forced to answer from nothing.
    """
    for msg in messages:
        if (
            msg.type == "tool"
            and msg.name == "search_company_documents"
            and msg.content
            and str(msg.content).strip()
            and "No matching" not in str(msg.content)
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Agent node — standalone function (testable in isolation)
# Fix #3: reads system prompt from ContextVar (thread-safe)
# Fix #4: bound LLM passed in — .bind_tools() not called every loop pass
# ---------------------------------------------------------------------------

async def agent_node(
    state: AgentState,
    *,
    bound_llm: Any,
) -> dict:
    """Core LLM node with deduplication guard.

    Args:
        state:     Current LangGraph agent state.
        bound_llm: LLM already bound to the full tool list. Passed in so
                   .bind_tools() is called once per graph compilation, not
                   on every loop iteration.
    """
    # Fix #3: read prompt from ContextVar — safe for concurrent async requests
    system_prompt = _current_system_prompt.get()
    messages_with_system = [SystemMessage(content=system_prompt)] + state["messages"]
    response = await _astream_complete(bound_llm, messages_with_system)

    # ── Deduplication guard ──────────────────────────────────────────────
    if getattr(response, "tool_calls", None):
        already_used = _get_previous_tool_calls(state["messages"])

        blocked = False
        filtered_calls: list = []

        for tc in response.tool_calls:
            name  = tc.get("name", "")
            args  = tc.get("args", {})
            query = (
                args.get("query")
                or args.get("document_name")
                or args.get("expression")
                or ""
            )
            key = f"{name}::{query.lower().strip()}"

            is_redundant_search = (
                name == "search_company_documents"
                and _first_search_had_results(state["messages"])
            )

            if key in already_used or is_redundant_search:
                logger.warning(
                    "Dedup guard: blocked redundant tool call %s(%s). Forcing direct answer.",
                    name, query,
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
                response = await _astream_complete(bare_llm, direct_messages)

        # Fix #5: compare lengths — clearer intent than value equality on dicts
        elif len(filtered_calls) != len(list(response.tool_calls)):
            response.tool_calls = filtered_calls

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph compilation — dict cache with bounded size
# Fix #2: cap _graph_cache at _GRAPH_CACHE_MAX_SIZE entries
# Fix #4: bind LLM to tools once at compile time
# ---------------------------------------------------------------------------

_graph_cache: dict[tuple, Any] = {}


def _get_compiled_graph(tool_tuple: tuple, checkpointer) -> Any:
    """Return a compiled graph, building it only when the tool set changes.

    Keyed on (sorted tool names, checkpointer id). Evicts the oldest entry
    when the cache exceeds _GRAPH_CACHE_MAX_SIZE.
    """
    cache_key = (
        tuple(sorted(t.name for t in tool_tuple)),
        id(checkpointer),
    )
    if cache_key not in _graph_cache:
        # Fix #2: evict oldest entry before inserting a new one
        if len(_graph_cache) >= _GRAPH_CACHE_MAX_SIZE:
            oldest_key = next(iter(_graph_cache))
            del _graph_cache[oldest_key]
            logger.debug("Graph cache evicted oldest entry (cache full)")

        logger.info(
            "Compiling new graph for tool-set: %s",
            [t.name for t in tool_tuple],
        )
        _graph_cache[cache_key] = _compile_graph(tool_tuple, checkpointer)
    return _graph_cache[cache_key]


def _compile_graph(tools: tuple, checkpointer) -> Any:
    """Build and compile the StateGraph. Called only on cache miss.

    Fix #4: bound_llm is created once here so agent_node does not call
    .bind_tools() on every loop iteration.
    Fix #3: agent_node reads the system prompt from _current_system_prompt
    ContextVar — no shared mutable state on the graph object.
    """
    # Fix #4: bind tools once at compile time
    bound_llm = get_llm().bind_tools(list(tools), parallel_tool_calls=False)
    tool_node  = ToolNode(list(tools))

    async def _agent_node_wrapper(state: AgentState) -> dict:
        return await agent_node(state, bound_llm=bound_llm)

    builder = StateGraph(AgentState)
    builder.add_node("agent", _agent_node_wrapper)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Shared run-preparation helper — DRY + prompt set via ContextVar
# ---------------------------------------------------------------------------

def _prepare_run(
    question: str,
    session_id: str,
    checkpointer,
    history: list | None,
    all_tools: tuple,
) -> tuple[Any, dict, dict]:
    """Build the graph, initial state, and config for one agent run.

    Sets the system prompt on the ContextVar so concurrent async requests
    never share prompt state (Fix #3).

    Returns:
        (graph, initial_state, config)
    """
    system_prompt = _build_system_prompt()

    # Fix #3: set on ContextVar — each asyncio Task sees its own value
    _current_system_prompt.set(system_prompt)

    graph = _get_compiled_graph(all_tools, checkpointer)

    if checkpointer is not None:
        initial_state: dict = {"messages": [HumanMessage(content=question)]}
    else:
        prior_messages = history or []
        initial_state = {"messages": prior_messages + [HumanMessage(content=question)]}

    config: dict = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": _AGENT_RECURSION_LIMIT,
    }

    return graph, initial_state, config


# ---------------------------------------------------------------------------
# build_graph — public wrapper kept for backwards compatibility
# Fix #8: deprecation warning added so callers know to migrate
# ---------------------------------------------------------------------------

def build_graph(dynamic_tools: list, checkpointer=None) -> Any:
    """Return a compiled graph for the given tool list.

    .. deprecated::
        Use ``_get_compiled_graph(tuple(tools), checkpointer)`` directly.
        This wrapper is kept for backwards compatibility only and may be
        removed in a future release.
    """
    warnings.warn(
        "build_graph() is deprecated. Use _get_compiled_graph(tuple(tools), checkpointer) "
        "or let _prepare_run() handle graph creation.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _get_compiled_graph(tuple(dynamic_tools), checkpointer)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def aask(
    question: str,
    session_id: str = "default",
    checkpointer=None,
    history: list | None = None,
) -> dict:
    """Primary async entry point — called by the FastAPI /ask endpoint."""
    async with mcp_server_context() as mcp_tools:
        all_tools = tuple(LOCAL_TOOLS) + tuple(mcp_tools)
        graph, initial_state, config = _prepare_run(
            question, session_id, checkpointer, history, all_tools
        )
        result = await graph.ainvoke(initial_state, config=config)
        return parse_result(result)


def ask(
    question: str,
    session_id: str = "default",
    checkpointer=None,
    history: list | None = None,
) -> dict:
    """Synchronous wrapper around aask — for CLI scripts and unit tests.

    Fix #7: detects an already-running event loop and raises a clear error
    instead of crashing with a cryptic RuntimeError from asyncio.run().
    Use ``await aask(...)`` directly in async contexts (FastAPI, Jupyter).
    """
    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "ask() cannot be called from inside a running event loop. "
            "Use 'await aask(...)' instead."
        )
    except RuntimeError as exc:
        if "cannot be called" in str(exc):
            raise
    return asyncio.run(
        aask(question, session_id=session_id, checkpointer=checkpointer, history=history)
    )


class KnowledgeTransferAgent:
    """Streaming interface for the agent (token + tool events for SSE)."""

    def __init__(self, checkpointer=None) -> None:
        self.checkpointer = checkpointer

    def __repr__(self) -> str:
        # Fix #6: useful repr for debugging
        cp_name = type(self.checkpointer).__name__ if self.checkpointer else "None"
        return f"KnowledgeTransferAgent(checkpointer={cp_name})"

    async def run(
        self,
        question: str,
        session_id: str = "default",
        history: list | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE-ready events while the graph runs.

        Event types:
          - ``status``: ``{"type": "status", "stage": "thinking"}``
          - ``token``:  ``{"type": "token",  "text": "..."}``     — answer delta
          - ``tool``:   ``{"type": "tool",   "name": "..."}``     — tool invoked
          - ``done``:   ``{"type": "done",   "payload": {...}}``  — final result
          - ``error``:  ``{"type": "error",  "detail": "..."}``
        """
        yield {"type": "status", "stage": "thinking"}

        async with mcp_server_context() as mcp_tools:
            all_tools = tuple(LOCAL_TOOLS) + tuple(mcp_tools)
            graph, initial_state, config = _prepare_run(
                question, session_id, self.checkpointer, history, all_tools
            )

            final_values: dict | None = None
            emitted_tool_ids: set[str] = set()

            try:
                async for mode, data in graph.astream(
                    initial_state,
                    config=config,
                    stream_mode=["messages", "values"],
                ):
                    if mode == "messages":
                        token, metadata = data
                        if metadata.get("langgraph_node") != "agent":
                            continue
                        if _has_tool_call_chunks(token):
                            continue
                        text = _chunk_text(getattr(token, "content", None))
                        if text:
                            yield {"type": "token", "text": text}

                    elif mode == "values":
                        final_values = data
                        messages = (data or {}).get("messages") or []
                        if not messages:
                            continue
                        last = messages[-1]
                        if getattr(last, "tool_calls", None):
                            for tc in last.tool_calls:
                                name = tc.get("name") or ""
                                if not name:
                                    continue
                                tid = tc.get("id") or str(uuid.uuid4())
                                if tid in emitted_tool_ids:
                                    continue
                                emitted_tool_ids.add(tid)
                                yield {"type": "tool", "name": name}

                if not final_values:
                    yield {"type": "error", "detail": "Agent finished without a result."}
                    return

                yield {
                    "type": "done",
                    "payload": serialize_parse_result(parse_result(final_values)),
                }

            except asyncio.TimeoutError:
                logger.error("Agent timed out after %ds", _TOOL_CALL_TIMEOUT_SECS)
                yield {
                    "type": "error",
                    "detail": f"Request timed out after {_TOOL_CALL_TIMEOUT_SECS}s. Try a simpler question.",
                }
            except Exception as exc:
                logger.exception("Streaming agent failed")
                yield {"type": "error", "detail": str(exc)}
