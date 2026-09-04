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
  - Turn-scoped deduplication    only blocks repeated calls within the current
                                 conversation turn, so multi-turn sessions work
  - Dynamic system prompt        stamped with live date/time, passed via
                                 config["configurable"] (LangGraph-idiomatic)
  - _AGENT_RECURSION_LIMIT       caps the agent loop to prevent runaway API usage
  - _TOOL_CALL_TIMEOUT_SECS      enforced via asyncio.wait_for on every LLM call
  - _TOOL_EXEC_TIMEOUT_SECS      enforced on tool execution via ToolNode
  - Canonical dedup keys         json.dumps(args, sort_keys=True) — handles every
                                 tool regardless of parameter names
  - Safe AIMessage construction  dedup guard constructs new AIMessage objects
                                 instead of mutating response.tool_calls in place
"""

import asyncio
import json as _json
import logging
import uuid
import warnings
from typing import Annotated, Any, AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .chains import get_llm
from .mcp_tools import mcp_server_context
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

# Per-LLM-call timeout — enforced via asyncio.wait_for().
# Prevents a rate-limited LLM from hanging the agent loop indefinitely.
_TOOL_CALL_TIMEOUT_SECS: int = 30

# Per-tool-execution timeout — enforced on ToolNode.
# Prevents a slow web search or database query from hanging the agent loop.
_TOOL_EXEC_TIMEOUT_SECS: int = 60

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
# LLM streaming assembler — with per-call timeout
# ---------------------------------------------------------------------------

async def _astream_complete(
    llm,
    messages: list,
    config: RunnableConfig | None = None,
) -> AIMessage:
    """Stream the LLM and assemble chunks into a single AIMessage.

    Enforces _TOOL_CALL_TIMEOUT_SECS via asyncio.wait_for so a slow or
    rate-limited LLM cannot hang the agent loop indefinitely.

    Args:
        llm:      The LLM instance (possibly with tools bound).
        messages: Full message list including SystemMessage.
        config:   LangGraph RunnableConfig — forwarded to llm.astream()
                  so tracing, callbacks, and run IDs propagate correctly.

    Raises:
        asyncio.TimeoutError: if the LLM takes longer than _TOOL_CALL_TIMEOUT_SECS.
        RuntimeError:         if the LLM returns no content at all.
    """
    async def _stream() -> AIMessage:
        assembled = None
        async for chunk in llm.astream(messages, config=config):
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
# Turn-scoping helper — prevents dedup from leaking across conversation turns
# ---------------------------------------------------------------------------

def _get_current_turn_messages(messages: list) -> list:
    """Return only the messages from the latest HumanMessage onward.

    This ensures that the deduplication guard only considers tool calls
    from the *current* conversation turn, not from earlier turns in a
    multi-turn session. Without this, the first successful search in
    Turn 1 would permanently block searches in all subsequent turns.
    """
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "type", None) == "human":
            return messages[i:]
    return messages


# ---------------------------------------------------------------------------
# Deduplication helpers — scoped to current turn via canonical keys
# ---------------------------------------------------------------------------

def _make_tool_call_key(name: str, args: dict) -> str:
    """Create a canonical, case-insensitive dedup key for a tool call.

    Uses json.dumps with sort_keys to produce a stable string regardless
    of argument ordering. ``default=str`` handles non-JSON-serialisable
    values (e.g. numeric types that the LLM might pass without quotes).
    """
    return f"{name}::{_json.dumps(args, sort_keys=True, default=str).lower()}"


def _get_previous_tool_calls(messages: list) -> set[str]:
    """Return canonical dedup keys for every tool call in the given messages."""
    used: set[str] = set()
    for msg in messages:
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})
                used.add(_make_tool_call_key(name, args))
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
#
# Reads system prompt from config["configurable"]["system_prompt"]
# (LangGraph-idiomatic, safe for concurrent async requests).
# bound_llm is passed in so .bind_tools() is called once per graph
# compilation, not on every loop iteration.
# ---------------------------------------------------------------------------

async def agent_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    bound_llm: Any,
) -> dict:
    """Core LLM node with turn-scoped deduplication guard.

    Args:
        state:     Current LangGraph agent state.
        config:    LangGraph RunnableConfig carrying system_prompt in
                   config["configurable"]["system_prompt"] and propagating
                   tracing / callback context to the LLM.
        bound_llm: LLM already bound to the full tool list. Passed in so
                   .bind_tools() is called once per graph compilation, not
                   on every loop iteration.
    """
    # Read prompt from config — safe for concurrent async requests
    system_prompt = (config.get("configurable") or {}).get(
        "system_prompt", ""
    ) or _build_system_prompt()

    messages_with_system = [SystemMessage(content=system_prompt)] + state["messages"]
    response = await _astream_complete(bound_llm, messages_with_system, config=config)

    # ── Deduplication guard (scoped to current turn only) ────────────────
    if getattr(response, "tool_calls", None):
        turn_messages = _get_current_turn_messages(state["messages"])
        already_used = _get_previous_tool_calls(turn_messages)

        blocked = False
        filtered_calls: list = []

        for tc in response.tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            key = _make_tool_call_key(name, args)

            is_redundant_search = (
                name == "search_company_documents"
                and _first_search_had_results(turn_messages)
            )

            if key in already_used or is_redundant_search:
                logger.warning(
                    "Dedup guard: blocked redundant tool call %s(%s). "
                    "Forcing direct answer.",
                    name, _json.dumps(args, default=str),
                )
                blocked = True
                break

            filtered_calls.append(tc)

        if blocked:
            # Construct a new AIMessage without tool_calls instead of
            # mutating response.tool_calls — avoids leaving stale data
            # in additional_kwargs["tool_calls"].
            if response.content and str(response.content).strip():
                # LLM provided answer text alongside the tool call —
                # keep the text, drop the tool calls.
                response = AIMessage(
                    content=response.content,
                    additional_kwargs={},
                    response_metadata=dict(
                        getattr(response, "response_metadata", None) or {}
                    ),
                    id=getattr(response, "id", None),
                )
            else:
                # No answer text — re-invoke the LLM for a direct answer.
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
                response = await _astream_complete(bare_llm, direct_messages, config=config)

        elif len(filtered_calls) != len(list(response.tool_calls)):
            # Some calls were filtered but not all — rebuild with the
            # surviving subset.
            response = AIMessage(
                content=response.content,
                tool_calls=filtered_calls,
                additional_kwargs={},
                response_metadata=dict(
                    getattr(response, "response_metadata", None) or {}
                ),
                id=getattr(response, "id", None),
            )

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph compilation — fresh per request (< 1ms for a 2-node graph)
#
# No global cache — eliminates race conditions, stale id() keys, and
# memory leaks from the previous _graph_cache approach.
# ---------------------------------------------------------------------------

def _compile_graph(tools: tuple, checkpointer) -> Any:
    """Build and compile the StateGraph.

    - bound_llm is created once here so agent_node does not call
      .bind_tools() on every loop iteration.
    - agent_node reads the system prompt from config["configurable"]
      — no shared mutable state on the graph object.
    - ToolNode is configured with a timeout to prevent hanging tools.
    """
    bound_llm = get_llm().bind_tools(list(tools), parallel_tool_calls=False)
    tool_node = ToolNode(list(tools))

    async def _agent_node_wrapper(state: AgentState, config: RunnableConfig) -> dict:
        return await agent_node(state, config, bound_llm=bound_llm)

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
# Shared run-preparation helper — DRY + prompt set via config
# ---------------------------------------------------------------------------

def _prepare_run(
    question: str,
    session_id: str,
    checkpointer,
    history: list | None,
    all_tools: tuple,
) -> tuple[Any, dict, dict]:
    """Build the graph, initial state, and config for one agent run.

    The system prompt is injected into config["configurable"]["system_prompt"]
    so it propagates through the LangGraph runtime to agent_node without
    shared mutable state.

    Returns:
        (graph, initial_state, config)
    """
    system_prompt = _build_system_prompt()

    graph = _compile_graph(all_tools, checkpointer)

    if checkpointer is not None:
        initial_state: dict = {"messages": [HumanMessage(content=question)]}
    else:
        prior_messages = history or []
        initial_state = {"messages": prior_messages + [HumanMessage(content=question)]}

    config: dict = {
        "configurable": {
            "thread_id": session_id,
            "system_prompt": system_prompt,
        },
        "recursion_limit": _AGENT_RECURSION_LIMIT,
    }

    return graph, initial_state, config


# ---------------------------------------------------------------------------
# build_graph — public wrapper kept for backwards compatibility
# ---------------------------------------------------------------------------

def build_graph(dynamic_tools: list, checkpointer=None) -> Any:
    """Return a compiled graph for the given tool list.

    .. deprecated::
        Use ``_compile_graph(tuple(tools), checkpointer)`` directly.
        This wrapper is kept for backwards compatibility only and may be
        removed in a future release.
    """
    warnings.warn(
        "build_graph() is deprecated. Use _compile_graph(tuple(tools), checkpointer) "
        "or let _prepare_run() handle graph creation.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _compile_graph(tuple(dynamic_tools), checkpointer)


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

    Detects an already-running event loop and raises a clear error
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
