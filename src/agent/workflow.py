"""
LangGraph agentic workflow for the CTE Knowledge Transfer Assistant.

Architecture overview:
  User question
       |
       v
  [ agent ]  <--------------------+
       |                          |
       | tool_calls present?      |
      yes                         |
       |                          |
       v                          |
  [ tools ]  (ToolNode + timeout) |
       |                          |
       +--------------------------+  (loop back with ToolMessages)
       |
       | no tool_calls (final answer)
       v
  [ END ]  -> parse_result() -> FastAPI response

Design notes:
  - parallel_tool_calls=False    one tool at a time
  - Turn-scoped dedup            identical calls in the same turn get a
                                 ToolMessage skip result (valid tool protocol)
  - System prompt                passed via config["configurable"] (serialisable)
  - Tools / bound LLM            per-request ContextVars so the compiled graph
                                 can be reused without putting tool objects in
                                 the checkpointer config
  - Graph compiled once          per checkpointer instance
  - _AGENT_RECURSION_LIMIT       LangGraph supersteps (node visits), not tool count
  - _TOOL_CALL_TIMEOUT_SECS      asyncio.wait_for around each LLM call
  - _TOOL_EXEC_TIMEOUT_SECS      asyncio.wait_for around ToolNode; timeouts
                                 still return ToolMessages so the loop can end
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, AsyncIterator, Literal
from weakref import WeakKeyDictionary

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .chains import get_llm
from .mcp_tools import mcp_server_context
from .parser import parse_result, serialize_parse_result
from .prompt import _build_system_prompt
from .tools import (
    EMPTY_COMPANY_SEARCH_RESULT,
    analyse_csv,
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

# LangGraph counts node visits. One tool round is agent → tools → agent (~3).
# 25 allows about eight tool rounds plus a final answer before GraphRecursionError.
_AGENT_RECURSION_LIMIT: int = 25

# Per-LLM-call timeout — enforced via asyncio.wait_for().
_TOOL_CALL_TIMEOUT_SECS: int = 90

# Per-ToolNode-invocation timeout — enforced via asyncio.wait_for().
_TOOL_EXEC_TIMEOUT_SECS: int = 60

_REDUNDANT_TOOL_RESULT = (
    "Skipped: this tool was already used in the current turn with the same "
    "arguments, or company document search already returned results. "
    "Answer the user using the existing tool outputs. Do not retry this call."
)

# ---------------------------------------------------------------------------
# Static tool list — tuple prevents accidental mutation
# ---------------------------------------------------------------------------

LOCAL_TOOLS: tuple[BaseTool, ...] = (
    search_company_documents,
    search_web,
    summarise_document,
    extract_structured_data,
    calculate,
    generate_chart,
    analyse_csv,
)

# Per-request injection. Not stored on the compiled graph or checkpointer config.
_bound_llm_var: ContextVar[Any] = ContextVar("kt_bound_llm")
_tools_var: ContextVar[tuple[BaseTool, ...]] = ContextVar("kt_tools")

_graph_lock = threading.Lock()
_graph_without_checkpointer: Any = None
_graphs_by_checkpointer: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(MessagesState):
    """Messages-only state; ``messages`` uses the ``add_messages`` reducer."""


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
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
    llm: BaseChatModel | Any,
    messages: list[BaseMessage],
    config: RunnableConfig | None = None,
) -> AIMessage:
    """Stream the LLM and assemble chunks into a single AIMessage.

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
# Turn-scoping and tool-call helpers
# ---------------------------------------------------------------------------

def _get_current_turn_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return only the messages from the latest HumanMessage onward."""
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "type", None) == "human":
            return messages[i:]
    return messages


def _tool_call_parts(tc: Any) -> tuple[str, dict[str, Any], str]:
    """Normalise a tool_call (dict or object) to (name, args, id)."""
    if isinstance(tc, dict):
        name = str(tc.get("name") or "")
        args = tc.get("args") or {}
        tid = str(tc.get("id") or "")
    else:
        name = str(getattr(tc, "name", "") or "")
        args = getattr(tc, "args", None) or {}
        tid = str(getattr(tc, "id", "") or "")
    if not isinstance(args, dict):
        args = {"_args": args}
    return name, args, tid


def _make_tool_call_key(name: str, args: dict) -> str:
    """Canonical dedup key: case-insensitive name, order-stable JSON args."""
    return f"{name.lower()}::{json.dumps(args, sort_keys=True, default=str)}"


def _get_previous_tool_calls(messages: list[BaseMessage]) -> set[str]:
    """Return canonical dedup keys for every tool call in the given messages."""
    used: set[str] = set()
    for msg in messages:
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name, args, _ = _tool_call_parts(tc)
                used.add(_make_tool_call_key(name, args))
    return used


def _first_search_had_results(messages: list[BaseMessage]) -> bool:
    """True if search_company_documents already returned a non-empty hit.

    Empty is the exact sentinel from tools.search_company_documents, or
    whitespace-only content — not a substring match.
    """
    for msg in messages:
        if msg.type != "tool" or msg.name != "search_company_documents":
            continue
        text = str(msg.content or "").strip()
        if text and text != EMPTY_COMPANY_SEARCH_RESULT:
            return True
    return False


def _is_redundant_tool_call(
    name: str,
    args: dict[str, Any],
    turn_messages: list[BaseMessage],
) -> bool:
    """True if this call should be skipped and answered from prior results."""
    key = _make_tool_call_key(name, args)
    if key in _get_previous_tool_calls(turn_messages):
        return True
    return name == "search_company_documents" and _first_search_had_results(
        turn_messages
    )


def _error_payload(detail: str) -> dict[str, Any]:
    """Shape compatible with parse_result / QuestionResponse when the graph aborts."""
    return {
        "generation": detail,
        "datasource": "direct_llm",
        "tools_used": [],
        "citations": [],
        "chart_data": None,
    }


# ---------------------------------------------------------------------------
# Agent node — LLM only; does not strip tool_calls
# ---------------------------------------------------------------------------

async def agent_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    bound_llm: Any,
) -> dict[str, list[AIMessage]]:
    """Core LLM node. Dedup happens in the tools node via ToolMessages."""
    system_prompt = (config.get("configurable") or {}).get(
        "system_prompt", ""
    ) or _build_system_prompt()

    messages_with_system: list[BaseMessage] = [
        SystemMessage(content=system_prompt)
    ] + list(state["messages"])
    response = await _astream_complete(bound_llm, messages_with_system, config=config)
    return {"messages": [response]}


async def _agent_node_runtime(
    state: AgentState, config: RunnableConfig
) -> dict[str, list[AIMessage]]:
    try:
        bound_llm = _bound_llm_var.get()
    except LookupError as exc:
        raise RuntimeError(
            "Agent graph invoked without a bound LLM. Use aask(), ask(), or "
            "KnowledgeTransferAgent.run() so tools are installed for the request."
        ) from exc
    return await agent_node(state, config, bound_llm=bound_llm)


# ---------------------------------------------------------------------------
# Tools node — protocol-safe dedup + execution timeout
# ---------------------------------------------------------------------------

def _synthetic_tool_message(name: str, tool_call_id: str, content: str) -> ToolMessage:
    return ToolMessage(
        content=content,
        name=name or "unknown_tool",
        tool_call_id=tool_call_id or str(uuid.uuid4()),
    )


async def run_tools_node(
    state: AgentState,
    config: RunnableConfig,
    tools: tuple[BaseTool, ...],
) -> dict[str, list[BaseMessage]]:
    """Execute allowed tool calls; skip redundant ones with matching ToolMessages."""
    messages = list(state["messages"])
    last = messages[-1]
    raw_calls = list(getattr(last, "tool_calls", None) or [])
    prior_turn = _get_current_turn_messages(messages[:-1])

    allowed: list[Any] = []
    results: list[ToolMessage] = []

    for tc in raw_calls:
        name, args, tid = _tool_call_parts(tc)
        if _is_redundant_tool_call(name, args, prior_turn):
            logger.warning(
                "Dedup guard: skipped redundant tool call %s(%s)",
                name,
                json.dumps(args, default=str),
            )
            results.append(
                _synthetic_tool_message(name, tid, _REDUNDANT_TOOL_RESULT)
            )
        else:
            allowed.append(tc)

    if not allowed:
        return {"messages": results}

    subset = AIMessage(
        content=getattr(last, "content", "") or "",
        tool_calls=allowed,
        additional_kwargs=dict(getattr(last, "additional_kwargs", None) or {}),
        response_metadata=dict(getattr(last, "response_metadata", None) or {}),
        id=getattr(last, "id", None),
    )
    tool_state: AgentState = {"messages": [subset]}
    tool_node = ToolNode(list(tools), handle_tool_errors=True)

    try:
        executed = await asyncio.wait_for(
            tool_node.ainvoke(tool_state, config),
            timeout=_TOOL_EXEC_TIMEOUT_SECS,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Tool execution timed out after %ds", _TOOL_EXEC_TIMEOUT_SECS
        )
        timeout_msgs: list[ToolMessage] = []
        for tc in allowed:
            name, _, tid = _tool_call_parts(tc)
            timeout_msgs.append(
                _synthetic_tool_message(
                    name,
                    tid,
                    f"Tool '{name}' timed out after {_TOOL_EXEC_TIMEOUT_SECS}s.",
                )
            )
        return {"messages": results + timeout_msgs}

    executed_messages = list((executed or {}).get("messages") or [])
    return {"messages": results + executed_messages}


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
    """Build the 2-node ReAct graph. Tools are injected per request."""
    builder = StateGraph(AgentState)
    builder.add_node("agent", _agent_node_runtime)
    builder.add_node("tools", _tools_node_runtime)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "agent")
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
    bound_llm = get_llm().bind_tools(
        list(all_tools), parallel_tool_calls=False
    )
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
) -> tuple[Any, dict[str, list[BaseMessage]], dict[str, Any]]:
    """Build initial state and config. Caller must enter ``_request_tool_context``."""
    graph = _get_compiled_graph(checkpointer)

    if checkpointer is not None:
        initial_state: dict[str, list[BaseMessage]] = {
            "messages": [HumanMessage(content=question)]
        }
    else:
        prior_messages = list(history or [])
        initial_state = {
            "messages": prior_messages + [HumanMessage(content=question)]
        }

    config: dict[str, Any] = {
        "configurable": {
            "thread_id": session_id,
            "system_prompt": _build_system_prompt(),
        },
        "recursion_limit": _AGENT_RECURSION_LIMIT,
    }
    return graph, initial_state, config


def build_graph(_dynamic_tools: list | None = None, checkpointer: Any = None) -> Any:
    """Return the compiled ReAct graph for ``checkpointer``.

    ``_dynamic_tools`` is accepted for call-site compatibility; tools are
    injected per request via ``_request_tool_context``, not compiled in.
    """
    return _get_compiled_graph(checkpointer)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def aask(
    question: str,
    session_id: str = "default",
    checkpointer: Any = None,
    history: list[BaseMessage] | None = None,
) -> dict[str, Any]:
    """Primary async entry point — called by the FastAPI /ask endpoint."""
    async with mcp_server_context() as mcp_tools:
        all_tools = tuple(LOCAL_TOOLS) + tuple(mcp_tools)
        graph, initial_state, config = _prepare_run(
            question, session_id, checkpointer, history
        )
        with _request_tool_context(all_tools):
            try:
                result = await graph.ainvoke(initial_state, config=config)
            except asyncio.TimeoutError:
                logger.error(
                    "Agent timed out after %ds", _TOOL_CALL_TIMEOUT_SECS
                )
                return _error_payload(
                    f"Request timed out after {_TOOL_CALL_TIMEOUT_SECS}s. "
                    "Try a simpler question."
                )
            except GraphRecursionError:
                logger.error(
                    "Agent hit recursion_limit=%s", _AGENT_RECURSION_LIMIT
                )
                return _error_payload(
                    "Stopped after too many steps. Try a more specific question."
                )
            return parse_result(result)


def ask(
    question: str,
    session_id: str = "default",
    checkpointer: Any = None,
    history: list[BaseMessage] | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper around aask — for CLI scripts and unit tests.

    Use ``await aask(...)`` directly in async contexts (FastAPI, Jupyter).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            aask(
                question,
                session_id=session_id,
                checkpointer=checkpointer,
                history=history,
            )
        )
    raise RuntimeError(
        "ask() cannot be called from inside a running event loop. "
        "Use 'await aask(...)' instead."
    )


class KnowledgeTransferAgent:
    """Streaming interface for the agent (token + tool events for SSE)."""

    def __init__(self, checkpointer: Any = None) -> None:
        self.checkpointer = checkpointer

    def __repr__(self) -> str:
        cp_name = type(self.checkpointer).__name__ if self.checkpointer else "None"
        return f"KnowledgeTransferAgent(checkpointer={cp_name})"

    async def run(
        self,
        question: str,
        session_id: str = "default",
        history: list[BaseMessage] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE-ready events while the graph runs.

        Event types:
          - ``status``: ``{"type": "status", "stage": "thinking"}``
          - ``token``:  ``{"type": "token",  "text": "..."}``
          - ``tool``:   ``{"type": "tool",   "name": "..."}``
          - ``done``:   ``{"type": "done",   "payload": {...}}``
          - ``error``:  ``{"type": "error",  "detail": "..."}``
        """
        yield {"type": "status", "stage": "thinking"}

        async with mcp_server_context() as mcp_tools:
            all_tools = tuple(LOCAL_TOOLS) + tuple(mcp_tools)
            graph, initial_state, config = _prepare_run(
                question, session_id, self.checkpointer, history
            )

            final_values: dict | None = None
            emitted_tool_ids: set[str] = set()

            try:
                with _request_tool_context(all_tools):
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
                                    name, _, tid = _tool_call_parts(tc)
                                    if not name or not tid or tid in emitted_tool_ids:
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
                    "detail": (
                        f"Request timed out after {_TOOL_CALL_TIMEOUT_SECS}s. "
                        "Try a simpler question."
                    ),
                }
            except GraphRecursionError:
                logger.error(
                    "Agent hit recursion_limit=%s", _AGENT_RECURSION_LIMIT
                )
                yield {
                    "type": "error",
                    "detail": (
                        "Stopped after too many steps. Try a more specific question."
                    ),
                }
            except Exception as exc:
                logger.exception("Streaming agent failed")
                yield {"type": "error", "detail": str(exc)}
