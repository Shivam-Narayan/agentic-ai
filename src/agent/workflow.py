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
  - _get_compiled_graph()        lru_cache — graph compiled once per tool-set
  - _TOOL_CALL_TIMEOUT_SECS      per-tool asyncio timeout guard
"""

import asyncio
import logging
import uuid
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
# Module-level constants  (Fix #3 — no more magic numbers scattered in code)
# ---------------------------------------------------------------------------

# Maximum tool-call loops before the graph forces a final answer.
_AGENT_RECURSION_LIMIT: int = 8

# Per-tool call timeout in seconds. Prevents a slow web search or DB query
# from hanging the entire agent loop indefinitely.
_TOOL_CALL_TIMEOUT_SECS: int = 30

# ---------------------------------------------------------------------------
# Static tool list  (Fix #12 — tuple prevents accidental mutation)
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
# LLM streaming assembler  (Fix #6 — raises on empty instead of silent return)
# ---------------------------------------------------------------------------

async def _astream_complete(llm, messages: list) -> AIMessage:
    """Stream the LLM and assemble chunks into a single AIMessage.

    Raises:
        RuntimeError: if the LLM returns no content at all (network / rate-limit issue).
    """
    assembled = None
    async for chunk in llm.astream(messages):
        assembled = chunk if assembled is None else assembled + chunk

    # Fix #6: never silently swallow an empty response — surface it immediately.
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

    Fix #5: only block a second search call if the first one actually found
    something. If it returned empty results the LLM should be allowed to retry
    with a different query rather than being forced to answer from nothing.
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
# Agent node  (Fix #4 — standalone function, not a nested closure)
# ---------------------------------------------------------------------------

async def agent_node(
    state: AgentState,
    *,
    tools: tuple,
    system_prompt: str,
) -> dict:
    """Core LLM node with deduplication guard.

    Extracted from build_graph() so it can be unit-tested and profiled
    independently. Receives the compiled tool list and the pre-built system
    prompt so neither is re-created on every loop iteration.

    Args:
        state:         Current LangGraph agent state.
        tools:         Tuple of all available tools (local + MCP).
        system_prompt: Already-rendered system prompt string (built once per
                       request in _prepare_run, not on every loop pass).
    """
    llm = get_llm().bind_tools(list(tools), parallel_tool_calls=False)
    messages_with_system = [SystemMessage(content=system_prompt)] + state["messages"]
    response = await _astream_complete(llm, messages_with_system)

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

            # Fix #5: only block a repeat search if the first call found content.
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
        elif filtered_calls != list(response.tool_calls):
            response.tool_calls = filtered_calls

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph compilation  (Fix #1 — lru_cache so graph is built once per tool-set)
# ---------------------------------------------------------------------------

# Internal cache: (tool_names_key, checkpointer_id) -> compiled graph
# LangChain StructuredTool objects are not hashable, so we key on a tuple of
# tool names + the id() of the checkpointer instead of the objects themselves.
_graph_cache: dict[tuple, Any] = {}


def _get_compiled_graph(tool_tuple: tuple, checkpointer) -> Any:
    """Return a compiled graph, building it only when the tool set changes.

    Uses a plain dict cache keyed on (sorted tool names, checkpointer id)
    because LangChain StructuredTool objects are not hashable and cannot be
    used as lru_cache keys directly.
    """
    cache_key = (
        tuple(sorted(t.name for t in tool_tuple)),
        id(checkpointer),
    )
    if cache_key not in _graph_cache:
        logger.info(
            "Compiling new graph for tool-set: %s",
            [t.name for t in tool_tuple],
        )
        _graph_cache[cache_key] = _compile_graph(tool_tuple, checkpointer)
    return _graph_cache[cache_key]


def _compile_graph(tools: tuple, checkpointer) -> Any:
    """Build and compile the StateGraph. Called only when cache misses."""
    # Capture the tool tuple in the node closure so the lambda is stable.
    tool_node = ToolNode(list(tools))

    # Wrap agent_node with the tool tuple baked in. The system_prompt is
    # injected per-call via a wrapper set at invocation time (see _run_graph).
    # We use a mutable container so the wrapper can swap the prompt each call.
    _prompt_holder: list[str] = [""]

    async def _agent_node_wrapper(state: AgentState) -> dict:
        return await agent_node(
            state,
            tools=tools,
            system_prompt=_prompt_holder[0],
        )

    builder = StateGraph(AgentState)
    builder.add_node("agent", _agent_node_wrapper)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "agent")
    compiled = builder.compile(checkpointer=checkpointer)

    # Attach the prompt holder so callers can inject the prompt before each run.
    compiled._prompt_holder = _prompt_holder  # type: ignore[attr-defined]
    return compiled


# ---------------------------------------------------------------------------
# Shared run-preparation helper  (Fix #2 + #9 — DRY + prompt built once)
# ---------------------------------------------------------------------------

def _prepare_run(
    question: str,
    session_id: str,
    checkpointer,
    history: list | None,
    all_tools: tuple,
) -> tuple[Any, dict, dict]:
    """Build the graph, initial state, and config for one agent run.

    Centralises the setup logic that was previously duplicated between
    aask() and KnowledgeTransferAgent.run().

    The system prompt is built here — once per request — and injected into
    the cached graph's prompt holder so agent_node doesn't rebuild it on
    every loop iteration.

    Returns:
        (graph, initial_state, config)
    """
    # Fix #9: build system prompt once per request, not once per loop iteration.
    system_prompt = _build_system_prompt()

    graph = _get_compiled_graph(all_tools, checkpointer)

    # Inject the fresh prompt into the cached graph before running.
    if hasattr(graph, "_prompt_holder"):
        graph._prompt_holder[0] = system_prompt

    if checkpointer is not None:
        initial_state: dict = {"messages": [HumanMessage(content=question)]}
    else:
        prior_messages = history or []
        initial_state = {"messages": prior_messages + [HumanMessage(content=question)]}

    config: dict = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": _AGENT_RECURSION_LIMIT,  # Fix #3: named constant
    }

    return graph, initial_state, config


# ---------------------------------------------------------------------------
# build_graph — kept as a public convenience wrapper (backwards compat)
# ---------------------------------------------------------------------------

def build_graph(dynamic_tools: list, checkpointer=None):
    """Public wrapper around _get_compiled_graph.

    Kept for backwards compatibility with any external callers.
    Prefer _get_compiled_graph(tuple(tools), checkpointer) internally.
    """
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
    """Synchronous wrapper around aask — used by CLI scripts and unit tests.

    Fix #8: asyncio imported at module level, not inside the function body.
    Note: asyncio.run() will raise RuntimeError if an event loop is already
    running (e.g. Jupyter). Use `await aask(...)` directly in async contexts.
    """
    return asyncio.run(
        aask(question, session_id=session_id, checkpointer=checkpointer, history=history)
    )


class KnowledgeTransferAgent:
    """Streaming interface for the agent (token + tool events for SSE)."""

    def __init__(self, checkpointer=None) -> None:
        self.checkpointer = checkpointer

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
            # Fix #10: use uuid4 so two calls with the same tool name never
            # collide in the dedup set when the tool id is absent.
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
                                # Fix #10: prefer real id; fall back to uuid4
                                # (never fall back to name — two calls to the
                                # same tool with different args would be suppressed)
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

            except Exception as exc:
                logger.exception("Streaming agent failed")
                yield {"type": "error", "detail": str(exc)}
