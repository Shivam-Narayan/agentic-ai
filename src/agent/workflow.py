"""
LangGraph agentic workflow for the CTE Knowledge Transfer Assistant.

This file contains the complete agent workflow in one place:
  - State definition
  - Node implementations (planner, agent, tools, reflection)
  - Graph compilation and routing
  - Public entry points (aask, ask, KnowledgeTransferAgent)

Architecture overview (with Planner + Reflection patterns):

  User question
       |
       v
  [ complexity_router ]
       |
       +------ simple -------> [ agent ]  <----------+
       |                           |                  |
       +------ complex -----> [ planner ]             |
                                   |          tool_calls present?
                                   v                  |
                               [ agent ] ----------yes+
                                   |
                               no tool_calls (draft answer)
                                   |
                                   v
                           [ reflection ]  (critique + improve if complex)
                                   |
                                   v
                               [ END ]  -> parse_result() -> response

Design principles:
  - complexity_router        keyword + length heuristic; skips planner for simple questions
  - planner node             one LLM call that writes a numbered step list into state["plan"]
  - parallel_tool_calls=False  one tool at a time
  - Turn-scoped dedup        identical calls in the same turn get a skip ToolMessage
  - reflection node          Pattern 1: critique and rewrite draft if needed
  - System prompt            passed via config["configurable"] (serializable)
  - Tools / bound LLM        per-request ContextVars so the compiled graph can be reused
  - Graph compiled once      per checkpointer instance
  - _AGENT_RECURSION_LIMIT   LangGraph supersteps (node visits), not tool count
  - _TOOL_CALL_TIMEOUT_SECS  asyncio.wait_for around each LLM call
  - _TOOL_EXEC_TIMEOUT_SECS  asyncio.wait_for around ToolNode
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Annotated, Any, AsyncIterator, Literal
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
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.tools.mcp_tools import mcp_server_context
from src.tools.tools import (
    EMPTY_COMPANY_SEARCH_RESULT,
    analyse_csv,
    calculate,
    extract_structured_data,
    generate_chart,
    search_company_documents,
    search_web,
    summarise_document,
)

from .chains import get_llm
from .parser import parse_result, serialize_parse_result
from .prompt import (
    build_planner_prompt,
    build_reflection_prompt,
    build_system_prompt,
)

logger = logging.getLogger(__name__)

# ===========================================================================
# CONSTANTS
# ===========================================================================

# LangGraph counts node visits. One tool round is agent → tools → agent (~3).
# 35 allows ~10 tool rounds (planner + reflection add 2 extra node visits).
AGENT_RECURSION_LIMIT: int = 35

# Per-LLM-call timeout — enforced via asyncio.wait_for().
TOOL_CALL_TIMEOUT_SECS: int = 90

# Per-ToolNode-invocation timeout — enforced via asyncio.wait_for().
TOOL_EXEC_TIMEOUT_SECS: int = 60

REDUNDANT_TOOL_RESULT = (
    "Skipped: this tool was already used in the current turn with the same "
    "arguments, or company document search already returned results. "
    "Answer the user using the existing tool outputs. Do not retry this call."
)

# Keywords that signal a complex, multi-step question requiring the planner.
COMPLEX_KEYWORDS = (
    "compare", "analyse", "analyze", "breakdown", "break down",
    "how many", "trend", "correlation", "relationship between",
    "summarise and", "summarize and", "extract and", "find and",
    "step by step", "detailed report", "full report",
    "what caused", "why did", "predict", "forecast",
)

# Questions shorter than this word count skip the planner even with a keyword.
MIN_WORDS_FOR_PLANNER: int = 8

# Static tool list — tuple prevents accidental mutation
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


# ===========================================================================
# STATE DEFINITION
# ===========================================================================

class AgentState(TypedDict):
    """Extended state that carries plan and reflection outcome alongside messages.
    
    Fields:
      - messages: Full conversation (uses add_messages reducer)
      - plan: List of step strings from planner node (Pattern 4)
      - reflection_status: "pass", "improved", or "" (Pattern 1)
      - is_complex: True if planner ran for this request
    """
    messages: Annotated[list[BaseMessage], add_messages]
    plan: list[str]
    reflection_status: str
    is_complex: bool


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def _chunk_text(content: Any) -> str:
    """Normalize a streaming chunk's content into a plain string."""
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


def _error_payload(detail: str) -> dict[str, Any]:
    """Shape compatible with parse_result / QuestionResponse when the graph aborts."""
    return {
        "generation": detail,
        "datasource": "direct_llm",
        "tools_used": [],
        "citations": [],
        "chart_data": None,
    }


def _get_current_turn_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return only the messages from the latest HumanMessage onward."""
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "type", None) == "human":
            return messages[i:]
    return messages


def _tool_call_parts(tc: Any) -> tuple[str, dict[str, Any], str]:
    """Normalize a tool_call (dict or object) to (name, args, id)."""
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
    """True if search_company_documents already returned a non-empty hit."""
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


def _synthetic_tool_message(name: str, tool_call_id: str, content: str) -> ToolMessage:
    return ToolMessage(
        content=content,
        name=name or "unknown_tool",
        tool_call_id=tool_call_id or str(uuid.uuid4()),
    )


# ===========================================================================
# LLM CALL HELPERS
# ===========================================================================

async def _astream_complete(
    llm: BaseChatModel | Any,
    messages: list[BaseMessage],
    config: RunnableConfig | None = None,
) -> AIMessage:
    """Stream the LLM and assemble chunks into a single AIMessage.

    Raises:
        asyncio.TimeoutError: if the LLM takes longer than TOOL_CALL_TIMEOUT_SECS.
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

        if isinstance(assembled, AIMessage) and not type(assembled).__name__.endswith(
            "Chunk"
        ):
            return assembled

        return AIMessage(
            content=assembled.content,
            tool_calls=list(getattr(assembled, "tool_calls", None) or []),
            additional_kwargs=dict(
                getattr(assembled, "additional_kwargs", None) or {}
            ),
            response_metadata=dict(
                getattr(assembled, "response_metadata", None) or {}
            ),
            id=getattr(assembled, "id", None),
        )

    return await asyncio.wait_for(_stream(), timeout=TOOL_CALL_TIMEOUT_SECS)


async def _acall_plain(
    prompt: str,
    config: RunnableConfig | None = None,
) -> str:
    """Call the base LLM (no tools bound) and return the response as plain text."""
    base_llm = get_llm()
    response = await asyncio.wait_for(
        base_llm.ainvoke([HumanMessage(content=prompt)], config=config),
        timeout=TOOL_CALL_TIMEOUT_SECS,
    )
    content = getattr(response, "content", "") or ""
    if isinstance(content, list):
        content = " ".join(
            str(b.get("text", "") if isinstance(b, dict) else b) for b in content
        )
    return str(content).strip()


# ===========================================================================
# GRAPH NODES
# ===========================================================================

# ---------------------------------------------------------------------------
# Pattern 4 — Planner node
# ---------------------------------------------------------------------------

async def planner_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """Decompose the question into an ordered plan and store it in state.

    The planner calls the base LLM (no tools) and writes a numbered list of
    steps into state["plan"]. The executor (agent node) then works through
    the plan implicitly — the plan is prepended to the system prompt.
    """
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    question = str(last_human.content) if last_human else ""

    logger.info("planner_node: generating plan for %.80s…", question)
    try:
        raw_plan = await _acall_plain(build_planner_prompt(question), config=config)
    except asyncio.TimeoutError:
        logger.warning("planner_node timed out — skipping to direct agent")
        return {"plan": [], "is_complex": False}
    except Exception:
        logger.exception("planner_node failed — skipping")
        return {"plan": [], "is_complex": False}

    # Parse the numbered list into individual step strings.
    steps: list[str] = []
    for line in raw_plan.splitlines():
        line = line.strip()
        # Match "1. Step text", "1) Step text", or "• Step text"
        m = re.match(r"^(?:\d+[.)]\s*|[-•]\s*)(.+)", line)
        if m:
            steps.append(m.group(1).strip())

    # Fallback: if the LLM returned no numbered/bulleted list at all,
    # treat each non-empty line as a step (handles free-form responses).
    if not steps:
        steps = [line.strip() for line in raw_plan.splitlines() if line.strip()]

    logger.info("planner_node: %d steps → %s", len(steps), steps)
    return {"plan": steps, "is_complex": True}


# ---------------------------------------------------------------------------
# Agent node — LLM executor
# ---------------------------------------------------------------------------

async def agent_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    bound_llm: Any,
) -> dict[str, Any]:
    """Core LLM node. When a plan exists, it is appended to the system prompt."""
    system_prompt: str = (config.get("configurable") or {}).get(
        "system_prompt", ""
    ) or build_system_prompt()

    # Pattern 4: inject the plan so the executor knows which steps to follow.
    plan = state.get("plan") or []
    if plan:
        plan_block = "\n\nEXECUTION PLAN (follow these steps in order):\n" + "\n".join(
            f"  {i + 1}. {step}" for i, step in enumerate(plan)
        )
        system_prompt = system_prompt + plan_block

    messages_with_system: list[BaseMessage] = [
        SystemMessage(content=system_prompt)
    ] + list(state["messages"])

    response = await _astream_complete(bound_llm, messages_with_system, config=config)
    return {"messages": [response]}


async def _agent_node_runtime(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """Runtime wrapper that resolves the bound LLM from ContextVar."""
    try:
        bound_llm = _bound_llm_var.get()
    except LookupError as exc:
        raise RuntimeError(
            "Agent graph invoked without a bound LLM. Use aask(), ask(), or "
            "KnowledgeTransferAgent.run() so tools are installed for the request."
        ) from exc
    return await agent_node(state, config, bound_llm=bound_llm)


# ---------------------------------------------------------------------------
# Tools node
# ---------------------------------------------------------------------------

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
                _synthetic_tool_message(name, tid, REDUNDANT_TOOL_RESULT)
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
    tool_state: dict = {"messages": [subset]}
    tool_node = ToolNode(list(tools), handle_tool_errors=True)

    try:
        executed = await asyncio.wait_for(
            tool_node.ainvoke(tool_state, config),
            timeout=TOOL_EXEC_TIMEOUT_SECS,
        )
    except asyncio.TimeoutError:
        logger.error("Tool execution timed out after %ds", TOOL_EXEC_TIMEOUT_SECS)
        timeout_msgs: list[ToolMessage] = []
        for tc in allowed:
            name, _, tid = _tool_call_parts(tc)
            timeout_msgs.append(
                _synthetic_tool_message(
                    name,
                    tid,
                    f"Tool '{name}' timed out after {TOOL_EXEC_TIMEOUT_SECS}s.",
                )
            )
        return {"messages": results + timeout_msgs}

    executed_messages = list((executed or {}).get("messages") or [])
    return {"messages": results + executed_messages}


async def _tools_node_runtime(
    state: AgentState, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Runtime wrapper that resolves tools from ContextVar."""
    try:
        tools = _tools_var.get()
    except LookupError as exc:
        raise RuntimeError(
            "Agent graph invoked without tools. Use aask(), ask(), or "
            "KnowledgeTransferAgent.run()."
        ) from exc
    return await run_tools_node(state, config, tools)


# ---------------------------------------------------------------------------
# Pattern 1 — Reflection node
# ---------------------------------------------------------------------------

_REFLECTION_IMPROVED_PREFIX = "REFLECTION_IMPROVED:"
_REFLECTION_PASS_PREFIX = "REFLECTION_PASS:"


def _extract_reflection_content(raw: str) -> tuple[str, str]:
    """Parse the reflection LLM output.

    Returns:
        (status, answer) where status is "improved", "pass", or "pass" (fallback).
    """
    stripped = raw.strip()
    if stripped.upper().startswith("REFLECTION_IMPROVED:"):
        answer = stripped[len("REFLECTION_IMPROVED:"):].strip()
        return "improved", answer
    if stripped.upper().startswith("REFLECTION_PASS:"):
        answer = stripped[len("REFLECTION_PASS:"):].strip()
        return "pass", answer
    # Fallback: the LLM didn't follow the protocol — keep the original draft.
    logger.warning(
        "reflection_node: LLM did not follow prefix protocol; keeping draft."
    )
    return "pass", ""


async def reflection_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """Critique the draft answer and rewrite it if quality is insufficient.

    Reads the last AIMessage (the draft), calls the base LLM with the critique
    prompt, and either replaces the last message content (improved) or leaves
    messages unchanged (pass).
    """
    messages = list(state["messages"])

    # Find the last AI message that isn't a tool call — that's the draft answer.
    draft_msg: AIMessage | None = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            draft_msg = msg
            break

    if draft_msg is None:
        logger.warning("reflection_node: no draft AIMessage found; skipping.")
        return {"reflection_status": "pass"}

    draft_text = _chunk_text(draft_msg.content)
    if not draft_text.strip():
        return {"reflection_status": "pass"}

    # Retrieve the original question for the reflection prompt.
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    question = str(last_human.content) if last_human else ""

    # Collect tool output context so the reflector can judge completeness.
    tool_context_parts: list[str] = []
    _MAX_TOOL_CONTEXT = 1500
    for msg in messages:
        if msg.type == "tool" and msg.content:
            content_str = str(msg.content).strip()
            if content_str and content_str != REDUNDANT_TOOL_RESULT:
                tool_name = getattr(msg, "name", "tool")
                # Smart truncation: keep head + tail for context completeness.
                if len(content_str) > _MAX_TOOL_CONTEXT:
                    half = _MAX_TOOL_CONTEXT // 2
                    snippet = content_str[:half] + "\n...[truncated]...\n" + content_str[-half:]
                else:
                    snippet = content_str
                tool_context_parts.append(f"[{tool_name}]: {snippet}")
    tool_context = "\n".join(tool_context_parts) if tool_context_parts else None

    logger.info("reflection_node: critiquing draft (%.80s…)", draft_text)
    try:
        raw_reflection = await _acall_plain(
            build_reflection_prompt(question, draft_text, tool_context=tool_context),
            config=config,
        )
    except asyncio.TimeoutError:
        logger.warning("reflection_node timed out — keeping draft.")
        return {"reflection_status": "pass"}
    except Exception:
        logger.exception("reflection_node failed — keeping draft.")
        return {"reflection_status": "pass"}

    status, improved_text = _extract_reflection_content(raw_reflection)

    if status == "improved" and improved_text:
        # Replace the draft message content in-place.
        logger.info("reflection_node: draft improved by reflection.")
        improved_msg = AIMessage(
            content=improved_text,
            tool_calls=[],
            additional_kwargs=dict(draft_msg.additional_kwargs or {}),
            response_metadata=dict(draft_msg.response_metadata or {}),
            id=draft_msg.id,
        )
        # Return only the replacement message; add_messages reducer will
        # match on id and update it in the state list.
        return {
            "messages": [improved_msg],
            "reflection_status": "improved",
        }

    logger.info("reflection_node: draft passed quality check.")
    return {"reflection_status": "pass"}


# ===========================================================================
# GRAPH ROUTING
# ===========================================================================

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


def should_continue(state: AgentState) -> Literal["tools", "reflection", "__end__"]:
    """Go to tools if the LLM requested a tool call; reflect only on complex answers."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    
    # Skip reflection for simple direct answers — no tools were used and the
    # planner didn't run, so the LLM answered from its own knowledge.
    if not state.get("is_complex"):
        turn_msgs = _get_current_turn_messages(list(state["messages"]))
        has_tool_use = any(m.type == "tool" for m in turn_msgs)
        if not has_tool_use:
            return "__end__"
    return "reflection"


# ===========================================================================
# GRAPH COMPILATION
# ===========================================================================

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


def _prepare_run(
    question: str,
    session_id: str,
    checkpointer: Any,
    history: list[BaseMessage] | None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Build initial state and config. Caller must enter _request_tool_context."""
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
    """Return the compiled graph for checkpointer.

    _dynamic_tools is accepted for call-site compatibility; tools are
    injected per request via _request_tool_context, not compiled in.
    """
    return _get_compiled_graph(checkpointer)


# ===========================================================================
# PUBLIC ENTRY POINTS
# ===========================================================================

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
                logger.error("Agent timed out after %ds", TOOL_CALL_TIMEOUT_SECS)
                return _error_payload(
                    f"Request timed out after {TOOL_CALL_TIMEOUT_SECS}s. "
                    "Try a simpler question."
                )
            except GraphRecursionError:
                logger.error(
                    "Agent hit recursion_limit=%s", AGENT_RECURSION_LIMIT
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

    Use await aask(...) directly in async contexts (FastAPI, Jupyter).
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


# ===========================================================================
# STREAMING INTERFACE
# ===========================================================================

class KnowledgeTransferAgent:
    """Streaming interface for the agent (token + tool + plan + reflection events)."""

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
          - status:     {"type": "status",     "stage": "thinking"|"planning"|"reflecting"}
          - plan:       {"type": "plan",       "steps": ["step 1", ...]}
          - token:      {"type": "token",      "text": "..."}
          - tool:       {"type": "tool",       "name": "..."}
          - reflection: {"type": "reflection", "status": "pass"|"improved"}
          - done:       {"type": "done",       "payload": {...}}
          - error:      {"type": "error",      "detail": "..."}
        """
        yield {"type": "status", "stage": "thinking"}

        async with mcp_server_context() as mcp_tools:
            all_tools = tuple(LOCAL_TOOLS) + tuple(mcp_tools)
            graph, initial_state, config = _prepare_run(
                question, session_id, self.checkpointer, history
            )

            final_values: dict | None = None
            emitted_tool_ids: set[str] = set()
            plan_emitted: bool = False
            reflection_emitted: bool = False

            try:
                with _request_tool_context(all_tools):
                    async for mode, data in graph.astream(
                        initial_state,
                        config=config,
                        stream_mode=["messages", "values"],
                    ):
                        if mode == "messages":
                            token, metadata = data
                            node = metadata.get("langgraph_node")

                            # ---- token streaming (executor agent only) ----
                            if node == "agent":
                                if _has_tool_call_chunks(token):
                                    continue
                                text = _chunk_text(
                                    getattr(token, "content", None)
                                )
                                if text:
                                    yield {"type": "token", "text": text}

                        elif mode == "values":
                            final_values = data
                            messages = (data or {}).get("messages") or []

                            # ---- plan event (emitted once after planner runs) ----
                            plan = (data or {}).get("plan") or []
                            if plan and not plan_emitted:
                                plan_emitted = True
                                yield {"type": "status", "stage": "planning"}
                                yield {"type": "plan", "steps": plan}

                            # ---- tool events ----
                            if messages:
                                last = messages[-1]
                                if getattr(last, "tool_calls", None):
                                    for tc in last.tool_calls:
                                        name, _, tid = _tool_call_parts(tc)
                                        if (
                                            not name
                                            or not tid
                                            or tid in emitted_tool_ids
                                        ):
                                            continue
                                        emitted_tool_ids.add(tid)
                                        yield {"type": "tool", "name": name}

                            # ---- reflection event ----
                            ref_status = (data or {}).get("reflection_status", "")
                            if ref_status and not reflection_emitted:
                                reflection_emitted = True
                                yield {
                                    "type": "reflection",
                                    "status": ref_status,
                                }

                if not final_values:
                    yield {
                        "type": "error",
                        "detail": "Agent finished without a result.",
                    }
                    return

                # Emit reflection status before done if not yet emitted
                ref_status = (final_values or {}).get("reflection_status", "")
                if ref_status and not reflection_emitted:
                    yield {"type": "reflection", "status": ref_status}

                yield {
                    "type": "done",
                    "payload": serialize_parse_result(parse_result(final_values)),
                }

            except asyncio.TimeoutError:
                logger.error("Agent timed out after %ds", TOOL_CALL_TIMEOUT_SECS)
                yield {
                    "type": "error",
                    "detail": (
                        f"Request timed out after {TOOL_CALL_TIMEOUT_SECS}s. "
                        "Try a simpler question."
                    ),
                }
            except GraphRecursionError:
                logger.error(
                    "Agent hit recursion_limit=%s", AGENT_RECURSION_LIMIT
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


# ===========================================================================
# LEGACY EXPORTS (backward compatibility with old split-file structure)
# ===========================================================================

# Export all public symbols that tests and app.py may import
__all__ = [
    # State
    "AgentState",
    "LOCAL_TOOLS",
    "AGENT_RECURSION_LIMIT",
    "TOOL_CALL_TIMEOUT_SECS",
    "TOOL_EXEC_TIMEOUT_SECS",
    "REDUNDANT_TOOL_RESULT",
    "COMPLEX_KEYWORDS",
    "MIN_WORDS_FOR_PLANNER",
    # Nodes
    "planner_node",
    "agent_node",
    "run_tools_node",
    "reflection_node",
    # Routing
    "complexity_router",
    "should_continue",
    # Graph
    "build_graph",
    # Entry points
    "aask",
    "ask",
    "KnowledgeTransferAgent",
    # Internal (used by tests)
    "_extract_reflection_content",
    "_classify_complexity",
    "_is_redundant_tool_call",
    "_chunk_text",
    "_error_payload",
]
