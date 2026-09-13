"""Graph node implementations — planner, agent executor, tools, and reflection."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from .llm_helpers import _acall_plain, _astream_complete
from .prompt import (
    _build_planner_prompt,
    _build_reflection_prompt,
    _build_system_prompt,
)
from .state import (
    REDUNDANT_TOOL_RESULT,
    TOOL_EXEC_TIMEOUT_SECS,
    AgentState,
    _chunk_text,
    _get_current_turn_messages,
    _is_redundant_tool_call,
    _tool_call_parts,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern 4 — Planner node
# ---------------------------------------------------------------------------

async def planner_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """Decompose the question into an ordered plan and store it in state.

    The planner calls the base LLM (no tools) and writes a numbered list of
    steps into ``state["plan"]``.  The executor (agent node) then works through
    the plan implicitly — the plan is prepended to the system prompt so the
    agent knows what steps to follow.
    """
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    question = str(last_human.content) if last_human else ""

    logger.info("planner_node: generating plan for %.80s…", question)
    try:
        raw_plan = await _acall_plain(_build_planner_prompt(question), config=config)
    except asyncio.TimeoutError:
        logger.warning("planner_node timed out — skipping to direct agent")
        return {"plan": [], "is_complex": False}
    except Exception:
        logger.exception("planner_node failed — skipping")
        return {"plan": [], "is_complex": False}

    # Parse the numbered list into individual step strings.
    # Regex-first pass: extract only properly formatted steps.
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
# Agent node — LLM executor; receives plan via system prompt when available
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
    ) or _build_system_prompt()

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
    return "pass", ""  # empty → caller uses original draft


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
    draft_idx: int = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            draft_msg = msg
            draft_idx = i
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
    for msg in messages:
        if msg.type == "tool" and msg.content:
            content_str = str(msg.content).strip()
            if content_str and content_str != REDUNDANT_TOOL_RESULT:
                tool_name = getattr(msg, "name", "tool")
                # Truncate very long tool outputs to avoid prompt bloat.
                snippet = content_str[:500] + ("…" if len(content_str) > 500 else "")
                tool_context_parts.append(f"[{tool_name}]: {snippet}")
    tool_context = "\n".join(tool_context_parts) if tool_context_parts else None

    logger.info("reflection_node: critiquing draft (%.80s…)", draft_text)
    try:
        raw_reflection = await _acall_plain(
            _build_reflection_prompt(question, draft_text, tool_context=tool_context),
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
