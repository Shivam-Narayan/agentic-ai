"""Agent state definition and module-level constants."""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

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

# ---------------------------------------------------------------------------
# Agent state  (Pattern 4: plan field; Pattern 1: reflection fields)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Extended state that carries the plan and reflection outcome alongside messages."""

    # Full conversation — uses LangGraph's add_messages reducer so messages
    # are appended rather than overwritten on each node return.
    messages: Annotated[list[BaseMessage], add_messages]

    # Pattern 4 — Planner output: list of step strings, e.g.
    #   ["Search company docs for X", "Calculate Y from result"]
    # Empty list means no plan was created (simple question path).
    plan: list[str]

    # Pattern 1 — Reflection outcome tag: "pass", "improved", or "" (not yet run).
    reflection_status: str

    # Tracks whether the planner ran for this request (controls routing).
    is_complex: bool


# ---------------------------------------------------------------------------
# Helpers used across modules
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
