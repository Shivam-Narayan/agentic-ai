"""
LangGraph agentic workflow for the CTE Knowledge Transfer Assistant.

Architecture overview (with Planner + Reflection):

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
                            is_complex OR tools_used?
                              /          \\
                           yes            no
                            |              |
                            v              v
                     [ reflection ]     [ END ]  (skip reflection for
                            |                     simple direct answers)
                            v
                         [ END ]  -> parse_result() -> response

Module layout:
  - state.py       AgentState, constants, shared helpers (dedup, turn-scoping)
  - llm_helpers.py LLM streaming assembler and plain-text invoke
  - nodes.py       Graph node implementations (planner, agent, tools, reflection)
  - graph.py       Graph compilation, caching, routing, ContextVar wiring
  - workflow.py    (this file) Public API: aask(), ask(), KnowledgeTransferAgent
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from langchain_core.messages import BaseMessage
from langgraph.errors import GraphRecursionError

from .graph import _prepare_run, _request_tool_context, build_graph  # noqa: F401
from src.tools.mcp_tools import mcp_server_context
from .parser import parse_result, serialize_parse_result
from .state import (
    AGENT_RECURSION_LIMIT,
    LOCAL_TOOLS,
    TOOL_CALL_TIMEOUT_SECS,
    _chunk_text,
    _error_payload,
    _has_tool_call_chunks,
    _tool_call_parts,
)

# Re-export everything that was importable from the old monolithic workflow.py.
# This ensures backward compatibility for all callers (tests, app.py, __init__.py).
from .graph import (  # noqa: F401
    _classify_complexity,
    complexity_router,
    should_continue,
)
from .nodes import (  # noqa: F401
    _extract_reflection_content,
    agent_node,
    planner_node,
    reflection_node,
    run_tools_node,
)
from .state import (  # noqa: F401
    REDUNDANT_TOOL_RESULT as _REDUNDANT_TOOL_RESULT,
    AgentState,
    _first_search_had_results,
    _get_previous_tool_calls,
    _is_redundant_tool_call,
    _make_tool_call_key,
)

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Streaming agent — emits plan steps + reflection stage events
# ---------------------------------------------------------------------------

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
          - ``status``:     ``{"type": "status",     "stage": "thinking"|"planning"|"reflecting"}``
          - ``plan``:       ``{"type": "plan",        "steps": ["step 1", "step 2", ...]}``
          - ``token``:      ``{"type": "token",       "text": "..."}``
          - ``tool``:       ``{"type": "tool",        "name": "..."}``
          - ``reflection``: ``{"type": "reflection",  "status": "pass"|"improved"}``
          - ``done``:       ``{"type": "done",        "payload": {...}}``
          - ``error``:      ``{"type": "error",       "detail": "..."}``
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
