"""LLM call helpers — streaming assembler and plain-text invoke."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from .chains import get_llm
from .state import TOOL_CALL_TIMEOUT_SECS

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# LLM plain-text call helper (no streaming needed for planner / reflection)
# ---------------------------------------------------------------------------

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
