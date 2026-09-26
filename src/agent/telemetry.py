"""
Token counting, cost tracking, and per-request usage metrics.

Every LLM call in the agent accumulates token usage from
response_metadata (populated by LangChain for all major providers).
Cost is calculated using a static price table — update _COST_TABLE
when provider pricing changes.

Usage:
    tracker = UsageTracker()
    tracker.record(ai_message)          # call once per AIMessage
    metrics = tracker.to_metrics()      # get final UsageMetrics
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost table  (USD per 1 000 tokens)
# Update these when providers change pricing.
# ---------------------------------------------------------------------------

_COST_TABLE: dict[str, dict[str, float]] = {
    # Groq  ─  https://groq.com/pricing
    "openai/gpt-oss-120b":     {"prompt": 0.0009, "completion": 0.0009},
    "llama3-70b-8192":         {"prompt": 0.0006, "completion": 0.0008},
    "llama3-8b-8192":          {"prompt": 0.00005,"completion": 0.00008},
    "mixtral-8x7b-32768":      {"prompt": 0.0006, "completion": 0.0006},

    # Google Gemini  ─  https://ai.google.dev/pricing
    "gemini-1.5-flash":        {"prompt": 0.000075, "completion": 0.0003},
    "gemini-1.5-pro":          {"prompt": 0.00125,  "completion": 0.005},
    "gemini-2.0-flash":        {"prompt": 0.0001,   "completion": 0.0004},

    # OpenAI  ─  https://openai.com/pricing
    "gpt-4o":                  {"prompt": 0.0025,  "completion": 0.010},
    "gpt-4o-mini":             {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4-turbo":             {"prompt": 0.010,   "completion": 0.030},
    "gpt-3.5-turbo":           {"prompt": 0.0005,  "completion": 0.0015},

    # Azure OpenAI  ─  same as OpenAI by default (region pricing may differ)
    "gpt-4o-azure":            {"prompt": 0.0025,  "completion": 0.010},
    "gpt-4o-mini-azure":       {"prompt": 0.00015, "completion": 0.0006},

    # Cohere  ─  https://cohere.com/pricing
    "command-r-plus":          {"prompt": 0.003,   "completion": 0.015},
    "command-r":               {"prompt": 0.00015, "completion": 0.0006},
}

# Fallback price when the model isn't in the table.
_FALLBACK_COST = {"prompt": 0.001, "completion": 0.002}


def _cost_per_1k(model: str) -> dict[str, float]:
    """Return prompt/completion cost per 1k tokens for a model name."""
    # Exact match first
    if model in _COST_TABLE:
        return _COST_TABLE[model]
    # Substring match — handles versioned names like "gemini-1.5-flash-001"
    model_lower = model.lower()
    for key, prices in _COST_TABLE.items():
        if key in model_lower:
            return prices
    logger.debug("Model '%s' not in cost table — using fallback price", model)
    return _FALLBACK_COST


# ---------------------------------------------------------------------------
# Core data class
# ---------------------------------------------------------------------------

@dataclass
class UsageMetrics:
    """Per-request token and cost summary."""

    prompt_tokens:     int   = 0       # total prompt tokens across all LLM calls
    completion_tokens: int   = 0       # total completion tokens
    total_tokens:      int   = 0       # prompt + completion
    cost_usd:          float = 0.0     # estimated USD cost
    model:             str   = ""      # last model name seen
    latency_ms:        int   = 0       # wall-clock ms from first to last token
    llm_calls:         int   = 0       # number of individual LLM invocations

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
            "cost_usd":          round(self.cost_usd, 6),
            "model":             self.model,
            "latency_ms":        self.latency_ms,
            "llm_calls":         self.llm_calls,
        }

    def log_summary(self, session_id: str = "") -> None:
        """Write a one-line cost+token summary to the logger."""
        prefix = f"[session={session_id}] " if session_id else ""
        logger.info(
            "%sTokens: %d prompt + %d completion = %d total | "
            "Cost: $%.6f | Latency: %dms | Calls: %d | Model: %s",
            prefix,
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
            self.cost_usd,
            self.latency_ms,
            self.llm_calls,
            self.model or "unknown",
        )


# ---------------------------------------------------------------------------
# Tracker — accumulates usage across multiple LLM calls in one request
# ---------------------------------------------------------------------------

class UsageTracker:
    """Accumulates token usage from every AIMessage in a single agent run."""

    def __init__(self) -> None:
        self._prompt_tokens:     int   = 0
        self._completion_tokens: int   = 0
        self._model:             str   = ""
        self._calls:             int   = 0
        self._start_ms:          float = time.monotonic() * 1000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, ai_message: Any) -> None:
        """Extract and accumulate token counts from an AIMessage.

        Works with every major LangChain provider — Groq, Google Gemini,
        OpenAI, Azure OpenAI, and Cohere all populate response_metadata
        with token usage, though the key names differ slightly.
        """
        metadata: dict = getattr(ai_message, "response_metadata", None) or {}
        usage = self._extract_usage(metadata)

        if usage["prompt"] == 0 and usage["completion"] == 0:
            # Some streaming responses put usage on additional_kwargs
            extra: dict = getattr(ai_message, "additional_kwargs", None) or {}
            usage = self._extract_usage(extra) or usage

        self._prompt_tokens     += usage["prompt"]
        self._completion_tokens += usage["completion"]
        self._calls             += 1

        # Capture model name from metadata if available
        if not self._model:
            model = (
                metadata.get("model")
                or metadata.get("model_name")
                or metadata.get("model_id")
                or ""
            )
            if model:
                self._model = str(model)

    def to_metrics(self, latency_ms: int | None = None) -> UsageMetrics:
        """Build the final UsageMetrics object."""
        total = self._prompt_tokens + self._completion_tokens
        prices = _cost_per_1k(self._model)
        cost = (
            self._prompt_tokens     / 1000 * prices["prompt"]
            + self._completion_tokens / 1000 * prices["completion"]
        )
        elapsed = int(time.monotonic() * 1000 - self._start_ms)
        return UsageMetrics(
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=total,
            cost_usd=round(cost, 6),
            model=self._model,
            latency_ms=latency_ms if latency_ms is not None else elapsed,
            llm_calls=self._calls,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_usage(d: dict) -> dict[str, int]:
        """Normalise provider-specific token usage keys into prompt/completion."""
        if not d:
            return {"prompt": 0, "completion": 0}

        # ── OpenAI / Azure / Groq ──────────────────────────────────────
        # response_metadata → token_usage → { prompt_tokens, completion_tokens }
        token_usage = d.get("token_usage") or d.get("usage") or {}
        if token_usage:
            return {
                "prompt":     int(token_usage.get("prompt_tokens",     0)),
                "completion": int(token_usage.get("completion_tokens", 0)),
            }

        # Groq also puts it directly on response_metadata at top level
        if "prompt_tokens" in d:
            return {
                "prompt":     int(d.get("prompt_tokens",     0)),
                "completion": int(d.get("completion_tokens", 0)),
            }

        # ── Google Gemini ──────────────────────────────────────────────
        # response_metadata → usage_metadata → { prompt_token_count, candidates_token_count }
        usage_metadata = d.get("usage_metadata") or {}
        if usage_metadata:
            return {
                "prompt":     int(usage_metadata.get("prompt_token_count",     0)),
                "completion": int(usage_metadata.get("candidates_token_count", 0)),
            }

        # ── Cohere ────────────────────────────────────────────────────
        # response_metadata → meta → billed_units → { input_tokens, output_tokens }
        meta = d.get("meta") or {}
        billed = meta.get("billed_units") or {}
        if billed:
            return {
                "prompt":     int(billed.get("input_tokens",  0)),
                "completion": int(billed.get("output_tokens", 0)),
            }

        return {"prompt": 0, "completion": 0}
