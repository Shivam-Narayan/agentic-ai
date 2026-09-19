"""Pydantic models for API request/response contracts and structured LLM output."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Datasource type — single source of truth
#
# Used by DatasourceClassification, StructuredAnswer, and QuestionResponse.
# Changing a label here propagates everywhere automatically.
# ---------------------------------------------------------------------------

DatasourceType = Literal[
    "direct_llm",
    "company_docs",
    "database",
    "web_search",
    "calculation",
    "chart",
    "multiple",
]


# ---------------------------------------------------------------------------
# API request
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    """Incoming question payload for the /ask endpoint."""

    question:   str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(
        default="default",
        description="Session ID for conversation memory (letters, digits, hyphens, underscores; max 64 chars)",
    )


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A source reference attached to an agent answer."""

    source: str = Field(..., description="Filename, table name, or URL that was consulted")
    detail: str = Field(default="", description="Extra detail e.g. page number, SQL used, search snippet")


# ---------------------------------------------------------------------------
# Usage metrics  (token counting + cost tracking)
# ---------------------------------------------------------------------------

class UsageMetrics(BaseModel):
    """Per-request token counts and estimated USD cost.

    Populated from LLM response_metadata — works with Groq, Gemini,
    OpenAI, Azure OpenAI, and Cohere.
    """

    prompt_tokens:     int   = Field(default=0,   description="Tokens in the prompt(s)")
    completion_tokens: int   = Field(default=0,   description="Tokens in the completion(s)")
    total_tokens:      int   = Field(default=0,   description="Total tokens (prompt + completion)")
    cost_usd:          float = Field(default=0.0, description="Estimated USD cost for this request")
    model:             str   = Field(default="",  description="LLM model name")
    latency_ms:        int   = Field(default=0,   description="End-to-end latency in milliseconds")
    llm_calls:         int   = Field(default=0,   description="Number of LLM invocations in this request")


# ---------------------------------------------------------------------------
# API response
# ---------------------------------------------------------------------------

class QuestionResponse(BaseModel):
    """Full response payload returned by /ask."""

    answer:         str
    datasource:     Optional[DatasourceType]   = None
    tools_used:     list[str]                  = Field(default_factory=list)
    citations:      list[Citation]             = Field(default_factory=list)
    chart_data:     Optional[dict[str, Any]]   = Field(
        default=None,
        description="Plotly figure JSON when a chart was generated, else null",
    )
    usage:          Optional[UsageMetrics]     = Field(
        default=None,
        description="Token counts and cost for this request",
    )
    prompt_version: Optional[str]              = Field(
        default=None,
        description="Version tag of the system prompt that produced this answer",
    )


# ---------------------------------------------------------------------------
# Structured LLM output  (used with with_structured_output())
#
# These models enforce compile-time safety on the datasource field via
# the shared DatasourceType Literal — invalid values are rejected by Pydantic
# before they ever reach the rest of the application.
# ---------------------------------------------------------------------------

class DatasourceClassification(BaseModel):
    """Structured output for classifying which datasource answered a question.

    Use with: llm.with_structured_output(DatasourceClassification)

    Example::

        classifier = get_llm().with_structured_output(DatasourceClassification)
        result = classifier.invoke([HumanMessage(content=question)])
        print(result.datasource)  # "company_docs"
    """

    datasource: DatasourceType = Field(
        ...,
        description=(
            "Which datasource was used to answer the question. "
            "Must be one of: direct_llm, company_docs, database, "
            "web_search, calculation, chart, multiple"
        ),
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score 0–1 for this classification",
    )
    reasoning: str = Field(
        default="",
        description="One sentence explaining why this datasource was chosen",
    )


class StructuredAnswer(BaseModel):
    """Structured output for responses that need typed fields from the LLM.

    Used with with_structured_output(StructuredAnswer) in evaluation
    or when a caller needs a machine-readable breakdown of the answer.

    Example::

        extractor = get_llm().with_structured_output(StructuredAnswer)
        result = extractor.invoke(messages)
        print(result.datasource, result.confidence)
    """

    answer:      str            = Field(..., description="The final answer text")
    datasource:  DatasourceType = Field(..., description="Datasource used to produce this answer")
    tools_used:  list[str]      = Field(default_factory=list, description="Tool names that were called")
    citations:   list[str]      = Field(default_factory=list, description="Source references (filenames, URLs, SQL)")
    has_chart:   bool           = Field(default=False,        description="True if a chart was generated")
    confidence:  float          = Field(default=1.0, ge=0.0, le=1.0, description="Answer confidence 0–1")
