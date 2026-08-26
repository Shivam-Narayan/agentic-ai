from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", description="Session ID for conversation memory")


class Citation(BaseModel):
    source: str = Field(..., description="Filename, table name, or URL that was consulted")
    detail: str = Field(default="", description="Extra detail e.g. page number, SQL used, search snippet")


class QuestionResponse(BaseModel):
    answer: str
    datasource: str | None = None
    tools_used: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    chart_data: dict[str, Any] | None = Field(
        default=None,
        description="Plotly figure JSON when a chart was generated, else null",
    )
