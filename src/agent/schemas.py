from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class QuestionResponse(BaseModel):
    answer: str
    datasource: str | None = None
    tools_used: list[str] = Field(default_factory=list)
