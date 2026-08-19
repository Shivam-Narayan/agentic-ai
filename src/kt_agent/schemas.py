from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class QuestionResponse(BaseModel):
    answer: str
    datasource: str | None = None
