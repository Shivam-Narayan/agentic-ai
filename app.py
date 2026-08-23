import logging

from fastapi import FastAPI, HTTPException

from src.agent.config import setup_logging
from src.agent.schemas import QuestionRequest, QuestionResponse
from src.agent.workflow import aask

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LangGraph KT Assistant",
    version="1.0.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    try:
        logger.info("Received question: %s", request.question)
        result = await aask(request.question)
        return QuestionResponse(
            answer=result.get("generation", ""),
            datasource=result.get("datasource"),
            tools_used=result.get("tools_used") or [],
        )
    except Exception as exc:
        logger.exception("Failed to process question")
        raise HTTPException(
            status_code=500,
            detail="Failed to process request",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
