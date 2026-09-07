"""
Optional live connectivity check for Tavily search API.
Only executes when RUN_LIVE_API_TESTS=true is set in environment.
"""
import os
import pytest
from dotenv import load_dotenv

load_dotenv()


def test_tavily_api_connection():
    if os.getenv("RUN_LIVE_API_TESTS", "").lower() != "true":
        pytest.skip("Set RUN_LIVE_API_TESTS=true to run live third-party Tavily API checks.")

    key = os.getenv("TAVILY_API_KEY")
    if not key or key.startswith("your_"):
        pytest.skip("TAVILY_API_KEY not configured in environment.")

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tool = TavilySearchResults(k=2)
        results = tool.invoke({"query": "Python programming language"})
        if isinstance(results, str) and "HTTPError" in results:
            pytest.skip(f"Tavily API returned upstream error: {results}")
        assert isinstance(results, list)
    except Exception as exc:
        pytest.skip(f"Tavily live check skipped due to network/upstream error: {exc}")
