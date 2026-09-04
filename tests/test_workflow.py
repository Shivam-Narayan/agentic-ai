"""
Unit tests for LangGraph agent workflow components, parsers, and state helpers.
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.agent.parser import parse_result, serialize_parse_result
from src.agent.workflow import (
    _make_tool_call_key,
    _get_previous_tool_calls,
    _first_search_had_results,
)


def test_make_tool_call_key_deterministic():
    k1 = _make_tool_call_key("search_company_documents", {"query": "revenue", "page": 1})
    k2 = _make_tool_call_key("search_company_documents", {"page": 1, "query": "revenue"})
    assert k1 == k2


def test_get_previous_tool_calls():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "calculate", "args": {"expression": "2+2"}, "id": "call_1"}],
    )
    calls = _get_previous_tool_calls([msg])
    expected_key = _make_tool_call_key("calculate", {"expression": "2+2"})
    assert expected_key in calls


def test_first_search_had_results():
    empty_tool = ToolMessage(
        content="No matching company documents were found.",
        name="search_company_documents",
        tool_call_id="call_1",
    )
    assert _first_search_had_results([empty_tool]) is False

    valid_tool = ToolMessage(
        content="[Source: financial_report.pdf, page 2]\nRevenue was $10M.",
        name="search_company_documents",
        tool_call_id="call_2",
    )
    assert _first_search_had_results([valid_tool]) is True


def test_parse_result_direct_llm():
    messages = [
        HumanMessage(content="Hello!"),
        AIMessage(content="Hello! How can I assist you today?"),
    ]
    parsed = parse_result({"messages": messages})
    assert parsed["generation"] == "Hello! How can I assist you today?"
    assert parsed["datasource"] == "direct_llm"
    assert parsed["tools_used"] == []


def test_parse_result_with_citations_and_tools():
    messages = [
        HumanMessage(content="What was Q1 revenue?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_company_documents", "args": {"query": "Q1 revenue"}, "id": "1"}],
        ),
        ToolMessage(
            content="[Source: report.pdf, page 4]\nQ1 revenue was $5 million.",
            name="search_company_documents",
            tool_call_id="1",
        ),
        AIMessage(content="According to the report, Q1 revenue was $5 million."),
    ]
    parsed = parse_result({"messages": messages})
    assert "Q1 revenue was $5 million" in parsed["generation"]
    assert parsed["datasource"] == "company_docs"
    assert "search_company_documents" in parsed["tools_used"]
    assert len(parsed["citations"]) == 1
    assert parsed["citations"][0]["source"] == "report.pdf, page 4"


def test_serialize_parse_result_backward_compatibility():
    parsed = {
        "generation": "Test response",
        "datasource": "database",
        "tools_used": ["query_company_database"],
        "citations": [{"source": "SQL", "detail": "SELECT *"}],
        "chart_data": None,
    }
    serialized = serialize_parse_result(parsed)
    # Both 'answer' and 'generation' must exist for backward-compatibility with UI
    assert serialized["answer"] == "Test response"
    assert serialized["generation"] == "Test response"
    assert serialized["datasource"] == "database"
