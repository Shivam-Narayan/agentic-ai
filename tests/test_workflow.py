"""
Unit tests for LangGraph agent workflow components, parsers, and state helpers.
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.agent.parser import parse_result, serialize_parse_result
from src.agent.tools import EMPTY_COMPANY_SEARCH_RESULT
from src.agent.workflow import (
    _REDUNDANT_TOOL_RESULT,
    _first_search_had_results,
    _get_previous_tool_calls,
    _is_redundant_tool_call,
    _make_tool_call_key,
    run_tools_node,
    should_continue,
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
        content=EMPTY_COMPANY_SEARCH_RESULT,
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


def test_first_search_had_results_ignores_substring():
    msg = ToolMessage(
        content="Policy note: No matching jackets were sold last quarter.",
        name="search_company_documents",
        tool_call_id="call_3",
    )
    assert _first_search_had_results([msg]) is True


def test_is_redundant_identical_call_in_turn():
    prior = [
        HumanMessage(content="q"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "calculate", "args": {"expression": "2+2"}, "id": "1"}
            ],
        ),
        ToolMessage(content="4", name="calculate", tool_call_id="1"),
    ]
    assert _is_redundant_tool_call("calculate", {"expression": "2+2"}, prior) is True
    assert _is_redundant_tool_call("calculate", {"expression": "3+3"}, prior) is False


def test_is_redundant_search_after_successful_hit():
    prior = [
        HumanMessage(content="q"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_company_documents",
                    "args": {"query": "revenue"},
                    "id": "1",
                }
            ],
        ),
        ToolMessage(
            content="[Source: report.pdf]\nRevenue was $10M.",
            name="search_company_documents",
            tool_call_id="1",
        ),
    ]
    assert (
        _is_redundant_tool_call(
            "search_company_documents", {"query": "other"}, prior
        )
        is True
    )


def test_should_continue_routes_on_tool_calls():
    with_tools = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "calculate", "args": {}, "id": "1"}],
            )
        ]
    }
    without = {"messages": [AIMessage(content="done")]}
    assert should_continue(with_tools) == "tools"
    assert should_continue(without) == "__end__"


def test_redundant_skip_message_is_stable():
    assert "Do not retry this call" in _REDUNDANT_TOOL_RESULT


def test_run_tools_node_returns_skip_tool_messages():
    import asyncio

    prior_ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "calculate", "args": {"expression": "1+1"}, "id": "old"}
        ],
    )
    prior_tool = ToolMessage(content="2", name="calculate", tool_call_id="old")
    new_ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "calculate", "args": {"expression": "1+1"}, "id": "new"}
        ],
    )
    state = {
        "messages": [
            HumanMessage(content="q"),
            prior_ai,
            prior_tool,
            new_ai,
        ]
    }

    result = asyncio.run(run_tools_node(state, {}, ()))
    msgs = result["messages"]
    assert len(msgs) == 1
    assert msgs[0].tool_call_id == "new"
    assert msgs[0].content == _REDUNDANT_TOOL_RESULT


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
