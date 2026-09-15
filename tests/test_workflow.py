"""
Unit tests for LangGraph agent workflow components, parsers, and state helpers.
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.agent.parser import parse_result, serialize_parse_result
from src.tools.tools import EMPTY_COMPANY_SEARCH_RESULT
from src.agent.workflow import (
    _REDUNDANT_TOOL_RESULT,
    _classify_complexity,
    _extract_reflection_content,
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
    """Agent with pending tool calls should route to tools node."""
    with_tools = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "calculate", "args": {}, "id": "1"}],
            )
        ]
    }
    assert should_continue(with_tools) == "tools"


def test_should_continue_skips_reflection_for_simple_direct():
    """Simple direct answers (no tools, not complex) skip reflection → __end__."""
    simple_direct = {
        "messages": [
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language."),
        ],
        "is_complex": False,
    }
    assert should_continue(simple_direct) == "__end__"


def test_should_continue_reflects_when_complex():
    """Complex questions (planner ran) always go through reflection."""
    complex_no_tools = {
        "messages": [
            HumanMessage(content="Compare revenue trends"),
            AIMessage(content="Based on the data..."),
        ],
        "is_complex": True,
    }
    assert should_continue(complex_no_tools) == "reflection"


def test_should_continue_reflects_when_tools_used():
    """Even simple-path questions reflect if tools were used in this turn."""
    simple_with_tools = {
        "messages": [
            HumanMessage(content="What is in my resume?"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_company_documents", "args": {"query": "resume"}, "id": "1"}],
            ),
            ToolMessage(content="Skills: Python, FastAPI", name="search_company_documents", tool_call_id="1"),
            AIMessage(content="Your resume lists Python and FastAPI."),
        ],
        "is_complex": False,
    }
    assert should_continue(simple_with_tools) == "reflection"


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


# ---------------------------------------------------------------------------
# Pattern 4 — Complexity classifier tests
# ---------------------------------------------------------------------------

def test_classify_complexity_simple_short():
    """Short questions without keywords should be classified as simple."""
    assert _classify_complexity("What is Python?") is False
    assert _classify_complexity("Hello") is False


def test_classify_complexity_short_with_keyword():
    """Short questions with keywords but under word threshold stay simple."""
    assert _classify_complexity("Compare A B") is False


def test_classify_complexity_long_with_keyword():
    """Long questions with complexity keywords should be classified as complex."""
    assert _classify_complexity("Compare the revenue trends between Q1 and Q2 for the company") is True
    assert _classify_complexity("Analyse the breakdown of expenses by department over the last year") is True
    assert _classify_complexity("What caused the decline in sales during the third quarter period?") is True


def test_classify_complexity_long_without_keyword():
    """Long questions without complexity keywords should be simple."""
    assert _classify_complexity("Tell me about the company history and when it was founded") is False


# ---------------------------------------------------------------------------
# Pattern 1 — Reflection content extraction tests
# ---------------------------------------------------------------------------

def test_extract_reflection_content_pass():
    """REFLECTION_PASS prefix should return status=pass with the original text."""
    status, text = _extract_reflection_content("REFLECTION_PASS: The answer is correct.")
    assert status == "pass"
    assert text == "The answer is correct."


def test_extract_reflection_content_improved():
    """REFLECTION_IMPROVED prefix should return status=improved with rewritten text."""
    status, text = _extract_reflection_content("REFLECTION_IMPROVED: Here is a better answer.")
    assert status == "improved"
    assert text == "Here is a better answer."


def test_extract_reflection_content_case_insensitive():
    """Prefix detection should be case-insensitive."""
    status, text = _extract_reflection_content("reflection_pass: Still good.")
    assert status == "pass"
    assert text == "Still good."

    status, text = _extract_reflection_content("Reflection_Improved: Better version.")
    assert status == "improved"
    assert text == "Better version."


def test_extract_reflection_content_no_prefix():
    """Missing prefix should fallback to pass with empty text (keep original draft)."""
    status, text = _extract_reflection_content("The answer looks fine to me.")
    assert status == "pass"
    assert text == ""


def test_extract_reflection_content_multiline():
    """Multi-line improved answer should be captured correctly."""
    raw = "REFLECTION_IMPROVED: Line one.\nLine two.\nLine three."
    status, text = _extract_reflection_content(raw)
    assert status == "improved"
    assert "Line one." in text


def test_extract_reflection_content_whitespace():
    """Leading/trailing whitespace should be stripped."""
    status, text = _extract_reflection_content("  REFLECTION_PASS:   Clean answer.  ")
    assert status == "pass"
    assert text == "Clean answer."


# ---------------------------------------------------------------------------
# Reflection prompt builder tests
# ---------------------------------------------------------------------------

def test_reflection_prompt_without_tool_context():
    """Prompt should work without tool context (backward compatible)."""
    from src.agent.prompt import build_reflection_prompt
    prompt = build_reflection_prompt("What is AI?", "AI is artificial intelligence.")
    assert "ORIGINAL QUESTION:" in prompt
    assert "DRAFT ANSWER:" in prompt
    assert "TOOL OUTPUTS" not in prompt


def test_reflection_prompt_with_tool_context():
    """Prompt should include tool outputs when provided."""
    from src.agent.prompt import build_reflection_prompt
    prompt = build_reflection_prompt(
        "What is in the report?",
        "The report shows revenue of $10M.",
        tool_context="[search_company_documents]: Revenue was $10M in Q1."
    )
    assert "TOOL OUTPUTS" in prompt
    assert "[search_company_documents]:" in prompt
