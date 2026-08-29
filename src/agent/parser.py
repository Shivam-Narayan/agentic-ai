"""Parsing logic for LangGraph agent outputs."""

import json
import logging
import re

logger = logging.getLogger(__name__)

# Maps each tool name to a UI datasource label shown in the Streamlit frontend
_TOOL_DATASOURCE = {
    "search_company_documents": "company_docs",
    "summarise_document":        "company_docs",
    "extract_structured_data":   "company_docs",
    "search_web":                "web_search",
    "calculate":                 "calculation",
    "generate_chart":            "chart",
    # MCP database tools — support both naming conventions (hyphen and underscore)
    "query_company_database":    "database",
    "query-database":            "database",
    "read-query":                "database",
    "list_database_tables":      "database",
    "list-tables":               "database",
    "describe_database_table":   "database",
}


def _extract_citations(messages: list) -> list[dict]:
    """
    Walk all ToolMessages and build a deduplicated list of citation dicts.

    Each citation is {"source": str, "detail": str}.  Sources are extracted:
      - search_company_documents / search_web: from [Source: <value>] markers
        that the tools embed in their output strings
      - query_company_database: SQL query pulled from the preceding AI message's
        tool_call args (so we show the exact query that ran)
      - calculate: the expression + result string
      - summarise_document: the filename parsed from the "[Full text of ...]" prefix
      - extract_structured_data: same [Source: ...] pattern as search tools
      - schema tools (list_tables, describe_table): labelled as "schema lookup"
    """
    citations: list[dict] = []
    seen: set[str] = set()

    def _add(source: str, detail: str = "") -> None:
        """Add a citation only if we haven't seen this source+detail pair before."""
        key = f"{source}::{detail}"
        if key not in seen:
            seen.add(key)
            citations.append({"source": source, "detail": detail})

    for msg in messages:
        # Only ToolMessages carry citation-relevant information
        if msg.type != "tool":
            continue

        name = msg.name or ""
        content = msg.content or ""

        if name in ("search_company_documents", "search_web"):
            # Tools embed "[Source: filename_or_url]" at the start of each snippet
            for match in re.finditer(r"\[Source:\s*(.+?)\]", content):
                _add(match.group(1).strip())

        elif name in ("query_company_database", "query-database", "read-query"):
            # The SQL is in the AI message's tool_call args, not in the tool output
            for ai_msg in messages:
                if ai_msg.type == "ai" and getattr(ai_msg, "tool_calls", None):
                    for tc in ai_msg.tool_calls:
                        if tc.get("name") == name:
                            sql = (
                                tc.get("args", {}).get("sql_query")
                                or tc.get("args", {}).get("query")
                                or ""
                            )
                            if sql:
                                _add("company database", sql.strip())

        elif name == "calculate":
            # Tool returns "expression = result" — cite the full expression
            _add("calculator", content.strip())

        elif name == "summarise_document":
            # Tool wraps output with "[Full text of <filename>]"
            match = re.match(r"\[Full text of (.+?)\]", content)
            if match:
                _add(match.group(1).strip())
            else:
                _add("company documents")

        elif name == "extract_structured_data":
            # Same [Source: ...] markers as search tools
            for match in re.finditer(r"\[Source:\s*(.+?)\]", content):
                _add(match.group(1).strip())

        elif name in ("list_database_tables", "list-tables", "describe_database_table"):
            _add("company database", f"schema lookup via {name}")

    return citations


def _extract_chart(messages: list) -> dict | None:
    """
    Return the Plotly figure as a JSON dict if generate_chart was used.

    generate_chart prefixes its output with "CHART_JSON::" followed by the
    serialised Plotly figure so parse_result can detect and forward it to
    the Streamlit frontend for inline rendering.
    """
    for msg in messages:
        if msg.type == "tool" and msg.name == "generate_chart":
            content = msg.content or ""
            if content.startswith("CHART_JSON::"):
                try:
                    return json.loads(content[len("CHART_JSON::"):])
                except json.JSONDecodeError:
                    logger.warning("Could not parse chart JSON from generate_chart output")
    return None


def parse_result(result: dict) -> dict:
    """
    Convert the raw LangGraph output into a clean response dict for FastAPI.

    Returns:
        {
            "generation":  str   — the final answer text
            "datasource":  str   — which data source was used (for UI badge)
            "tools_used":  list  — names of every tool that ran
            "citations":   list  — list of {"source", "detail"} dicts
            "chart_data":  dict  — Plotly figure JSON, or None
        }
    """
    messages = result["messages"]

    # Walk backwards through messages to find the last non-empty AI response.
    # We can't just take messages[-1] because the dedup guard might have
    # produced an empty AI message before the re-invoked direct answer.
    answer = ""
    for msg in reversed(messages):
        if msg.type == "ai" and msg.content and str(msg.content).strip():
            answer = str(msg.content).strip()
            break

    if not answer:
        answer = "I was unable to generate an answer. Please try rephrasing your question."

    # Collect tool names and map them to datasource categories
    tools_used: list[str] = []
    datasources: set[str] = set()

    for msg in messages:
        if msg.type == "tool":
            name = msg.name or ""
            tools_used.append(name)
            ds = _TOOL_DATASOURCE.get(name)
            if ds:
                datasources.add(ds)

    # If multiple datasource types were used, label it "multiple"
    if len(datasources) > 1:
        datasource = "multiple"
    elif datasources:
        datasource = next(iter(datasources))
    else:
        # No tools ran — the LLM answered from its own knowledge
        datasource = "direct_llm"

    citations  = _extract_citations(messages)
    chart_data = _extract_chart(messages)

    return {
        "generation":  answer,
        "datasource":  datasource,
        "tools_used":  tools_used,
        "citations":   citations,
        "chart_data":  chart_data,
    }


def serialize_parse_result(parsed: dict) -> dict:
    """Shape parse_result() output for JSON / SSE clients.

    Moved here from workflow.py so it lives next to parse_result()
    and can be tested independently.
    """
    return {
        "answer":     parsed.get("generation", ""),
        "datasource": parsed.get("datasource"),
        "tools_used": parsed.get("tools_used") or [],
        "citations":  parsed.get("citations") or [],
        "chart_data": parsed.get("chart_data"),
    }
