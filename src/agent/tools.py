"""LangChain tools for the KT Agent.

Six local tools covering document search, web search, calculation, charting,
document summarisation, and structured data extraction. Database tools live
in mcp_client.py.

Quality improvements vs original:
  - Full type annotations on all functions and helpers
  - search_web: simplified — _FallbackSearchTool in chains.py already handles
    all response formats; just call .invoke() and stringify
  - summarise_document: configurable MAX_SUMMARY_CHARS constant (was hardcoded 6000)
  - generate_chart: validates chart_type and data before calling Plotly
  - _safe_eval: explicit return type annotation; ValueError message improved
  - Unused imports removed (base64, io)
"""

import ast
import json
import logging
import operator
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool

from .config import DATA_DIR
from .rag import retrieve_documents

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Characters returned per document chunk in search / extract tools.
MAX_CHUNK_CHARS: int = 700

# Maximum characters loaded for full-document summarisation.
# Increase for longer documents — stays within most LLM context windows at 8000.
MAX_SUMMARY_CHARS: int = 8_000

# Allowed chart types for generate_chart input validation.
_VALID_CHART_TYPES: frozenset[str] = frozenset({"bar", "line", "pie", "scatter"})


# ---------------------------------------------------------------------------
# 1. search_company_documents
# ---------------------------------------------------------------------------

@tool
def search_company_documents(query: str) -> str:
    """Search indexed company documents (PDFs, Word docs, Excel files, CSVs).

    Use this for any question about internal company knowledge, projects,
    reports, policies, or data stored in uploaded files.
    Always note the source filename in your answer.

    Args:
        query: The search query describing what to look for.
    """
    logger.info("Tool search_company_documents: %s", query)

    documents: list[Document] = retrieve_documents(query)
    if not documents:
        return "No matching company documents were found."

    snippets: list[str] = []
    for doc in documents[:4]:
        text = " ".join(doc.page_content.split())
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0] + "..."

        # Surface page number when available (LlamaIndex sets page_label)
        meta     = doc.metadata or {}
        source   = meta.get("file_name") or meta.get("source") or "company documents"
        page     = meta.get("page_label") or meta.get("page")
        citation = f"{source}, page {page}" if page else source

        snippets.append(f"[Source: {citation}]\n{text}")

    return "\n\n---\n\n".join(snippets)


# ---------------------------------------------------------------------------
# 2. search_web
# ---------------------------------------------------------------------------

@tool
def search_web(query: str) -> str:
    """Search the live web for facts the LLM does not know.

    Use this for current events, prices, news, weather, or any real-time
    information. Always cite the URL source in your answer.
    Do NOT quote dates from search snippets — only quote factual values
    and cite the source URL.

    Args:
        query: The web search query.
    """
    logger.info("Tool search_web: %s", query)

    from .chains import get_web_search_tool  # lazy — avoids circular import

    # _FallbackSearchTool handles all provider response formats internally.
    # It cascades Tavily → Serper → DuckDuckGo and always returns a string.
    result = get_web_search_tool().invoke(query)
    return str(result)


# ---------------------------------------------------------------------------
# 3. summarise_document
# ---------------------------------------------------------------------------

@tool
def summarise_document(filename: str) -> str:
    """Summarise the full contents of a specific company document by filename.

    Use this when the user asks for an overview or summary of a particular
    file (e.g. "summarise the annual report", "overview of handbook.docx").
    The filename must match a file in the data directory.

    Args:
        filename: The document filename (e.g. 'annual_report.pdf').
    """
    logger.info("Tool summarise_document: %s", filename)

    candidate = DATA_DIR / filename
    if not candidate.exists():
        # Case-insensitive fallback
        matches = [p for p in DATA_DIR.iterdir() if p.name.lower() == filename.lower()]
        if not matches:
            available = sorted(p.name for p in DATA_DIR.iterdir() if p.is_file())
            return (
                f"File '{filename}' not found in the data directory. "
                f"Available files: {', '.join(available)}"
            )
        candidate = matches[0]

    try:
        from llama_index.core import SimpleDirectoryReader

        docs = SimpleDirectoryReader(input_files=[str(candidate)]).load_data()
        if not docs:
            return f"Could not extract text from '{filename}'."

        full_text = "\n\n".join(d.text for d in docs)
        if len(full_text) > MAX_SUMMARY_CHARS:
            full_text = (
                full_text[:MAX_SUMMARY_CHARS]
                + f"\n\n[... document truncated at {MAX_SUMMARY_CHARS:,} characters ...]"
            )

        return f"[Full text of {candidate.name} for summarisation]\n\n{full_text}"

    except Exception as exc:
        logger.exception("summarise_document failed for %s", filename)
        return f"Error reading '{filename}': {exc}"


# ---------------------------------------------------------------------------
# 4. extract_structured_data
# ---------------------------------------------------------------------------

@tool
def extract_structured_data(document_query: str, fields: str) -> str:
    """Extract specific fields or facts from company documents.

    Use this when the user wants specific values pulled from a document —
    dates, names, amounts, contract terms, or key figures.

    Args:
        document_query: Search terms to locate relevant chunks
                        (e.g. 'project deadline budget').
        fields:         Comma-separated field names to extract
                        (e.g. 'start_date, budget, project_name, owner').
    """
    logger.info(
        "Tool extract_structured_data: query=%s fields=%s",
        document_query, fields,
    )

    documents: list[Document] = retrieve_documents(document_query)
    if not documents:
        return json.dumps({"error": "No relevant documents found for the given query."})

    context = "\n\n".join(doc.page_content[:800] for doc in documents[:4])
    field_list = [f.strip() for f in fields.split(",") if f.strip()]
    field_template = json.dumps(
        {f: "<extracted value or null>" for f in field_list},
        indent=2,
    )

    return (
        f"Document context for extraction:\n\n{context}\n\n"
        f"Fields to extract (fill in from context above):\n{field_template}\n\n"
        "Instructions: Fill in each field value from the document context. "
        "Use null if the field is not mentioned."
    )


# ---------------------------------------------------------------------------
# 5. calculate
# ---------------------------------------------------------------------------

# Maps ast operator nodes to their Python equivalents.
# Only arithmetic operators are allowed — no function calls, attributes, etc.
_SAFE_OPERATORS: dict[type, Any] = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.Pow:      operator.pow,
    ast.Mod:      operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub:     operator.neg,
    ast.UAdd:     operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a numeric arithmetic AST node.

    Only numeric constants and the operators in _SAFE_OPERATORS are permitted.
    All other node types raise ValueError — no function calls, imports, or
    attribute access can execute through this evaluator.

    Raises:
        ValueError: if the expression contains unsupported syntax.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](
            _safe_eval(node.left),
            _safe_eval(node.right),
        )

    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))

    raise ValueError(
        f"Unsupported expression node '{type(node).__name__}'. "
        "Only numeric constants and arithmetic operators (+, -, *, /, **, %, //) are allowed."
    )


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression accurately.

    Use this for any calculation, arithmetic, percentage, or numeric
    computation. Never compute numbers in your head — always use this tool.

    Args:
        expression: A math expression (e.g. '1500 * 12 / 100',
                    '(250 + 300) * 0.18', '85000 * 0.15').
    """
    logger.info("Tool calculate: %s", expression)

    cleaned = expression.replace(",", "").strip()
    try:
        tree   = ast.parse(cleaned, mode="eval")
        result = _safe_eval(tree.body)

        # Integer formatting if result is whole; 6 significant figures otherwise
        if result == int(result):
            return f"{expression} = {int(result):,}"
        return f"{expression} = {round(result, 6)}"

    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


# ---------------------------------------------------------------------------
# 6. generate_chart
# ---------------------------------------------------------------------------

@tool
def generate_chart(data_json: str, chart_type: str, title: str) -> str:
    """Generate an interactive chart from tabular data.

    Use this when the user asks to visualise data, create a chart, or when
    query results would be clearer as a visual. The UI renders it inline.

    Args:
        data_json:  JSON data in one of two formats:
                    - List of dicts: '[{"month":"Jan","sales":1200},...]'
                    - Dict of lists: '{"month":["Jan","Feb"],"sales":[1200,1500]}'
        chart_type: Chart style — one of: 'bar', 'line', 'pie', 'scatter'.
        title:      Chart title (e.g. 'Monthly Sales 2025').
    """
    logger.info("Tool generate_chart: type=%s title=%s", chart_type, title)

    # Input validation before touching Plotly
    chart_type = chart_type.lower().strip()
    if chart_type not in _VALID_CHART_TYPES:
        return (
            f"Error: unsupported chart_type '{chart_type}'. "
            f"Choose one of: {', '.join(sorted(_VALID_CHART_TYPES))}"
        )

    if not title or not title.strip():
        return "Error: title must not be empty."

    try:
        import plotly.graph_objects as go

        # Parse and normalise data
        raw: Any = json.loads(data_json)

        if isinstance(raw, list):
            if not raw:
                return "Error: data_json list is empty."
            if not isinstance(raw[0], dict):
                return "Error: data_json list items must be dicts."
            keys = list(raw[0].keys())
            data_dict: dict[str, list] = {k: [row.get(k) for row in raw] for k in keys}

        elif isinstance(raw, dict):
            if not raw:
                return "Error: data_json dict is empty."
            data_dict = raw

        else:
            return "Error: data_json must be a list of dicts or a dict of lists."

        cols = list(data_dict.keys())
        if len(cols) < 2:
            return "Error: need at least 2 columns (x-axis and one y-axis value)."

        x_col  = cols[0]
        y_cols = cols[1:]
        x_vals = data_dict[x_col]

        if not x_vals:
            return "Error: x-axis column is empty."

        fig = go.Figure()

        if chart_type == "pie":
            fig.add_trace(go.Pie(
                labels=x_vals,
                values=data_dict[y_cols[0]],
                name=y_cols[0],
                textfont=dict(color="white"),
            ))
        elif chart_type == "scatter":
            for y_col in y_cols:
                fig.add_trace(go.Scatter(
                    x=x_vals, y=data_dict[y_col],
                    mode="markers", name=y_col,
                ))
        elif chart_type == "line":
            for y_col in y_cols:
                fig.add_trace(go.Scatter(
                    x=x_vals, y=data_dict[y_col],
                    mode="lines+markers", name=y_col,
                ))
        else:  # bar (default)
            for y_col in y_cols:
                fig.add_trace(go.Bar(
                    x=x_vals, y=data_dict[y_col], name=y_col,
                ))

        fig.update_layout(
            title=title.strip(),
            xaxis_title=x_col,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right",  x=1,
            ),
        )

        # CHART_JSON:: prefix is detected by parse_result() in parser.py
        return f"CHART_JSON::{fig.to_json()}"

    except ImportError:
        return "Error: plotly is not installed. Run: pip install plotly"
    except json.JSONDecodeError as exc:
        return f"Error: could not parse data_json — {exc}"
    except Exception as exc:
        logger.exception("generate_chart failed")
        return f"Error generating chart: {exc}"
