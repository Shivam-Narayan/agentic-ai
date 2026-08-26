"""Tools the LLM uses for company docs, live web data, calculations, charts and data extraction.

Database queries are handled dynamically by the MCP server (SQLite/Postgres).
The MCP server auto-discovers the database schema and provides query tools to the LLM.
"""

import ast
import base64
import io
import json
import logging
import operator
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool

from .config import DATA_DIR
from .rag import retrieve_documents

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. search_company_documents
# ---------------------------------------------------------------------------

@tool
def search_company_documents(query: str) -> str:
    """Search indexed company / project documents (PDFs, Word docs, Excel files, CSVs).
    Use this for any question about internal company knowledge, projects, reports,
    policies, or data stored in uploaded files.
    Always note the source filename in your answer."""
    logger.info("Tool search_company_documents: %s", query)
    documents: List[Document] = retrieve_documents(query)
    if not documents:
        return "No matching company documents were found."

    snippets = []
    for i, doc in enumerate(documents[:4]):
        text = " ".join(doc.page_content.split())
        if len(text) > 700:
            text = text[:700].rsplit(" ", 1)[0] + "..."
        source = doc.metadata.get("file_name") or doc.metadata.get("source") or "company documents"
        snippets.append(f"[Source: {source}]\n{text}")
    return "\n\n---\n\n".join(snippets)


# ---------------------------------------------------------------------------
# 2. search_web
# ---------------------------------------------------------------------------

@tool
def search_web(query: str) -> str:
    """Search the live web for facts the LLM does not know (weather, news, current events, prices).
    Always cite the URL source in your answer."""
    logger.info("Tool search_web: %s", query)
    from .chains import get_web_search_tool

    docs = get_web_search_tool().invoke({"query": query})
    if docs and isinstance(docs, list):
        if isinstance(docs[0], dict):
            parts = []
            for d in docs:
                url = d.get("url", "")
                content = d.get("content", str(d))
                parts.append(f"[Source: {url}]\n{content}" if url else content)
            return "\n\n".join(parts)
        return "\n".join(str(d) for d in docs)
    return str(docs)


# ---------------------------------------------------------------------------
# 3. summarise_document  (Tool #5 from suggestions)
# ---------------------------------------------------------------------------

@tool
def summarise_document(filename: str) -> str:
    """Summarise the full contents of a specific company document by its filename.
    Use this when the user asks for an overview or summary of a particular file
    (e.g. 'summarise the annual report' or 'give me an overview of handbook.docx').
    The filename should match a file in the data directory (e.g. 'report.pdf')."""
    logger.info("Tool summarise_document: %s", filename)

    # Find the file in data/
    candidate = DATA_DIR / filename
    if not candidate.exists():
        # Try case-insensitive search
        matches = [p for p in DATA_DIR.iterdir() if p.name.lower() == filename.lower()]
        if not matches:
            available = [p.name for p in DATA_DIR.iterdir() if p.is_file()]
            return (
                f"File '{filename}' not found in the data directory. "
                f"Available files: {', '.join(available)}"
            )
        candidate = matches[0]

    try:
        from llama_index.core import SimpleDirectoryReader
        loader = SimpleDirectoryReader(input_files=[str(candidate)])
        docs = loader.load_data()
        if not docs:
            return f"Could not extract text from '{filename}'."

        # Concatenate all text, cap at ~6000 chars to stay within context
        full_text = "\n\n".join(d.text for d in docs)
        if len(full_text) > 6000:
            full_text = full_text[:6000] + "\n\n[... document truncated for summary ...]"

        return f"[Full text of {candidate.name} for summarisation]\n\n{full_text}"
    except Exception as exc:
        logger.exception("summarise_document failed for %s", filename)
        return f"Error reading '{filename}': {exc}"


# ---------------------------------------------------------------------------
# 4. extract_structured_data  (Tool #8 from suggestions)
# ---------------------------------------------------------------------------

@tool
def extract_structured_data(document_query: str, fields: str) -> str:
    """Extract specific fields or facts from company documents.
    Use this when the user wants specific values pulled out of a document,
    such as dates, names, amounts, contract terms, or key figures.

    Args:
        document_query: What to search for in the documents (e.g. 'project deadline budget')
        fields: Comma-separated list of fields to extract (e.g. 'start_date, budget, project_name, owner')

    Returns a JSON object with the extracted field values.
    Example: extract_structured_data('project timeline cost', 'project_name, start_date, end_date, total_cost')
    """
    logger.info("Tool extract_structured_data: query=%s fields=%s", document_query, fields)

    documents: List[Document] = retrieve_documents(document_query)
    if not documents:
        return json.dumps({"error": "No relevant documents found for the given query."})

    context = "\n\n".join(
        doc.page_content[:800] for doc in documents[:4]
    )
    field_list = [f.strip() for f in fields.split(",")]
    field_json = json.dumps({f: "<extracted value or null>" for f in field_list}, indent=2)

    # Return the context and field template — the LLM calling this tool
    # will synthesise the extraction in its next reasoning step
    return (
        f"Document context for extraction:\n\n{context}\n\n"
        f"Fields to extract (fill in from context above):\n{field_json}\n\n"
        f"Instructions: Fill in each field value from the document context. "
        f"Use null if the field is not mentioned."
    )


# ---------------------------------------------------------------------------
# 5. calculate  (Tool #6 from suggestions)
# ---------------------------------------------------------------------------

# Safe allowed operators for the expression evaluator
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a safe arithmetic AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression accurately.
    Use this whenever the user asks for a calculation, arithmetic, percentage,
    or any numeric computation. Do NOT compute numbers in your head — always use this tool.

    Args:
        expression: A math expression string, e.g. '1500 * 12 / 100' or '(250 + 300) * 0.18'

    Returns the numeric result as a string.
    """
    logger.info("Tool calculate: %s", expression)
    # Strip common extras like commas in numbers
    cleaned = expression.replace(",", "").strip()
    try:
        tree = ast.parse(cleaned, mode="eval")
        result = _safe_eval(tree.body)
        # Format: integer if whole number, else 6 significant figures
        if result == int(result):
            return f"{expression} = {int(result)}"
        return f"{expression} = {round(result, 6)}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as exc:
        return f"Error evaluating expression '{expression}': {exc}"


# ---------------------------------------------------------------------------
# 6. generate_chart  (Tool #7 from suggestions)
# ---------------------------------------------------------------------------

@tool
def generate_chart(data_json: str, chart_type: str, title: str) -> str:
    """Generate a chart from tabular data and return it as a Plotly JSON figure.
    Use this when the user asks to visualise data, create a chart, plot a graph,
    or when presenting query results that would be clearer as a visual.

    Args:
        data_json: JSON string of the data. Two formats accepted:
            - List of dicts: '[{"month": "Jan", "sales": 1200}, {"month": "Feb", "sales": 1500}]'
            - Dict of lists:  '{"month": ["Jan", "Feb"], "sales": [1200, 1500]}'
        chart_type: One of 'bar', 'line', 'pie', 'scatter'
        title: Chart title (e.g. 'Monthly Sales 2025')

    Returns a JSON string representing a Plotly figure that the UI will render.
    """
    logger.info("Tool generate_chart: type=%s title=%s", chart_type, title)

    try:
        import plotly.graph_objects as go

        # Parse the data
        raw = json.loads(data_json)
        if isinstance(raw, list) and raw:
            # List of dicts → convert to dict of lists
            keys = list(raw[0].keys())
            data_dict: dict = {k: [row.get(k) for row in raw] for k in keys}
        elif isinstance(raw, dict):
            data_dict = raw
        else:
            return "Error: data_json must be a list of dicts or a dict of lists."

        cols = list(data_dict.keys())
        if len(cols) < 2:
            return "Error: Need at least 2 columns (x and y) to generate a chart."

        x_col = cols[0]
        y_cols = cols[1:]
        x_vals = data_dict[x_col]
        chart_type = chart_type.lower().strip()

        fig = go.Figure()

        if chart_type == "pie":
            y_vals = data_dict[y_cols[0]]
            fig.add_trace(go.Pie(labels=x_vals, values=y_vals, name=y_cols[0]))
        elif chart_type == "scatter":
            for y_col in y_cols:
                fig.add_trace(go.Scatter(
                    x=x_vals, y=data_dict[y_col], mode="markers", name=y_col
                ))
        elif chart_type == "line":
            for y_col in y_cols:
                fig.add_trace(go.Scatter(
                    x=x_vals, y=data_dict[y_col], mode="lines+markers", name=y_col
                ))
        else:  # default: bar
            for y_col in y_cols:
                fig.add_trace(go.Bar(x=x_vals, y=data_dict[y_col], name=y_col))

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        figure_json = fig.to_json()
        # Return a compact marker the parse_result() function can detect
        return f"CHART_JSON::{figure_json}"

    except ImportError:
        return "Error: plotly is not installed. Run: pip install plotly"
    except json.JSONDecodeError as exc:
        return f"Error: Could not parse data_json — {exc}"
    except Exception as exc:
        logger.exception("generate_chart failed")
        return f"Error generating chart: {exc}"
