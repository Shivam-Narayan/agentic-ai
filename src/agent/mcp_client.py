"""MCP-style database tools — implemented as native Python tools.

The @modelcontextprotocol/server-sqlite npm package does not exist on the registry.
We implement equivalent database access using Python's built-in sqlite3 module,
wrapped as LangChain @tool functions and served via the same asynccontextmanager
interface so workflow.py requires zero changes.
"""

import logging
import sqlite3
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from langchain_core.tools import BaseTool, tool

from .config import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = str(DATA_DIR / "company.db")


@tool
def list_database_tables() -> str:
    """List all tables available in the company database."""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if not rows:
            return "No tables found in the database."
        return "Tables: " + ", ".join(r[0] for r in rows)
    finally:
        conn.close()


@tool
def describe_database_table(table_name: str) -> str:
    """Describe the columns and schema of a table in the company database."""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        if not rows:
            return f"Table '{table_name}' not found or has no columns."
        cols = [f"{r[1]} ({r[2]})" for r in rows]
        return f"Table '{table_name}' columns: " + ", ".join(cols)
    finally:
        conn.close()


@tool
def query_company_database(sql: str) -> str:
    """Run a read-only SQL SELECT query against the company database and return the results."""
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT"):
        return "Only SELECT queries are allowed for safety."
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        if not rows:
            return "Query returned no results."
        lines = [", ".join(cols)]
        for row in rows:
            lines.append(", ".join(str(v) for v in row))
        return "\n".join(lines)
    except Exception as e:
        return f"Query error: {e}"
    finally:
        conn.close()


@asynccontextmanager
async def mcp_server_context() -> AsyncGenerator[List[BaseTool], None]:
    """Yields database tools to the LangGraph workflow.
    
    Mirrors the MCP interface (asynccontextmanager yielding a list of tools)
    so workflow.py needs zero changes when a real MCP server replaces this.
    """
    logger.info("Initializing SQLite database tools (Python-native).")
    yield [list_database_tables, describe_database_table, query_company_database]
