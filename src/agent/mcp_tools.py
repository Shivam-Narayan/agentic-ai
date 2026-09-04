"""MCP-compatible database tools for the KT Agent.

Provides three LangChain tools that give the agent read (and optionally
write) access to the company database. The interface mirrors the MCP
asynccontextmanager pattern so workflow.py needs zero changes if a real
MCP server replaces this module later.

Backend selection (controlled via .env):
  USE_PGVECTOR=false  →  SQLite      (data/company.db)   default
  USE_PGVECTOR=true   →  PostgreSQL  (POSTGRES_URL)
"""

import concurrent.futures
import logging
import os
import re
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List

from langchain_core.tools import BaseTool, tool

from .config import DATA_DIR, POSTGRES_URL, USE_PGVECTOR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum rows returned by any SELECT — prevents LLM context window overflow.
MAX_ROWS: int = 100

# Maximum characters per cell value before truncation.
MAX_CELL_LEN: int = 80

# Allow INSERT / UPDATE / DELETE (default off — read-only is safer).
# Set ALLOW_DB_WRITES=true in .env to enable.
ALLOW_DB_WRITES: bool = os.getenv("ALLOW_DB_WRITES", "false").lower() == "true"

# SQLite database path — only used when USE_PGVECTOR=false.
_SQLITE_PATH: Path = DATA_DIR / "company.db"

# Statements that are always blocked regardless of ALLOW_DB_WRITES.
_ALWAYS_BLOCKED: tuple[str, ...] = ("DROP", "TRUNCATE", "ALTER", "CREATE")

# ---------------------------------------------------------------------------
# Schema Cache (avoids redundant queries during agent tool invocation loops)
# ---------------------------------------------------------------------------
_SCHEMA_CACHE: dict[str, tuple[float, str]] = {}
_SCHEMA_CACHE_TTL: float = float(os.getenv("DB_SCHEMA_CACHE_TTL", "300"))  # 5 minutes default

def get_cached_schema(key: str) -> str | None:
    """Retrieve an item from the schema cache if still valid."""
    if key in _SCHEMA_CACHE:
        cached_time, val = _SCHEMA_CACHE[key]
        if time.time() - cached_time < _SCHEMA_CACHE_TTL:
            return val
        del _SCHEMA_CACHE[key]
    return None

def set_cached_schema(key: str, val: str) -> None:
    """Store an item in the schema cache with the current timestamp."""
    _SCHEMA_CACHE[key] = (time.time(), val)

def clear_schema_cache() -> None:
    """Clear all cached database schema information."""
    _SCHEMA_CACHE.clear()

# ---------------------------------------------------------------------------
# Formatters (module-level — no closures)
# ---------------------------------------------------------------------------

def _trunc(val: object) -> str:
    """Truncate a cell value to MAX_CELL_LEN characters."""
    s = "NULL" if val is None else str(val)
    return s[:MAX_CELL_LEN] + "…" if len(s) > MAX_CELL_LEN else s


def _format_rows(columns: list[str], rows: list[list]) -> str:
    """Render query results as an aligned ASCII table with a row-count footer.

    Args:
        columns: Column header names.
        rows:    List of rows, each a list of cell values.

    Returns:
        Multi-line string table, or a "no results" message if rows is empty.
    """
    if not rows:
        return "Query returned no results."

    str_rows: list[list[str]] = [[_trunc(cell) for cell in row] for row in rows]

    widths: list[int] = [
        max(len(col), max(len(r[i]) for r in str_rows))
        for i, col in enumerate(columns)
    ]

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def _row_line(vals: list[str]) -> str:
        return "|" + "|".join(f" {v:<{widths[i]}} " for i, v in enumerate(vals)) + "|"

    lines = [sep, _row_line(columns), sep]
    lines += [_row_line(r) for r in str_rows]
    lines.append(sep)

    truncated = len(rows) >= MAX_ROWS
    footer = f"  {len(rows)} row(s) returned"
    if truncated:
        footer += f" (limited to {MAX_ROWS} — add a WHERE clause to narrow results)"
    lines.append(footer)

    return "\n".join(lines)


def _sanitise_identifier(name: str) -> str:
    """Validate a table or column name contains only [a-zA-Z0-9_].

    Raises:
        ValueError: if the name contains any other characters, preventing
                    SQL injection via PRAGMA or unparameterised table names.
    """
    if not re.match(r"^\w+$", name):
        raise ValueError(
            f"Invalid identifier {name!r} — only letters, digits, "
            "and underscores are allowed."
        )
    return name


def _is_blocked_statement(sql_upper: str) -> bool:
    """Return True if the first meaningful SQL token is a blocked DDL keyword.

    Only inspects the *first* token so that blocked keywords appearing inside
    string literals (e.g. WHERE bio = 'DROP TABLE ...') do not trigger a false
    positive. DDL is always the leading verb, never a column value.
    """
    # Strip leading comments (-- ... and /* ... */) before checking first token
    stripped = re.sub(r"/\*.*?\*/", "", sql_upper, flags=re.DOTALL)
    stripped = re.sub(r"--[^\n]*", "", stripped).strip()
    first_token = stripped.split()[0] if stripped.split() else ""
    return first_token in _ALWAYS_BLOCKED


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _get_sqlite_conn() -> sqlite3.Connection:
    """Open and return a SQLite connection.

    Raises:
        FileNotFoundError: if data/company.db does not exist.
    """
    if not _SQLITE_PATH.exists():
        raise FileNotFoundError(
            f"Company database not found: {_SQLITE_PATH}\n"
            "Place your SQLite database file at data/company.db"
        )
    conn = sqlite3.connect(str(_SQLITE_PATH))
    # Use index-based access for consistency with the psycopg3 cursor API.
    # Row objects would require row["col"] access — keep both backends uniform.
    return conn


def _get_pg_conn():
    """Open and return a psycopg3 PostgreSQL connection.

    psycopg is imported lazily here because it is an optional dependency —
    only required when USE_PGVECTOR=true. Importing at module level would
    cause an ImportError for users running the default SQLite backend.
    """
    import psycopg  # noqa: PLC0415 — intentional lazy import

    plain_url = POSTGRES_URL.replace("postgresql+psycopg://", "postgresql://")
    return psycopg.connect(plain_url)


def _get_conn():
    """Return a DB connection for the configured backend."""
    if USE_PGVECTOR:
        return _get_pg_conn()
    return _get_sqlite_conn()


# ---------------------------------------------------------------------------
# Async DB execution helpers
# ---------------------------------------------------------------------------

def _run_list_tables() -> str:
    """Blocking implementation of list_database_tables (runs in a thread)."""
    conn = _get_conn()
    try:
        if USE_PGVECTOR:
            cur = conn.execute(
                """
                SELECT table_name
                FROM   information_schema.tables
                WHERE  table_schema = 'public'
                  AND  table_type   = 'BASE TABLE'
                ORDER  BY table_name
                """
            )
            table_names: list[str] = [row[0] for row in cur.fetchall()]
        else:
            cur = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' ORDER BY name"
            )
            table_names = [row[0] for row in cur.fetchall()]

        if not table_names:
            return "No tables found in the database."

        lines: list[str] = ["Available tables:\n"]
        for name in table_names:
            safe = _sanitise_identifier(name)
            try:
                count_cur = conn.execute(f"SELECT COUNT(*) FROM {safe}")  # noqa: S608
                count: int = count_cur.fetchone()[0]
                lines.append(f"  \u2022 {name}  ({count:,} rows)")
            except Exception:
                lines.append(f"  \u2022 {name}")

        return "\n".join(lines)
    finally:
        conn.close()


def _run_describe_table(safe_name: str, table_name: str) -> str:
    """Blocking implementation of describe_database_table (runs in a thread)."""
    conn = _get_conn()
    try:
        if USE_PGVECTOR:
            cur = conn.execute(
                """
                SELECT column_name, data_type,
                       is_nullable, column_default
                FROM   information_schema.columns
                WHERE  table_schema = 'public'
                  AND  table_name   = %s
                ORDER  BY ordinal_position
                """,
                (safe_name,),
            )
            rows = cur.fetchall()
            if not rows:
                return f"Table '{table_name}' not found or has no columns."
            cols_info: list[str] = [
                f"  {r[0]}  {r[1]}"
                + (" NOT NULL"      if r[2] == "NO" else "")
                + (f"  DEFAULT {r[3]}" if r[3]        else "")
                for r in rows
            ]
        else:
            cur = conn.execute(f"PRAGMA table_info({safe_name})")  # noqa: S608
            rows = cur.fetchall()
            if not rows:
                return f"Table '{table_name}' not found or has no columns."
            cols_info = [
                f"  {r[1]}  {r[2]}"
                + (" NOT NULL"          if r[3]           else "")
                + (f"  DEFAULT {r[4]}"  if r[4] is not None else "")
                + (" PRIMARY KEY"        if r[5]           else "")
                for r in rows
            ]
        return f"Table '{table_name}' columns:\n" + "\n".join(cols_info)
    finally:
        conn.close()


def _run_query(stripped: str, is_select: bool) -> str:
    """Blocking implementation of query_company_database (runs in a thread)."""
    conn = _get_conn()
    try:
        if is_select:
            cursor = conn.execute(stripped)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return _format_rows(columns, [list(r) for r in rows])
        else:
            cursor = conn.execute(stripped)
            conn.commit()
            return f"Query executed. Rows affected: {cursor.rowcount}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def list_database_tables() -> str:
    """List all tables in the company database with their row counts.

    Always call this first when the user asks about the database, to
    discover what data is available before writing any queries.
    """
    logger.info("Tool list_database_tables")
    cached = get_cached_schema("all_tables")
    if cached is not None:
        logger.debug("Returning cached database tables list")
        return cached

    try:
        result = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(
            _run_list_tables
        ).result(timeout=30)
        set_cached_schema("all_tables", result)
        return result
    except Exception as exc:
        logger.exception("list_database_tables failed")
        return f"Error listing tables: {exc}"


@tool
def describe_database_table(table_name: str) -> str:
    """Describe the columns, types, and constraints of a database table.

    Always call this before writing a SELECT query so you know the exact
    column names and types.

    Args:
        table_name: Name of the table to describe (e.g. 'orders', 'customers').
    """
    logger.info("Tool describe_database_table: %s", table_name)
    try:
        safe_name = _sanitise_identifier(table_name)
        cached = get_cached_schema(f"table:{safe_name}")
        if cached is not None:
            logger.debug("Returning cached schema for %s", safe_name)
            return cached

        result = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(
            _run_describe_table, safe_name, table_name
        ).result(timeout=30)
        set_cached_schema(f"table:{safe_name}", result)
        return result

    except ValueError as exc:
        return f"Invalid table name: {exc}"
    except Exception as exc:
        logger.exception("describe_database_table failed for %s", table_name)
        return f"Error describing table '{table_name}': {exc}"


@tool
def query_company_database(sql: str) -> str:
    """Run a SQL query against the company database and return results.

    SELECT / WITH queries return up to 100 rows formatted as a table.
    INSERT / UPDATE / DELETE are only permitted when ALLOW_DB_WRITES=true.
    DROP / TRUNCATE / ALTER / CREATE are always blocked.

    Always call list_database_tables and describe_database_table first to
    discover available tables and column names before writing queries.

    Args:
        sql: SQL query string.
             Example: SELECT order_id, status FROM orders LIMIT 10
    """
    logger.info("Tool query_company_database: %s", sql)

    stripped  = sql.strip()
    sql_upper = stripped.upper()

    # Always-blocked statements — DDL is never permitted.
    if _is_blocked_statement(sql_upper):
        blocked = ", ".join(_ALWAYS_BLOCKED)
        return f"{blocked} statements are not allowed."

    is_select = sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")
    is_write  = any(
        sql_upper.startswith(kw)
        for kw in ("INSERT", "UPDATE", "DELETE")
    )

    if is_write and not ALLOW_DB_WRITES:
        return (
            "Only SELECT queries are allowed. "
            "Set ALLOW_DB_WRITES=true in .env to enable write operations."
        )

    # Auto-inject LIMIT to prevent context window overflow.
    if is_select and "LIMIT" not in sql_upper:
        stripped = f"{stripped.rstrip(';')} LIMIT {MAX_ROWS}"
        logger.debug("Auto-added LIMIT %d to query", MAX_ROWS)

    try:
        result = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(
            _run_query, stripped, is_select
        ).result(timeout=30)
        if not is_select:
            clear_schema_cache()
        return result
    except Exception as exc:
        logger.exception("query_company_database failed")
        return f"Query error: {exc}"


# ---------------------------------------------------------------------------
# MCP context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def mcp_server_context() -> AsyncGenerator[List[BaseTool], None]:
    """Yield database tools to the LangGraph workflow.

    Mirrors the MCP asynccontextmanager interface so workflow.py needs
    zero changes when a real MCP server replaces this module.

    Active backend: PostgreSQL when USE_PGVECTOR=true, SQLite otherwise.
    """
    backend = "PostgreSQL" if USE_PGVECTOR else "SQLite"
    logger.info("Initialising database tools (backend: %s)", backend)
    yield [list_database_tables, describe_database_table, query_company_database]
