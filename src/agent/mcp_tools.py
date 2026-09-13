"""Database tools for the KT Agent.

Provides three LangChain tools that give the agent read (and optionally
write) access to the company database via a simple asynccontextmanager
interface.

Backend selection (controlled via .env):
  USE_PGVECTOR=false  →  SQLite      (data/company.db)   default
  USE_PGVECTOR=true   →  PostgreSQL  (POSTGRES_URL)

Note: This is NOT an MCP (Model Context Protocol) implementation.
"""

import asyncio
import logging
import re
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from langchain_core.tools import BaseTool, tool

from .config import DATA_DIR, POSTGRES_URL, USE_PGVECTOR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

def _get_allow_db_writes() -> bool:
    """Check if database writes are allowed (import-time snapshot from config)."""
    from .config import require_env_var
    try:
        # Check config.py first for consistency
        import os
        return os.getenv("ALLOW_DB_WRITES", "false").lower() == "true"
    except Exception:
        return False

ALLOW_DB_WRITES = _get_allow_db_writes()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum rows returned by any SELECT — prevents LLM context window overflow.
MAX_ROWS: int = 100

# Maximum characters per cell value before truncation.
MAX_CELL_LEN: int = 80

# SQLite database path — only used when USE_PGVECTOR=false.
_SQLITE_PATH: Path = DATA_DIR / "company.db"

# Statements that are always blocked regardless of ALLOW_DB_WRITES.
_ALWAYS_BLOCKED: tuple[str, ...] = (
    "DROP", "TRUNCATE", "ALTER", "CREATE", "ATTACH", "DETACH",
    "VACUUM", "PRAGMA", "GRANT", "REVOKE", "CALL", "DO", "COPY"
)

# Read-only statements (allowed even when ALLOW_DB_WRITES=false).
_READ_ONLY: tuple[str, ...] = ("SELECT", "EXPLAIN", "SHOW", "DESCRIBE", "DESC")

# Write statements (require ALLOW_DB_WRITES=true).
_WRITE_STATEMENTS: tuple[str, ...] = (
    "INSERT", "UPDATE", "DELETE", "REPLACE", "MERGE", "WITH"
)

# ---------------------------------------------------------------------------
# Schema Cache (avoids redundant queries during agent tool invocation loops)
# ---------------------------------------------------------------------------
_SCHEMA_CACHE: dict[str, tuple[float, str]] = {}
_SCHEMA_CACHE_TTL: float = 300.0  # 5 minutes default
_SCHEMA_CACHE_LOCK = threading.Lock()

def get_cached_schema(key: str) -> str | None:
    """Retrieve an item from the schema cache if still valid."""
    with _SCHEMA_CACHE_LOCK:
        if key in _SCHEMA_CACHE:
            cached_time, val = _SCHEMA_CACHE[key]
            if time.time() - cached_time < _SCHEMA_CACHE_TTL:
                return val
            del _SCHEMA_CACHE[key]
    return None

def set_cached_schema(key: str, val: str) -> None:
    """Store an item in the schema cache with the current timestamp."""
    with _SCHEMA_CACHE_LOCK:
        _SCHEMA_CACHE[key] = (time.time(), val)

def clear_schema_cache() -> None:
    """Clear all cached database schema information."""
    with _SCHEMA_CACHE_LOCK:
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


def _is_blocked_statement(sql: str) -> bool:
    """Return True if the statement contains blocked DDL or administrative keywords.

    Strips comments and checks the first meaningful token. Also rejects stacked
    statements (multiple statements separated by semicolons).

    Args:
        sql: SQL statement to check (case-insensitive).

    Returns:
        True if the statement is blocked, False otherwise.
    """
    # Strip leading/trailing whitespace
    sql = sql.strip()
    
    # Check for multiple statements (semicolon not at end)
    # Strip trailing semicolon first, then check for any remaining
    sql_no_trailing_semi = sql.rstrip(";").rstrip()
    if ";" in sql_no_trailing_semi:
        logger.warning("Blocked: multiple statements detected")
        return True
    
    # Strip comments before checking first token
    sql_upper = sql.upper()
    # Remove /* ... */ style comments
    sql_upper = re.sub(r"/\*.*?\*/", "", sql_upper, flags=re.DOTALL)
    # Remove -- ... style comments
    sql_upper = re.sub(r"--[^\n]*", "", sql_upper)
    sql_upper = sql_upper.strip()
    
    if not sql_upper:
        return False
    
    # Extract first meaningful token
    first_token = sql_upper.split()[0] if sql_upper.split() else ""
    
    return first_token in _ALWAYS_BLOCKED


def _classify_statement(sql: str) -> tuple[bool, bool]:
    """Classify a SQL statement as read-only or write operation.

    Returns:
        (is_read_only, is_write): Tuple of two booleans.
            - is_read_only: True for SELECT, EXPLAIN, SHOW, etc.
            - is_write: True for INSERT, UPDATE, DELETE, WITH, etc.

    Note: WITH clauses can contain writes (WITH ... INSERT/UPDATE/DELETE),
    so WITH is classified as a write operation requiring permissions.
    """
    sql_upper = sql.strip().upper()
    
    # Strip comments
    sql_upper = re.sub(r"/\*.*?\*/", "", sql_upper, flags=re.DOTALL)
    sql_upper = re.sub(r"--[^\n]*", "", sql_upper)
    sql_upper = sql_upper.strip()
    
    if not sql_upper:
        return False, False
    
    first_token = sql_upper.split()[0] if sql_upper.split() else ""
    
    is_read_only = first_token in _READ_ONLY
    is_write = first_token in _WRITE_STATEMENTS
    
    return is_read_only, is_write


def _enforce_limit(sql: str) -> str:
    """Add LIMIT clause if missing and statement is SELECT/WITH.

    Performs numeric validation: if LIMIT exists, checks it's <= MAX_ROWS.

    Args:
        sql: SQL statement to enforce limit on.

    Returns:
        SQL with LIMIT clause added or validated.

    Raises:
        ValueError: if existing LIMIT exceeds MAX_ROWS.
    """
    sql_upper = sql.strip().upper()
    sql_stripped = sql.strip().rstrip(";")
    
    # Check if LIMIT already exists
    limit_match = re.search(r"\bLIMIT\s+(\d+)", sql_upper)
    
    if limit_match:
        limit_value = int(limit_match.group(1))
        if limit_value > MAX_ROWS:
            raise ValueError(
                f"LIMIT {limit_value} exceeds maximum allowed {MAX_ROWS}. "
                f"Please use a smaller LIMIT or add WHERE clauses to narrow results."
            )
        return sql  # Already has acceptable LIMIT
    
    # Add LIMIT
    return f"{sql_stripped} LIMIT {MAX_ROWS}"


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

# Shared connection pool for Postgres (lazy-initialized)
_pg_pool = None
_pg_pool_lock = threading.Lock()

def _get_pg_pool():
    """Get or create the PostgreSQL connection pool (lazy initialization)."""
    global _pg_pool
    
    if _pg_pool is None:
        with _pg_pool_lock:
            if _pg_pool is None:  # Double-check after acquiring lock
                import psycopg_pool  # noqa: PLC0415
                
                plain_url = POSTGRES_URL.replace("postgresql+psycopg://", "postgresql://")
                _pg_pool = psycopg_pool.ConnectionPool(
                    plain_url,
                    min_size=1,
                    max_size=10,
                    timeout=10.0,
                    open=True,
                )
                logger.info("PostgreSQL connection pool created")
    
    return _pg_pool


def _get_sqlite_conn() -> sqlite3.Connection:
    """Open and return a SQLite connection with timeout.

    Raises:
        FileNotFoundError: if data/company.db does not exist.
    """
    if not _SQLITE_PATH.exists():
        raise FileNotFoundError(
            f"Company database not found: {_SQLITE_PATH}\n"
            "Place your SQLite database file at data/company.db"
        )
    # Add timeout to prevent "database is locked" under concurrent requests
    return sqlite3.connect(str(_SQLITE_PATH), timeout=10.0)


def _get_conn():
    """Return a DB connection for the configured backend.
    
    For Postgres, returns a connection from the pool.
    For SQLite, returns a new connection.
    """
    if USE_PGVECTOR:
        pool = _get_pg_pool()
        return pool.getconn()
    return _get_sqlite_conn()


def _release_conn(conn) -> None:
    """Release a connection back to the pool (Postgres) or close it (SQLite)."""
    if USE_PGVECTOR:
        pool = _get_pg_pool()
        pool.putconn(conn)
    else:
        conn.close()


# ---------------------------------------------------------------------------
# Async DB execution helpers (proper async using asyncio.to_thread)
# ---------------------------------------------------------------------------

async def _run_list_tables() -> str:
    """List all tables in the database with row counts."""
    def _blocking():
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
                    lines.append(f"  • {name}  ({count:,} rows)")
                except Exception:
                    lines.append(f"  • {name}")

            return "\n".join(lines)
        finally:
            _release_conn(conn)
    
    return await asyncio.to_thread(_blocking)


async def _run_describe_table(safe_name: str, table_name: str) -> str:
    """Describe a table's schema."""
    def _blocking():
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
            _release_conn(conn)
    
    return await asyncio.to_thread(_blocking)


async def _run_query(sql: str, is_read_only: bool, allow_writes: bool) -> str:
    """Execute a SQL query."""
    def _blocking():
        conn = _get_conn()
        try:
            # For Postgres, prevent multiple statements at driver level
            if USE_PGVECTOR:
                # psycopg3 execute() only runs the first statement by default,
                # but we've already validated in query_company_database
                pass
            
            cursor = conn.execute(sql)
            
            if is_read_only or sql.strip().upper().startswith(("SELECT", "EXPLAIN")):
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return _format_rows(columns, [list(r) for r in rows])
            else:
                conn.commit()
                return f"Query executed. Rows affected: {cursor.rowcount}"
        finally:
            _release_conn(conn)
    
    return await asyncio.to_thread(_blocking)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
async def list_database_tables() -> str:
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
        result = await _run_list_tables()
        set_cached_schema("all_tables", result)
        return result
    except Exception as exc:
        logger.exception("list_database_tables failed")
        return f"Error listing tables: {exc}"


@tool
async def describe_database_table(table_name: str) -> str:
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

        result = await _run_describe_table(safe_name, table_name)
        set_cached_schema(f"table:{safe_name}", result)
        return result

    except ValueError as exc:
        return f"Invalid table name: {exc}"
    except Exception as exc:
        logger.exception("describe_database_table failed for %s", table_name)
        return f"Error describing table '{table_name}': {exc}"


@tool
async def query_company_database(sql: str) -> str:
    """Run a SQL query against the company database and return results.

    SELECT / EXPLAIN queries return up to 100 rows formatted as a table.
    INSERT / UPDATE / DELETE / WITH are only permitted when ALLOW_DB_WRITES=true.
    DROP / TRUNCATE / ALTER / CREATE are always blocked.
    Multiple statements (stacked with semicolons) are rejected.

    Always call list_database_tables and describe_database_table first to
    discover available tables and column names before writing queries.

    Args:
        sql: SQL query string.
             Example: SELECT order_id, status FROM orders WHERE created > '2024-01-01' LIMIT 10
    """
    logger.info("Tool query_company_database: %s", sql[:100])

    sql = sql.strip()

    # 1. Check for blocked statements (DDL, admin commands, stacked statements)
    if _is_blocked_statement(sql):
        blocked = ", ".join(_ALWAYS_BLOCKED)
        return (
            f"This query is not allowed. Blocked operations include: {blocked}. "
            f"Multiple statements separated by semicolons are also blocked."
        )

    # 2. Classify the statement
    is_read_only, is_write = _classify_statement(sql)

    # 3. Enforce write permissions
    if is_write and not ALLOW_DB_WRITES:
        return (
            "Only SELECT and EXPLAIN queries are allowed. "
            "Set ALLOW_DB_WRITES=true in .env to enable write operations "
            "(INSERT, UPDATE, DELETE, WITH clauses)."
        )

    # 4. Auto-inject or validate LIMIT for read-only queries
    if is_read_only:
        try:
            sql = _enforce_limit(sql)
        except ValueError as exc:
            return f"Query rejected: {exc}"

    # 5. Execute the query
    try:
        result = await _run_query(sql, is_read_only, ALLOW_DB_WRITES)
        if is_write:
            clear_schema_cache()
        return result
    except Exception as exc:
        logger.exception("query_company_database failed")
        return f"Query error: {exc}"


# ---------------------------------------------------------------------------
# Context manager for database tools
# ---------------------------------------------------------------------------

@asynccontextmanager
async def mcp_server_context() -> AsyncGenerator[list[BaseTool], None]:
    """Yield database tools to the LangGraph workflow.

    Simple asynccontextmanager interface that provides three database tools.
    Backend: PostgreSQL (with connection pooling) when USE_PGVECTOR=true,
             SQLite (with timeout) otherwise.
    
    Note: This is NOT an MCP (Model Context Protocol) server. The name is
    retained for backward compatibility with existing workflow code.
    """
    backend = "PostgreSQL" if USE_PGVECTOR else "SQLite"
    writes = "enabled" if ALLOW_DB_WRITES else "disabled"
    logger.info("Database tools ready (backend: %s, writes: %s)", backend, writes)
    
    try:
        yield [list_database_tables, describe_database_table, query_company_database]
    finally:
        # Cleanup: close connection pool if using Postgres
        global _pg_pool
        if _pg_pool is not None:
            with _pg_pool_lock:
                if _pg_pool is not None:
                    _pg_pool.close()
                    _pg_pool = None
                    logger.info("PostgreSQL connection pool closed")
