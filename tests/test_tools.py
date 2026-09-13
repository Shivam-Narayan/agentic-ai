"""
Unit tests for local and database tools.
"""
import json
import pytest
from src.agent.mcp_tools import (
    _sanitise_identifier,
    _is_blocked_statement,
    _classify_statement,
    _enforce_limit,
    _format_rows,
    get_cached_schema,
    set_cached_schema,
    clear_schema_cache,
    query_company_database,
    MAX_ROWS,
)
from src.agent.tools import calculate, generate_chart


def test_sanitise_identifier_valid():
    assert _sanitise_identifier("users") == "users"
    assert _sanitise_identifier("order_items_2024") == "order_items_2024"


def test_sanitise_identifier_invalid():
    with pytest.raises(ValueError):
        _sanitise_identifier("users; DROP TABLE users;")
    with pytest.raises(ValueError):
        _sanitise_identifier("users table")
    with pytest.raises(ValueError):
        _sanitise_identifier("users'--")


def test_is_blocked_statement():
    # DDL statements are blocked
    assert _is_blocked_statement("DROP TABLE users") is True
    assert _is_blocked_statement("TRUNCATE TABLE logs") is True
    assert _is_blocked_statement("ALTER TABLE users ADD COLUMN age INT") is True
    assert _is_blocked_statement("CREATE TABLE new_tbl (id INT)") is True
    assert _is_blocked_statement("VACUUM") is True
    assert _is_blocked_statement("ATTACH DATABASE foo AS bar") is True
    
    # Stacked statements are blocked
    assert _is_blocked_statement("SELECT * FROM users; DROP TABLE users") is True
    assert _is_blocked_statement("SELECT 1; SELECT 2") is True
    
    # Comments are stripped before checking
    assert _is_blocked_statement("-- comment\nDROP TABLE users") is True
    assert _is_blocked_statement("/* comment */ DROP TABLE users") is True
    
    # Valid statements are not blocked
    assert _is_blocked_statement("SELECT id, name FROM users") is False
    assert _is_blocked_statement("SELECT * FROM users WHERE name = 'DROP TABLE'") is False


def test_classify_statement():
    # Read-only statements
    is_read, is_write = _classify_statement("SELECT * FROM users")
    assert is_read is True
    assert is_write is False
    
    is_read, is_write = _classify_statement("EXPLAIN SELECT * FROM users")
    assert is_read is True
    assert is_write is False
    
    # Write statements
    is_read, is_write = _classify_statement("INSERT INTO users VALUES (1, 'test')")
    assert is_read is False
    assert is_write is True
    
    is_read, is_write = _classify_statement("UPDATE users SET name = 'new'")
    assert is_read is False
    assert is_write is True
    
    is_read, is_write = _classify_statement("DELETE FROM users WHERE id = 1")
    assert is_read is False
    assert is_write is True
    
    # WITH is classified as write (can contain INSERT/UPDATE/DELETE)
    is_read, is_write = _classify_statement("WITH cte AS (SELECT * FROM users) SELECT * FROM cte")
    assert is_read is False
    assert is_write is True


def test_enforce_limit():
    # Adds LIMIT if missing
    result = _enforce_limit("SELECT * FROM users")
    assert f"LIMIT {MAX_ROWS}" in result
    
    # Accepts existing LIMIT within bounds
    result = _enforce_limit(f"SELECT * FROM users LIMIT {MAX_ROWS - 10}")
    assert result == f"SELECT * FROM users LIMIT {MAX_ROWS - 10}"
    
    # Rejects LIMIT that's too high
    with pytest.raises(ValueError, match="exceeds maximum"):
        _enforce_limit(f"SELECT * FROM users LIMIT {MAX_ROWS + 1}")
    
    # Handles LIMIT with semicolon
    result = _enforce_limit("SELECT * FROM users;")
    assert f"LIMIT {MAX_ROWS}" in result
    assert result.count(";") == 0  # Semicolon should be stripped


def test_schema_caching():
    clear_schema_cache()
    assert get_cached_schema("test_key") is None
    
    set_cached_schema("test_key", "sample_schema_data")
    assert get_cached_schema("test_key") == "sample_schema_data"
    
    clear_schema_cache()
    assert get_cached_schema("test_key") is None


def test_calculate_valid():
    res = calculate.invoke({"expression": "2 + 3 * 4"})
    assert "14" in res
    
    res = calculate.invoke({"expression": "(100 - 25) / 5"})
    assert "15" in res


def test_calculate_rejects_unsafe_code():
    res = calculate.invoke({"expression": "__import__('os').system('ls')"})
    assert "Calculation error" in res or "Unsupported" in res


def test_generate_chart_validation():
    # Valid chart call
    chart_data = json.dumps([
        {"quarter": "Q1", "revenue": 100.0},
        {"quarter": "Q2", "revenue": 150.0},
    ])
    valid_res = generate_chart.invoke({
        "chart_type": "bar",
        "title": "Quarterly Revenue",
        "data_json": chart_data,
    })
    assert "CHART_JSON::" in valid_res
    
    # Invalid chart type
    invalid_type = generate_chart.invoke({
        "chart_type": "3d_donut",
        "title": "Invalid",
        "data_json": chart_data,
    })
    assert "unsupported chart_type" in invalid_type


@pytest.mark.anyio
async def test_query_company_database_blocks_ddl():
    res = await query_company_database.ainvoke({"sql": "DROP TABLE customers"})
    assert "not allowed" in res


@pytest.mark.anyio
async def test_query_company_database_blocks_stacked():
    res = await query_company_database.ainvoke({"sql": "SELECT 1; DROP TABLE customers"})
    assert "not allowed" in res or "Multiple statements" in res


@pytest.mark.anyio
async def test_query_company_database_blocks_excessive_limit():
    res = await query_company_database.ainvoke({"sql": f"SELECT * FROM users LIMIT {MAX_ROWS + 1000}"})
    assert "exceeds maximum" in res or "rejected" in res
