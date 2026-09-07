"""
Unit tests for local and database tools.
"""
import json
import pytest
from src.agent.mcp_tools import (
    _sanitise_identifier,
    _is_blocked_statement,
    _format_rows,
    get_cached_schema,
    set_cached_schema,
    clear_schema_cache,
    query_company_database,
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
    assert _is_blocked_statement("DROP TABLE users") is True
    assert _is_blocked_statement("SELECT * FROM users; DROP TABLE users") is True
    assert _is_blocked_statement("TRUNCATE TABLE logs") is True
    assert _is_blocked_statement("ALTER TABLE users ADD COLUMN age INT") is True
    assert _is_blocked_statement("CREATE TABLE new_tbl (id INT)") is True
    assert _is_blocked_statement("SELECT id, name FROM users") is False


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


def test_query_company_database_blocks_ddl():
    res = query_company_database.invoke({"sql": "DROP TABLE customers"})
    assert "statements are not allowed" in res
