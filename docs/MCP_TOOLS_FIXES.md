# Database Tools (mcp_tools.py) - Security and Performance Fixes

## Summary

Fixed critical security vulnerabilities and performance issues in the database tools module. The module is now properly documented as **NOT an MCP implementation** but rather a set of LangChain SQL tools.

## What Was Fixed

### 1. **SQL Injection and Safety Issues**

#### Stacked Statement Detection
- **Before**: Only checked first token, allowing `SELECT 1; DROP TABLE users`
- **After**: Detects and blocks multiple statements separated by semicolons

#### WITH Clause Classification
- **Before**: Treated all WITH as SELECT (read-only)
- **After**: Classifies WITH as write operation (can contain INSERT/UPDATE/DELETE)

#### LIMIT Validation
- **Before**: Substring check failed on `LIMITED_ORDERS` and accepted `LIMIT 1000000`
- **After**: Numeric validation ensures LIMIT ≤ 100, proper regex matching

#### Expanded Blocklist
- **Added**: ATTACH, DETACH, VACUUM, PRAGMA, GRANT, REVOKE, CALL, DO, COPY
- **Before**: Only blocked DROP, TRUNCATE, ALTER, CREATE

#### Comment Stripping
- **Before**: Only stripped comments from blocklist check
- **After**: Strips comments before all classification checks

### 2. **Runtime and Connection Management**

#### PostgreSQL Connection Pooling
- **Before**: New connection per tool call (expensive, no timeout)
- **After**: Shared `ConnectionPool` with:
  - `min_size=1`, `max_size=10`
  - 10-second connection timeout
  - Proper connection reuse via `getconn()`/`putconn()`
  - Cleanup on context manager exit

#### SQLite Timeout
- **Before**: `sqlite3.connect()` with no timeout → "database is locked" under load
- **After**: `timeout=10.0` for concurrent request handling

#### Async Execution
- **Before**: New `ThreadPoolExecutor(max_workers=1)` per tool call (leak-ish churn)
- **After**: Proper async with `asyncio.to_thread()` (ToolNode already provides executor)

### 3. **Thread Safety**

#### Schema Cache Locking
- **Before**: Global dict with no lock (race conditions possible)
- **After**: `threading.Lock()` protects all cache operations

### 4. **Documentation Corrections**

#### Module Docstring
- **Before**: "MCP-compatible" / "mirrors MCP asynccontextmanager pattern"
- **After**: Clearly states "This is NOT an MCP (Model Context Protocol) implementation"

#### Context Manager
- **Before**: Logged "Initialising" on every request
- **After**: Logs once at startup with backend + write permissions status

#### Tool Descriptions
- **Updated**: All tools now async with proper documentation of validation rules

### 5. **Configuration Handling**

#### ALLOW_DB_WRITES
- **Before**: Raw `os.getenv()` at module level (inconsistent with other config)
- **After**: Centralized import-time snapshot with clear function

## Code Structure Changes

### New Helper Functions

```python
_classify_statement(sql) -> tuple[bool, bool]
    Returns (is_read_only, is_write) with proper WITH handling

_enforce_limit(sql) -> str
    Adds LIMIT or validates existing LIMIT numerically

_get_pg_pool() -> ConnectionPool
    Lazy-initializes shared PostgreSQL connection pool

_release_conn(conn) -> None
    Unified connection cleanup (pool return or close)
```

### Updated Functions

```python
_is_blocked_statement(sql) -> bool
    Now detects stacked statements and strips comments properly

list_database_tables() -> str (now async)
describe_database_table(table_name) -> str (now async)
query_company_database(sql) -> str (now async)
    Proper async execution, enhanced validation
```

## Test Coverage Updates

New test cases added to `tests/test_tools.py`:

```python
test_is_blocked_statement()
    - Stacked statements
    - Comment handling
    - Extended blocklist (VACUUM, ATTACH, etc.)

test_classify_statement()
    - Read-only vs write detection
    - WITH clause classification
    - EXPLAIN, SHOW handling

test_enforce_limit()
    - Auto-inject LIMIT
    - Validate existing LIMIT
    - Reject excessive LIMIT

test_query_company_database_blocks_stacked() (async)
test_query_company_database_blocks_excessive_limit() (async)
```

## Migration Notes

### Breaking Changes

1. **All three @tool functions are now async**
   - `list_database_tables()` → `async def list_database_tables()`
   - `describe_database_table()` → `async def describe_database_table()`
   - `query_company_database()` → `async def query_company_database()`

2. **WITH clauses now require ALLOW_DB_WRITES=true**
   - Previously treated as SELECT (allowed)
   - Now classified as write operation for safety

3. **LIMIT validation is now strict**
   - `LIMIT 1000` will be rejected (max is 100)
   - Use WHERE clauses to narrow results instead

### Dependencies

No new dependencies required:
- `psycopg[binary,pool]` already in requirements.txt (line 213)
- Uses `psycopg_pool.ConnectionPool` for Postgres pooling

### Environment Variables

```bash
# Existing (no changes)
USE_PGVECTOR=false          # true for PostgreSQL, false for SQLite
ALLOW_DB_WRITES=false       # true to enable INSERT/UPDATE/DELETE/WITH
POSTGRES_URL=postgresql+psycopg://postgres:password@localhost:5432/datadialogue
```

## Performance Impact

### Before
- New DB connection per tool call
- New thread pool per tool call
- No connection timeout
- N+1 queries for table listing
- Potential "database is locked" errors

### After
- Connection pooling (Postgres) / timeout (SQLite)
- Shared async executor via `asyncio.to_thread()`
- 10-second connection timeout
- N+1 queries remain (but cached for 5 minutes)
- Graceful handling of concurrent requests

## Security Impact

### Vulnerabilities Fixed

1. **SQL Injection via Stacked Statements**: High severity
   - `SELECT 1; DROP TABLE users` now blocked

2. **WITH Clause Write Bypass**: Medium severity
   - `WITH ... DELETE FROM users` now requires write permissions

3. **LIMIT Overflow**: Low severity
   - `LIMIT 999999` no longer accepted

4. **Extended DDL Blocklist**: Medium severity
   - ATTACH, VACUUM, PRAGMA, etc. now blocked

### Remaining Limitations

- **SQLite-specific**: Stacked statement protection is more reliable on Postgres
- **Parameterization**: Table names still use f-strings (mitigated by `_sanitise_identifier()`)
- **Rate limiting**: Not implemented (consider adding per-user limits)

## Testing

Run the updated test suite:

```bash
# All tests
pytest tests/test_tools.py -v

# Just database tool tests
pytest tests/test_tools.py::test_is_blocked_statement -v
pytest tests/test_tools.py::test_classify_statement -v
pytest tests/test_tools.py::test_enforce_limit -v

# Async integration tests (requires database)
pytest tests/test_tools.py::test_query_company_database_blocks_ddl -v
pytest tests/test_tools.py::test_query_company_database_blocks_stacked -v
pytest tests/test_tools.py::test_query_company_database_blocks_excessive_limit -v
```

## Rollback Plan

If issues arise, the previous version can be restored:

1. Revert `src/agent/mcp_tools.py` to the commit before these changes
2. Revert `tests/test_tools.py` to match
3. No database migration or config changes needed

## Future Improvements

Consider for later:

1. **True MCP Implementation**: If needed, use `langchain-mcp-adapters` with subprocess MCP server
2. **Query Plan Analysis**: Check estimated row counts before execution
3. **Rate Limiting**: Per-user query quotas
4. **Audit Logging**: Track all queries for security review
5. **Read Replicas**: Route SELECT to replica, writes to primary
6. **Prepared Statements**: For frequently-used queries
7. **Query Timeout**: Cancel queries running > 30 seconds

## References

- [psycopg3 Connection Pooling](https://www.psycopg.org/psycopg3/docs/api/pool.html)
- [LangChain Tools Documentation](https://python.langchain.com/docs/how_to/custom_tools/)
- [SQL Injection Prevention (OWASP)](https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html)
