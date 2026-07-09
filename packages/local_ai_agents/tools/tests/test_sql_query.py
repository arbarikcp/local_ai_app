import sqlite3

import pytest

from local_ai_agents.tools.sql_query import (
    SqlQueryArgs,
    UnsafeQueryError,
    make_sql_query_tool,
    run_read_only_query,
    validate_read_only_query,
)


def make_fixture_db(tmp_path):
    db_path = tmp_path / "support.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE tickets (id INTEGER PRIMARY KEY, subject TEXT, status TEXT)")
    conn.executemany(
        "INSERT INTO tickets (subject, status) VALUES (?, ?)",
        [("Password reset", "open"), ("Billing question", "closed"), ("API rate limit", "open")],
    )
    conn.commit()
    conn.close()
    return db_path


class TestValidateReadOnlyQuery:
    def test_accepts_a_plain_select(self):
        validate_read_only_query("SELECT * FROM tickets")

    def test_accepts_a_select_with_leading_whitespace(self):
        validate_read_only_query("  SELECT id FROM tickets")

    def test_rejects_insert(self):
        with pytest.raises(UnsafeQueryError):
            validate_read_only_query("INSERT INTO tickets (subject) VALUES ('x')")

    def test_rejects_delete(self):
        with pytest.raises(UnsafeQueryError):
            validate_read_only_query("DELETE FROM tickets")

    def test_rejects_drop(self):
        with pytest.raises(UnsafeQueryError):
            validate_read_only_query("DROP TABLE tickets")

    def test_rejects_multiple_statements(self):
        with pytest.raises(UnsafeQueryError):
            validate_read_only_query("SELECT * FROM tickets; DROP TABLE tickets;")

    def test_rejects_attach(self):
        with pytest.raises(UnsafeQueryError):
            validate_read_only_query("ATTACH DATABASE 'other.db' AS other")


class TestRunReadOnlyQuery:
    def test_returns_matching_rows(self, tmp_path):
        db_path = make_fixture_db(tmp_path)
        rows = run_read_only_query(db_path, "SELECT subject, status FROM tickets WHERE status = 'open'")
        assert len(rows) == 2
        assert all(r["status"] == "open" for r in rows)

    def test_respects_max_rows(self, tmp_path):
        db_path = make_fixture_db(tmp_path)
        rows = run_read_only_query(db_path, "SELECT * FROM tickets", max_rows=1)
        assert len(rows) == 1

    def test_rejects_a_write_query_before_touching_the_database(self, tmp_path):
        db_path = make_fixture_db(tmp_path)
        with pytest.raises(UnsafeQueryError):
            run_read_only_query(db_path, "DELETE FROM tickets")
        # The rejected query never reached the database - data is untouched.
        rows = run_read_only_query(db_path, "SELECT * FROM tickets")
        assert len(rows) == 3

    def test_the_readonly_connection_itself_rejects_writes_as_a_second_layer(self, tmp_path):
        db_path = make_fixture_db(tmp_path)
        # Bypass this tool's own query-text validation entirely and open the
        # exact same mode=ro connection this tool uses, to prove the second
        # defense layer (SQLite's own read-only enforcement) works independently.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM tickets")
        conn.close()


class TestMakeSqlQueryTool:
    async def test_handler_returns_rows(self, tmp_path):
        db_path = make_fixture_db(tmp_path)
        tool = make_sql_query_tool(db_path)
        rows = await tool.handler(SqlQueryArgs(query="SELECT * FROM tickets"))
        assert len(rows) == 3

    def test_tool_is_not_dangerous(self, tmp_path):
        db_path = make_fixture_db(tmp_path)
        tool = make_sql_query_tool(db_path)
        assert tool.dangerous is False
