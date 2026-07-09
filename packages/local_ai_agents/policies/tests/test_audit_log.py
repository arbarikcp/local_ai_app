from local_ai_agents.policies.audit_log import AuditLog


class TestRecord:
    def test_recorded_entry_is_retrievable(self, tmp_path):
        log = AuditLog(tmp_path / "audit.db")
        log.record("trace-1", "calculator", {"expression": "2+2"}, "success", "returned 4")
        entries = log.entries_for_trace("trace-1")
        assert len(entries) == 1
        assert entries[0].tool_name == "calculator"
        assert entries[0].arguments == {"expression": "2+2"}
        log.close()

    def test_entries_for_trace_only_returns_that_trace(self, tmp_path):
        log = AuditLog(tmp_path / "audit.db")
        log.record("trace-1", "calculator", {}, "success")
        log.record("trace-2", "search_files", {}, "success")
        assert len(log.entries_for_trace("trace-1")) == 1
        log.close()

    def test_all_entries_returns_everything_in_insertion_order(self, tmp_path):
        log = AuditLog(tmp_path / "audit.db")
        log.record("trace-1", "calculator", {}, "success")
        log.record("trace-1", "search_files", {}, "denied")
        entries = log.all_entries()
        assert [e.tool_name for e in entries] == ["calculator", "search_files"]
        log.close()

    def test_denied_and_error_outcomes_are_recorded_distinctly(self, tmp_path):
        log = AuditLog(tmp_path / "audit.db")
        log.record("trace-1", "write_file", {}, "denied", "no approval")
        log.record("trace-1", "sql_query", {}, "error", "unsafe query")
        entries = log.entries_for_trace("trace-1")
        assert entries[0].outcome == "denied"
        assert entries[1].outcome == "error"
        log.close()


class TestPersistenceAcrossRestart:
    def test_entries_survive_a_close_and_reopen(self, tmp_path):
        db_path = tmp_path / "audit.db"
        log1 = AuditLog(db_path)
        log1.record("trace-1", "calculator", {"expression": "2+2"}, "success")
        log1.close()

        log2 = AuditLog(db_path)
        entries = log2.entries_for_trace("trace-1")
        assert len(entries) == 1
        assert entries[0].tool_name == "calculator"
        log2.close()
