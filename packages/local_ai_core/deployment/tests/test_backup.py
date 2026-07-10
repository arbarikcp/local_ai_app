import sqlite3

from local_ai_agents.policies.audit_log import AuditLog

from local_ai_core.deployment.backup import backup_sqlite_db, list_backups, restore_sqlite_db


class TestBackupSqliteDb:
    def test_backs_up_a_real_audit_log_with_real_entries(self, tmp_path):
        db_path = tmp_path / "audit.db"
        audit_log = AuditLog(db_path)
        audit_log.record("trace-1", "lookup_order", {"order_id": "123"}, "success", "")

        backup_dir = tmp_path / "backups"
        backup_path = backup_sqlite_db(db_path, backup_dir)

        assert backup_path.exists()
        assert backup_path.parent == backup_dir

    def test_the_backup_contains_the_real_data(self, tmp_path):
        db_path = tmp_path / "audit.db"
        audit_log = AuditLog(db_path)
        audit_log.record("trace-1", "lookup_order", {"order_id": "123"}, "success", "")

        backup_path = backup_sqlite_db(db_path, tmp_path / "backups")

        conn = sqlite3.connect(str(backup_path))
        rows = conn.execute("SELECT trace_id, tool_name FROM audit_entries").fetchall()
        conn.close()
        assert rows == [("trace-1", "lookup_order")]


class TestRestoreSqliteDb:
    def test_a_full_backup_then_restore_round_trip_preserves_real_data(self, tmp_path):
        original_path = tmp_path / "audit.db"
        audit_log = AuditLog(original_path)
        audit_log.record("trace-1", "lookup_order", {"order_id": "123"}, "success", "")
        audit_log.record("trace-2", "cancel_order", {"order_id": "456"}, "denied", "not permitted")

        backup_path = backup_sqlite_db(original_path, tmp_path / "backups")

        restored_path = tmp_path / "restored.db"
        restore_sqlite_db(backup_path, restored_path)

        restored_log = AuditLog(restored_path)
        entries = restored_log.all_entries()
        assert len(entries) == 2
        assert entries[0].trace_id == "trace-1"
        assert entries[1].outcome == "denied"


class TestListBackups:
    def test_lists_backups_in_sorted_order(self, tmp_path):
        db_path = tmp_path / "audit.db"
        AuditLog(db_path)
        backup_dir = tmp_path / "backups"

        first = backup_sqlite_db(db_path, backup_dir)
        second = backup_sqlite_db(db_path, backup_dir)

        backups = list_backups(backup_dir)
        assert set(backups) == {first, second}
        assert backups == sorted(backups)

    def test_empty_or_missing_backup_dir_returns_empty_list(self, tmp_path):
        assert list_backups(tmp_path / "does_not_exist") == []
