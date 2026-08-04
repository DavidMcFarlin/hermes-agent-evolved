"""Tool-telemetry ledger: every dispatch records an outcome row, failures
carry drill-in evidence, and recording never breaks a tool call."""
import json

import pytest

from tools import tool_telemetry
from tools.registry import ToolRegistry, tool_error


@pytest.fixture
def telemetry_db(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_telemetry, "get_hermes_home", lambda: tmp_path)
    return tmp_path / "tool_telemetry.db"


def _schema(name):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}


class TestClassifyResult:
    def test_tool_error_envelope_is_failure(self):
        ok, etype, err = tool_telemetry.classify_result(tool_error("boom"))
        assert (ok, etype, err) == (False, "tool_error", "boom")

    def test_plain_string_is_success(self):
        assert tool_telemetry.classify_result("all good")[0] is True

    def test_json_without_error_is_success(self):
        assert tool_telemetry.classify_result('{"success": true}')[0] is True

    def test_multimodal_dict_is_success(self):
        assert tool_telemetry.classify_result({"_multimodal": True, "content": []})[0] is True


class TestLedger:
    def test_record_stats_failures_roundtrip(self, telemetry_db):
        tool_telemetry.record("alpha", True, elapsed_ms=12)
        tool_telemetry.record("alpha", False, error_type="tool_error",
                              error="file not found", elapsed_ms=5)
        tool_telemetry.record("beta", True, elapsed_ms=3)

        stats = {r["tool"]: r for r in tool_telemetry.stats(days=1)}
        assert stats["alpha"]["calls"] == 2 and stats["alpha"]["failures"] == 1
        assert stats["alpha"]["failure_rate"] == 0.5
        assert stats["beta"]["failure_rate"] == 0.0

        fails = tool_telemetry.failures(tool="alpha", days=1)
        assert len(fails) == 1 and fails[0]["error"] == "file not found"

    def test_worst_failure_rate_sorts_first(self, telemetry_db):
        tool_telemetry.record("good", True)
        tool_telemetry.record("bad", False, error_type="tool_error", error="x")
        assert tool_telemetry.stats(days=1)[0]["tool"] == "bad"

    def test_queries_survive_missing_db_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tool_telemetry, "get_hermes_home", lambda: tmp_path / "nope" / "deep")
        assert tool_telemetry.record("x", True) is None  # swallowed, no raise
        assert tool_telemetry.stats() == []
        assert tool_telemetry.failures() == []


class TestDispatchIntegration:
    def test_dispatch_records_success_and_failure(self, telemetry_db):
        reg = ToolRegistry()
        reg.register(name="works", toolset="core", schema=_schema("works"),
                     handler=lambda args, **kw: json.dumps({"ok": True}))
        reg.register(name="breaks", toolset="core", schema=_schema("breaks"),
                     handler=lambda args, **kw: (_ for _ in ()).throw(RuntimeError("kaput")))

        reg.dispatch("works", {})
        reg.dispatch("breaks", {})

        stats = {r["tool"]: r for r in tool_telemetry.stats(days=1)}
        assert stats["works"]["failures"] == 0
        assert stats["breaks"]["failures"] == 1
        assert "kaput" in tool_telemetry.failures(tool="breaks")[0]["error"]

    def test_unknown_tool_is_recorded_as_failure(self, telemetry_db):
        reg = ToolRegistry()
        reg.dispatch("ghost", {})
        assert tool_telemetry.stats(days=1)[0]["tool"] == "ghost"
        assert tool_telemetry.stats(days=1)[0]["failures"] == 1


class TestTargetAttribution:
    def test_target_extracted_and_recorded(self, telemetry_db):
        assert tool_telemetry.target_from_args({"name": "gif-search"}) == "gif-search"
        assert tool_telemetry.target_from_args({"skill": "plan"}) == "plan"
        assert tool_telemetry.target_from_args({"other": 1}) is None
        assert tool_telemetry.target_from_args("not-a-dict") is None

        tool_telemetry.record("skill_manage", False, error_type="tool_error",
                              error="refused", target="gif-search")
        fails = tool_telemetry.failures(tool="skill_manage", days=1)
        assert fails[0]["target"] == "gif-search"

    def test_dispatch_records_target(self, telemetry_db):
        reg = ToolRegistry()
        reg.register(name="skillish", toolset="core", schema=_schema("skillish"),
                     handler=lambda args, **kw: tool_error("nope"))
        reg.dispatch("skillish", {"name": "obsidian"})
        assert tool_telemetry.failures(days=1)[0]["target"] == "obsidian"

    def test_v1_db_migrates_in_place(self, telemetry_db):
        import sqlite3
        conn = sqlite3.connect(telemetry_db)
        conn.execute("CREATE TABLE tool_calls (ts REAL NOT NULL, tool TEXT NOT NULL,"
                     " ok INTEGER NOT NULL, error_type TEXT, error TEXT,"
                     " elapsed_ms INTEGER, session TEXT)")
        conn.commit(); conn.close()
        tool_telemetry.record("x", True, target="t")  # must not raise on old schema
        assert tool_telemetry.stats(days=1)[0]["tool"] == "x"
