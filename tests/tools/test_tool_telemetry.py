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
