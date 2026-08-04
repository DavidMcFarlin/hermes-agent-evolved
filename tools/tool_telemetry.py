"""Per-tool invocation ledger — the system's real failure telemetry.

Every Registry.dispatch() call is recorded here with its outcome, so
"which tools/skills actually fail, how often, and why" is answerable from
data instead of heuristics (the old triage guessed failure rates from
session length and substring matches, which is why everything read ~50%).

Design constraints:
- Recording must NEVER break or slow a tool call: every public function
  swallows exceptions; writes are one INSERT on a WAL sqlite db.
- Rows carry a truncated error snippet so failures are drill-in-able
  without hunting through journals.
- Retention: pruned to ``RETENTION_DAYS`` opportunistically on write.
"""

import json
import logging
import random
import sqlite3
import time
from pathlib import Path

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

RETENTION_DAYS = 90
_ERROR_SNIPPET_CHARS = 300


def _db_path() -> Path:
    return get_hermes_home() / "tool_telemetry.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tool_calls ("
        " ts REAL NOT NULL,"          # unix seconds
        " tool TEXT NOT NULL,"
        " ok INTEGER NOT NULL,"       # 1 success, 0 failure
        " error_type TEXT,"           # exception class or 'tool_error'
        " error TEXT,"                # truncated message for drill-in
        " elapsed_ms INTEGER,"
        " session TEXT)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_ts ON tool_calls(tool, ts)")
    # v2: what the call operated on (skill name etc.) for per-target attribution.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tool_calls)")}
    if "target" not in cols:
        conn.execute("ALTER TABLE tool_calls ADD COLUMN target TEXT")
    return conn


# Argument keys whose value names the artifact a call operates on. Kept small
# and generic — per-target failure attribution (e.g. which SKILL a
# skill_view/skill_manage touched) needs this recorded at dispatch time.
_TARGET_ARG_KEYS = ("name", "skill")


def target_from_args(args) -> str | None:
    if not isinstance(args, dict):
        return None
    for key in _TARGET_ARG_KEYS:
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()[:100]
    return None


def classify_result(result) -> tuple[bool, str | None, str | None]:
    """Map a normalized dispatch result to (ok, error_type, error_snippet).

    Failures surface as tool_error() JSON envelopes ('{"error": ...}').
    Multimodal dict envelopes and non-envelope strings are successes.
    """
    if isinstance(result, str) and result.startswith("{") and '"error"' in result[:200]:
        try:
            payload = json.loads(result)
        except ValueError:
            return True, None, None
        if isinstance(payload, dict) and payload.get("error"):
            return False, "tool_error", str(payload["error"])[:_ERROR_SNIPPET_CHARS]
    return True, None, None


def _current_session() -> str | None:
    try:
        from hermes_logging import _session_context
        return getattr(_session_context, "session_id", None)
    except Exception:  # noqa: BLE001
        return None


def record(tool: str, ok: bool, *, error_type: str | None = None,
           error: str | None = None, elapsed_ms: int | None = None,
           target: str | None = None) -> None:
    """Append one invocation row. Best-effort — never raises."""
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO tool_calls (ts, tool, ok, error_type, error, elapsed_ms, session, target)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), tool, 1 if ok else 0, error_type,
                 (error or "")[:_ERROR_SNIPPET_CHARS] or None,
                 elapsed_ms, _current_session(), target),
            )
            # Opportunistic retention prune (~1% of writes).
            if random.random() < 0.01:
                conn.execute("DELETE FROM tool_calls WHERE ts < ?",
                             (time.time() - RETENTION_DAYS * 86400,))
    except Exception:  # noqa: BLE001
        logger.debug("tool telemetry write failed", exc_info=True)


def stats(days: float = 7) -> list[dict]:
    """Per-tool aggregates over the window, worst failure rate first."""
    try:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT tool, COUNT(*) AS calls, SUM(1 - ok) AS failures,"
                "       CAST(AVG(elapsed_ms) AS INTEGER) AS avg_ms,"
                "       MAX(ts) AS last_used"
                " FROM tool_calls WHERE ts >= ?"
                " GROUP BY tool",
                (time.time() - days * 86400,),
            ).fetchall()
        out = [{**dict(r), "failure_rate": round(r["failures"] / r["calls"], 3)} for r in rows]
        out.sort(key=lambda r: (-r["failure_rate"], -r["calls"]))
        return out
    except Exception:  # noqa: BLE001
        logger.debug("tool telemetry stats failed", exc_info=True)
        return []


def failures(tool: str | None = None, days: float = 7, limit: int = 50) -> list[dict]:
    """Recent failure rows (newest first) for drill-in, optionally per tool."""
    try:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            q = ("SELECT ts, tool, target, error_type, error, elapsed_ms, session"
                 " FROM tool_calls WHERE ok = 0 AND ts >= ?")
            args: list = [time.time() - days * 86400]
            if tool:
                q += " AND tool = ?"
                args.append(tool)
            q += " ORDER BY ts DESC LIMIT ?"
            args.append(max(1, min(int(limit), 500)))
            return [dict(r) for r in conn.execute(q, args).fetchall()]
    except Exception:  # noqa: BLE001
        logger.debug("tool telemetry failures query failed", exc_info=True)
        return []
