import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict


class SQLiteSessionStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    session_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        sessions: Dict[str, Dict[str, Any]] = {}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, payload FROM analysis_sessions"
            ).fetchall()
        for row in rows:
            try:
                sessions[row["session_id"]] = json.loads(row["payload"])
            except json.JSONDecodeError:
                continue
        return sessions

    def save_all(self, sessions: Dict[str, Dict[str, Any]]) -> None:
        now = datetime.datetime.now().isoformat()
        session_ids = list(sessions.keys())

        with self._connect() as conn:
            for session_id, payload in sessions.items():
                conn.execute(
                    """
                    INSERT INTO analysis_sessions(session_id, payload, updated_at)
                    VALUES(?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        payload=excluded.payload,
                        updated_at=excluded.updated_at
                    """,
                    (session_id, json.dumps(payload, ensure_ascii=False), now),
                )

            if session_ids:
                placeholders = ",".join("?" for _ in session_ids)
                conn.execute(
                    f"DELETE FROM analysis_sessions WHERE session_id NOT IN ({placeholders})",
                    session_ids,
                )
            else:
                conn.execute("DELETE FROM analysis_sessions")
            conn.commit()

    def migrate_from_json_file(self, json_file: Path) -> bool:
        if not json_file.exists():
            return False

        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) AS c FROM analysis_sessions").fetchone()["c"]
        if count > 0:
            return False

        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            return False

        sessions = payload.get("sessions", {})
        if not isinstance(sessions, dict):
            return False

        cleaned: Dict[str, Dict[str, Any]] = {}
        for session_id, session_data in sessions.items():
            if not isinstance(session_data, dict):
                continue
            cleaned[session_id] = session_data

        self.save_all(cleaned)
        return True
