"""
alpha/pipelines/pipeline_state.py

Persist PipelineState to app SQLite DB (app/data/sessions.db) so it survives restarts
without relying on a standalone JSON file.
"""
from __future__ import annotations

import datetime
import json
import logging
import sqlite3
import threading
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_STATE_FILENAME = "pipeline_state.json"
_STATE_KEY = "alpha_pipeline"


class PipelineState:
    """Thread-safe, DB-backed state for DailyPipeline."""

    def __init__(self, state_dir: str | None = None, db_path: str | None = None):
        # Keep state_dir for backward compatibility and one-time migration from legacy JSON.
        base_dir = Path(__file__).resolve().parents[2]
        default_db = base_dir / "app" / "data" / "sessions.db"
        self._db_path = Path(db_path) if db_path else default_db
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        legacy_dir = Path(state_dir) if state_dir else (base_dir / "alpha" / "pipelines")
        self._legacy_path = legacy_dir / _STATE_FILENAME

        self._lock = threading.RLock()
        self._init_db()
        self._data: dict = self._load()

    @property
    def last_run_date(self) -> Optional[date]:
        with self._lock:
            raw = self._data.get("last_run_date")
            if not raw:
                return None
            try:
                return date.fromisoformat(raw)
            except ValueError:
                return None

    @last_run_date.setter
    def last_run_date(self, value: date) -> None:
        with self._lock:
            self._data["last_run_date"] = str(value)
            self._save()

    def already_ran_today(self) -> bool:
        lrd = self.last_run_date
        return lrd is not None and lrd == date.today()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_state (
                    state_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _load(self) -> dict:
        data = self._load_from_db()
        if data is not None:
            return data

        # One-time migration path from legacy JSON file if it exists.
        if not self._legacy_path.exists():
            return {}
        try:
            with self._legacy_path.open("r", encoding="utf-8") as f:
                legacy = json.load(f)
                legacy = legacy if isinstance(legacy, dict) else {}
                self._save_to_db(legacy)
                return legacy
        except Exception as exc:
            logger.warning("Không đọc được pipeline state (legacy json): %s", exc)
            return {}

    def _save(self) -> None:
        self._save_to_db(self._data)

    def _load_from_db(self) -> dict | None:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT payload FROM app_state WHERE state_key = ?",
                    (_STATE_KEY,),
                ).fetchone()
            if row is None:
                return {}
            payload = json.loads(row["payload"])
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            logger.error("Không đọc được pipeline state từ DB: %s", exc)
            return None

    def _save_to_db(self, payload: dict) -> None:
        try:
            now = datetime.datetime.now().isoformat()
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO app_state(state_key, payload, updated_at)
                    VALUES(?, ?, ?)
                    ON CONFLICT(state_key) DO UPDATE SET
                        payload=excluded.payload,
                        updated_at=excluded.updated_at
                    """,
                    (_STATE_KEY, json.dumps(payload, ensure_ascii=False), now),
                )
                conn.commit()
        except Exception as exc:
            logger.error("Không lưu được pipeline state vào DB: %s", exc)