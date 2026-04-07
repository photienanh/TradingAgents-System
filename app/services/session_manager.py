"""
app/services/session_manager.py

Thread-safe session manager thay thế dict + lock rải rác trong main.py.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple


class SessionManager:
    """
    Thread-safe wrapper cho analysis_sessions dict.
    Gộp lock vào đây để không bị quên acquire ở bất kỳ chỗ nào.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()  # RLock cho phép re-entrant trong cùng thread

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def set(self, session_id: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self._sessions[session_id] = data

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._sessions.get(session_id)

    def update(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Cập nhật một phần session. Trả về False nếu session không tồn tại."""
        with self._lock:
            if session_id not in self._sessions:
                return False
            self._sessions[session_id].update(updates)
            return True

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def contains(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def all_items(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Snapshot an toàn của toàn bộ sessions."""
        with self._lock:
            return list(self._sessions.items())

    def all_sessions_copy(self) -> Dict[str, Dict[str, Any]]:
        """Trả về shallow copy của toàn bộ sessions dict."""
        with self._lock:
            return dict(self._sessions)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ------------------------------------------------------------------
    # Helper: set nested field
    # ------------------------------------------------------------------

    def set_field(self, session_id: str, field: str, value: Any) -> bool:
        """Shortcut để set một field trong session."""
        with self._lock:
            if session_id not in self._sessions:
                return False
            self._sessions[session_id][field] = value
            return True

    def get_field(self, session_id: str, field: str, default: Any = None) -> Any:
        """Shortcut để đọc một field trong session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return default
            return session.get(field, default)