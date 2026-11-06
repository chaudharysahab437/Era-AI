"""Visual context memory.

Holds:
  current_state       - the most recent ContextEntry (or None)
  recent_states        - a bounded deque of past ContextEntry objects
  important_events     - a bounded list of ImportantEvent (errors, app
                          open/close, etc.) derived from state transitions
  task_context          - free-text description of what the user is
                          currently asking ERA to do (set externally;
                          Phase 2 does not set this itself)

Nothing here stores raw screenshots -- only the structured ScreenState and
WindowInfo produced by the analyzer. Every entry has a timestamp and is
dropped once it exceeds `ttl_seconds`, independent of the max history size.
This is intentionally an in-memory structure only; nothing is persisted to
disk (Phase 2 requirement: no permanent screenshot storage, and by default
no persistence of context either).
"""

from __future__ import annotations

import threading
import time
from collections import deque

from vision.models import ContextEntry, ImportantEvent


class ContextManager:
    def __init__(self, history_size: int, ttl_seconds: float) -> None:
        self._history_size = history_size
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._recent: deque[ContextEntry] = deque(maxlen=history_size)
        self._current: ContextEntry | None = None
        self._important_events: deque[ImportantEvent] = deque(maxlen=100)
        self._task_context: str | None = None

    # -- writes -----------------------------------------------------------

    def add_observation(self, entry: ContextEntry) -> list[ImportantEvent]:
        new_events: list[ImportantEvent] = []
        with self._lock:
            previous = self._current
            self._current = entry
            self._recent.append(entry)
            self._purge_expired_locked()

            if previous is not None and previous.window.process_name != entry.window.process_name:
                new_events.append(
                    ImportantEvent(
                        kind="app_changed",
                        description=f"Active app changed: {previous.window.process_name!r} -> {entry.window.process_name!r}",
                    )
                )
            if entry.state.errors:
                for err in entry.state.errors:
                    new_events.append(ImportantEvent(kind="error_detected", description=err))

            for ev in new_events:
                self._important_events.append(ev)

        return new_events

    def set_task_context(self, text: str | None) -> None:
        with self._lock:
            self._task_context = text

    # -- reads --------------------------------------------------------------

    def _purge_expired_locked(self) -> None:
        now = time.time()
        while self._recent and self._recent[0].is_expired(self._ttl_seconds, now):
            self._recent.popleft()
        if self._current is not None and self._current.is_expired(self._ttl_seconds, now):
            self._current = None

    def get_current(self) -> ContextEntry | None:
        with self._lock:
            self._purge_expired_locked()
            return self._current

    def get_recent(self) -> list[ContextEntry]:
        with self._lock:
            self._purge_expired_locked()
            return list(self._recent)

    def get_important_events(self, limit: int = 20) -> list[ImportantEvent]:
        with self._lock:
            return list(self._important_events)[-limit:]

    def get_task_context(self) -> str | None:
        with self._lock:
            return self._task_context

    def entry_count(self) -> int:
        with self._lock:
            self._purge_expired_locked()
            return len(self._recent)

    def clear(self) -> None:
        with self._lock:
            self._recent.clear()
            self._current = None
            self._important_events.clear()

    def snapshot(self) -> dict:
        """JSON-serializable snapshot for GET /v1/vision/context."""
        with self._lock:
            self._purge_expired_locked()
            return {
                "current_state": self._current.model_dump(mode="json") if self._current else None,
                "recent_states": [e.model_dump(mode="json") for e in self._recent],
                "important_events": [e.model_dump(mode="json") for e in self._important_events],
                "task_context": self._task_context,
            }
