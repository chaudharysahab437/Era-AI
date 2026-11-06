"""Application/window exclusion rules.

Wraps the exclusion lists that started as environment-variable defaults
(config.py) in a small object that can be updated at runtime via the API
(`POST /v1/vision/privacy/exclusions`) and persisted to a local JSON file
so exclusions survive a restart. This file only ever contains process
names and title keywords chosen by the user -- never screenshots or
observed content.

`WindowDetector` (vision/windows.py) is constructed once with the initial
list from Settings; PrivacyStore additionally exposes a live view so the
API layer and diagnostics can report/update the current set of rules. For
Phase 2, updating the store also rebuilds the WindowDetector's rules via
`apply_to`.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger("era_vision.privacy")


class PrivacyStore:
    def __init__(
        self,
        initial_processes: tuple[str, ...],
        initial_title_keywords: tuple[str, ...],
        persist_path: str | Path = "data/exclusions.json",
    ) -> None:
        self._lock = threading.Lock()
        self._path = Path(persist_path)
        self._processes: set[str] = {p.lower() for p in initial_processes}
        self._title_keywords: set[str] = {k.lower() for k in initial_title_keywords}
        self._load_persisted()

    def _load_persisted(self) -> None:
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._processes |= {p.lower() for p in data.get("processes", [])}
            self._title_keywords |= {k.lower() for k in data.get("title_keywords", [])}
        except Exception:
            logger.exception("Failed to load persisted exclusions from %s", self._path)

    def _persist_locked(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({"processes": sorted(self._processes), "title_keywords": sorted(self._title_keywords)}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("Failed to persist exclusions to %s", self._path)

    def add_process(self, name: str) -> None:
        with self._lock:
            self._processes.add(name.strip().lower())
            self._persist_locked()

    def remove_process(self, name: str) -> None:
        with self._lock:
            self._processes.discard(name.strip().lower())
            self._persist_locked()

    def add_title_keyword(self, keyword: str) -> None:
        with self._lock:
            self._title_keywords.add(keyword.strip().lower())
            self._persist_locked()

    def remove_title_keyword(self, keyword: str) -> None:
        with self._lock:
            self._title_keywords.discard(keyword.strip().lower())
            self._persist_locked()

    def snapshot(self) -> dict:
        with self._lock:
            return {"processes": sorted(self._processes), "title_keywords": sorted(self._title_keywords)}

    def as_tuples(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        with self._lock:
            return tuple(sorted(self._processes)), tuple(sorted(self._title_keywords))
