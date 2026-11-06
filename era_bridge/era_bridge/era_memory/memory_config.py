"""Configuration for the ERA memory engine.

The package is deliberately independent of the Bridge: it only *reads*
optional attributes from a settings object (see ``from_settings``) and
works with sensible defaults when given nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional


@dataclass
class MemoryConfig:
    db_path: Path = Path("data") / "memory.db"

    # --- feature switches -------------------------------------------------
    enabled: bool = True                 # master switch for *automatic* memory
    auto_memory_default: bool = True     # initial value of the runtime toggle
    store_turns: bool = True             # keep a scrubbed, short-lived turn buffer
    always_include_communication_prefs: bool = True

    # --- retrieval limits / budgets --------------------------------------
    max_context_chars: int = 2000
    max_retrieved: int = 8
    max_comm_prefs: int = 3
    max_vector_scan: int = 5000          # rows whose embedding is compared per query
    fts_candidates: int = 40
    vector_candidates: int = 40
    context_timeout_s: float = 2.0       # never make the caller wait longer than this

    # --- scoring ------------------------------------------------------------
    keyword_gate: float = 0.34           # min keyword overlap to count as relevant
    semantic_gate: float = 0.42          # min cosine similarity to count as relevant
    w_relevance: float = 0.55
    w_recency: float = 0.12
    w_importance: float = 0.20
    w_confidence: float = 0.08
    w_project: float = 0.05
    recency_half_life_days: float = 45.0

    # --- dedupe / correction matching -----------------------------------
    duplicate_similarity: float = 0.93   # cosine above which two same-type memories are one
    correction_match: float = 0.55       # keyword overlap needed to treat a slot-less correction as replacing an older memory

    # --- retention -----------------------------------------------------------
    retention_days: int = 14             # working memory + raw turns
    max_turns_per_conversation: int = 40

    # --- summaries -----------------------------------------------------------
    summary_interval_turns: int = 6
    summary_max_chars: int = 600
    episode_min_turns: int = 4           # closing a conversation shorter than this makes no episodic memory

    # --- access control ---------------------------------------------------
    local_only: bool = True

    @classmethod
    def from_settings(cls, settings: Any, **overrides: Any) -> "MemoryConfig":
        g = lambda name, default: getattr(settings, name, default)  # noqa: E731
        cfg = cls(
            db_path=Path(g("MEMORY_DB_PATH", cls.db_path)),
            enabled=bool(g("MEMORY_ENABLED", True)),
            auto_memory_default=bool(g("MEMORY_AUTO_DEFAULT", True)),
            store_turns=bool(g("MEMORY_STORE_TURNS", True)),
            max_context_chars=int(g("MEMORY_MAX_CONTEXT_CHARS", 2000)),
            max_retrieved=int(g("MEMORY_MAX_RETRIEVED", 8)),
            retention_days=int(g("MEMORY_RETENTION_DAYS", 14)),
            local_only=bool(g("MEMORY_LOCAL_ONLY", True)),
        )
        return replace(cfg, **overrides) if overrides else cfg

    def with_db(self, path: Path) -> "MemoryConfig":
        return replace(self, db_path=Path(path))
