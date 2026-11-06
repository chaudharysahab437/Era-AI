"""SQLite persistence for ERA memory, with versioned migrations.

* Data lives in a single local file (WAL mode) and survives restarts.
* The schema is applied from ``migrations/NNN_name.sql`` files; applied
  versions are recorded in ``schema_migrations``. A migration whose first
  line contains ``-- optional`` may fail (e.g. FTS5 missing) without
  stopping the engine.
* Initialisation is lazy -- constructing a store touches nothing on disk
  until the first operation.
* Every failure surfaces as ``MemoryStoreError`` so callers can degrade
  gracefully. Log messages never contain memory content.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .memory_models import Category, Conversation, EmbeddingRecord, Memory, MemoryType, Project, SourceRef

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_UPDATABLE = {
    "type", "category", "content", "content_hash", "slot", "updated_at", "importance", "confidence",
    "status", "superseded_by", "conflict_with", "sensitivity", "share_with_model", "project_id",
    "times_seen", "embedding", "embedding_model", "embedding_provider", "embedding_dim", "metadata", "source_kind", "conversation_id", "source_ref",
}


class MemoryStoreError(Exception):
    """Any storage-level failure (I/O, corruption, locking...)."""


class MemoryStore:
    def __init__(self, db_path: Path | str, migrations_dir: Optional[Path] = None) -> None:
        self.db_path = Path(db_path)
        self.migrations_dir = Path(migrations_dir) if migrations_dir else MIGRATIONS_DIR
        self._lock = threading.RLock()
        self._ready = False
        self.fts_available = False

    # ------------------------------------------------------------------ setup
    def _raw_connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass
        return conn

    def _ensure(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = self._raw_connect()
                try:
                    self._migrate(conn)
                    row = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
                    ).fetchone()
                    self.fts_available = row is not None
                    conn.commit()
                finally:
                    conn.close()
            except (sqlite3.Error, OSError) as exc:
                raise MemoryStoreError(f"memory database unavailable ({type(exc).__name__})") from exc
            self._ready = True

    def _migrate(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT, applied_at TEXT)"
        )
        applied = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
        for path in sorted(self.migrations_dir.glob("*.sql")):
            m = re.match(r"(\d+)_", path.name)
            if not m:
                continue
            version = int(m.group(1))
            if version in applied:
                continue
            sql = path.read_text(encoding="utf-8")
            optional = "-- optional" in sql.splitlines()[0].lower() if sql.strip() else False
            try:
                conn.executescript(sql)
            except sqlite3.Error as exc:
                if optional:
                    logger.warning("Optional memory migration %s skipped (%s).", path.name, type(exc).__name__)
                    continue
                raise
            conn.execute(
                "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?, ?, datetime('now'))",
                (version, path.name),
            )

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        self._ensure()
        with self._lock:
            try:
                conn = self._raw_connect()
            except sqlite3.Error as exc:
                raise MemoryStoreError(f"cannot open memory database ({type(exc).__name__})") from exc
            try:
                yield conn
                conn.commit()
            except sqlite3.Error as exc:
                conn.rollback()
                raise MemoryStoreError(f"memory database error ({type(exc).__name__})") from exc
            finally:
                conn.close()

    def schema_version(self) -> int:
        with self._conn() as c:
            row = c.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        return int(row[0] or 0)

    # ------------------------------------------------------------- converters
    @staticmethod
    def _to_memory(row: sqlite3.Row) -> Memory:
        try:
            meta = json.loads(row["metadata"] or "{}")
        except json.JSONDecodeError:
            meta = {}
        return Memory(
            id=row["id"],
            type=MemoryType(row["type"]),
            content=row["content"],
            category=Category(row["category"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            source=SourceRef(row["source_kind"], row["conversation_id"], row["source_ref"]),
            importance=row["importance"],
            confidence=row["confidence"],
            status=row["status"],
            superseded_by=row["superseded_by"],
            conflict_with=row["conflict_with"],
            slot=row["slot"],
            content_hash=row["content_hash"],
            sensitivity=row["sensitivity"],
            share_with_model=bool(row["share_with_model"]),
            project_id=row["project_id"],
            times_seen=row["times_seen"],
            metadata=meta,
        )

    # --------------------------------------------------------------- memories
    def add_memory(self, m: Memory, emb: Optional[EmbeddingRecord], fts_text: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO memories (id, type, category, content, content_hash, slot, created_at, updated_at,
                   source_kind, conversation_id, source_ref, importance, confidence, status, superseded_by,
                   conflict_with, sensitivity, share_with_model, project_id, times_seen, embedding,
                   embedding_model, embedding_provider, embedding_dim, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    m.id, m.type.value, m.category.value, m.content, m.content_hash, m.slot, m.created_at,
                    m.updated_at, m.source.kind, m.source.conversation_id, m.source.reference, m.importance,
                    m.confidence, m.status, m.superseded_by, m.conflict_with, m.sensitivity,
                    int(m.share_with_model), m.project_id, m.times_seen,
                    emb.blob if emb else None, emb.model if emb else None, emb.provider if emb else None,
                    emb.dim if emb else None, json.dumps(m.metadata),
                ),
            )
            if self.fts_available and m.status == "active":
                c.execute("INSERT INTO memories_fts(memory_id, content) VALUES (?, ?)", (m.id, fts_text))

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._to_memory(row) if row else None

    def get_many(self, ids: Sequence[str]) -> List[Memory]:
        if not ids:
            return []
        out: List[Memory] = []
        with self._conn() as c:
            for i in range(0, len(ids), 500):
                chunk = list(ids[i : i + 500])
                q = ",".join("?" * len(chunk))
                out += [self._to_memory(r) for r in c.execute(f"SELECT * FROM memories WHERE id IN ({q})", chunk)]
        return out

    def update_memory(self, memory_id: str, fields: Dict[str, Any], fts_text: Optional[str] = None) -> bool:
        bad = set(fields) - _UPDATABLE
        if bad:
            raise ValueError(f"cannot update fields: {sorted(bad)}")
        cols, vals = [], []
        for k, v in fields.items():
            if k == "metadata":
                v = json.dumps(v)
            elif k == "share_with_model":
                v = int(v)
            elif k in ("type", "category") and hasattr(v, "value"):
                v = v.value
            cols.append(f"{k} = ?")
            vals.append(v)
        if not cols:
            return False
        with self._conn() as c:
            cur = c.execute(f"UPDATE memories SET {', '.join(cols)} WHERE id = ?", vals + [memory_id])
            if cur.rowcount and self.fts_available and fts_text is not None:
                c.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
                c.execute("INSERT INTO memories_fts(memory_id, content) VALUES (?, ?)", (memory_id, fts_text))
            return bool(cur.rowcount)

    def mark_superseded(self, old_id: str, new_id: str, when: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE memories SET status='superseded', superseded_by=?, conflict_with=NULL, updated_at=? WHERE id=?",
                (new_id, when, old_id),
            )
            if self.fts_available:
                c.execute("DELETE FROM memories_fts WHERE memory_id = ?", (old_id,))

    def list_memories(
        self,
        *,
        types: Optional[Sequence[str]] = None,
        categories: Optional[Sequence[str]] = None,
        project_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        include_history: bool = False,
        text: Optional[str] = None,
        order: str = "updated_desc",
        limit: int = 50,
        offset: int = 0,
    ) -> List[Memory]:
        where, args = [], []
        if not include_history:
            where.append("status = 'active'")
        if types:
            where.append(f"type IN ({','.join('?' * len(types))})")
            args += list(types)
        if categories:
            where.append(f"category IN ({','.join('?' * len(categories))})")
            args += list(categories)
        if project_id:
            where.append("project_id = ?")
            args.append(project_id)
        if conversation_id:
            where.append("conversation_id = ?")
            args.append(conversation_id)
        if text:
            where.append("content LIKE ? ESCAPE '\\'")
            args.append("%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        order_sql = {
            "updated_desc": "updated_at DESC",
            "created_asc": "created_at ASC",
            "importance_desc": "importance DESC, updated_at DESC",
        }.get(order, "updated_at DESC")
        sql = "SELECT * FROM memories"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {order_sql} LIMIT ? OFFSET ?"
        with self._conn() as c:
            rows = c.execute(sql, args + [int(limit), int(offset)]).fetchall()
        return [self._to_memory(r) for r in rows]

    def count_memories(self, status: str = "active") -> Dict[str, int]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT type, COUNT(*) AS n FROM memories WHERE status = ? GROUP BY type", (status,)
            ).fetchall()
        return {r["type"]: r["n"] for r in rows}

    def find_active_by_hash(self, content_hash: str) -> Optional[Memory]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM memories WHERE content_hash = ? AND status = 'active' ORDER BY updated_at DESC LIMIT 1",
                (content_hash,),
            ).fetchone()
        return self._to_memory(row) if row else None

    def find_active_by_slot(self, slot: str) -> List[Memory]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM memories WHERE slot = ? AND status = 'active' ORDER BY updated_at DESC", (slot,)
            ).fetchall()
        return [self._to_memory(r) for r in rows]

    # -------------------------------------------------------- candidate search
    def fts_candidates(self, stems: Sequence[str], limit: int) -> List[str]:
        stems = [s for s in stems if s.isalnum()][:12]
        if not stems:
            return []
        if self.fts_available:
            query = " OR ".join(f'"{s}"' for s in stems)
            with self._conn() as c:
                rows = c.execute(
                    "SELECT memory_id FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                    (query, int(limit)),
                ).fetchall()
            return [r[0] for r in rows]
        clause = " OR ".join("content LIKE ?" for _ in stems)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT id FROM memories WHERE status='active' AND ({clause}) ORDER BY updated_at DESC LIMIT ?",
                [f"%{s}%" for s in stems] + [int(limit)],
            ).fetchall()
        return [r[0] for r in rows]

    _KEY_SQL = "embedding_provider = ? AND embedding_model = ? AND embedding_dim = ?"

    def embedding_rows(self, key: Tuple[str, str, int], limit: int) -> List[Tuple[str, bytes]]:
        """Vectors that were produced by EXACTLY this provider/model/dimension -- nothing else."""
        with self._conn() as c:
            rows = c.execute(
                f"SELECT id, embedding FROM memories WHERE status='active' AND {self._KEY_SQL} "
                "AND embedding IS NOT NULL ORDER BY updated_at DESC LIMIT ?",
                (*key, int(limit)),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def embeddings_for(self, ids: Sequence[str]) -> Dict[str, Tuple[Tuple[str, str, int], bytes]]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        with self._conn() as c:
            rows = c.execute(
                f"SELECT id, embedding_provider, embedding_model, embedding_dim, embedding FROM memories "
                f"WHERE id IN ({q}) AND embedding IS NOT NULL",
                list(ids),
            ).fetchall()
        return {r[0]: ((r[1], r[2], r[3]), r[4]) for r in rows}

    def rows_needing_embedding(self, key: Tuple[str, str, int], limit: int = 500) -> List[Tuple[str, str]]:
        """Active memories with no vector, or a vector from a different provider/model/dimension."""
        with self._conn() as c:
            rows = c.execute(
                f"SELECT id, content FROM memories WHERE status='active' AND "
                f"(embedding IS NULL OR NOT ({self._KEY_SQL})) LIMIT ?",
                (*key, int(limit)),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def embedding_census(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT embedding_provider AS provider, embedding_model AS model, embedding_dim AS dim, COUNT(*) AS n "
                "FROM memories WHERE status='active' GROUP BY 1,2,3"
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- deletion
    def delete_memory(self, memory_id: str) -> List[str]:
        """Hard-delete a memory *and* the superseded history that led to it,
        so a deleted value can't linger as 'history'. Returns deleted ids."""
        with self._conn() as c:
            if not c.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone():
                return []
            ids, frontier = [memory_id], [memory_id]
            while frontier:
                q = ",".join("?" * len(frontier))
                nxt = [r[0] for r in c.execute(f"SELECT id FROM memories WHERE superseded_by IN ({q})", frontier)]
                nxt = [i for i in nxt if i not in ids]
                ids += nxt
                frontier = nxt
            q = ",".join("?" * len(ids))
            c.execute(f"DELETE FROM memories WHERE id IN ({q})", ids)
            c.execute(f"UPDATE memories SET conflict_with = NULL WHERE conflict_with IN ({q})", ids)
            if self.fts_available:
                c.execute(f"DELETE FROM memories_fts WHERE memory_id IN ({q})", ids)
            return ids

    def clear_all(self) -> int:
        with self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            for table in ("memories", "conversations", "projects", "turns"):
                c.execute(f"DELETE FROM {table}")
            if self.fts_available:
                c.execute("DELETE FROM memories_fts")
        return int(n)

    def purge_expired(self, working_before: str) -> Dict[str, int]:
        """Delete short-lived data older than the cutoff (ISO timestamp)."""
        with self._conn() as c:
            ids = [
                r[0]
                for r in c.execute(
                    "SELECT id FROM memories WHERE type = 'working' AND updated_at < ?", (working_before,)
                )
            ]
            if ids:
                q = ",".join("?" * len(ids))
                c.execute(f"DELETE FROM memories WHERE id IN ({q})", ids)
                if self.fts_available:
                    c.execute(f"DELETE FROM memories_fts WHERE memory_id IN ({q})", ids)
            turns = c.execute("DELETE FROM turns WHERE created_at < ?", (working_before,)).rowcount
        return {"working_memories": len(ids), "turns": turns}

    # -------------------------------------------------------------------- meta
    def get_meta(self, key: str) -> Optional[str]:
        with self._conn() as c:
            row = c.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # ----------------------------------------------------------- conversations
    @staticmethod
    def _to_conv(r: sqlite3.Row) -> Conversation:
        return Conversation(
            id=r["id"], created_at=r["created_at"], updated_at=r["updated_at"], title=r["title"],
            project_id=r["project_id"], active_task=r["active_task"], summary=r["summary"],
            summary_updated_at=r["summary_updated_at"], turn_count=r["turn_count"], closed_at=r["closed_at"],
        )

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        return self._to_conv(r) if r else None

    def ensure_conversation(self, conv_id: str, now: str, project_id: Optional[str] = None) -> Conversation:
        with self._conn() as c:
            r = c.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            if r is None:
                c.execute(
                    "INSERT INTO conversations(id, created_at, updated_at, project_id) VALUES (?,?,?,?)",
                    (conv_id, now, now, project_id),
                )
            elif project_id and not r["project_id"]:
                c.execute("UPDATE conversations SET project_id = ? WHERE id = ?", (project_id, conv_id))
            r = c.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        return self._to_conv(r)

    def update_conversation(self, conv_id: str, **fields: Any) -> None:
        allowed = {"title", "updated_at", "project_id", "active_task", "summary", "summary_updated_at", "turn_count", "closed_at"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        if not fields:
            return
        with self._conn() as c:
            c.execute(
                f"UPDATE conversations SET {', '.join(k + ' = ?' for k in fields)} WHERE id = ?",
                list(fields.values()) + [conv_id],
            )

    def recent_conversations(self, exclude_id: Optional[str], limit: int = 3) -> List[Conversation]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM conversations WHERE id != ? AND active_task IS NOT NULL ORDER BY updated_at DESC LIMIT ?",
                (exclude_id or "", int(limit)),
            ).fetchall()
        return [self._to_conv(r) for r in rows]

    # ------------------------------------------------------------------- turns
    def add_turn(self, conv_id: str, role: str, content: str, now: str, keep: int) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO turns(conversation_id, role, content, created_at) VALUES (?,?,?,?)",
                (conv_id, role, content, now),
            )
            c.execute(
                "DELETE FROM turns WHERE conversation_id = ? AND id NOT IN "
                "(SELECT id FROM turns WHERE conversation_id = ? ORDER BY id DESC LIMIT ?)",
                (conv_id, conv_id, int(keep)),
            )

    def get_turns(self, conv_id: str) -> List[Dict[str, str]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT role, content, created_at FROM turns WHERE conversation_id = ? ORDER BY id", (conv_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- projects
    @staticmethod
    def _to_project(r: sqlite3.Row) -> Project:
        try:
            aliases = json.loads(r["aliases"] or "[]")
        except json.JSONDecodeError:
            aliases = []
        return Project(
            id=r["id"], name=r["name"], created_at=r["created_at"], updated_at=r["updated_at"], aliases=aliases,
            summary=r["summary"], summary_updated_at=r["summary_updated_at"], status=r["status"],
        )

    def get_project(self, project_id: str) -> Optional[Project]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return self._to_project(r) if r else None

    def list_projects(self) -> List[Project]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        return [self._to_project(r) for r in rows]

    def add_project(self, p: Project) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO projects(id, name, aliases, created_at, updated_at, status) VALUES (?,?,?,?,?,?)",
                (p.id, p.name, json.dumps(p.aliases), p.created_at, p.updated_at, p.status),
            )

    def update_project(self, project_id: str, **fields: Any) -> None:
        allowed = {"name", "aliases", "updated_at", "summary", "summary_updated_at", "status"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        if "aliases" in fields:
            fields["aliases"] = json.dumps(fields["aliases"])
        if not fields:
            return
        with self._conn() as c:
            c.execute(
                f"UPDATE projects SET {', '.join(k + ' = ?' for k in fields)} WHERE id = ?",
                list(fields.values()) + [project_id],
            )
