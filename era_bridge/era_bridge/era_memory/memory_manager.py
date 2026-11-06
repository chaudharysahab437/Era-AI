"""MemoryManager -- the single entry point for ERA's persistent memory.

Flow for one chat turn (driven by the Gemini Live frontend through Bridge):
  1. ``build_context(message, conversation_id)``  -> compact, budgeted text the
     frontend prepends to the model prompt (retrieval happens BEFORE the reply);
  2. the model answers;
  3. ``ingest_turn(user_message, assistant_message, ...)`` extracts and saves
     anything worth keeping, supersedes corrected facts, and skips duplicates.

Safety rules baked in:
* automatic paths (context / ingest) never raise: on any storage error they
  return an empty result with ``error`` set, so voice and automation commands
  are never blocked by memory problems;
* secrets are never stored; sensitive data is stored only on explicit request;
* log lines carry ids/counts only, never memory content.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from .memory_config import MemoryConfig
from .memory_extractor import MemoryExtractor
from .memory_models import (
    Candidate, Category, ContextPackage, IngestResult, LEVEL_TO_SCORE, Confidence, Memory, MemoryType,
    Project, ScoredMemory, SourceRef, UpsertOutcome,
)
from .memory_retriever import EmbeddingService, MemoryRetriever
from .memory_store import MemoryStore, MemoryStoreError
from .memory_summarizer import MemorySummarizer
from .privacy import classify_sensitive, find_secrets, scrub
from .text_utils import content_hash, overlap, slugify, token_set, tokens, truncate

logger = logging.getLogger(__name__)

_CONTINUATION = re.compile(
    r"\b(continue|resume|carry on|pick up|where (?:were|did) we|last time|left off|unfinished|what was i (?:doing|working on)|go on with|keep going)\b",
    re.I,
)
AUTO_KEY = "auto_memory"


class MemoryError_(Exception):
    """Base class for memory-engine errors surfaced to callers."""


class MemoryUnavailableError(MemoryError_):
    """The memory database can't be used right now."""


class MemoryNotFoundError(MemoryError_):
    pass


class SecretContentError(MemoryError_):
    def __init__(self, kinds: Sequence[str]) -> None:
        super().__init__("secrets/credentials are never stored")
        self.kinds = list(kinds)


class SensitiveContentError(MemoryError_):
    def __init__(self, categories: Sequence[str]) -> None:
        super().__init__("sensitive content needs allow_sensitive=true")
        self.categories = list(categories)


class MemoryManager:
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        *,
        store: Optional[MemoryStore] = None,
        embedder: Optional[EmbeddingService] = None,
        extractor: Optional[MemoryExtractor] = None,
        summarizer: Optional[MemorySummarizer] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.store = store or MemoryStore(self.config.db_path)
        self.embedder = embedder or EmbeddingService()
        self.extractor = extractor or MemoryExtractor()
        self.summarizer = summarizer or MemorySummarizer(max_chars=self.config.summary_max_chars)
        self.retriever = MemoryRetriever(self.store, self.embedder, self.config, self._clock)
        self._purged = False

    # ---------------------------------------------------------------- helpers
    def _now(self) -> str:
        return self._clock().isoformat()

    @staticmethod
    def _new_id() -> str:
        return "mem_" + uuid.uuid4().hex[:16]

    def _guard(self):
        return _StoreGuard()

    def _maybe_purge(self) -> None:
        if self._purged:
            return
        cutoff = (self._clock() - timedelta(days=self.config.retention_days)).isoformat()
        stats = self.store.purge_expired(cutoff)
        self._purged = True
        if stats["working_memories"] or stats["turns"]:
            logger.info("Memory retention purge: %s", stats)

    # --------------------------------------------------------------- settings
    def _auto_status(self) -> str:
        """'ok' | 'disabled_by_config' | 'disabled_by_user'. Raises MemoryStoreError if the DB is unreachable
        -- an unreadable switch is reported as an ERROR, never silently as 'off'."""
        if not self.config.enabled:
            return "disabled_by_config"
        v = self.store.get_meta(AUTO_KEY)
        on = self.config.auto_memory_default if v is None else v == "1"
        return "ok" if on else "disabled_by_user"

    def auto_enabled(self) -> bool:
        """Convenience for callers that only need a boolean (unreachable DB -> False)."""
        try:
            return self._auto_status() == "ok"
        except MemoryStoreError:
            return False

    @staticmethod
    def _describe(exc: Exception) -> tuple:
        """(status, human-readable error) for a failure. Never includes memory content."""
        if isinstance(exc, MemoryStoreError):
            return "unavailable", f"memory_unavailable: {exc}"
        return "internal_error", f"internal_error: {type(exc).__name__}"

    def set_auto(self, enabled: bool) -> bool:
        with self._guard():
            self.store.set_meta(AUTO_KEY, "1" if enabled else "0")
            return self._auto_status() == "ok"

    def settings(self) -> Dict[str, Any]:
        with self._guard():                       # unreachable DB -> MemoryUnavailableError (HTTP 503), not "off"
            status = self._auto_status()
        return {
            "auto_memory": status == "ok",
            "status": status,
            "master_switch": self.config.enabled,
            "max_context_chars": self.config.max_context_chars,
            "max_retrieved": self.config.max_retrieved,
            "retention_days": self.config.retention_days,
            "store_turns": self.config.store_turns,
            "local_only": self.config.local_only,
        }

    # --------------------------------------------------------------- projects
    def get_or_create_project(self, name_or_id: str, create: bool = True) -> Optional[Project]:
        pid = slugify(name_or_id)
        proj = self.store.get_project(pid)
        if proj:
            return proj
        for p in self.store.list_projects():
            if p.name.lower() == name_or_id.lower() or name_or_id.lower() in [a.lower() for a in p.aliases]:
                return p
        if not create:
            return None
        now = self._now()
        proj = Project(id=pid, name=name_or_id.strip(), created_at=now, updated_at=now)
        self.store.add_project(proj)
        return proj

    def _detect_project(self, message: str, conversation_id: Optional[str]) -> Optional[str]:
        for p in self.store.list_projects():
            names = [p.name] + p.aliases
            if any(re.search(rf"(?<![\w]){re.escape(n)}(?![\w])", message, re.I) for n in names if n):
                return p.id
        if conversation_id:
            conv = self.store.get_conversation(conversation_id)
            if conv and conv.project_id:
                return conv.project_id
        return None

    # ------------------------------------------------------------------ write
    def _embed(self, content: str):
        return self.embedder.embed(content)

    def _build_memory(self, c: Candidate, conversation_id: Optional[str], project_id: Optional[str]) -> Memory:
        now = self._now()
        return Memory(
            id=self._new_id(), type=c.type, content=c.content, category=c.category, created_at=now, updated_at=now,
            source=SourceRef(c.source_kind, conversation_id), importance=min(1.0, c.importance), confidence=c.confidence,
            slot=c.slot, content_hash=content_hash(c.content), sensitivity=c.sensitivity,
            share_with_model=c.share_with_model, project_id=project_id, metadata=dict(c.metadata),
        )

    def _insert(self, mem: Memory) -> None:
        self.store.add_memory(mem, self._embed(mem.content), " ".join(tokens(mem.content)))

    def _upsert(self, c: Candidate, conversation_id: Optional[str], project_id: Optional[str]) -> UpsertOutcome:
        """Save a candidate, or dedupe / supersede / flag a conflict."""
        h = content_hash(c.content)
        dup = self.store.find_active_by_hash(h)
        if dup is not None:
            self._bump(dup, c)
            return UpsertOutcome("duplicate", dup)

        # 1) single-valued facts share a slot: newer explicit statement wins
        if c.slot:
            existing = [m for m in self.store.find_active_by_slot(c.slot) if m.project_id == project_id or c.slot.startswith(("user:", "comm:"))]
            if existing:
                strongest = max(existing, key=lambda m: m.confidence)
                if c.is_correction or c.confidence >= strongest.confidence:
                    return self._supersede(existing, c, conversation_id, project_id)
                # weaker, uncorrected claim against a confirmed fact: don't overwrite -- flag it
                return self._conflict(strongest, c, conversation_id, project_id)

        # 2) slot-less near duplicates (same type, almost identical embedding)
        if not c.slot:
            new_toks = token_set(c.content)
            for other, sim in self.retriever.similar_by_embedding(c.content, c.type, limit=2):
                if sim >= self.config.duplicate_similarity and not c.is_correction and self._same_statement(new_toks, token_set(other.content)):
                    self._bump(other, c)
                    return UpsertOutcome("duplicate", other)

        # 3) slot-less correction: replace the closest older memory of the same kind
        if c.is_correction and not c.slot:
            q = token_set(c.content)
            best, best_ov = None, 0.0
            for cand in self.store.list_memories(types=[c.type.value], limit=200):
                if cand.category != c.category and c.category not in (Category.FACT,):
                    continue
                ov = overlap(q, token_set(cand.content))
                if ov > best_ov:
                    best, best_ov = cand, ov
            if best is not None and best_ov >= self.config.correction_match:
                return self._supersede([best], c, conversation_id, project_id)

        mem = self._build_memory(c, conversation_id, project_id)
        self._insert(mem)
        return UpsertOutcome("created", mem)

    @staticmethod
    def _same_statement(a: set, b: set) -> bool:
        """Guard against merging DISTINCT facts that merely look alike to an embedding
        (e.g. 'rack 3 ...' vs 'rack 4 ...'): the word sets must almost coincide and every
        number must be identical."""
        if not a or not b:
            return False
        jaccard = len(a & b) / len(a | b)
        nums = lambda t: {x for x in t if any(ch.isdigit() for ch in x)}  # noqa: E731
        return jaccard >= 0.9 and nums(a) == nums(b)

    def _bump(self, existing: Memory, c: Candidate) -> None:
        fields: Dict[str, Any] = {"times_seen": existing.times_seen + 1, "updated_at": self._now()}
        if c.confidence > existing.confidence:
            fields["confidence"] = c.confidence
        fields["importance"] = max(existing.importance, min(1.0, c.importance))
        self.store.update_memory(existing.id, fields)

    def _supersede(self, olds: List[Memory], c: Candidate, conversation_id, project_id) -> UpsertOutcome:
        mem = self._build_memory(c, conversation_id, project_id)
        if c.is_correction:
            mem.confidence = max(mem.confidence, 0.9)
            mem.metadata["correction"] = True
        self._insert(mem)
        now = self._now()
        for old in olds:
            self.store.mark_superseded(old.id, mem.id, now)
        return UpsertOutcome("superseded", mem, [o.id for o in olds])

    def _conflict(self, existing: Memory, c: Candidate, conversation_id, project_id) -> UpsertOutcome:
        mem = self._build_memory(c, conversation_id, project_id)
        mem.conflict_with = existing.id
        self._insert(mem)
        self.store.update_memory(existing.id, {"conflict_with": mem.id})
        return UpsertOutcome("conflict", mem)

    def save(
        self,
        content: str,
        *,
        type: MemoryType = MemoryType.SEMANTIC,
        category: Category = Category.NOTE,
        importance: float = 0.6,
        confidence: float = 0.95,
        project: Optional[str] = None,
        conversation_id: Optional[str] = None,
        allow_sensitive: bool = False,
        share_with_model: Optional[bool] = None,
        slot: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UpsertOutcome:
        """Explicit save (user/API initiated). Works even when automatic memory is off."""
        content = (content or "").strip()
        if not content:
            raise ValueError("content must not be empty")
        secrets = find_secrets(content)
        if secrets:
            raise SecretContentError(secrets)
        cats = classify_sensitive(content)
        if cats and not allow_sensitive:
            raise SensitiveContentError(cats)
        meta = dict(metadata or {})
        sens = "normal"
        share = True if share_with_model is None else share_with_model
        if cats:
            sens, meta["sensitive_categories"] = "sensitive", cats
            share = bool(share_with_model)  # sensitive memories are private to the model unless opted in
        cand = Candidate(
            content=truncate(content, 1000), type=type, category=category, slot=slot, confidence=confidence,
            importance=importance, sensitivity=sens, share_with_model=share, source_kind="manual", metadata=meta,
        )
        with self._guard():
            pid = self.get_or_create_project(project).id if project else None
            out = self._upsert(cand, conversation_id, pid)
            if pid:
                self._refresh_project_summary(pid)
        logger.info("Memory saved via API: id=%s action=%s type=%s", out.memory.id, out.action, out.memory.type.value)
        return out

    # ------------------------------------------------------------------- read
    def get(self, memory_id: str) -> Memory:
        with self._guard():
            m = self.store.get_memory(memory_id)
        if m is None:
            raise MemoryNotFoundError(memory_id)
        return m

    def list(self, **kw: Any) -> List[Memory]:
        with self._guard():
            return self.store.list_memories(**kw)

    def search(self, query: str, *, limit: int = 10, project: Optional[str] = None, types: Optional[Sequence[MemoryType]] = None) -> List[ScoredMemory]:
        with self._guard():
            pid = None
            if project:
                p = self.get_or_create_project(project, create=False)
                pid = p.id if p else None
            return self.retriever.retrieve(query, project_id=pid, types=types, limit=limit, include_other_working=True)

    # ----------------------------------------------------------------- update
    def update(self, memory_id: str, **changes: Any) -> Memory:
        """User edit of a saved memory. Editing counts as verification."""
        with self._guard():
            m = self.store.get_memory(memory_id)
            if m is None:
                raise MemoryNotFoundError(memory_id)
            allow_sensitive = bool(changes.pop("allow_sensitive", False))
            fields: Dict[str, Any] = {"updated_at": self._now()}
            fts_text = None
            if changes.get("content") is not None:
                content = str(changes["content"]).strip()
                if not content:
                    raise ValueError("content must not be empty")
                secrets = find_secrets(content)
                if secrets:
                    raise SecretContentError(secrets)
                cats = classify_sensitive(content)
                if cats and not (allow_sensitive or m.is_sensitive):
                    raise SensitiveContentError(cats)
                rec = self._embed(content)
                fields.update(content=truncate(content, 1000), content_hash=content_hash(content),
                              embedding=rec.blob if rec else None, embedding_model=rec.model if rec else None,
                              embedding_provider=rec.provider if rec else None, embedding_dim=rec.dim if rec else None)
                fts_text = " ".join(tokens(content))
                if cats:
                    fields["sensitivity"] = "sensitive"
                    fields["metadata"] = {**m.metadata, "sensitive_categories": cats}
                    fields.setdefault("share_with_model", m.share_with_model if m.is_sensitive else False)
                if "confidence" not in changes and "confidence_level" not in changes:
                    fields["confidence"] = max(m.confidence, 0.95)
            if changes.get("type") is not None:
                fields["type"] = MemoryType(changes["type"])
            if changes.get("category") is not None:
                fields["category"] = Category(changes["category"])
            if changes.get("importance") is not None:
                fields["importance"] = max(0.0, min(1.0, float(changes["importance"])))
            if changes.get("confidence") is not None:
                fields["confidence"] = max(0.0, min(1.0, float(changes["confidence"])))
            if changes.get("confidence_level") is not None:
                fields["confidence"] = LEVEL_TO_SCORE[Confidence(changes["confidence_level"])]
            if changes.get("share_with_model") is not None:
                fields["share_with_model"] = bool(changes["share_with_model"])
            self.store.update_memory(memory_id, fields, fts_text=fts_text)
            if m.project_id:
                self._refresh_project_summary(m.project_id)
            return self.store.get_memory(memory_id)  # type: ignore[return-value]

    def resolve_conflict(self, keep_id: str) -> Memory:
        """User says which side of a flagged conflict is right; the other is superseded."""
        with self._guard():
            keep = self.store.get_memory(keep_id)
            if keep is None:
                raise MemoryNotFoundError(keep_id)
            now = self._now()
            if keep.conflict_with:
                other = self.store.get_memory(keep.conflict_with)
                if other is not None and other.status == "active":
                    self.store.mark_superseded(other.id, keep.id, now)
            self.store.update_memory(keep.id, {"conflict_with": None, "confidence": max(keep.confidence, 0.95), "updated_at": now})
            return self.store.get_memory(keep.id)  # type: ignore[return-value]

    # ----------------------------------------------------------------- delete
    def delete(self, memory_id: str) -> bool:
        with self._guard():
            m = self.store.get_memory(memory_id)
            ids = self.store.delete_memory(memory_id)
            if m and m.project_id:
                self._refresh_project_summary(m.project_id)
        logger.info("Memory deleted: id=%s (removed %d row(s) incl. history)", memory_id, len(ids))
        return bool(ids)

    def clear_all(self) -> int:
        with self._guard():
            n = self.store.clear_all()
        logger.info("All memories cleared (%d rows).", n)
        return n

    # ---------------------------------------------------------------- summaries
    def _refresh_project_summary(self, project_id: str) -> Optional[str]:
        proj = self.store.get_project(project_id)
        if proj is None:
            return None
        mems = self.store.list_memories(types=[MemoryType.PROJECT.value], project_id=project_id, limit=200)
        text = self.summarizer.summarize_project(proj.name, mems)
        self.store.update_project(project_id, summary=text, summary_updated_at=self._now(), updated_at=self._now())
        return text

    def project_summary(self, project: str) -> Dict[str, Any]:
        with self._guard():
            p = self.get_or_create_project(project, create=False)
            if p is None:
                raise MemoryNotFoundError(project)
            text = self._refresh_project_summary(p.id) or ""
            mems = self.store.list_memories(types=[MemoryType.PROJECT.value], project_id=p.id, limit=200)
            return {
                "project_id": p.id, "name": p.name, "summary": text, "memory_count": len(mems),
                "conversations": sorted({m.conversation_id for m in mems if m.conversation_id}),
                "first_activity": min((m.created_at for m in mems), default=None),
                "last_activity": max((m.updated_at for m in mems), default=None),
            }

    def list_projects(self) -> List[Project]:
        with self._guard():
            return self.store.list_projects()

    def conversation(self, conversation_id: str):
        with self._guard():
            c = self.store.get_conversation(conversation_id)
        if c is None:
            raise MemoryNotFoundError(conversation_id)
        return c

    def close_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Finalise a conversation: refresh its summary and, if it was substantial, save an episodic memory."""
        with self._guard():
            conv = self.store.get_conversation(conversation_id)
            if conv is None:
                raise MemoryNotFoundError(conversation_id)
            turns = self.store.get_turns(conversation_id)
            summary = self.summarizer.summarize_conversation(turns, conv.summary) if turns else (conv.summary or "")
            now = self._now()
            self.store.update_conversation(conversation_id, summary=summary, summary_updated_at=now, closed_at=now)
            episode_id = None
            if summary and conv.turn_count >= self.config.episode_min_turns and self.auto_enabled():
                cand = Candidate(
                    content=f"Conversation on {now[:10]}: {summary}", type=MemoryType.EPISODIC, category=Category.SUMMARY,
                    slot=f"episode:{conversation_id}", confidence=0.7, importance=0.5, is_correction=True, source_kind="summary",
                )
                episode_id = self._upsert(cand, conversation_id, conv.project_id).memory.id
            return {"conversation_id": conversation_id, "summary": summary, "episode_memory_id": episode_id}

    # ------------------------------------------------------------------ ingest
    def ingest_turn(
        self,
        user_message: str,
        assistant_message: Optional[str] = None,
        *,
        conversation_id: str,
        project: Optional[str] = None,
    ) -> IngestResult:
        """Extract + persist from one turn. Never raises."""
        res = IngestResult(conversation_id=conversation_id)
        try:
            status = self._auto_status()
            if status != "ok":
                res.enabled, res.status = False, status
                return res
            self._maybe_purge()
            now = self._now()
            pid = self.get_or_create_project(project).id if project else self._detect_project(user_message, conversation_id)
            ex = self.extractor.extract(user_message, assistant_message, project_known=bool(pid), today=self._clock().date())
            if ex.project_name and not pid:
                pid = self.get_or_create_project(ex.project_name).id
            conv = self.store.ensure_conversation(conversation_id, now, pid)
            pid = pid or conv.project_id
            res.project_id, res.held_back, res.secrets_blocked = pid, ex.held_back, ex.secrets_blocked

            if self.config.store_turns:  # short-lived buffer, scrubbed of secrets/sensitive sentences
                keep = self.config.max_turns_per_conversation
                self.store.add_turn(conversation_id, "user", scrub(user_message), now, keep)
                if assistant_message:
                    self.store.add_turn(conversation_id, "assistant", scrub(assistant_message), now, keep)

            for cand in ex.candidates:
                out = self._upsert(cand, conversation_id, pid if cand.type == MemoryType.PROJECT else pid)
                if out.action == "created":
                    res.saved.append(out.memory.id)
                elif out.action == "duplicate":
                    res.updated.append(out.memory.id)
                elif out.action == "superseded":
                    res.saved.append(out.memory.id)
                    res.superseded += [{"old": o, "new": out.memory.id} for o in out.superseded_ids]
                elif out.action == "conflict":
                    res.saved.append(out.memory.id)
                    res.conflicts.append(out.memory.id)

            if ex.active_task:
                res.active_task = ex.active_task
                self.store.update_conversation(conversation_id, active_task=ex.active_task)
                task = Candidate(
                    content=f"Active task: {ex.active_task}", type=MemoryType.WORKING, category=Category.TASK,
                    slot=f"task:{conversation_id}", confidence=0.9, importance=0.6, is_correction=True,
                )
                self._upsert(task, conversation_id, pid)

            turn_count = conv.turn_count + 1
            self.store.update_conversation(conversation_id, turn_count=turn_count, updated_at=now, project_id=pid)
            if turn_count % self.config.summary_interval_turns == 0:
                summ = self.summarizer.summarize_conversation(self.store.get_turns(conversation_id), conv.summary)
                self.store.update_conversation(conversation_id, summary=summ, summary_updated_at=now)
            if pid and any(c.type == MemoryType.PROJECT for c in ex.candidates):
                self._refresh_project_summary(pid)
            if res.saved or res.superseded:
                logger.info("Memory ingest: conv=%s saved=%d updated=%d superseded=%d held_back=%s secrets_blocked=%d",
                            conversation_id, len(res.saved), len(res.updated), len(res.superseded), res.held_back, res.secrets_blocked)
        except Exception as exc:  # noqa: BLE001 - memory must never break the assistant
            res.status, res.error = self._describe(exc)
            res.available, res.enabled = False, None
            logger.warning("Memory ingest failed safely (%s).", res.error)
        return res

    # ------------------------------------------------------------------ context
    def build_context(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        project: Optional[str] = None,
    ) -> ContextPackage:
        """Retrieve relevant memories and build the budgeted context text. Never raises."""
        pkg = ContextPackage(conversation_id=conversation_id)
        try:
            status = self._auto_status()
            if status != "ok":
                pkg.enabled, pkg.status = False, status
                return pkg
            self._maybe_purge()
            cfg = self.config
            pid = None
            if project:
                p = self.get_or_create_project(project, create=False)
                pid = p.id if p else None
            pid = pid or self._detect_project(message, conversation_id)
            pkg.project_id = pid
            continuation = bool(_CONTINUATION.search(message))
            pkg.continuation = continuation

            hits = self.retriever.retrieve(
                message, project_id=pid, conversation_id=conversation_id, limit=cfg.max_retrieved,
                for_model=True, include_other_working=continuation,
            )
            chosen: Dict[str, Memory] = {h.memory.id: h.memory for h in hits}
            scores = {h.memory.id: h.score for h in hits}

            comm: List[Memory] = []
            if cfg.always_include_communication_prefs:
                comm = [m for m in self.store.list_memories(categories=[Category.COMMUNICATION.value], order="importance_desc", limit=cfg.max_comm_prefs)
                        if not (m.is_sensitive and not m.share_with_model)]
                for m in comm:
                    chosen.pop(m.id, None)

            open_items: List[Memory] = []
            if pid or continuation:
                if pid:
                    open_items = self.store.list_memories(categories=[Category.BLOCKER.value, Category.NEXT_STEP.value], project_id=pid, limit=6)
                if continuation:
                    for conv in self.store.recent_conversations(conversation_id, 2):
                        wm = self.store.list_memories(types=[MemoryType.WORKING.value], conversation_id=conv.id, limit=1)
                        open_items += wm
                    if not pid:
                        open_items += self.store.list_memories(categories=[Category.NEXT_STEP.value, Category.BLOCKER.value], limit=4)
                seen = set()
                open_items = [m for m in open_items if not (m.id in seen or seen.add(m.id))]
                for m in open_items:
                    chosen.pop(m.id, None)

            # sensitive memories that aren't shared with the model are excluded (just counted)
            pkg.withheld_sensitive = self._count_withheld(message)

            conflicts = self._conflicts(list(chosen.values()) + comm + open_items)
            pkg.conflicts = conflicts

            lines, used, truncated = self._format(comm, list(chosen.values()), scores, open_items, pid, conflicts, pkg)
            pkg.text, pkg.truncated = lines, truncated
            pkg.char_count = len(lines)
        except Exception as exc:  # noqa: BLE001
            status, error = self._describe(exc)
            logger.warning("Memory context failed safely (%s).", error)
            pkg = ContextPackage(enabled=None, status=status, available=False, conversation_id=conversation_id, error=error)
        return pkg

    def _count_withheld(self, message: str) -> int:
        try:
            all_hits = self.retriever.retrieve(message, limit=self.config.max_retrieved, for_model=False, include_other_working=True)
            return sum(1 for h in all_hits if h.memory.is_sensitive and not h.memory.share_with_model)
        except Exception:  # noqa: BLE001
            return 0

    def _conflicts(self, mems: List[Memory]) -> List[Dict[str, Any]]:
        out, seen = [], set()
        for m in mems:
            if m.conflict_with and m.id not in seen:
                other = self.store.get_memory(m.conflict_with)
                if other and other.status == "active" and not (m.is_sensitive or other.is_sensitive):
                    seen.update({m.id, other.id})
                    out.append({"a": m.id, "b": other.id, "a_content": m.content, "b_content": other.content})
        return out

    def _label(self, m: Memory) -> str:
        tag = "confirmed" if m.confidence >= 0.8 else ("unverified" if m.confidence >= 0.5 else "uncertain")
        return f"({tag}, {m.updated_at[:10]})"

    def _format(self, comm, retrieved, scores, open_items, pid, conflicts, pkg: ContextPackage):
        budget = self.config.max_context_chars
        header = ("[ERA MEMORY -- saved notes about the user, background only. Use them only if relevant. "
                  "Do not claim to remember anything that is not listed here; treat (unverified)/(uncertain) items as possibly wrong.]")
        footer = "[END ERA MEMORY]"
        remaining = budget - len(header) - len(footer) - 4
        blocks: List[str] = []
        truncated = False
        added_any = False

        def add_section(title: str, items: List[str]) -> None:
            nonlocal remaining, truncated, added_any
            if not items:
                return
            chunk = [title]
            cost = len(title) + 1
            for it in items:
                if cost + len(it) + 1 > remaining:
                    truncated = True
                    break
                chunk.append(it)
                cost += len(it) + 1
            if len(chunk) > 1:
                blocks.append("\n".join(chunk))
                remaining -= cost + 1
                added_any = True
            else:
                truncated = True

        def line(m: Memory, prefix: str = "- ") -> str:
            return f"{prefix}{truncate(m.content, 220)} {self._label(m)}"

        retrieved = sorted(retrieved, key=lambda m: scores.get(m.id, 0), reverse=True)
        add_section("Communication preferences:", [line(m) for m in comm])
        add_section("Relevant memories:", [line(m) for m in retrieved])
        if pid:
            proj = self.store.get_project(pid)
            if proj and proj.summary:
                add_section(f"Active project ({proj.name}):", [truncate(proj.summary.replace("\n", " / "), 420)])
        add_section("Open items / unfinished work:", [line(m) for m in open_items])
        if conflicts:
            add_section("CONFLICTING memories -- ask the user which is correct before relying on either:",
                        [f'- "{truncate(c["a_content"], 100)}" vs "{truncate(c["b_content"], 100)}"' for c in conflicts])

        if not added_any:
            return "", False, False
        text = "\n".join([header] + blocks + [footer])
        pkg.items = [
            {"id": m.id, "type": m.type.value, "category": m.category.value, "content": m.content,
             "confidence_level": m.confidence_level.value, "updated_at": m.updated_at, "score": round(scores.get(m.id, 0.0), 3)}
            for m in comm + retrieved + open_items if m.content in text
        ]
        return text, True, truncated

    # --------------------------------------------------------- async wrappers
    async def abuild_context(self, message: str, conversation_id: Optional[str] = None, project: Optional[str] = None) -> ContextPackage:
        """Non-blocking; bounded by ``context_timeout_s`` so the reply is never held up."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.build_context, message, conversation_id, project), timeout=self.config.context_timeout_s
            )
        except asyncio.TimeoutError:
            logger.warning("Memory context timed out after %.1fs; continuing without it.", self.config.context_timeout_s)
            return ContextPackage(enabled=None, status="timeout", available=False, conversation_id=conversation_id,
                                  error=f"timeout: context lookup exceeded {self.config.context_timeout_s:g}s")

    async def aingest_turn(self, *a: Any, **kw: Any) -> IngestResult:
        return await asyncio.to_thread(self.ingest_turn, *a, **kw)

    def reembed(self, limit: int = 500) -> Dict[str, Any]:
        """Re-embed memories whose vector is missing or came from a different provider/model/dimension,
        using the current primary provider. Until then such memories are still found by keyword."""
        with self._guard():
            key = self.embedder.primary_key()
            if key is None:
                return {"reembedded": 0, "failed": 0, "remaining": None, "error": "no embedding provider is available"}
            done = failed = 0
            for mid, content in self.store.rows_needing_embedding(key, limit):
                rec = self.embedder.embed(content)
                if rec is None:
                    failed += 1
                    continue
                self.store.update_memory(mid, {"embedding": rec.blob, "embedding_model": rec.model,
                                               "embedding_provider": rec.provider, "embedding_dim": rec.dim})
                done += 1
            return {"reembedded": done, "failed": failed, "remaining": len(self.store.rows_needing_embedding(key, 1_000_000)),
                    "provider": key[0], "model": key[1], "dim": key[2]}

    def stats(self) -> Dict[str, Any]:
        with self._guard():
            counts = self.store.count_memories()
            key = self.embedder.primary_key()
            census = self.store.embedding_census()
            other = sum(r["n"] for r in census if key is None or (r["provider"], r["model"], r["dim"]) != key)
            return {
                "memories_by_type": counts, "total_active": sum(counts.values()),
                "projects": len(self.store.list_projects()), "schema_version": self.store.schema_version(),
                "keyword_index": "fts5" if self.store.fts_available else "like-fallback",
                "embedding_providers": [p.name for p in self.embedder.providers],
                "embedding_primary": None if key is None else {"provider": key[0], "model": key[1], "dim": key[2]},
                "memories_with_other_or_missing_embeddings": other,
                "auto_memory": self._auto_status() == "ok", "status": self._auto_status(),
            }


class _StoreGuard:
    """Turns low-level storage errors into MemoryUnavailableError for explicit operations."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and issubclass(exc_type, MemoryStoreError):
            raise MemoryUnavailableError(str(exc)) from exc
        return False
