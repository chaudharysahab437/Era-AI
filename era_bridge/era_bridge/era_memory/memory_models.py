"""Typed data models for the ERA memory engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(str, Enum):
    WORKING = "working"        # current conversation / active task (short-lived)
    EPISODIC = "episodic"      # significant past conversations and events
    SEMANTIC = "semantic"      # established facts about the user / world
    PREFERENCE = "preference"  # stable user + communication preferences
    PROJECT = "project"        # requirements, decisions, progress, blockers, next steps


class Category(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    COMMUNICATION = "communication"
    DECISION = "decision"
    GOAL = "goal"
    EVENT = "event"
    PROJECT_UPDATE = "project_update"
    BLOCKER = "blocker"
    NEXT_STEP = "next_step"
    TASK = "task"
    SUMMARY = "summary"
    NOTE = "note"


class Confidence(str, Enum):
    CONFIRMED = "confirmed"   # the user said it plainly
    INFERRED = "inferred"     # derived / lower certainty
    UNCERTAIN = "uncertain"   # hedged ("I think", "maybe") or unverified


CONFIRMED_AT = 0.8
INFERRED_AT = 0.5
LEVEL_TO_SCORE = {Confidence.CONFIRMED: 0.9, Confidence.INFERRED: 0.6, Confidence.UNCERTAIN: 0.3}


def confidence_level(score: float) -> Confidence:
    if score >= CONFIRMED_AT:
        return Confidence.CONFIRMED
    if score >= INFERRED_AT:
        return Confidence.INFERRED
    return Confidence.UNCERTAIN


@dataclass
class SourceRef:
    kind: str = "chat"                       # chat | manual | api | summary
    conversation_id: Optional[str] = None
    reference: Optional[str] = None          # free-form pointer (never raw message text)


@dataclass
class Memory:
    id: str
    type: MemoryType
    content: str
    category: Category = Category.FACT
    created_at: str = ""
    updated_at: str = ""
    source: SourceRef = field(default_factory=SourceRef)
    importance: float = 0.5
    confidence: float = 0.9
    status: str = "active"                   # active | superseded
    superseded_by: Optional[str] = None
    conflict_with: Optional[str] = None
    slot: Optional[str] = None               # "single-valued fact" key, e.g. user:location
    content_hash: str = ""
    sensitivity: str = "normal"              # normal | sensitive
    share_with_model: bool = True
    project_id: Optional[str] = None
    times_seen: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_level(self) -> Confidence:
        return confidence_level(self.confidence)

    @property
    def conversation_id(self) -> Optional[str]:
        return self.source.conversation_id

    @property
    def is_sensitive(self) -> bool:
        return self.sensitivity == "sensitive"

    def masked_content(self) -> str:
        cats = ", ".join(self.metadata.get("sensitive_categories", [])) or "sensitive"
        return f"[hidden: {cats} memory -- request with reveal=true to view]"

    def to_dict(self, reveal: bool = False) -> Dict[str, Any]:
        show = reveal or not self.is_sensitive
        return {
            "id": self.id,
            "type": self.type.value,
            "category": self.category.value,
            "content": self.content if show else self.masked_content(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": {
                "kind": self.source.kind,
                "conversation_id": self.source.conversation_id,
                "reference": self.source.reference,
            },
            "importance": round(self.importance, 3),
            "confidence": round(self.confidence, 3),
            "confidence_level": self.confidence_level.value,
            "status": self.status,
            "superseded_by": self.superseded_by,
            "conflict_with": self.conflict_with,
            "slot": self.slot,
            "sensitivity": self.sensitivity,
            "share_with_model": self.share_with_model,
            "project_id": self.project_id,
            "times_seen": self.times_seen,
            "metadata": self.metadata if show else {k: v for k, v in self.metadata.items() if k == "sensitive_categories"},
        }


@dataclass
class EmbeddingRecord:
    """A stored vector plus the exact configuration that produced it."""

    blob: bytes
    provider: str
    model: str
    dim: int

    @property
    def key(self):
        return (self.provider, self.model, self.dim)


@dataclass
class Conversation:
    id: str
    created_at: str
    updated_at: str
    title: Optional[str] = None
    project_id: Optional[str] = None
    active_task: Optional[str] = None
    summary: Optional[str] = None
    summary_updated_at: Optional[str] = None
    turn_count: int = 0
    closed_at: Optional[str] = None


@dataclass
class Project:
    id: str
    name: str
    created_at: str
    updated_at: str
    aliases: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    summary_updated_at: Optional[str] = None
    status: str = "active"


@dataclass
class ScoredMemory:
    memory: Memory
    score: float
    relevance: float
    keyword: float
    semantic: float


@dataclass
class Candidate:
    """A memory-to-be, produced by the extractor or an explicit save."""

    content: str
    type: MemoryType = MemoryType.SEMANTIC
    category: Category = Category.FACT
    slot: Optional[str] = None
    confidence: float = 0.9
    importance: float = 0.5
    is_correction: bool = False
    sensitivity: str = "normal"
    share_with_model: bool = True
    source_kind: str = "chat"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeldBack:
    category: str


@dataclass
class ExtractionResult:
    candidates: List[Candidate] = field(default_factory=list)
    held_back: List[str] = field(default_factory=list)   # categories only, never content
    secrets_blocked: int = 0
    project_name: Optional[str] = None
    active_task: Optional[str] = None


@dataclass
class UpsertOutcome:
    action: str                                  # created | duplicate | superseded | conflict
    memory: Memory
    superseded_ids: List[str] = field(default_factory=list)


@dataclass
class IngestResult:
    enabled: Optional[bool] = True             # automatic memory switch; None = could not be determined
    status: str = "ok"                         # ok | disabled_by_user | disabled_by_config | unavailable | internal_error
    available: bool = True                     # False when the memory database/engine itself failed
    saved: List[str] = field(default_factory=list)
    updated: List[str] = field(default_factory=list)       # duplicates that were refreshed
    superseded: List[Dict[str, str]] = field(default_factory=list)   # {"old": id, "new": id}
    conflicts: List[str] = field(default_factory=list)
    held_back: List[str] = field(default_factory=list)
    secrets_blocked: int = 0
    project_id: Optional[str] = None
    conversation_id: Optional[str] = None
    active_task: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        if self.held_back:
            d["note"] = (
                "Some content looked sensitive and was NOT saved. To keep it, save it explicitly "
                "with POST /v1/memory and allow_sensitive=true."
            )
        return d


@dataclass
class ContextPackage:
    enabled: Optional[bool] = True
    status: str = "ok"                         # ok | disabled_by_user | disabled_by_config | unavailable | timeout | internal_error
    available: bool = True
    text: str = ""
    items: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    withheld_sensitive: int = 0
    truncated: bool = False
    char_count: int = 0
    project_id: Optional[str] = None
    conversation_id: Optional[str] = None
    continuation: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)
