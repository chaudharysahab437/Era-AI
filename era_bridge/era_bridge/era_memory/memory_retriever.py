"""Hybrid memory retrieval: keyword + semantic + recency + importance.

Embedding providers are pluggable. The built-in ``HashingEmbedding`` needs
no model, no network and no API key: it hashes word, word-pair and
character-trigram features into a fixed vector. It catches morphological
and spelling similarity ("colour"/"color", "deploy"/"deployment") but it
is NOT a true semantic model -- it will not link unrelated synonyms.
Plug in a real provider (anything with ``name``, ``embed(text)``) to get
that; if it fails, the engine silently falls back to the hashing provider.

Only relevant memories are returned: a memory must clear a keyword or
semantic relevance gate before recency/importance can lift it, so
"recent but unrelated" memories never leak into a response.
"""

from __future__ import annotations

import logging
import math
import struct
import zlib
from datetime import datetime, timezone
from operator import mul
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from .memory_config import MemoryConfig
from .memory_models import EmbeddingRecord, Memory, MemoryType, ScoredMemory
from .memory_store import MemoryStore
from .text_utils import overlap, token_set, tokens

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, text: str) -> Optional[Sequence[float]]: ...


class HashingEmbedding:
    """Dependency-free fallback embedding (feature hashing, 256 dims)."""

    name = "hashing-v1"
    provider = "builtin"
    model = "hashing-v1"
    dim = 256

    def embed(self, text: str) -> Optional[List[float]]:
        toks = tokens(text)
        if not toks:
            return None
        vec = [0.0] * self.dim

        def add(feature: str, weight: float) -> None:
            h = zlib.crc32(feature.encode("utf-8"))
            vec[h % self.dim] += weight if (h >> 16) & 1 else -weight

        for t in toks:
            add("w:" + t, 1.0)
            padded = f"#{t}#"
            for i in range(len(padded) - 2):
                add("c:" + padded[i : i + 3], 0.3)
        for a, b in zip(toks, toks[1:]):
            add(f"b:{a}_{b}", 0.6)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return None
        return [v / norm for v in vec]


class CallableEmbedding:
    """Wrap any ``fn(text) -> list[float]`` (e.g. a local sentence-transformer)."""

    def __init__(self, name: str, fn: Callable[[str], Sequence[float]]) -> None:
        self.name = name
        self._fn = fn

    def embed(self, text: str) -> Optional[Sequence[float]]:
        return self._fn(text)


def _meta(p: "EmbeddingProvider") -> Tuple[str, str]:
    """(provider, model) identity of a provider. Providers may expose ``provider`` and ``model``
    attributes; both default to ``name``."""
    name = getattr(p, "name", "unknown")
    return str(getattr(p, "provider", name)), str(getattr(p, "model", name))


class EmbeddingService:
    """Tries providers in order; a failing or misbehaving provider is skipped, never fatal.

    Every vector is tagged with (provider, model, dimension). Retrieval only ever compares
    vectors carrying the same tag, so switching providers or models can never produce
    meaningless similarity scores -- affected memories are simply found by keyword until
    they are re-embedded (see ``MemoryManager.reembed``)."""

    def __init__(self, providers: Optional[Sequence[EmbeddingProvider]] = None) -> None:
        self.providers: List[EmbeddingProvider] = list(providers) if providers else []
        if not any(p.name == HashingEmbedding.name for p in self.providers):
            self.providers.append(HashingEmbedding())

    def _vector(self, p: EmbeddingProvider, text: str) -> Optional[List[float]]:
        try:
            vec = p.embed(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding provider '%s' failed (%s); trying next.", getattr(p, "name", "?"), type(exc).__name__)
            return None
        if vec is None:
            return None
        try:
            vec = [float(x) for x in vec]
        except (TypeError, ValueError):
            logger.warning("Embedding provider '%s' returned a non-numeric vector; ignoring it.", getattr(p, "name", "?"))
            return None
        if not vec or not all(math.isfinite(x) for x in vec):
            logger.warning("Embedding provider '%s' returned an empty/non-finite vector; ignoring it.", getattr(p, "name", "?"))
            return None
        unit = _unit(vec)
        return unit if math.sqrt(sum(map(mul, unit, unit))) > 0.5 else None

    def embed(self, text: str) -> Optional[EmbeddingRecord]:
        for p in self.providers:
            vec = self._vector(p, text)
            if vec:
                provider, model = _meta(p)
                return EmbeddingRecord(pack(vec), provider, model, len(vec))
        return None

    def embed_all(self, text: str) -> List[Tuple[Tuple[str, str, int], List[float]]]:
        """Query vectors from every working provider, each tagged with its (provider, model, dim) key."""
        out = []
        for p in self.providers:
            vec = self._vector(p, text)
            if vec:
                provider, model = _meta(p)
                out.append(((provider, model, len(vec)), vec))
        return out

    def primary_key(self) -> Optional[Tuple[str, str, int]]:
        rec = self.embed("embedding configuration probe")
        return rec.key if rec else None


def _unit(vec: Sequence[float]) -> List[float]:
    """Unit-length copy, so similarity is a plain dot product (stored vectors are pre-normalised)."""
    n = math.sqrt(sum(map(mul, vec, vec)))
    return [v / n for v in vec] if n else list(vec)


def pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def unpack(blob: bytes) -> List[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of two UNIT-length vectors (== cosine similarity)."""
    if len(a) != len(b):
        return 0.0
    return sum(map(mul, a, b))


def parse_iso(ts: str) -> datetime:
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class MemoryRetriever:
    def __init__(self, store: MemoryStore, embedder: EmbeddingService, config: MemoryConfig, clock: Callable[[], datetime]) -> None:
        self.store, self.embedder, self.config, self._clock = store, embedder, config, clock

    def similar_by_embedding(self, content: str, mtype: Optional[MemoryType] = None, limit: int = 3) -> List[Tuple[Memory, float]]:
        """Nearest active memories by embedding (used for duplicate detection).

        Near-duplicates necessarily share keywords, so only the keyword
        candidates are compared -- O(candidates), not O(database)."""
        ids = self.store.fts_candidates(sorted(token_set(content)), self.config.fts_candidates)
        stored = self.store.embeddings_for(ids)
        if not stored:
            return []
        sims: Dict[str, float] = {}
        for model, qvec in self.embedder.embed_all(content):
            for mid, (row_key, blob) in stored.items():
                if tuple(row_key) == tuple(model):        # identical provider + model + dimension only
                    sims[mid] = max(sims.get(mid, -1.0), cosine(qvec, unpack(blob)))
        top = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)[: limit * 3]
        mems = {m.id: m for m in self.store.get_many([i for i, _ in top])}
        out = [(mems[i], s) for i, s in top if i in mems and (mtype is None or mems[i].type == mtype)]
        return out[:limit]

    def _semantic_scores(self, query: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for key, qvec in self.embedder.embed_all(query):
            for mid, blob in self.store.embedding_rows(key, self.config.max_vector_scan):
                if len(blob) != 4 * len(qvec):            # belt and braces: never compare unequal dimensions
                    continue
                s = cosine(qvec, unpack(blob))
                if s > scores.get(mid, -1.0):
                    scores[mid] = s
        return scores

    def retrieve(
        self,
        query: str,
        *,
        project_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        types: Optional[Sequence[MemoryType]] = None,
        limit: Optional[int] = None,
        for_model: bool = False,
        include_other_working: bool = False,
    ) -> List[ScoredMemory]:
        cfg = self.config
        limit = limit or cfg.max_retrieved
        q_tokens = token_set(query)
        if not q_tokens:
            return []

        # 1) cheap candidate generation: keyword index + nearest embeddings
        ids = list(dict.fromkeys(self.store.fts_candidates(sorted(q_tokens), cfg.fts_candidates)))
        sem = self._semantic_scores(query)
        top_sem = [i for i, s in sorted(sem.items(), key=lambda kv: kv[1], reverse=True)[: cfg.vector_candidates] if s >= cfg.semantic_gate]
        ids = list(dict.fromkeys(ids + top_sem))
        if not ids:
            return []

        # 2) score only the candidates
        now = self._clock()
        scored: List[ScoredMemory] = []
        for m in self.store.get_many(ids):
            if m.status != "active":
                continue
            if types and m.type not in types:
                continue
            if m.type == MemoryType.WORKING and conversation_id != m.conversation_id and not include_other_working:
                continue
            if for_model and m.is_sensitive and not m.share_with_model:
                continue
            kw = overlap(q_tokens, token_set(m.content))
            sm = max(0.0, sem.get(m.id, 0.0))
            if kw < cfg.keyword_gate and sm < cfg.semantic_gate:
                continue
            relevance = max(kw, sm)
            age_days = max(0.0, (now - parse_iso(m.updated_at)).total_seconds() / 86400.0)
            recency = 0.5 ** (age_days / cfg.recency_half_life_days)
            proj = 1.0 if project_id and m.project_id == project_id else 0.0
            score = (
                cfg.w_relevance * relevance
                + cfg.w_recency * recency
                + cfg.w_importance * m.importance
                + cfg.w_confidence * m.confidence
                + cfg.w_project * proj
            )
            scored.append(ScoredMemory(m, score, relevance, kw, sm))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:limit]
