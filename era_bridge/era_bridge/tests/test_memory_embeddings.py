"""Embeddings from different providers/models/dimensions must never be compared."""

from __future__ import annotations

import sqlite3

import pytest

from era_memory import CallableEmbedding, EmbeddingService, MemoryConfig, MemoryManager
from era_memory.memory_retriever import HashingEmbedding


class Provider:
    def __init__(self, name, vecs, provider=None, model=None, default=None):
        self.name, self.vecs, self.default = name, vecs, default
        if provider:
            self.provider = provider
        if model:
            self.model = model

    def embed(self, text):
        for word, vec in self.vecs.items():
            if word in text.lower():
                return vec
        return self.default


def _mgr(db_path, clock, *providers):
    return MemoryManager(MemoryConfig(db_path=db_path), embedder=EmbeddingService(list(providers)), clock=clock)


def _meta(db_path):
    with sqlite3.connect(db_path) as c:
        return c.execute("SELECT embedding_provider, embedding_model, embedding_dim FROM memories").fetchall()


def test_provider_model_and_dimension_are_stored_with_every_embedding(db_path, clock):
    mm = _mgr(db_path, clock, Provider("acme", {"car": [1, 0, 0]}, provider="acme-cloud", model="acme-embed-3", default=[0, 0, 1]))
    mm.save("I drive a car to work")
    assert _meta(db_path) == [("acme-cloud", "acme-embed-3", 3)]
    plain = MemoryManager(MemoryConfig(db_path=db_path.with_name("b.db")), clock=clock)
    plain.save("I like tea in the evening")
    with sqlite3.connect(db_path.with_name("b.db")) as c:
        assert c.execute("SELECT embedding_provider, embedding_model, embedding_dim FROM memories").fetchone() == ("builtin", "hashing-v1", 256)


def test_different_provider_vectors_are_never_compared_and_keyword_search_still_works(db_path, clock):
    old = _mgr(db_path, clock, Provider("prov-a", {"car": [1, 0, 0]}, default=[0, 0, 1]))
    old.save("I drive an automobile car every day")
    # ERA restarts configured with a different provider whose vectors happen to have the SAME length
    # and would score ~1.0 against the old ones if (wrongly) compared.
    new = _mgr(db_path, clock, Provider("prov-b", {"zzz": [1, 0, 0]}, default=[1, 0, 0]))
    hits = new.search("which car do I drive")               # shares the keyword 'car'
    assert hits and hits[0].semantic == 0.0                 # found by KEYWORD; no cross-provider similarity used
    assert new.search("banana smoothie recipe") == []       # and an unrelated query does NOT match by accident


def test_same_provider_name_but_different_dimension_is_incompatible(db_path, clock):
    _mgr(db_path, clock, Provider("prov", {"car": [1, 0, 0]}, default=[0, 0, 1])).save("I drive an automobile")
    upgraded = _mgr(db_path, clock, Provider("prov", {"car": [1, 0, 0, 0, 0]}, default=[0, 0, 0, 0, 1]))   # dim 3 -> 5
    assert upgraded.search("my car") == []                  # no crash, no bogus semantic match
    assert upgraded.search("automobile")                    # keyword fallback still finds it


def test_same_provider_different_model_is_incompatible(db_path, clock):
    _mgr(db_path, clock, Provider("p", {"car": [1, 0, 0]}, provider="x", model="v1", default=[0, 0, 1])).save("I drive an automobile")
    v2 = _mgr(db_path, clock, Provider("p", {"car": [1, 0, 0]}, provider="x", model="v2", default=[0, 0, 1]))
    assert v2.search("my car") == []                        # v1 vectors are not comparable with v2


def test_reembed_makes_incompatible_memories_semantically_searchable_again(db_path, clock):
    _mgr(db_path, clock, Provider("old", {"automobile": [1, 0, 0]}, default=[0, 1, 0])).save("I drive an automobile")
    new = _mgr(db_path, clock, Provider("new", {"automobile": [0, 0, 0, 1], "car": [0, 0, 0, 1]}, default=[0, 1, 0, 0]))
    assert new.search("my car") == []
    assert new.stats()["memories_with_other_or_missing_embeddings"] == 1
    out = new.reembed()
    assert out["reembedded"] == 1 and out["remaining"] == 0 and out["dim"] == 4
    hits = new.search("my car")
    assert hits and hits[0].keyword == 0.0 and hits[0].semantic > 0.9        # true semantic hit, no shared words
    assert new.stats()["memories_with_other_or_missing_embeddings"] == 0


def test_provider_outage_at_save_time_leaves_no_bad_vector_and_is_recoverable(db_path, clock):
    def down(_t):
        raise RuntimeError("provider down")

    # no fallback provider except the built-in hashing one, which is always appended
    mm = _mgr(db_path, clock, CallableEmbedding("remote", down))
    mm.save("ERA keeps memories in SQLite")
    assert _meta(db_path)[0][:2] == ("builtin", "hashing-v1")        # tagged honestly as the fallback


@pytest.mark.parametrize("bad", [[], [float("nan"), 1.0], [float("inf")], ["a", "b"], [0.0, 0.0, 0.0]])
def test_malformed_vectors_are_rejected_not_stored(db_path, clock, bad):
    mm = _mgr(db_path, clock, Provider("bad", {}, default=bad))
    mm.save("Some perfectly ordinary memory text")
    assert _meta(db_path) == [("builtin", "hashing-v1", 256)]        # the bad provider was skipped


def test_legacy_rows_are_backfilled_by_the_migration(tmp_path):
    """A database created before migration 003 (no provider/dim columns) upgrades in place."""
    import shutil
    from era_memory.memory_store import MIGRATIONS_DIR, MemoryStore

    old_dir = tmp_path / "mig_v2"
    old_dir.mkdir()
    for f in MIGRATIONS_DIR.glob("00[12]_*.sql"):
        shutil.copy(f, old_dir / f.name)
    db = tmp_path / "legacy.db"
    MemoryStore(db, migrations_dir=old_dir).schema_version()
    blob = b"\x00\x00\x80?" * 8                                        # 8 floats
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO memories (id,type,category,content,content_hash,created_at,updated_at,embedding,embedding_model) "
                  "VALUES ('m1','semantic','fact','x','h','t','t',?,?)", (blob, "hashing-v1"))
    upgraded = MemoryStore(db)                                        # real migrations dir -> applies 003
    assert upgraded.schema_version() == 3
    assert _meta(db) == [("hashing-v1", "hashing-v1", 8)]
