"""Storage layer: persistence, migrations, restart survival, safe failure."""

from __future__ import annotations

import sqlite3

import pytest

from era_memory import MemoryConfig, MemoryManager, MemoryType, MemoryUnavailableError
from era_memory.memory_store import MemoryStore, MemoryStoreError


def test_save_and_retrieve_a_memory(mm):
    out = mm.save("The project deadline is 15 October", type=MemoryType.SEMANTIC)
    assert out.action == "created"
    got = mm.get(out.memory.id)
    assert got.content == "The project deadline is 15 October"
    assert got.type is MemoryType.SEMANTIC
    assert got.created_at and got.updated_at and got.source.kind == "manual"
    assert 0 <= got.importance <= 1 and got.confidence_level.value == "confirmed"


def test_memory_survives_restart(make_manager):
    first = make_manager()
    mid = first.save("I keep my notes in Obsidian").memory.id
    del first                                   # "close ERA completely"
    second = make_manager()                     # "start ERA again": brand-new objects, same file
    assert second.get(mid).content == "I keep my notes in Obsidian"
    assert [m.id for m in second.list()] == [mid]


def test_migrations_are_versioned_and_idempotent(db_path):
    s = MemoryStore(db_path)
    assert s.schema_version() >= 2 and s.fts_available
    s2 = MemoryStore(db_path)                   # second start must not re-apply anything
    assert s2.schema_version() == s.schema_version()
    with sqlite3.connect(db_path) as c:
        versions = [r[0] for r in c.execute("SELECT version FROM schema_migrations ORDER BY version")]
    assert versions == sorted(set(versions))


def test_store_is_lazy_and_touches_no_file_until_used(tmp_path):
    p = tmp_path / "sub" / "never.db"
    MemoryManager(MemoryConfig(db_path=p))
    assert not p.exists()


def test_falls_back_to_like_search_without_fts(db_path, clock):
    mm = MemoryManager(MemoryConfig(db_path=db_path), clock=clock)
    mm.store._ensure()
    mm.store.fts_available = False              # simulate an SQLite build without FTS5
    mm.save("ERA uses SQLite for persistent storage")
    hits = mm.search("what storage does ERA use")
    assert hits and "SQLite" in hits[0].memory.content


def test_delete_removes_memory_and_its_history(mm):
    old = mm.save("I live in Delhi", slot="user:location").memory
    new = mm.save("I live in Pune", slot="user:location").memory
    assert mm.get(old.id).status == "superseded"
    assert mm.delete(new.id) is True
    assert mm.list(include_history=True) == []  # the superseded old value must not linger


def test_delete_unknown_id_returns_false(mm):
    assert mm.delete("mem_nope") is False


def test_clear_all_wipes_everything(mm):
    mm.save("fact one about apples")
    mm.ingest_turn("I am working on ERA memory", conversation_id="c1")
    assert mm.clear_all() >= 1
    assert mm.list(include_history=True) == []
    assert mm.list_projects() == []


def test_corrupt_database_is_handled_safely(tmp_path, clock):
    bad = tmp_path / "corrupt.db"
    bad.write_bytes(b"this is definitely not a sqlite database" * 50)
    mm = MemoryManager(MemoryConfig(db_path=bad), clock=clock)
    # automatic paths never raise ...
    assert mm.build_context("what is my name?", "c1").text == ""
    assert mm.ingest_turn("My name is Shavan", conversation_id="c1").saved == []
    # ... explicit calls raise a clean, catchable error
    with pytest.raises(MemoryUnavailableError):
        mm.save("anything")
    assert bad.read_bytes().startswith(b"this is definitely")   # and the file was not clobbered


def test_unwritable_location_is_handled_safely(tmp_path, clock):
    blocker = tmp_path / "iamafile"
    blocker.write_text("x")
    mm = MemoryManager(MemoryConfig(db_path=blocker / "memory.db"), clock=clock)  # parent is a file
    assert mm.ingest_turn("I live in Agra", conversation_id="c").error is not None
    assert mm.build_context("where do I live", "c").error is not None or mm.build_context("x", "c").text == ""


def test_store_error_type(tmp_path):
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"garbage" * 100)
    with pytest.raises(MemoryStoreError):
        MemoryStore(bad).schema_version()


def test_retention_purges_old_working_memory_and_turns(make_manager, clock):
    mm = make_manager(retention_days=14)
    mm.ingest_turn("I am working on ERA", conversation_id="old")
    assert mm.list(types=["working"])
    clock.advance(days=20)
    fresh = make_manager(retention_days=14)     # new process purges on first use
    fresh.ingest_turn("Hello there my friend today", conversation_id="new")
    assert fresh.list(types=["working"], conversation_id="old") == []
    assert fresh.store.get_turns("old") == []
