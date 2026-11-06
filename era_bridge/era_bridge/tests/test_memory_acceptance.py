"""Mandatory acceptance criteria 3, 5, 6, 8 -- each criterion has explicit, named tests.
(1, 2, 4, 7, 9, 10 are covered in test_memory_integration / _updates / _privacy; the mapping is in README.)"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import bridge_server
from config import settings
from core.session_store import SessionStore
from era_memory import MemoryConfig, MemoryManager
from tests.test_bridge_api import FakeAgentClient

H = {"X-Era-Bridge-Key": settings.BRIDGE_API_KEY}


# --- 3. relevant old info is connected automatically; memories are NOT dumped into the prompt -------------
def test_criterion3_context_contains_only_relevant_memories_not_the_whole_store(make_manager):
    mm = make_manager(max_context_chars=4000)
    for i in range(60):
        mm.save(f"Unrelated note {i}: the office plant number {i} needs watering on day {i}")
    mm.ingest_turn("I'm building Atlas and we decided to use PostgreSQL for the Atlas database.", conversation_id="old")
    pkg = mm.build_context("which database did we choose for Atlas?", conversation_id="brand-new")
    assert "PostgreSQL" in pkg.text
    assert "office plant" not in pkg.text                       # none of the 60 unrelated notes
    assert len(pkg.items) <= 3 and pkg.char_count < 900         # a handful of lines, not the database
    total = len(mm.list(limit=200))
    assert total > 60 and len(pkg.items) < total / 10


def test_criterion3_unrelated_message_gets_no_memory_context_at_all(mm):
    mm.ingest_turn("I'm building Atlas and we decided to use PostgreSQL for the Atlas database.", conversation_id="old")
    assert mm.build_context("what's the weather like on Mars?", conversation_id="new").text == ""


# --- 5. duplicates do not accumulate ---------------------------------------------------------------------
def test_criterion5_repeating_the_same_fact_many_times_keeps_one_memory(mm):
    variants = ["My favorite color is teal.", "my favorite color is teal", "My favourite colour is teal!"]
    for i in range(24):
        mm.ingest_turn(variants[i % 2], conversation_id=f"conv-{i}")     # 2 spellings x 12 conversations
    facts = [m for m in mm.list(limit=100) if "teal" in m.content]
    assert len(facts) == 1 and facts[0].times_seen == 24


def test_criterion5_explicit_and_automatic_paths_do_not_duplicate_each_other(mm):
    mm.ingest_turn("I live in Agra.", conversation_id="a")
    assert mm.save("User lives in Agra.", slot="user:location").action == "duplicate"
    assert len(mm.list()) == 1


# --- 6. secrets never stored automatically ---------------------------------------------------------------
def test_criterion6_secret_in_the_same_message_as_a_normal_fact_only_blocks_the_secret(mm):
    r = mm.ingest_turn("My name is Shavan and my api key is sk-abcdefghijklmnopqrstuvwxyz1234.", conversation_id="c")
    stored = " ".join(m.content for m in mm.list(include_history=True))
    assert "Shavan" in stored and "sk-" not in stored and r.secrets_blocked == 1
    assert "sk-" not in str(r.to_dict())


def test_criterion6_secrets_never_reach_the_database_file_at_all(mm, db_path):
    mm.ingest_turn("Remember my password is Tr0ub4dor&3xyz and token ghp_abcdefghijklmnopqrstuvwxyz0123456789", "Ok, noted.", conversation_id="c")
    mm.close_conversation("c")
    raw = b"".join(p.read_bytes() for p in db_path.parent.glob("memory.db*"))
    assert b"Tr0ub4dor" not in raw and b"ghp_abcdefgh" not in raw


# --- 8. failures never crash/block ERA, and the API reports the REAL error, never a false "off" ----------
@pytest.fixture
def broken_client(tmp_path, clock):
    bad = tmp_path / "broken.db"
    bad.write_bytes(b"not a database" * 100)
    bridge_server.session_store = SessionStore(db_path=tmp_path / "s.db")
    bridge_server.agent_client = FakeAgentClient()
    bridge_server.memory_manager = MemoryManager(MemoryConfig(db_path=bad), clock=clock)
    return TestClient(bridge_server.app)


def test_criterion8_context_reports_the_real_error_not_memory_off(broken_client):
    body = broken_client.post("/v1/memory/context", headers=H, json={"message": "what is my name please"}).json()
    assert body["status"] == "unavailable" and body["available"] is False
    assert body["error"].startswith("memory_unavailable:") and "DatabaseError" in body["error"]
    assert body["enabled"] is None                              # unknown -- explicitly NOT False ("off")
    assert body["text"] == ""


def test_criterion8_ingest_reports_the_real_error(broken_client):
    body = broken_client.post("/v1/memory/ingest", headers=H, json={"user_message": "My name is Shavan", "conversation_id": "c"}).json()
    assert body["status"] == "unavailable" and body["available"] is False and body["enabled"] is None
    assert body["error"].startswith("memory_unavailable:")


def test_criterion8_settings_and_stats_return_503_with_the_cause_not_auto_memory_false(broken_client):
    for url in ("/v1/memory/settings", "/v1/memory/stats", "/v1/memory"):
        r = broken_client.get(url, headers=H)
        assert r.status_code == 503, url
        d = r.json()["detail"]
        assert d["code"] == "memory_unavailable" and "DatabaseError" in d["error"]
        assert "auto_memory" not in r.text


def test_criterion8_genuinely_disabled_is_reported_as_disabled_with_the_reason(mm):
    mm.set_auto(False)
    pkg = mm.build_context("anything at all here", "c")
    assert pkg.enabled is False and pkg.status == "disabled_by_user" and pkg.available is True and pkg.error is None
    off = MemoryManager(MemoryConfig(db_path=mm.config.db_path, enabled=False))
    assert off.build_context("anything at all here", "c").status == "disabled_by_config"


def test_criterion8_commands_keep_working_while_memory_is_broken(broken_client):
    for _ in range(3):
        assert broken_client.post("/v1/execute", headers=H, json={"command": "open_notepad"}).json()["state"] == "success"
        assert broken_client.post("/v1/memory/context", headers=H, json={"message": "hello there friend"}).status_code == 200
    assert broken_client.get("/v1/status").status_code == 200


def test_criterion8_unexpected_internal_bug_is_reported_as_internal_error_and_does_not_raise(mm, monkeypatch):
    monkeypatch.setattr(mm.retriever, "retrieve", lambda *a, **k: 1 / 0)
    pkg = mm.build_context("what is my name", "c")
    assert pkg.status == "internal_error" and pkg.error == "internal_error: ZeroDivisionError" and pkg.available is False


def test_criterion8_slow_memory_lookup_times_out_instead_of_blocking(make_manager):
    import asyncio, time
    mm = make_manager(context_timeout_s=0.2)
    mm.build_context = lambda *a, **k: time.sleep(1.0)
    pkg = asyncio.run(mm.abuild_context("hello there", "c"))
    assert pkg.status == "timeout" and pkg.error.startswith("timeout:") and pkg.text == ""
