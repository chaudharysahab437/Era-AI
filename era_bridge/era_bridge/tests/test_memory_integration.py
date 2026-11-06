"""End-to-end through Bridge's HTTP API, including the acceptance flow A-E
across genuinely separate Python processes, and proof that existing Bridge
functionality (execute/confirm/status/commands) is unaffected by memory."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import bridge_server
from config import settings
from core.session_store import SessionStore
from era_memory import MemoryConfig, MemoryManager
from tests.test_bridge_api import FakeAgentClient

H = {"X-Era-Bridge-Key": settings.BRIDGE_API_KEY}
BRIDGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(tmp_path, db_path, clock):
    assert str(db_path).startswith(str(tmp_path))          # tests only ever use a temp database
    bridge_server.session_store = SessionStore(db_path=tmp_path / "sessions.db")
    bridge_server.agent_client = FakeAgentClient()
    bridge_server.memory_manager = MemoryManager(MemoryConfig(db_path=db_path), clock=clock)
    return TestClient(bridge_server.app)


def restart_bridge(db_path, clock):
    """Simulates closing ERA completely and starting it again over the same database."""
    bridge_server.memory_manager = MemoryManager(MemoryConfig(db_path=db_path), clock=clock)
    return TestClient(bridge_server.app)


# ---------------------------------------------------------------- acceptance --
def test_acceptance_A_to_E_over_http(client, db_path, clock):
    # A. tell ERA something important in one conversation
    r = client.post("/v1/memory/ingest", headers=H, json={
        "user_message": "I'm working on Atlas, an inventory app. We decided to use SQLite for Atlas storage.",
        "assistant_message": "Nice, SQLite is a solid choice.", "conversation_id": "conv-1"})
    assert r.status_code == 200 and r.json()["saved"]

    # B. close ERA completely and restart it
    clock.advance(days=2)
    client = restart_bridge(db_path, clock)

    # C. open a NEW conversation and ask a related question
    ctx = client.post("/v1/memory/context", headers=H, json={
        "message": "Which database are we using for Atlas?", "conversation_id": "conv-2"}).json()

    # D. ERA retrieves the saved information
    assert ctx["enabled"] and "SQLite" in ctx["text"] and ctx["project_id"] == "atlas"

    # E. correct it; subsequent conversations (even after another restart) use the corrected version
    fix = client.post("/v1/memory/ingest", headers=H, json={
        "user_message": "Actually, we decided to use PostgreSQL for the Atlas storage.", "conversation_id": "conv-2"}).json()
    assert fix["superseded"]
    client = restart_bridge(db_path, clock)
    ctx3 = client.post("/v1/memory/context", headers=H, json={
        "message": "Which database are we using for Atlas?", "conversation_id": "conv-3"}).json()
    assert "PostgreSQL" in ctx3["text"] and "SQLite" not in ctx3["text"]


PHASE = textwrap.dedent(
    """
    import json, sys
    from fastapi.testclient import TestClient
    import bridge_server
    from config import settings
    H = {"X-Era-Bridge-Key": settings.BRIDGE_API_KEY}
    c = TestClient(bridge_server.app)
    phase = sys.argv[1]
    if phase == "tell":
        out = c.post("/v1/memory/ingest", headers=H, json={"user_message": "My name is Shavan. I'm building Atlas, and we decided to use SQLite for Atlas.", "conversation_id": "one"}).json()
    elif phase == "ask":
        out = c.post("/v1/memory/context", headers=H, json={"message": "what database does Atlas use?", "conversation_id": "two"}).json()
    elif phase == "correct":
        out = c.post("/v1/memory/ingest", headers=H, json={"user_message": "Actually, we decided to use Postgres for Atlas.", "conversation_id": "two"}).json()
    elif phase == "ask_again":
        out = c.post("/v1/memory/context", headers=H, json={"message": "what database does Atlas use?", "conversation_id": "three"}).json()
    print("@@" + json.dumps(out))
    """
)


def _run_phase(phase, db, tmp_path):
    env = {**os.environ, "ERA_MEMORY_DB_PATH": str(db), "BRIDGE_API_KEY": settings.BRIDGE_API_KEY, "PYTHONPATH": str(BRIDGE_ROOT)}
    script = tmp_path / "phase.py"
    script.write_text(PHASE)
    p = subprocess.run([sys.executable, str(script), phase], cwd=BRIDGE_ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.split("@@")[-1])


def test_acceptance_with_real_separate_processes(tmp_path):
    db = tmp_path / "proc_memory.db"
    assert _run_phase("tell", db, tmp_path)["saved"]
    ask = _run_phase("ask", db, tmp_path)                        # fresh interpreter, new conversation
    assert "SQLite" in ask["text"]
    assert _run_phase("correct", db, tmp_path)["superseded"]
    again = _run_phase("ask_again", db, tmp_path)                # another fresh interpreter
    assert "Postgres" in again["text"] and "SQLite" not in again["text"]


# --------------------------------------------------------------- API surface --
def test_memory_routes_require_the_api_key(client):
    assert client.get("/v1/memory").status_code == 401
    assert client.post("/v1/memory/context", json={"message": "hi there"}).status_code == 401
    assert client.get("/v1/memory", headers={"X-Era-Bridge-Key": "wrong"}).status_code == 401


def test_memory_routes_are_local_only(tmp_path, db_path, clock):
    bridge_server.memory_manager = MemoryManager(MemoryConfig(db_path=db_path), clock=clock)
    remote = TestClient(bridge_server.app, client=("203.0.113.9", 5555))
    assert remote.get("/v1/memory", headers=H).status_code == 403
    assert remote.get("/v1/status").status_code == 200           # non-memory routes are unchanged


def test_full_crud_search_and_project_endpoints(client):
    r = client.post("/v1/memory", headers=H, json={"content": "Atlas needs offline mode", "type": "project",
                                                   "category": "note", "project": "Atlas"})
    assert r.status_code == 201
    mid = r.json()["memory"]["id"]
    assert client.get(f"/v1/memory/{mid}", headers=H).json()["content"] == "Atlas needs offline mode"
    assert [m["id"] for m in client.get("/v1/memory", headers=H, params={"type": "project"}).json()["memories"]] == [mid]
    assert client.get("/v1/memory/search", headers=H, params={"q": "offline mode"}).json()["results"][0]["id"] == mid
    up = client.patch(f"/v1/memory/{mid}", headers=H, json={"content": "Atlas needs offline mode and sync"})
    assert up.status_code == 200 and "sync" in up.json()["content"]
    summ = client.get("/v1/memory/projects/Atlas/summary", headers=H).json()
    assert summ["memory_count"] == 1 and "offline mode and sync" in summ["summary"]
    assert client.get("/v1/memory/projects", headers=H).json()["projects"][0]["id"] == "atlas"
    assert client.delete(f"/v1/memory/{mid}", headers=H).status_code == 200
    assert client.get(f"/v1/memory/{mid}", headers=H).status_code == 404
    assert client.delete(f"/v1/memory/{mid}", headers=H).status_code == 404


def test_settings_endpoint_toggles_and_persists(client, db_path, clock):
    assert client.get("/v1/memory/settings", headers=H).json()["auto_memory"] is True
    assert client.post("/v1/memory/settings", headers=H, json={"auto_memory": False}).json()["auto_memory"] is False
    client = restart_bridge(db_path, clock)
    assert client.get("/v1/memory/settings", headers=H).json()["auto_memory"] is False
    ctx = client.post("/v1/memory/context", headers=H, json={"message": "anything at all"}).json()
    assert ctx["enabled"] is False


def test_clear_all_needs_explicit_confirmation(client):
    client.post("/v1/memory", headers=H, json={"content": "temporary note to wipe"})
    assert client.delete("/v1/memory", headers=H).status_code == 400
    assert client.get("/v1/memory", headers=H).json()["count"] == 1
    assert client.delete("/v1/memory", headers=H, params={"confirm": "true"}).json()["deleted"] == 1
    assert client.get("/v1/memory", headers=H).json()["count"] == 0


def test_api_privacy_behaviour(client):
    secret = client.post("/v1/memory", headers=H, json={"content": "the password is hunter2hunter2"})
    assert secret.status_code == 422 and secret.json()["detail"]["code"] == "secret_not_storable"
    assert "hunter2" not in secret.text
    need_opt_in = client.post("/v1/memory", headers=H, json={"content": "I have diabetes"})
    assert need_opt_in.status_code == 422 and need_opt_in.json()["detail"]["categories"] == ["health"]
    ok = client.post("/v1/memory", headers=H, json={"content": "I have diabetes", "allow_sensitive": True})
    assert ok.status_code == 201 and "diabetes" not in ok.text                       # masked in the response
    mid = ok.json()["memory"]["id"]
    assert "diabetes" not in client.get("/v1/memory", headers=H).text
    assert "diabetes" in client.get(f"/v1/memory/{mid}", headers=H, params={"reveal": "true"}).text
    ing = client.post("/v1/memory/ingest", headers=H, json={"user_message": "My card number is 4111 1111 1111 1111", "conversation_id": "z"}).json()
    assert ing["held_back"] == ["financial"] and "4111" not in json.dumps(ing)


# ------------------------------------------- existing Bridge is unaffected ----
def test_existing_execute_flow_still_works_alongside_memory(client):
    assert client.get("/v1/status").status_code == 200
    assert "open_notepad" in client.get("/v1/commands", headers=H).json()["commands"]
    r = client.post("/v1/execute", headers=H, json={"command": "open_notepad"})
    assert r.json()["state"] == "success"
    r = client.post("/v1/execute", headers=H, json={"command": "shutdown_pc"})
    assert r.json()["state"] == "not_enabled"


def test_broken_memory_database_never_blocks_commands(tmp_path, clock):
    bad = tmp_path / "broken.db"
    bad.write_bytes(b"not a database" * 100)
    bridge_server.session_store = SessionStore(db_path=tmp_path / "s.db")
    bridge_server.agent_client = FakeAgentClient()
    bridge_server.memory_manager = MemoryManager(MemoryConfig(db_path=bad), clock=clock)
    c = TestClient(bridge_server.app)
    assert c.post("/v1/execute", headers=H, json={"command": "open_calculator"}).json()["state"] == "success"
    ctx = c.post("/v1/memory/context", headers=H, json={"message": "hello there friend"})
    assert ctx.status_code == 200 and ctx.json()["text"] == "" and ctx.json()["error"]
    assert c.post("/v1/memory/ingest", headers=H, json={"user_message": "My name is Shavan", "conversation_id": "x"}).status_code == 200
    assert c.post("/v1/memory", headers=H, json={"content": "some note here"}).status_code == 503
    assert c.get("/v1/memory", headers=H).status_code == 503


def test_memory_calls_do_not_block_command_execution(client):
    """Memory work runs off the event loop: interleave many memory calls with commands."""
    for i in range(20):
        client.post("/v1/memory/ingest", headers=H, json={"user_message": f"My favorite number is {i}", "conversation_id": "n"})
        assert client.post("/v1/execute", headers=H, json={"command": "open_notepad"}).json()["state"] == "success"


def test_local_agent_source_is_untouched():
    """Bridge must not import from or modify Local Agent; memory adds no coupling to it."""
    for py in (BRIDGE_ROOT / "era_memory").rglob("*.py"):
        text = py.read_text()
        assert "api_server" not in text and "system_commands" not in text
