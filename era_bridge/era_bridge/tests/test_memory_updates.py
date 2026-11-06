"""Corrections, duplicates, conflicts, edits, on/off switch."""

from __future__ import annotations

import pytest

from era_memory import MemoryConfig, MemoryManager, MemoryNotFoundError, MemoryType


def _active_text(mm):
    return [m.content for m in mm.list(limit=200)]


def test_user_correction_supersedes_the_old_fact(mm):
    mm.ingest_turn("I live in Agra.", conversation_id="c1")
    r = mm.ingest_turn("Actually, I live in Pune now.", conversation_id="c2")
    assert len(r.superseded) == 1
    texts = _active_text(mm)
    assert "User lives in Pune." in texts and "User lives in Agra." not in texts
    old = mm.get(r.superseded[0]["old"])
    assert old.status == "superseded" and old.superseded_by == r.superseded[0]["new"]     # history kept, not active
    ctx = mm.build_context("where do I live?", "c3").text
    assert "Pune" in ctx and "Agra" not in ctx


def test_correction_with_not_x_keeps_only_the_new_value(mm):
    mm.ingest_turn("My name is Sam.", conversation_id="a")
    mm.ingest_turn("No, my name is Shavan, not Sam.", conversation_id="b")
    assert "User's name is Shavan." in _active_text(mm)
    assert not any("Sam" in t for t in _active_text(mm))


def test_newer_explicit_statement_wins_without_correction_wording(mm):
    mm.ingest_turn("My laptop is a ThinkPad.", conversation_id="a")
    mm.ingest_turn("My laptop is a MacBook.", conversation_id="b")
    assert _active_text(mm) == ["User's laptop is a MacBook."]


def test_slotless_correction_replaces_the_closest_older_memory(mm):
    mm.ingest_turn("We decided to use SQLite for the Atlas storage.", conversation_id="a")
    r = mm.ingest_turn("Actually, we decided to use Postgres for the Atlas storage.", conversation_id="b")
    assert r.superseded
    texts = " ".join(_active_text(mm))
    assert "Postgres" in texts and "SQLite" not in texts


def test_duplicates_are_not_saved_twice(mm):
    a = mm.ingest_turn("My favorite color is teal.", conversation_id="a")
    b = mm.ingest_turn("my favorite color is teal!", conversation_id="b")
    assert len(a.saved) == 1 and b.saved == [] and len(b.updated) == 1
    mems = mm.list()
    assert len(mems) == 1 and mems[0].times_seen == 2


def test_explicit_save_of_existing_content_is_a_duplicate(mm):
    first = mm.save("Standup is at 9:30 every weekday")
    again = mm.save("standup is at 9:30 every weekday.")
    assert again.action == "duplicate" and again.memory.id == first.memory.id
    assert len(mm.list()) == 1


def test_questions_and_device_commands_are_not_saved(mm):
    r = mm.ingest_turn("What is the capital of France? Open notepad please. I want to open chrome.", conversation_id="c")
    assert r.saved == [] and mm.list() == []


def test_weak_claim_against_confirmed_fact_is_flagged_as_conflict_not_overwritten(mm):
    mm.ingest_turn("I live in Agra.", conversation_id="a")
    r = mm.ingest_turn("I think I live in Delhi.", conversation_id="b")            # hedged -> low confidence
    assert len(r.conflicts) == 1
    ctx = mm.build_context("where do I live", "c")
    assert ctx.conflicts and "CONFLICTING" in ctx.text                               # ERA is told to ask
    assert "Agra" in ctx.text and "Delhi" in ctx.text


def test_conflict_can_be_resolved_by_the_user(mm):
    mm.ingest_turn("I live in Agra.", conversation_id="a")
    weak = mm.ingest_turn("I think I live in Delhi.", conversation_id="b").conflicts[0]
    mm.resolve_conflict(weak)                                                        # user: Delhi is right
    assert _active_text(mm) == ["User lives in Delhi."]
    assert not mm.build_context("where do I live", "c").conflicts


def test_edit_updates_content_and_index(mm):
    mid = mm.save("The API port is 8000").memory.id
    mm.update(mid, content="The API port is 8100")
    assert mm.get(mid).content == "The API port is 8100"
    hits = mm.search("api port 8100")
    assert hits and hits[0].memory.id == mid
    assert mm.get(mid).confidence >= 0.9


def test_edit_missing_memory_raises(mm):
    with pytest.raises(MemoryNotFoundError):
        mm.update("mem_missing", content="x y z")


def test_disabling_auto_memory_stops_retrieval_and_extraction(mm):
    mm.ingest_turn("My favorite color is teal.", conversation_id="a")
    mm.set_auto(False)
    assert mm.build_context("favorite color?", "b").enabled is False
    r = mm.ingest_turn("My favorite fruit is mango.", conversation_id="b")
    assert r.enabled is False and r.saved == []
    assert len(mm.list()) == 1                                                        # nothing new was saved
    assert mm.save("Explicit saves still work while auto is off").action == "created"  # user-controlled
    mm.set_auto(True)
    assert "teal" in mm.build_context("favorite color?", "b").text


def test_auto_switch_persists_across_restart(make_manager):
    make_manager().set_auto(False)
    assert make_manager().auto_enabled() is False


def test_master_switch_in_config_wins(make_manager):
    off = make_manager(enabled=False)
    assert off.ingest_turn("My name is Shavan.", conversation_id="a").enabled is False
    assert off.set_auto(True) is False


def test_broken_llm_hook_does_not_break_extraction(make_manager):
    from era_memory.memory_extractor import MemoryExtractor

    def boom(_m, _a):
        raise RuntimeError("model offline")

    mm = make_manager()
    mm.extractor = MemoryExtractor(llm_hook=boom)
    assert mm.ingest_turn("My name is Shavan.", conversation_id="a").saved


def test_llm_hook_output_passes_the_same_privacy_filter(make_manager):
    from era_memory.memory_extractor import MemoryExtractor

    hook = lambda m, a: [  # noqa: E731
        {"content": "User is learning Rust", "confidence": 0.99},
        {"content": "User takes metformin for diabetes"},
        {"content": "api key is sk-abcdefghijklmnopqrstuv"},
    ]
    mm = make_manager()
    mm.extractor = MemoryExtractor(llm_hook=hook)
    r = mm.ingest_turn("ok cool thanks a lot", conversation_id="a")
    contents = _active_text(mm)
    assert contents == ["User is learning Rust"] and "health" in r.held_back and r.secrets_blocked == 1
    assert mm.list()[0].confidence <= 0.7                     # model output is never 'confirmed'


def test_conversation_summary_and_episode_on_close(mm):
    for text in ["I am working on the invoice parser.", "We decided to use regex first for the invoice parser.",
                 "I still need to handle multi page invoices.", "The blocker is the scanned PDFs quality."]:
        mm.ingest_turn(text, "Sure.", conversation_id="long")
    out = mm.close_conversation("long")
    assert out["summary"].startswith("Discussed:") and out["episode_memory_id"]
    assert mm.get(out["episode_memory_id"]).type is MemoryType.EPISODIC
    again = mm.close_conversation("long")                    # closing twice must not duplicate the episode
    assert len([m for m in mm.list(types=["episodic"])]) == 1
