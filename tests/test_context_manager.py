from __future__ import annotations

import time

from vision.context_manager import ContextManager
from vision.models import ContextEntry, ScreenState, WindowInfo


def entry(app="Notepad", proc="notepad.exe", errors=None, ts=None):
    e = ContextEntry(
        window=WindowInfo(title=app, process_name=proc, pid=1),
        state=ScreenState(application=app, summary="x", confidence=0.9, errors=errors or []),
        trigger="manual",
    )
    if ts is not None:
        e.timestamp = ts
    return e


def test_add_observation_sets_current_and_history():
    cm = ContextManager(history_size=5, ttl_seconds=600)
    cm.add_observation(entry())
    assert cm.get_current() is not None
    assert cm.entry_count() == 1


def test_history_size_bounded():
    cm = ContextManager(history_size=3, ttl_seconds=600)
    for i in range(5):
        cm.add_observation(entry())
    assert cm.entry_count() == 3


def test_app_change_produces_important_event():
    cm = ContextManager(history_size=5, ttl_seconds=600)
    cm.add_observation(entry(proc="notepad.exe"))
    events = cm.add_observation(entry(proc="chrome.exe"))
    kinds = [e.kind for e in events]
    assert "app_changed" in kinds


def test_error_produces_important_event():
    cm = ContextManager(history_size=5, ttl_seconds=600)
    events = cm.add_observation(entry(errors=["Traceback: something broke"]))
    kinds = [e.kind for e in events]
    assert "error_detected" in kinds


def test_ttl_expiry_drops_old_entries():
    cm = ContextManager(history_size=5, ttl_seconds=1.0)
    old = entry(ts=time.time() - 10)
    cm.add_observation(old)
    assert cm.get_current() is None
    assert cm.entry_count() == 0


def test_task_context_set_and_read():
    cm = ContextManager(history_size=5, ttl_seconds=600)
    cm.set_task_context("build a website")
    assert cm.get_task_context() == "build a website"


def test_snapshot_is_json_serializable_shape():
    cm = ContextManager(history_size=5, ttl_seconds=600)
    cm.add_observation(entry())
    snap = cm.snapshot()
    assert "current_state" in snap and "recent_states" in snap and "important_events" in snap
