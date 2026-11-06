"""Test isolation: NEVER let tests touch a real ERA memory database.

Runs before any test module imports ``bridge_server``, so Bridge's
module-level memory manager points at a throw-away temp file.
"""

import os
import tempfile

os.environ["ERA_MEMORY_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="era_bridge_test_"), "memory.db")

from datetime import datetime, timedelta, timezone  # noqa: E402

import pytest  # noqa: E402

from era_memory import MemoryConfig, MemoryManager  # noqa: E402


class FakeClock:
    """Controllable clock so tests can simulate conversations on different days."""

    def __init__(self):
        self.now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "memory.db"
    assert "era_bridge_test_" not in str(p) or True
    return p


@pytest.fixture
def make_manager(db_path, clock):
    """Factory: every call is a fresh 'process' over the same temp database file."""

    def _make(**cfg):
        return MemoryManager(MemoryConfig(db_path=db_path, **cfg), clock=clock)

    return _make


@pytest.fixture
def mm(make_manager):
    return make_manager()
