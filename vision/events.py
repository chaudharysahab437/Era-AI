"""Simple in-process event bus.

Used by VisionService to publish lifecycle/state events, and by the
WebSocket route to relay them to connected clients (ERA / Bridge / a debug
tool). Also keeps a bounded replay buffer so a client that connects after
an event fired can still see recent history via GET, or a WS client can
request replay-since-seq in a later phase.

No external dependency (no Redis, no asyncio pubsub library) -- this is
intentionally minimal for a single-process, single-machine agent.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any

from vision.models import EventEnvelope

# Event name constants (Phase 2 subset of the full spec's list -- only
# perception events exist; no ACTION_* / TASK_* events since Phase 2 does
# not execute or plan actions).
SCREEN_CHANGED = "SCREEN_CHANGED"
WINDOW_CHANGED = "WINDOW_CHANGED"
APP_OPENED = "APP_OPENED"  # reserved for a future phase with reliable open/close tracking
APP_CLOSED = "APP_CLOSED"  # reserved, same reason
ERROR_DETECTED = "ERROR_DETECTED"
ANALYSIS_STARTED = "ANALYSIS_STARTED"
ANALYSIS_COMPLETED = "ANALYSIS_COMPLETED"
ANALYSIS_FAILED = "ANALYSIS_FAILED"
VISION_STARTED = "VISION_STARTED"
VISION_STOPPED = "VISION_STOPPED"
VISION_PAUSED = "VISION_PAUSED"
VISION_RESUMED = "VISION_RESUMED"


class EventBus:
    def __init__(self, buffer_size: int = 200) -> None:
        self._buffer: deque[EventEnvelope] = deque(maxlen=buffer_size)
        self._seq = 0
        self._lock = threading.Lock()
        self._subscribers: list[asyncio.Queue] = []
        self._subscribers_lock = threading.Lock()

    def publish(self, event: str, data: dict[str, Any] | None = None) -> EventEnvelope:
        with self._lock:
            self._seq += 1
            envelope = EventEnvelope(event=event, data=data or {}, timestamp=time.time(), seq=self._seq)
            self._buffer.append(envelope)

        with self._subscribers_lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(envelope)
            except asyncio.QueueFull:
                pass  # slow subscriber; drop rather than block publishers
        return envelope

    def recent(self, limit: int = 50) -> list[EventEnvelope]:
        with self._lock:
            return list(self._buffer)[-limit:]

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        with self._subscribers_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._subscribers_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)
