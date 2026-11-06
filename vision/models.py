"""Structured data models shared across the Vision Agent."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class VisionMode(str, Enum):
    OFF = "off"
    RUNNING = "running"
    PAUSED = "paused"


class WindowInfo(BaseModel):
    """Active/foreground window, from Win32 (or a stub off-Windows)."""

    title: str = ""
    process_name: str = ""
    pid: int | None = None
    is_excluded: bool = False


class UIElement(BaseModel):
    label: str
    element_type: str = Field(description="button, field, menu, tab, link, dialog, checkbox, dropdown, etc.")
    approx_location: str | None = Field(
        default=None, description="Coarse location, e.g. 'center-right', 'top toolbar'."
    )


class ScreenState(BaseModel):
    """Structured description of what is currently on screen.

    This is intentionally tolerant of a partially-conforming model response:
    every field has a safe default, and `analyzer.py` treats a state that
    fails validation as a failed observation rather than crashing the loop.
    """

    application: str = "unknown"
    window_title: str = ""
    summary: str = ""
    visible_elements: list[UIElement] = Field(default_factory=list)
    important_text: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    current_activity: str = ""
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class ContextEntry(BaseModel):
    """One entry in the rolling visual-context history."""

    entry_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = Field(default_factory=time.time)
    window: WindowInfo
    state: ScreenState
    trigger: str = Field(description="what caused this observation: window_change, pixel_change, manual, etc.")
    provider: str = ""

    def is_expired(self, ttl_seconds: float, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return (now - self.timestamp) > ttl_seconds


class ImportantEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = Field(default_factory=time.time)
    kind: str  # e.g. "error_detected", "app_opened", "app_closed"
    description: str


class ChangeResult(BaseModel):
    changed: bool
    changed_fraction: float = 0.0
    changed_bbox: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 in source-frame pixels
    reason: str = ""


class VisionStatus(BaseModel):
    mode: VisionMode
    started_at: float | None = None
    last_observation_at: float | None = None
    frames_captured: int = 0
    analyses_performed: int = 0
    analyses_this_hour: int = 0
    max_analyses_per_hour: int = 0
    excluded_active: bool = False
    provider: str = ""
    provider_configured: bool = False


class DiagnosticsReport(BaseModel):
    """Backing model for GET /v1/vision/diagnostics."""

    agent_running: bool = True
    vision_mode: VisionMode
    capture_ok: bool
    capture_error: str | None = None
    active_window_detection_ok: bool
    active_window_error: str | None = None
    change_detector_ok: bool
    gemini_provider_configured: bool
    gemini_provider_reachable: bool | None = None  # None = not tested (no live call made)
    context_present: bool
    context_entry_count: int
    resource_ok: bool
    cpu_percent: float | None = None
    rss_mb: float | None = None
    timestamp: float = Field(default_factory=time.time)


class EventEnvelope(BaseModel):
    event: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    seq: int = 0
