from __future__ import annotations

import time

import pytest
from PIL import Image

from config import Settings
from vision.capture import Frame
from vision.models import ScreenState, WindowInfo
from vision.providers.base import VisionProvider


def make_frame(width: int = 320, height: int = 180, fill=(10, 20, 30)) -> Frame:
    img = Image.new("RGB", (width, height), color=fill)
    return Frame(image=img, width=width, height=height, captured_at=time.time())


class FakeCaptureWorker:
    """Stands in for vision.capture.CaptureWorker in tests that must not
    touch a real display."""

    def __init__(self, frames: list[Frame] | None = None, fail: bool = False):
        self._frames = frames or [make_frame()]
        self._idx = 0
        self.fail = fail
        self.closed = False

    def grab(self, timeout: float = 5.0) -> Frame:
        if self.fail:
            from vision.capture import CaptureError

            raise CaptureError("synthetic failure")
        frame = self._frames[min(self._idx, len(self._frames) - 1)]
        self._idx += 1
        return frame

    def close(self) -> None:
        self.closed = True


class FakeWindowDetector:
    def __init__(self, windows: list[WindowInfo] | None = None):
        self._windows = windows or [WindowInfo(title="Notepad", process_name="notepad.exe", pid=123)]
        self._idx = 0

    @property
    def backend_available(self) -> bool:
        return True

    def get_active_window(self) -> WindowInfo:
        w = self._windows[min(self._idx, len(self._windows) - 1)]
        self._idx += 1
        return w


class FakeProvider(VisionProvider):
    name = "fake"

    def __init__(self, configured: bool = True, result: ScreenState | None = None, error: Exception | None = None):
        self._configured = configured
        self._result = result or ScreenState(application="Notepad", summary="A text editor is open.", confidence=0.9)
        self._error = error
        self.calls = 0

    def is_configured(self) -> bool:
        return self._configured

    def analyze(self, image, *, prompt, roi_image=None) -> ScreenState:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._result


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        vision_agent_api_key="test-vision-key",
        gemini_api_key="test-gemini-key",
        log_dir=str(tmp_path / "logs"),
        context_ttl_seconds=600.0,
        max_gemini_calls_per_hour=120,
        min_seconds_between_analysis=0.0,
        debounce_seconds=0.0,
    )
