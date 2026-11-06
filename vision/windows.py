"""Active window and process detection.

Implemented with raw `ctypes` calls into user32/kernel32/psapi so the
Vision Agent does not need `pywin32` as a dependency. On non-Windows
platforms (used for development/testing here), a stub implementation is
used instead and clearly reports itself as such via `WindowsBackend.available`.

DPI awareness: `enable_dpi_awareness()` should be called once, early in
process startup (see main.py), so that window/monitor coordinates are not
scaled by the Windows DPI virtualization layer, which would otherwise
silently distort click/box coordinates on HiDPI displays. Phase 2 does not
click anything, but future phases depend on these coordinates being
correct, so it is set up correctly now.
"""

from __future__ import annotations

import ctypes
import logging
import platform
from dataclasses import dataclass

from vision.models import WindowInfo

logger = logging.getLogger("era_vision.windows")

IS_WINDOWS = platform.system() == "Windows"


def enable_dpi_awareness() -> None:
    """Best-effort. Safe no-op on non-Windows or if the call fails."""
    if not IS_WINDOWS:
        return
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
        except Exception:
            logger.warning("Could not set DPI awareness; window coordinates may be scaled.")


@dataclass
class RawWindowInfo:
    hwnd: int
    title: str
    pid: int
    process_name: str


class WindowsBackend:
    """Real Win32 backend. Only usable when IS_WINDOWS is True."""

    available = IS_WINDOWS

    def __init__(self) -> None:
        if not IS_WINDOWS:
            raise RuntimeError("WindowsBackend requires Windows")
        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32
        self.psapi = ctypes.windll.psapi

    def get_foreground_window(self) -> RawWindowInfo | None:
        hwnd = self.user32.GetForegroundWindow()
        if not hwnd:
            return None
        length = self.user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        self.user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value

        pid = ctypes.c_ulong()
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        process_name = self._process_name_for_pid(pid.value)
        return RawWindowInfo(hwnd=hwnd, title=title, pid=pid.value, process_name=process_name)

    def _process_name_for_pid(self, pid: int) -> str:
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h_process = self.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h_process:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(260)
            size = ctypes.c_ulong(260)
            # QueryFullProcessImageNameW is preferred over the deprecated
            # psapi GetModuleBaseName for processes we don't have full
            # access to.
            ok = self.kernel32.QueryFullProcessImageNameW(h_process, 0, buf, ctypes.byref(size))
            if not ok:
                return ""
            full_path = buf.value
            return full_path.rsplit("\\", 1)[-1].lower()
        finally:
            self.kernel32.CloseHandle(h_process)


class StubBackend:
    """Used on non-Windows platforms (development/testing environment).

    Returns a clearly-labelled synthetic window so tests and the rest of
    the pipeline can run, without pretending to have real desktop access.
    """

    available = False

    def get_foreground_window(self) -> RawWindowInfo | None:
        return RawWindowInfo(
            hwnd=0,
            title="[non-Windows stub: no real window]",
            pid=0,
            process_name="stub",
        )


def get_backend() -> WindowsBackend | StubBackend:
    if IS_WINDOWS:
        try:
            return WindowsBackend()
        except Exception:
            logger.exception("Failed to initialize Win32 backend; falling back to stub")
    return StubBackend()


class WindowDetector:
    def __init__(self, excluded_processes: tuple[str, ...], excluded_title_keywords: tuple[str, ...]) -> None:
        self._backend = get_backend()
        self._excluded_processes = {p.lower() for p in excluded_processes}
        self._excluded_title_keywords = [k.lower() for k in excluded_title_keywords]

    @property
    def backend_available(self) -> bool:
        return self._backend.available

    def get_active_window(self) -> WindowInfo:
        raw = self._backend.get_foreground_window()
        if raw is None:
            return WindowInfo(title="", process_name="", pid=None, is_excluded=False)
        excluded = self._is_excluded(raw.process_name, raw.title)
        return WindowInfo(
            title="" if excluded else raw.title,
            process_name=raw.process_name,
            pid=raw.pid,
            is_excluded=excluded,
        )

    def _is_excluded(self, process_name: str, title: str) -> bool:
        if process_name.lower() in self._excluded_processes:
            return True
        title_lower = title.lower()
        return any(kw in title_lower for kw in self._excluded_title_keywords)
