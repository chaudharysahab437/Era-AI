"""On-screen "ERA VISION ON" indicator.

Requirement #4: the indicator must provide an immediate way to stop
observation. It is a small always-on-top Tkinter window that:
  - is shown only while vision is RUNNING or PAUSED (started by the
    VISION_STARTED event, hidden by VISION_STOPPED),
  - changes text/color between RUNNING ("ERA VISION ON") and PAUSED
    ("ERA VISION PAUSED"),
  - has a single "Stop" button that calls POST /v1/vision/stop on the
    local agent itself (loopback), so a click always works regardless of
    what called start() in the first place.

Tkinter needs a display. On a headless machine (this sandbox, CI, a
Windows box running as a service with no interactive session) it will not
be available; IndicatorWindow.available reports that and the rest of the
agent works normally without it -- the indicator is a UX safety affordance
on top of the real stop mechanism (the API and the event bus), not a
replacement for it.
"""

from __future__ import annotations

import logging
import threading

import requests

logger = logging.getLogger("era_vision.indicator")

try:
    import tkinter as tk
except ImportError:  # pragma: no cover
    tk = None  # type: ignore[assignment]


class IndicatorWindow:
    available = tk is not None

    def __init__(self, local_base_url: str, vision_agent_api_key: str) -> None:
        self._base_url = local_base_url.rstrip("/")
        self._key = vision_agent_api_key
        self._root: "tk.Tk | None" = None
        self._label: "tk.Label | None" = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def _call_stop(self) -> None:
        try:
            requests.post(
                f"{self._base_url}/v1/vision/stop",
                headers={"X-Vision-Agent-Key": self._key},
                timeout=5.0,
            )
        except Exception:
            logger.exception("Indicator failed to call /v1/vision/stop")

    def _build(self) -> None:
        self._root = tk.Tk()
        self._root.title("ERA Vision")
        self._root.attributes("-topmost", True)
        self._root.geometry("240x60+40+40")
        self._root.resizable(False, False)
        self._label = tk.Label(self._root, text="ERA VISION ON", bg="#c0392b", fg="white", font=("Segoe UI", 12, "bold"))
        self._label.pack(fill="both", expand=True, side="top")
        stop_btn = tk.Button(self._root, text="Stop watching", command=self._on_stop_clicked)
        stop_btn.pack(fill="x", side="bottom")
        self._ready.set()
        self._root.mainloop()

    def _on_stop_clicked(self) -> None:
        threading.Thread(target=self._call_stop, daemon=True).start()

    def show_running(self) -> None:
        if not self.available:
            return
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._build, daemon=True)
            self._thread.start()
            self._ready.wait(timeout=2.0)
        self._set_text("ERA VISION ON", "#c0392b")

    def show_paused(self) -> None:
        self._set_text("ERA VISION PAUSED", "#e67e22")

    def hide(self) -> None:
        if self._root is not None:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass
            self._root = None
            self._label = None

    def _set_text(self, text: str, color: str) -> None:
        if self._root is not None and self._label is not None:
            try:
                self._root.after(0, lambda: self._label.config(text=text, bg=color))
            except Exception:
                pass
