"""VisionService: owns the observation loop and its lifecycle.

State machine: OFF -> RUNNING <-> PAUSED -> OFF

  start()  OFF -> RUNNING: spins up the capture worker + loop thread.
  pause()  RUNNING -> PAUSED: loop thread keeps running but skips capture
                                entirely (no frames are grabbed while paused).
  resume() PAUSED -> RUNNING.
  stop()   RUNNING/PAUSED -> OFF: tears down the capture worker and thread.

Vision is OFF at construction time and stays OFF until start() is called
explicitly via the API -- nothing in this class or its constructor begins
observing on its own (Phase 2 requirement).

The loop itself:
    OBSERVE (get active window) -> DETECT (grab frame, diff vs previous)
        -> POLICY (should we call the vision model?)
        -> [if yes] CAPTURE (encode) -> ANALYZE (provider call)
        -> UPDATE CONTEXT -> publish events

This entire loop performs no mouse/keyboard/system actions of any kind --
it only reads the screen and calls the configured vision provider.
"""

from __future__ import annotations

import logging
import threading
import time

from config import Settings
from vision.analyzer import Analyzer
from vision.capture import CaptureError, CaptureWorker, Frame, is_probably_blank
from vision.change_detector import ChangeDetector
from vision.context_manager import ContextManager
from vision.events import (
    ANALYSIS_COMPLETED,
    ANALYSIS_FAILED,
    ANALYSIS_STARTED,
    ERROR_DETECTED,
    EventBus,
    VISION_PAUSED,
    VISION_RESUMED,
    VISION_STARTED,
    VISION_STOPPED,
    WINDOW_CHANGED,
)
from vision.models import ContextEntry, VisionMode, VisionStatus, WindowInfo
from vision.policy import AnalysisPolicy
from vision.providers.base import ProviderError
from vision.windows import WindowDetector

logger = logging.getLogger("era_vision.service")


class VisionService:
    def __init__(
        self,
        settings: Settings,
        analyzer: Analyzer,
        context: ContextManager,
        events: EventBus,
        window_detector: WindowDetector | None = None,
        capture_worker_factory=CaptureWorker,
    ) -> None:
        self._settings = settings
        self._analyzer = analyzer
        self._context = context
        self._events = events
        self._window_detector = window_detector or WindowDetector(
            settings.excluded_process_names, settings.excluded_title_keywords
        )
        self._capture_worker_factory = capture_worker_factory

        self._change_detector = ChangeDetector(
            cols=settings.grid_cols, rows=settings.grid_rows, change_threshold=settings.change_threshold
        )
        self._policy = AnalysisPolicy(
            max_calls_per_hour=settings.max_gemini_calls_per_hour,
            min_seconds_between_analysis=settings.min_seconds_between_analysis,
        )

        self._mode = VisionMode.OFF
        self._mode_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture_worker: CaptureWorker | None = None

        self._started_at: float | None = None
        self._last_observation_at: float | None = None
        self._frames_captured = 0
        self._analyses_performed = 0
        self._last_window: WindowInfo | None = None
        self._last_frame_for_diagnostics: Frame | None = None
        self._last_error: str | None = None

    # -- lifecycle ------------------------------------------------------

    @property
    def mode(self) -> VisionMode:
        with self._mode_lock:
            return self._mode

    def start(self) -> VisionStatus:
        with self._mode_lock:
            if self._mode != VisionMode.OFF:
                return self.status()
            try:
                self._capture_worker = self._capture_worker_factory()
            except CaptureError as exc:
                self._last_error = str(exc)
                raise
            self._change_detector.reset()
            self._stop_event.clear()
            self._pause_event.clear()
            self._mode = VisionMode.RUNNING
            self._started_at = time.time()
            self._thread = threading.Thread(target=self._run_loop, name="vision-loop", daemon=True)
            self._thread.start()
        self._events.publish(VISION_STARTED, {})
        logger.info("Vision started")
        return self.status()

    def pause(self) -> VisionStatus:
        with self._mode_lock:
            if self._mode == VisionMode.RUNNING:
                self._mode = VisionMode.PAUSED
                self._pause_event.set()
                self._events.publish(VISION_PAUSED, {})
                logger.info("Vision paused")
        return self.status()

    def resume(self) -> VisionStatus:
        with self._mode_lock:
            if self._mode == VisionMode.PAUSED:
                self._mode = VisionMode.RUNNING
                self._pause_event.clear()
                self._events.publish(VISION_RESUMED, {})
                logger.info("Vision resumed")
        return self.status()

    def stop(self) -> VisionStatus:
        """This is also what the emergency 'ERA VISION ON' indicator calls
        (via its click handler -> HTTP POST /v1/vision/stop) for an
        immediate way to halt observation."""
        with self._mode_lock:
            if self._mode == VisionMode.OFF:
                return self.status()
            self._stop_event.set()
            self._pause_event.clear()
            thread = self._thread
            worker = self._capture_worker
            self._mode = VisionMode.OFF
        if thread is not None:
            thread.join(timeout=5.0)
        if worker is not None:
            try:
                worker.close()
            except Exception:
                logger.exception("Error closing capture worker")
        with self._mode_lock:
            self._thread = None
            self._capture_worker = None
        self._events.publish(VISION_STOPPED, {})
        logger.info("Vision stopped")
        return self.status()

    # -- loop -------------------------------------------------------------

    def _run_loop(self) -> None:
        settings = self._settings
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.25)
                continue

            loop_started = time.time()
            try:
                self._observe_once()
            except Exception:
                logger.exception("Unhandled error in vision loop iteration")

            elapsed = time.time() - loop_started
            interval = (
                settings.idle_capture_interval_seconds
                if self._change_detector.is_volatile
                else settings.capture_interval_seconds
            )
            sleep_for = max(0.05, interval - elapsed)
            self._stop_event.wait(timeout=sleep_for)

    def _observe_once(self) -> None:
        assert self._capture_worker is not None
        window = self._window_detector.get_active_window()

        try:
            frame = self._capture_worker.grab()
        except CaptureError as exc:
            self._last_error = str(exc)
            logger.warning("Capture failed this cycle: %s", exc)
            return

        self._frames_captured += 1
        self._last_observation_at = time.time()
        self._last_frame_for_diagnostics = frame

        if is_probably_blank(frame):
            logger.debug("Captured frame looks blank (DRM/exclusive-fullscreen content?)")

        window_changed = self._last_window is None or (
            self._last_window.process_name != window.process_name or self._last_window.title != window.title
        )
        if window_changed and self._last_window is not None:
            self._events.publish(WINDOW_CHANGED, {"from": self._last_window.process_name, "to": window.process_name})
        self._last_window = window

        if window.is_excluded:
            # Privacy: do not run change detection or analysis against an
            # excluded application's content. Still record that *something*
            # is active, without any title/content detail.
            self._change_detector.reset()
            return

        change = self._change_detector.detect(frame)
        self._events.publish("SCREEN_CHANGED_RAW", {"changed": change.changed, "fraction": change.changed_fraction})

        decision = self._policy.decide(
            window_changed=window_changed,
            pixel_changed=change.changed,
            is_volatile=self._change_detector.is_volatile,
        )
        if not decision.should_analyze:
            return

        if settings_debounce := self._settings.debounce_seconds:
            time.sleep(settings_debounce)

        self._policy.record_call()
        self._events.publish(ANALYSIS_STARTED, {"trigger": decision.reason})
        try:
            state = self._analyzer.analyze(
                frame,
                window,
                trigger=decision.reason,
                roi_bbox=change.changed_bbox,
                task_context=self._context.get_task_context(),
            )
        except ProviderError as exc:
            self._last_error = str(exc)
            self._events.publish(ANALYSIS_FAILED, {"error": str(exc)})
            logger.warning("Vision provider call failed: %s", exc)
            return

        self._analyses_performed += 1
        entry = ContextEntry(window=window, state=state, trigger=decision.reason, provider=self._analyzer.provider_name)
        new_events = self._context.add_observation(entry)
        self._events.publish(ANALYSIS_COMPLETED, {"entry_id": entry.entry_id, "application": state.application})
        for ev in new_events:
            if ev.kind == "error_detected":
                self._events.publish(ERROR_DETECTED, {"description": ev.description})

    # -- status / diagnostics --------------------------------------------

    def status(self) -> VisionStatus:
        return VisionStatus(
            mode=self.mode,
            started_at=self._started_at,
            last_observation_at=self._last_observation_at,
            frames_captured=self._frames_captured,
            analyses_performed=self._analyses_performed,
            analyses_this_hour=self._policy.calls_this_hour(),
            max_analyses_per_hour=self._settings.max_gemini_calls_per_hour,
            excluded_active=bool(self._last_window and self._last_window.is_excluded),
            provider=self._analyzer.provider_name,
            provider_configured=self._analyzer.provider_configured,
        )

    def diagnostics_capture_probe(self) -> tuple[bool, str | None]:
        """Used by GET /v1/vision/diagnostics. Performs a single real
        capture attempt using a short-lived capture worker if vision is
        currently OFF (so diagnostics work even before start() is called),
        or reuses the running worker if vision is already on."""
        worker = self._capture_worker
        owns_worker = False
        if worker is None:
            try:
                worker = self._capture_worker_factory()
                owns_worker = True
            except CaptureError as exc:
                return False, str(exc)
        try:
            worker.grab()
            return True, None
        except CaptureError as exc:
            return False, str(exc)
        finally:
            if owns_worker:
                try:
                    worker.close()
                except Exception:
                    logger.exception("Error closing diagnostic capture worker")

    def diagnostics_window_probe(self) -> tuple[bool, str | None]:
        try:
            self._window_detector.get_active_window()
            if not self._window_detector.backend_available:
                return False, "Not running on Windows: using stub window backend (dev environment)."
            return True, None
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def window_backend_available(self) -> bool:
        return self._window_detector.backend_available
