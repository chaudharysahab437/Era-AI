from __future__ import annotations

from tests.conftest import FakeCaptureWorker, FakeProvider, FakeWindowDetector, make_frame
from vision.analyzer import Analyzer
from vision.context_manager import ContextManager
from vision.events import EventBus
from vision.models import VisionMode, WindowInfo
from vision.service import VisionService


def _make_service(settings, frames=None, windows=None, provider=None):
    provider = provider or FakeProvider()
    analyzer = Analyzer(provider)  # type: ignore[arg-type]
    context = ContextManager(settings.context_history_size, settings.context_ttl_seconds)
    events = EventBus()
    wd = FakeWindowDetector(windows)
    svc = VisionService(
        settings=settings,
        analyzer=analyzer,
        context=context,
        events=events,
        window_detector=wd,
        capture_worker_factory=lambda: FakeCaptureWorker(frames),
    )
    return svc, context, events, provider


def test_starts_off(settings):
    svc, *_ = _make_service(settings)
    assert svc.mode == VisionMode.OFF


def test_start_stop_lifecycle(settings):
    svc, *_ = _make_service(settings)
    svc.start()
    assert svc.mode == VisionMode.RUNNING
    svc.stop()
    assert svc.mode == VisionMode.OFF


def test_observe_once_analyzes_on_window_change(settings):
    svc, context, events, provider = _make_service(
        settings,
        windows=[WindowInfo(title="Notepad", process_name="notepad.exe", pid=1)],
    )
    svc._capture_worker = FakeCaptureWorker([make_frame()])
    svc._observe_once()
    assert provider.calls == 1
    assert context.get_current() is not None


def test_excluded_window_skips_analysis(settings):
    svc, context, events, provider = _make_service(
        settings,
        windows=[WindowInfo(title="", process_name="keepass.exe", pid=1, is_excluded=True)],
    )
    svc._capture_worker = FakeCaptureWorker([make_frame()])
    svc._observe_once()
    assert provider.calls == 0
    assert context.get_current() is None


def test_capture_failure_does_not_crash_loop(settings):
    svc, context, events, provider = _make_service(settings)
    svc._capture_worker = FakeCaptureWorker(fail=True)
    svc._observe_once()  # should not raise
    assert provider.calls == 0


def test_provider_error_recorded_not_raised(settings):
    from vision.providers.base import ProviderResponseError

    provider = FakeProvider(error=ProviderResponseError("boom"))
    svc, context, events, _ = _make_service(settings, provider=provider)
    svc._capture_worker = FakeCaptureWorker([make_frame()])
    svc._observe_once()
    assert svc.last_error is not None
    assert context.get_current() is None


def test_no_action_capability_exists_on_service(settings):
    """Phase 2 safety invariant: VisionService must expose no click/type/
    system-control methods at all."""
    svc, *_ = _make_service(settings)
    forbidden = {"click", "type_text", "move_mouse", "press_key", "send_whatsapp", "shutdown", "restart"}
    assert not (forbidden & set(dir(svc)))
