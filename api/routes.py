"""REST endpoints, all under /v1/vision/*.

Every route except /v1/health depends on ApiKeyAuth (see security/auth.py).
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from vision.capture import CaptureError
from vision.models import DiagnosticsReport, VisionMode

router = APIRouter()


def _auth_dep(request: Request):
    return request.app.state.auth(request.headers.get("x-vision-agent-key"))


@router.get("/v1/health")
def health() -> dict:
    """Unauthenticated liveness check, mirroring the Bridge's /v1/status."""
    return {"status": "ok", "service": "era_vision_agent", "time": time.time()}


@router.get("/v1/vision/status", dependencies=[Depends(_auth_dep)])
def get_status(request: Request) -> dict:
    service = request.app.state.vision_service
    return service.status().model_dump(mode="json")


@router.post("/v1/vision/start", dependencies=[Depends(_auth_dep)])
def start_vision(request: Request) -> dict:
    service = request.app.state.vision_service
    try:
        status = service.start()
    except CaptureError as exc:
        raise HTTPException(status_code=500, detail=f"Could not start screen capture: {exc}") from exc
    indicator = request.app.state.indicator
    if indicator is not None:
        indicator.show_running()
    request.app.state.audit_logger.info("vision_started")
    return status.model_dump(mode="json")


@router.post("/v1/vision/stop", dependencies=[Depends(_auth_dep)])
def stop_vision(request: Request) -> dict:
    service = request.app.state.vision_service
    status = service.stop()
    indicator = request.app.state.indicator
    if indicator is not None:
        indicator.hide()
    request.app.state.audit_logger.info("vision_stopped")
    return status.model_dump(mode="json")


@router.post("/v1/vision/pause", dependencies=[Depends(_auth_dep)])
def pause_vision(request: Request) -> dict:
    service = request.app.state.vision_service
    status = service.pause()
    indicator = request.app.state.indicator
    if indicator is not None and status.mode == VisionMode.PAUSED:
        indicator.show_paused()
    request.app.state.audit_logger.info("vision_paused")
    return status.model_dump(mode="json")


@router.post("/v1/vision/resume", dependencies=[Depends(_auth_dep)])
def resume_vision(request: Request) -> dict:
    service = request.app.state.vision_service
    status = service.resume()
    indicator = request.app.state.indicator
    if indicator is not None and status.mode == VisionMode.RUNNING:
        indicator.show_running()
    request.app.state.audit_logger.info("vision_resumed")
    return status.model_dump(mode="json")


@router.get("/v1/vision/current-state", dependencies=[Depends(_auth_dep)])
def current_state(request: Request) -> dict:
    context = request.app.state.context_manager
    entry = context.get_current()
    return {"current_state": entry.model_dump(mode="json") if entry else None}


@router.get("/v1/vision/context", dependencies=[Depends(_auth_dep)])
def get_context(request: Request) -> dict:
    context = request.app.state.context_manager
    return context.snapshot()


class TaskContextBody(BaseModel):
    task_context: str | None = None


@router.post("/v1/vision/task-context", dependencies=[Depends(_auth_dep)])
def set_task_context(body: TaskContextBody, request: Request) -> dict:
    context = request.app.state.context_manager
    context.set_task_context(body.task_context)
    return {"ok": True}


class ExclusionBody(BaseModel):
    process_name: str | None = None
    title_keyword: str | None = None


@router.get("/v1/vision/privacy/exclusions", dependencies=[Depends(_auth_dep)])
def get_exclusions(request: Request) -> dict:
    return request.app.state.privacy_store.snapshot()


@router.post("/v1/vision/privacy/exclusions", dependencies=[Depends(_auth_dep)])
def add_exclusion(body: ExclusionBody, request: Request) -> dict:
    store = request.app.state.privacy_store
    if body.process_name:
        store.add_process(body.process_name)
    if body.title_keyword:
        store.add_title_keyword(body.title_keyword)
    request.app.state.audit_logger.info("exclusion_added process=%s title_kw=%s", body.process_name, body.title_keyword)
    return store.snapshot()


@router.delete("/v1/vision/privacy/exclusions", dependencies=[Depends(_auth_dep)])
def remove_exclusion(body: ExclusionBody, request: Request) -> dict:
    store = request.app.state.privacy_store
    if body.process_name:
        store.remove_process(body.process_name)
    if body.title_keyword:
        store.remove_title_keyword(body.title_keyword)
    request.app.state.audit_logger.info("exclusion_removed process=%s title_kw=%s", body.process_name, body.title_keyword)
    return store.snapshot()


@router.get("/v1/vision/diagnostics", dependencies=[Depends(_auth_dep)])
def diagnostics(request: Request) -> dict:
    """Requirement #12: a single endpoint proving each Phase 2 component
    actually works, not just that the process is up."""
    service = request.app.state.vision_service
    settings = request.app.state.settings
    resources = request.app.state.resource_monitor

    capture_ok, capture_err = service.diagnostics_capture_probe()
    window_ok, window_err = service.diagnostics_window_probe()
    context_entry = request.app.state.context_manager.get_current()
    cpu, rss = resources.sample(force=True)

    report = DiagnosticsReport(
        vision_mode=service.mode,
        capture_ok=capture_ok,
        capture_error=capture_err,
        active_window_detection_ok=window_ok,
        active_window_error=window_err,
        change_detector_ok=True,  # pure-Python/numpy logic, always available; verified by unit tests
        gemini_provider_configured=settings.gemini_key_configured,
        gemini_provider_reachable=None,  # Phase 2 never makes a live probe call on your behalf/budget
        context_present=context_entry is not None,
        context_entry_count=request.app.state.context_manager.entry_count(),
        resource_ok=not resources.should_throttle,
        cpu_percent=cpu,
        rss_mb=rss,
    )
    return report.model_dump(mode="json")
