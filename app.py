"""Builds the FastAPI application and wires every component together.

Vision is never started here. `Settings.vision_enabled_on_start` is always
False (enforced in config.py itself), and nothing in this module or in
`VisionService.__init__` calls `start()`. Vision only turns on in response
to an authenticated `POST /v1/vision/start` call.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from api.routes import router as rest_router
from api.websocket import router as ws_router
from config import Settings
from monitoring.logging_config import configure_logging
from monitoring.resources import ResourceMonitor
from security.auth import ApiKeyAuth
from security.indicator import IndicatorWindow
from security.privacy import PrivacyStore
from vision.analyzer import Analyzer
from vision.context_manager import ContextManager
from vision.events import EventBus
from vision.providers.gemini_provider import GeminiProvider
from vision.service import VisionService
from vision.windows import WindowDetector

logger = logging.getLogger("era_vision.app")


class LoopbackOnlyHostMiddleware:
    """Defence in depth: even though we bind the socket to 127.0.0.1, this
    rejects requests whose Host header doesn't look like localhost, which
    blocks a class of DNS-rebinding attacks against loopback services."""

    ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        host_header = headers.get(b"host", b"").decode("latin-1")
        host_only = host_header.split(":")[0]
        if host_only and host_only not in self.ALLOWED_HOSTS:
            response = JSONResponse({"detail": "Host not allowed."}, status_code=400)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def create_app(settings: Settings) -> FastAPI:
    configure_logging(
        settings.log_dir,
        settings.log_level,
        settings.log_max_bytes,
        settings.log_backup_count,
        secrets_to_redact=[settings.vision_agent_api_key, settings.gemini_api_key],
    )
    logger.info("Starting era_vision_agent with settings: %s", settings.summary())

    app = FastAPI(title="ERA Vision Agent", version="0.2.0-phase2")

    app.add_middleware(LoopbackOnlyHostMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["X-Vision-Agent-Key", "Content-Type"],
    )

    auth = ApiKeyAuth(settings)
    event_bus = EventBus(buffer_size=settings.event_buffer_size)
    context_manager = ContextManager(settings.context_history_size, settings.context_ttl_seconds)
    privacy_store = PrivacyStore(settings.excluded_process_names, settings.excluded_title_keywords)
    window_detector = WindowDetector(*privacy_store.as_tuples())
    resource_monitor = ResourceMonitor(settings.max_cpu_percent_sustained, settings.max_rss_mb, settings.resource_check_interval_seconds)

    provider = GeminiProvider(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        timeout_seconds=settings.gemini_timeout_seconds,
        max_retries=settings.gemini_max_retries,
    )
    analyzer = Analyzer(provider)

    vision_service = VisionService(
        settings=settings,
        analyzer=analyzer,
        context=context_manager,
        events=event_bus,
        window_detector=window_detector,
    )

    indicator = IndicatorWindow(f"http://{settings.host}:{settings.port}", settings.vision_agent_api_key)
    if not indicator.available:
        logger.info("Tkinter not available in this environment; the on-screen indicator is disabled (headless).")

    app.state.settings = settings
    app.state.auth = auth
    app.state.event_bus = event_bus
    app.state.context_manager = context_manager
    app.state.privacy_store = privacy_store
    app.state.resource_monitor = resource_monitor
    app.state.vision_service = vision_service
    app.state.indicator = indicator if indicator.available else None
    app.state.audit_logger = logging.getLogger("era_vision.audit")

    app.include_router(rest_router)
    app.include_router(ws_router)

    @app.on_event("shutdown")
    def _on_shutdown() -> None:
        # Always stop observing on process exit -- vision must not keep
        # running orphaned if the agent is killed/restarted.
        if vision_service.mode.value != "off":
            vision_service.stop()

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse({"detail": "Internal error."}, status_code=500)

    return app
