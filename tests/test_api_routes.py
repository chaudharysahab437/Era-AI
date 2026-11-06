from __future__ import annotations

from fastapi.testclient import TestClient

from app import create_app
from vision.models import VisionMode


def _client(settings):
    app = create_app(settings)
    return TestClient(app, base_url="http://127.0.0.1"), app


def test_health_requires_no_key(settings):
    c, _ = _client(settings)
    r = c.get("/v1/health")
    assert r.status_code == 200


def test_status_requires_key(settings):
    c, _ = _client(settings)
    assert c.get("/v1/vision/status").status_code == 401
    assert c.get("/v1/vision/status", headers={"X-Vision-Agent-Key": "wrong"}).status_code == 401
    r = c.get("/v1/vision/status", headers={"X-Vision-Agent-Key": settings.vision_agent_api_key})
    assert r.status_code == 200


def test_vision_is_off_by_default(settings):
    c, _ = _client(settings)
    r = c.get("/v1/vision/status", headers={"X-Vision-Agent-Key": settings.vision_agent_api_key})
    assert r.json()["mode"] == VisionMode.OFF.value


def test_non_loopback_host_header_rejected(settings):
    c, _ = _client(settings)
    r = c.get("/v1/health", headers={"Host": "evil.example.com"})
    assert r.status_code == 400


def test_diagnostics_returns_200_even_without_real_capture(settings):
    c, _ = _client(settings)
    r = c.get("/v1/vision/diagnostics", headers={"X-Vision-Agent-Key": settings.vision_agent_api_key})
    assert r.status_code == 200
    body = r.json()
    assert "capture_ok" in body and "gemini_provider_configured" in body


def test_context_endpoint_empty_by_default(settings):
    c, _ = _client(settings)
    r = c.get("/v1/vision/context", headers={"X-Vision-Agent-Key": settings.vision_agent_api_key})
    assert r.status_code == 200
    assert r.json()["current_state"] is None


def test_exclusions_add_and_list(settings):
    c, _ = _client(settings)
    headers = {"X-Vision-Agent-Key": settings.vision_agent_api_key}
    c.post("/v1/vision/privacy/exclusions", json={"process_name": "keepass.exe"}, headers=headers)
    r = c.get("/v1/vision/privacy/exclusions", headers=headers)
    assert "keepass.exe" in r.json()["processes"]


def test_agent_refuses_non_loopback_host_at_startup():
    from config import Settings

    try:
        Settings(host="0.0.0.0", vision_agent_api_key="x", gemini_api_key="y")
        assert False, "expected ValueError for non-loopback host"
    except ValueError:
        pass


def test_vision_enabled_on_start_cannot_be_overridden():
    from config import Settings

    s = Settings(vision_agent_api_key="x", gemini_api_key="y")
    assert s.vision_enabled_on_start is False
