"""API-key authentication.

Authenticates callers of this Vision Agent (ERA / Bridge / a developer
using curl) using `VISION_AGENT_API_KEY`. This is a completely different
secret from `GEMINI_API_KEY` (see config.py) and this module never touches
the Gemini key.

REST: header `X-Vision-Agent-Key: <key>`.
WebSocket: browsers cannot set custom headers on a WS handshake, so the key
is sent as the first text message immediately after the socket opens (see
api/websocket.py), which is what a native WS client (not just a browser)
can also do easily.

`GET /v1/health` is intentionally exempt so a caller (or a human with a
browser) can confirm the process is alive without a key, matching the
Bridge's existing pattern for its own `/v1/status`.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from config import Settings

_UNAUTHENTICATED_DETAIL = "Missing or invalid X-Vision-Agent-Key header."


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


class ApiKeyAuth:
    """Callable FastAPI dependency, constructed once with Settings so it
    always checks against the currently-configured key."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def __call__(self, x_vision_agent_key: str | None = Header(default=None)) -> None:
        expected = self._settings.vision_agent_api_key
        if not expected:
            # Fail closed: an agent with no key configured refuses all
            # authenticated calls rather than accepting anything.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="VISION_AGENT_API_KEY is not configured on this agent.",
            )
        if not x_vision_agent_key or not constant_time_equals(x_vision_agent_key, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHENTICATED_DETAIL)

    def check_token(self, token: str | None) -> bool:
        expected = self._settings.vision_agent_api_key
        if not expected or not token:
            return False
        return constant_time_equals(token, expected)
