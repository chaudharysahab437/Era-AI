"""WebSocket event stream at /v1/events.

Browsers cannot set custom headers on a WebSocket handshake, so
authentication happens via the first text message sent by the client
immediately after the connection opens:

    {"api_key": "<VISION_AGENT_API_KEY>"}

The server waits up to 5 seconds for that message; if it does not arrive,
or the key is wrong, the socket is closed with code 4401. On success the
server immediately sends the recent-event replay buffer, then streams new
events as they are published.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("era_vision.websocket")

router = APIRouter()


@router.websocket("/v1/events")
async def events_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    auth = websocket.app.state.auth
    events = websocket.app.state.event_bus

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        payload = json.loads(raw)
        token = payload.get("api_key")
    except (asyncio.TimeoutError, json.JSONDecodeError, KeyError):
        await websocket.close(code=4401, reason="Expected {'api_key': ...} within 5s")
        return

    if not auth.check_token(token):
        await websocket.close(code=4401, reason="Invalid api_key")
        return

    await websocket.send_json({"event": "AUTH_OK", "data": {}})
    for envelope in events.recent(limit=20):
        await websocket.send_json(envelope.model_dump(mode="json"))

    queue = events.subscribe()
    try:
        while True:
            envelope = await queue.get()
            await websocket.send_json(envelope.model_dump(mode="json"))
    except WebSocketDisconnect:
        pass
    finally:
        events.unsubscribe(queue)
