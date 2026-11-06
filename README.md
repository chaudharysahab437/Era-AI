# ERA Vision Agent — Phase 2 (Observe-Only)

Standalone perception service for ERA. **Phase 2 only observes the
screen** — no mouse, keyboard, WhatsApp, shutdown, or any control action
exists in this codebase. Vision is **OFF by default** and binds to
**127.0.0.1 only**.

## 1. Requirements

- Windows 10/11
- Python 3.10+ (uses `str | None` syntax)
- (Optional) Tesseract-OCR engine on PATH, for the `/ocr`-adjacent OCR helper

## 2. Setup

```bat
cd era_vision_agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```
VISION_AGENT_API_KEY=<generate below>
GEMINI_API_KEY=<from https://aistudio.google.com/app/apikey>
```

Generate a Vision Agent key (this is NOT your Gemini key — keep them separate):

```bat
python scripts\store_gemini_key.py --gen-vision-key
```

Paste the printed value into `.env` as `VISION_AGENT_API_KEY`.

## 3. Start / Stop

```bat
python main.py
```

Runs on `http://127.0.0.1:8200`. Vision does **not** start automatically —
you must call `/v1/vision/start`. Stop the process with Ctrl+C, or leave it
running and just call `/v1/vision/stop` to halt observation (process stays
up). The on-screen "ERA VISION ON" indicator also has a **Stop watching**
button for an immediate kill switch (requires a desktop session; not
available when run headless/as a service with no interactive session).

Windowless launch (no console): double-click `run_vision_agent.pyw`, or
register it with Task Scheduler.

## 4. Enable / disable vision

```bat
curl -X POST http://127.0.0.1:8200/v1/vision/start  -H "X-Vision-Agent-Key: <key>"
curl -X POST http://127.0.0.1:8200/v1/vision/pause   -H "X-Vision-Agent-Key: <key>"
curl -X POST http://127.0.0.1:8200/v1/vision/resume  -H "X-Vision-Agent-Key: <key>"
curl -X POST http://127.0.0.1:8200/v1/vision/stop    -H "X-Vision-Agent-Key: <key>"
```

## 5. Example API calls

```bat
curl http://127.0.0.1:8200/v1/health

curl http://127.0.0.1:8200/v1/vision/status -H "X-Vision-Agent-Key: <key>"

curl http://127.0.0.1:8200/v1/vision/diagnostics -H "X-Vision-Agent-Key: <key>"

curl http://127.0.0.1:8200/v1/vision/current-state -H "X-Vision-Agent-Key: <key>"

curl http://127.0.0.1:8200/v1/vision/context -H "X-Vision-Agent-Key: <key>"

curl -X POST http://127.0.0.1:8200/v1/vision/privacy/exclusions ^
  -H "X-Vision-Agent-Key: <key>" -H "Content-Type: application/json" ^
  -d "{\"process_name\": \"keepass.exe\"}"
```

`GET /v1/vision/diagnostics` is the single "is everything working" check:
it reports capture, active-window detection, change detector, Gemini
config, and current context state in one JSON response.

WebSocket: connect to `ws://127.0.0.1:8200/v1/events`, then immediately
send `{"api_key": "<key>"}` as the first message (browsers can't set
custom WS headers). You'll get `AUTH_OK`, a short replay buffer, then a
live event stream (`VISION_STARTED`, `WINDOW_CHANGED`, `ANALYSIS_STARTED`,
`ANALYSIS_COMPLETED`, `ANALYSIS_FAILED`, `ERROR_DETECTED`, etc.).

## 6. Expected output

After `/v1/vision/start`, within a few seconds of screen activity,
`GET /v1/vision/current-state` returns something like:

```json
{
  "current_state": {
    "window": {"title": "Untitled - Notepad", "process_name": "notepad.exe"},
    "state": {
      "application": "Notepad",
      "summary": "A blank text editor window is open.",
      "visible_elements": [...],
      "errors": [],
      "confidence": 0.85
    },
    "trigger": "window_changed"
  }
}
```

## 7. Privacy defaults

- Screenshots are **never written to disk** unless you explicitly set
  `STORE_SCREENSHOTS_TO_DISK=true` (not recommended; off by default).
- Visual context auto-expires after `CONTEXT_TTL_SECONDS` (default 600s).
- Excluded apps/titles (password managers, private/incognito browsing by
  default) are never analyzed or sent to Gemini.
- Only `VISION_AGENT_API_KEY` authenticates callers of this agent; only
  `GEMINI_API_KEY` authenticates this agent to Google. Never interchange them.

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `VISION_AGENT_API_KEY is not set` on startup | Generate one (`scripts\store_gemini_key.py --gen-vision-key`), put it in `.env`. |
| `401 Unauthorized` on every call | Wrong or missing `X-Vision-Agent-Key` header. |
| `400 Host not allowed` | You're calling via LAN IP/hostname, not `127.0.0.1`/`localhost`. This agent is loopback-only by design. |
| Screen analysis never happens | Check `GET /v1/vision/diagnostics` — `gemini_provider_configured` must be `true` and `capture_ok` must be `true`. |
| Capture keeps failing | Some windows (DRM video, some elevated windows, exclusive-fullscreen games) legitimately can't be captured — this is a Windows limitation, not a bug. |
| Runs under Task Scheduler but dies immediately | Make sure `run_vision_agent.pyw` is used (not a plain `.py` under `pythonw.exe`), and that `.env` values are reachable (or set as real environment variables) for the account the task runs as. |
| `pytesseract`/OCR not working | Tesseract engine must be installed separately (not just the pip package) and on PATH, or set `TESSERACT_CMD` in `.env`. OCR is optional — nothing else depends on it. |

## 9. What this Phase does NOT do

No clicking, typing, mouse movement, WhatsApp, file changes, or system
commands. No Bridge integration yet (standalone agent). No task
planning/execution loop. Those are later phases and require your explicit
approval, per the plan.

## 10. Tests

```bat
pip install -r requirements-dev.txt
pytest -q
```

43 tests, all runnable without Windows or a live Gemini key (capture and
Win32 window detection use fakes/stubs in tests; only real Windows capture
and a live Gemini call need verification on your machine — see final
report).
