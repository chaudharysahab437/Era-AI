# Era Bridge

Sits between **Era** (your Gemini/Google AI Studio assistant) and
**Era Local Agent**. You only ever talk to Era — Era calls Bridge,
and Bridge is the only thing that talks to Local Agent, entirely
through Local Agent's existing, unmodified HTTP API.

**Scope of this build (Phase 1 + 2, as approved):**
- Full session engine, parameter collection, and confirmation flow — built and tested.
- Automatic Local Agent detection, background startup, and health-check waiting — built and tested.
- Only the **5 commands Local Agent already supports over HTTP without parameters** (`open_notepad`, `open_calculator`, `open_file_explorer`, `open_chrome`, `open_camera`) execute end-to-end today.
- The other 7 commands return a clear `not_enabled` state — Phase 3 (an explicitly separate, approved change to Local Agent's `/execute` contract) is what would light these up. Nothing about them is broken or hidden; Bridge tells you plainly why.

**Nothing in this project modifies Era Local Agent.** Bridge only calls `GET /status` and `POST /execute` — the two endpoints Local Agent already exposes — or launches Local Agent's own existing start command as a subprocess when it isn't running.

## The "zero manual startup" behavior

You never need to open a terminal for either service. Every call to `/v1/execute` or `/v1/confirm`:

1. Checks whether Local Agent is already responding (`GET /status`, ~1.5s timeout).
2. If not, acquires a lock (so two near-simultaneous requests can't both try to launch it), re-checks in case another request just brought it up, then launches Local Agent's start command in the background — no visible console window.
3. Polls `/status` every 0.5s until it responds or a startup timeout (default 20s) is reached.
4. Only then proceeds with your actual command. If Local Agent still isn't up after the timeout, you get an honest `agent_unreachable` state — never a silent hang.

A PID file (`agent.pid`) records the process Bridge itself launched, so if Bridge is asked to ensure the Agent is running again shortly after, it recognizes "that process is already coming up" instead of launching a duplicate.

Bridge itself starts automatically at logon via Task Scheduler (see below) — so between that and the above, nothing needs a manual step.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. **Configure where your Local Agent project lives.** This wasn't part of the uploaded project, so Bridge doesn't assume a path — set it via environment variables (create a `.env` file, or set these in your shell/Task Scheduler action):

   ```
   AGENT_PROJECT_DIR=C:\path\to\your\era_agent
   AGENT_START_COMMAND=python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
   BRIDGE_API_KEY=choose-a-fixed-secret-here
   ```

   If `AGENT_PROJECT_DIR` is left unset, auto-start is simply skipped (logged as a warning) — Bridge still works against an already-running Agent, it just can't launch it for you.

   If `BRIDGE_API_KEY` is left unset, Bridge generates a random one each restart and logs it — fine for a quick test, but set a fixed one before pointing Era's tool config at Bridge permanently.

## Running

Development (visible console):
```
python main.py
```

Production (no console window — what Task Scheduler uses):
```
run_bridge.pyw
```

Auto-start at logon:
```
python startup/install_task.py install
```
This registers a separate Task Scheduler entry (`EraBridgeService`) from whatever entry Local Agent might have — installing or removing one never affects the other. Uses `/RL LIMITED`, no admin prompt.

## API

Every route except `GET /v1/status` requires the header `X-Era-Bridge-Key: <your key>`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/status` | Bridge's own health, plus a live check of Local Agent (no auth) |
| GET | `/v1/commands` | Every command, its params, confirmation requirement, and whether it's `direct` (enabled) or `pending_agent_extension` |
| POST | `/v1/execute` | Start/continue a command conversation |
| POST | `/v1/confirm` | Confirm or cancel a command that returned `needs_confirmation` |
| GET | `/docs` | Swagger UI |

### Response states

`needs_input` · `needs_confirmation` · `success` · `failure` · `error` · `not_enabled` · `agent_unreachable` · `cancelled`

`not_enabled` and `agent_unreachable` are new in Bridge — `not_enabled` means the command exists on Local Agent but Bridge can't deliver its parameters yet (Phase 3); `agent_unreachable` means Bridge tried to reach and auto-start Local Agent and it didn't come up in time.

### Example — a working command

```
POST /v1/execute
{ "command": "open_notepad" }

-> { "state": "success", "session_id": "...", "message": "Notepad opened successfully.", "command": "open_notepad" }
```
If Local Agent wasn't already running, this same call transparently started it first — no separate step needed.

### Example — a not-yet-enabled command

```
POST /v1/execute
{ "command": "search_file" }

-> { "state": "not_enabled", "session_id": "...", "message": "'search_file' exists on Era Local Agent but isn't enabled through Era Bridge yet -- it needs an explicitly approved Local Agent update first (Phase 3).", "command": "search_file" }
```

## Persistent memory (`era_memory`)

ERA remembers across conversations **and restarts**. Memory lives in Bridge (Python), in a local SQLite
file (`data/memory.db`). Nothing leaves your machine, there are no extra API keys and **no new
dependencies** -- it uses only the standard library plus the packages Bridge already requires.

### Install / start (nothing new to install)

```bash
cd era_bridge/era_bridge          # the Bridge root (where bridge_server.py lives)
pip install -r requirements.txt   # unchanged
python main.py                    # unchanged -- memory starts with Bridge
# optional: run the tests
pip install -r requirements-dev.txt
python -m pytest -q               # existing Bridge tests + the new memory tests
```

`data/memory.db` is created automatically on first use. Tests always use a temporary database and never touch it.

### How the frontend must use it (mandatory two-call contract)

Bridge never calls an LLM -- Gemini Live runs in the browser -- so **the frontend drives memory with two calls per turn**:

| When | Call | What to do with the result |
|---|---|---|
| **Before** the user's message is sent to Gemini Live | `POST /v1/memory/context` | If `text` is non-empty, give it to the model as background (see below). Ignore it when empty. |
| **After** the turn is complete (user message + assistant reply known) | `POST /v1/memory/ingest` | Fire-and-forget is fine; it never needs to block the voice flow. |

Both need the header `X-Era-Bridge-Key: <BRIDGE_API_KEY>` and must come from this machine (`MEMORY_LOCAL_ONLY`).
Use one stable `conversation_id` per chat (a new id for each new chat); pass `project` only if the UI knows it
(otherwise ERA detects project names mentioned in the message).

```http
POST /v1/memory/context
{"message": "which database are we using for Atlas?", "conversation_id": "chat-2024-05-02-a"}
-> {"status":"ok","enabled":true,"available":true,"text":"[ERA MEMORY ...]","items":[...],
    "conflicts":[], "withheld_sensitive":0, "error":null}

POST /v1/memory/ingest
{"user_message":"...", "assistant_message":"...", "conversation_id":"chat-2024-05-02-a"}
-> {"status":"ok","saved":[...],"updated":[...],"superseded":[...],"conflicts":[...],
    "held_back":["health"],"secrets_blocked":0,"error":null}
```

**Illustrative frontend sketch (TypeScript -- not tested against the real frontend, which was not part of the
uploaded code; adapt the two hook points to however the app sends a turn to Gemini Live):**

```ts
const BRIDGE = "http://127.0.0.1:8100";
const H = { "Content-Type": "application/json", "X-Era-Bridge-Key": BRIDGE_KEY };

async function memoryContext(message: string, conversationId: string): Promise<string> {
  try {
    const r = await fetch(`${BRIDGE}/v1/memory/context`, { method: "POST", headers: H,
      body: JSON.stringify({ message, conversation_id: conversationId }), signal: AbortSignal.timeout(2500) });
    const d = await r.json();
    if (d.error) console.warn("ERA memory:", d.status, d.error);   // real cause, not "memory off"
    return d.text || "";
  } catch { return ""; }                                            // memory must never block the turn
}

// 1) BEFORE sending the user's message to Gemini Live:
const ctx = await memoryContext(userText, conversationId);
session.sendClientContent({ turns: ctx ? [ctx, userText] : [userText], turnComplete: true });
// (voice turns: call memoryContext() with the finished input transcription, or at turn start with the topic,
//  and deliver `ctx` as extra client content / system-instruction text -- the exact hook depends on your app)

// 2) AFTER the turn is complete:
fetch(`${BRIDGE}/v1/memory/ingest`, { method: "POST", headers: H, keepalive: true,
  body: JSON.stringify({ user_message: userText, assistant_message: assistantText, conversation_id: conversationId }) })
  .catch(() => {});
```

The context text tells the model to use items only if relevant, not to claim memories that are not listed,
to treat `(unverified)`/`(uncertain)` items as possibly wrong, and to ask the user when it contains a
`CONFLICTING memories` block.

### What `/v1/memory/context` returns (not a dump)

Only memories that clear a relevance gate (keyword overlap or semantic similarity) are returned, ranked by
relevance + recency + importance + confidence, capped by `MEMORY_MAX_RETRIEVED` (8) and
`MEMORY_MAX_CONTEXT_CHARS` (2000). Always included on top of that: up to 3 *communication* preferences
("keep answers short"), the active project's summary, and -- for "continue / where were we" -- unfinished
tasks, next steps and blockers.

### Statuses and errors (never a false "memory off")

`status` is one of `ok`, `disabled_by_user`, `disabled_by_config`, `unavailable` (database problem),
`timeout`, `internal_error`. On failure `/context` and `/ingest` still return HTTP 200 with
`available:false`, `enabled:null` and `error:"memory_unavailable: memory database unavailable (DatabaseError)"`
so ERA keeps working; `/settings`, `/stats` and the CRUD endpoints return HTTP 503 with `detail.code =
"memory_unavailable"` and the cause.

### API

All under `/v1/memory`, all require the Bridge key and a local caller.

| Endpoint | Purpose |
|---|---|
| `POST /context` · `POST /ingest` | per-turn retrieval / extraction (above) |
| `POST /` · `GET /` · `GET /{id}` · `PATCH /{id}` · `DELETE /{id}` | save · list (`type,category,project,q,include_history`) · read · edit · delete |
| `GET /search?q=` | ranked search |
| `GET /projects` · `GET /projects/{id}/summary` | project summaries across conversations/dates |
| `GET /conversations/{id}` · `POST /conversations/{id}/close` | conversation summary; closing writes an episodic memory |
| `GET /settings` · `POST /settings {"auto_memory": false}` | turn automatic memory on/off (persisted) |
| `POST /{id}/resolve` | choose the correct side of a flagged conflict |
| `DELETE /?confirm=true` | clear **all** memory |
| `GET /stats` · `POST /reembed` | health / re-embed after changing embedding provider |

Memory types: `working`, `episodic`, `semantic`, `preference`, `project`. Each memory has id, type, content,
created/updated time, source, importance, confidence (`confirmed` / `inferred` / `uncertain`), status
(`active` / `superseded`) and optional project/conversation links.

### Privacy rules

* **Secrets** (API keys, passwords, tokens, private keys) are never stored -- not automatically and not even
  on an explicit save. Only a count is reported.
* **Sensitive personal data** (health, financial, government IDs) is never saved automatically; `/ingest`
  reports only the *category* (`held_back`). Save it deliberately with `POST /v1/memory` and
  `allow_sensitive=true`. Such memories are masked in responses unless `reveal=true`, and are **not sent to
  the AI model** unless you also set `share_with_model=true`.
* Raw turns are kept only briefly (`MEMORY_RETENTION_DAYS`, default 14), with secret/sensitive sentences
  already removed. Logs contain ids and counts only, never memory content.
* Deleting a memory also deletes its superseded history.

### Corrections, duplicates, conflicts

A newer explicit statement replaces the old value ("Actually I live in Pune now"); the old one is kept only as
inactive history. Re-stating a fact updates the existing memory instead of adding another. A hedged or weaker
claim that contradicts a confirmed fact is *not* allowed to overwrite it: both are flagged and ERA is told to
ask you (`POST /{id}/resolve` settles it).

### Embeddings

Default: a built-in, dependency-free hashing embedding (offline, no key). It matches spelling/word-form
similarity, **not true synonyms** ("database" vs "SQLite"). Every stored vector records its
`provider / model / dimension`; vectors are only compared with vectors from the identical configuration.
After switching provider, old memories are found by keyword until `POST /v1/memory/reembed`. A real provider
can be plugged in with `EmbeddingService([CallableEmbedding(name, fn)])` when constructing `MemoryManager`.

### Settings (environment variables, all optional)

`ERA_MEMORY_DB_PATH`, `MEMORY_ENABLED`, `MEMORY_AUTO_DEFAULT`, `MEMORY_MAX_CONTEXT_CHARS`, `MEMORY_MAX_RETRIEVED`,
`MEMORY_RETENTION_DAYS`, `MEMORY_STORE_TURNS`, `MEMORY_LOCAL_ONLY`.

### Known limitations

* Extraction is rule-based (English patterns); it will miss facts phrased unusually and cannot judge
  importance like an LLM. An optional hook (`MemoryExtractor(llm_hook=...)`) exists for a future upgrade.
* No natural-language "forget that" command yet -- use the delete endpoint.
* The database is a plain local file (not encrypted); use disk encryption for sensitive notes.
* Retrieval scans up to 5000 vectors per query (~100 ms at 3000 memories measured).

## Tests

```
pip install -r requirements-dev.txt
pytest tests/ -v
```

28 tests, all passing in this environment: session engine logic (unknown/disabled/param-collection/confirmation transitions), session store persistence, the agent client's health-check and auto-start behavior (including the no-duplicate-process guarantee, tested directly against the PID-liveness check), and full API integration tests with a fake Agent standing in for the real one. A live end-to-end smoke test (Bridge + a minimal stand-in HTTP server playing Local Agent's role) was also run manually to confirm the real request/response cycle over the network, not just mocks.

## Project structure

(New for memory: `era_memory/` package, `api/routes_memory.py`, `data/memory.db` at runtime, `tests/test_memory_*.py`.)


```
era_bridge/
├── main.py                    # dev entry point (visible console)
├── run_bridge.pyw               # production entry point (windowless)
├── bridge_server.py              # FastAPI app assembly, lifespan startup check
├── config/
│   └── settings.py                # all configuration, env-driven, no hardcoded paths
├── core/
│   ├── logging_config.py           # rotating file + console logging
│   ├── session_store.py            # SQLite multi-turn conversation state (Bridge-side)
│   ├── command_catalog.py          # metadata for all 12 Local Agent commands; enabled flags
│   ├── session_engine.py           # pure state-machine logic (unit-testable, no I/O)
│   ├── agent_client.py             # HTTP client + auto-start/health-check/dedup logic
│   └── auth.py                      # API key dependency
├── api/
│   ├── routes_execute.py            # POST /v1/execute, POST /v1/confirm
│   └── routes_meta.py               # GET /v1/status, GET /v1/commands
├── startup/
│   └── install_task.py               # Task Scheduler registration for Bridge itself
├── tests/
├── requirements.txt
├── requirements-dev.txt
└── pytest.ini
```

## Phase 3 (not part of this build)

When you're ready, extending Local Agent's `POST /execute` to accept an
optional `params` field (additive, non-breaking to the 5 commands
that already work) plus a `/confirm`-equivalent is the only change
needed on the Agent side. On Bridge's side, flipping `enabled=True`
for the relevant entries in `core/command_catalog.py` is the only
change needed here — the session engine, parameter collection, and
confirmation flow already fully support them; they're just switched
off until that Local Agent change is separately approved and made.
