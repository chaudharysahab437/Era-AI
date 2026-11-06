"""Era Bridge configuration -- entirely environment-driven.

Nothing here hardcodes a path to your Era Local Agent installation,
because that path is specific to your machine and wasn't part of the
uploaded project. Set AGENT_PROJECT_DIR (and AGENT_START_COMMAND, if
your setup differs from the default) via environment variables, or a
.env file loaded before this module is imported.

Every setting has a safe default so Bridge still starts without any
configuration -- auto-start of the Agent is simply skipped (with a
logged warning) until AGENT_PROJECT_DIR is set.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

# --- Bridge's own network settings ---------------------------------------
BRIDGE_HOST: str = os.environ.get("BRIDGE_HOST", "127.0.0.1")
BRIDGE_PORT: int = int(os.environ.get("BRIDGE_PORT", "8100"))

# --- Authentication ---------------------------------------------------------
# If you don't set BRIDGE_API_KEY yourself, Bridge generates a random one
# each time it starts and logs it clearly, so it's never silently
# unauthenticated -- but a fresh key each restart means anything calling
# Bridge (e.g. Era's tool config) needs to be updated too. Set your own
# fixed key via the BRIDGE_API_KEY environment variable to avoid that.
BRIDGE_API_KEY: str = os.environ.get("BRIDGE_API_KEY") or secrets.token_hex(16)
_API_KEY_WAS_GENERATED: bool = "BRIDGE_API_KEY" not in os.environ

# --- Era Local Agent connection ---------------------------------------------
AGENT_BASE_URL: str = os.environ.get("AGENT_BASE_URL", "http://127.0.0.1:8000")

# Path to your era_agent project folder (the one containing api_server.py).
# Required only for auto-start; leave unset to disable auto-start entirely.
_agent_dir_env = os.environ.get("AGENT_PROJECT_DIR")
AGENT_PROJECT_DIR: Path | None = Path(_agent_dir_env) if _agent_dir_env else None

# The command Bridge runs (with AGENT_PROJECT_DIR as its working directory)
# to bring the Agent up. Override this if you use a virtualenv's python,
# a different port, etc.
AGENT_START_COMMAND: str = os.environ.get(
    "AGENT_START_COMMAND", "python -m uvicorn api_server:app --host 127.0.0.1 --port 8000"
)

# --- Health check / auto-start tuning ---------------------------------------
AGENT_STATUS_CHECK_TIMEOUT: float = float(os.environ.get("AGENT_STATUS_CHECK_TIMEOUT", "1.5"))
AGENT_EXECUTE_TIMEOUT: float = float(os.environ.get("AGENT_EXECUTE_TIMEOUT", "10.0"))
AGENT_STARTUP_TIMEOUT: float = float(os.environ.get("AGENT_STARTUP_TIMEOUT", "20.0"))
AGENT_HEALTH_POLL_INTERVAL: float = float(os.environ.get("AGENT_HEALTH_POLL_INTERVAL", "0.5"))

# If true, Bridge also tries to warm up the Agent once at Bridge's own
# startup (in addition to the on-demand check before every request).
# Off by default -- auto-start remains purely on-demand, as requested.
AGENT_AUTOSTART_ON_BRIDGE_BOOT: bool = os.environ.get("AGENT_AUTOSTART_ON_BRIDGE_BOOT", "false").lower() == "true"

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
AGENT_PID_FILE: Path = PROJECT_ROOT / "agent.pid"
AGENT_PROCESS_LOG_FILE: Path = PROJECT_ROOT / "logs" / "agent_process_output.log"

# --- Persistent memory (era_memory) -------------------------------------------
# Everything is local by default: a SQLite file on this machine, no network
# calls, no extra API keys. See README.md ("Persistent memory").
MEMORY_DB_PATH: Path = Path(os.environ.get("ERA_MEMORY_DB_PATH") or (PROJECT_ROOT / "data" / "memory.db"))
# Master switch. False = automatic retrieval/extraction is off and cannot be
# turned on at runtime (explicit save/list/delete endpoints still work).
MEMORY_ENABLED: bool = os.environ.get("MEMORY_ENABLED", "true").lower() == "true"
# Default for the runtime "automatic memory" toggle (POST /v1/memory/settings
# overrides it and the choice is persisted in the memory database).
MEMORY_AUTO_DEFAULT: bool = os.environ.get("MEMORY_AUTO_DEFAULT", "true").lower() == "true"
MEMORY_MAX_CONTEXT_CHARS: int = int(os.environ.get("MEMORY_MAX_CONTEXT_CHARS", "2000"))
MEMORY_MAX_RETRIEVED: int = int(os.environ.get("MEMORY_MAX_RETRIEVED", "8"))
# Keep short-lived data (working memory, raw conversation turns) this long.
MEMORY_RETENTION_DAYS: int = int(os.environ.get("MEMORY_RETENTION_DAYS", "14"))
MEMORY_STORE_TURNS: bool = os.environ.get("MEMORY_STORE_TURNS", "true").lower() == "true"
# Only accept memory-management requests that originate from this machine.
MEMORY_LOCAL_ONLY: bool = os.environ.get("MEMORY_LOCAL_ONLY", "true").lower() == "true"
