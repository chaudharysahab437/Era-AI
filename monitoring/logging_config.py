"""Structured logging setup.

Two rotating log files under `log_dir`:
  vision_agent.log  - general application log (JSON lines)
  audit.log         - one line per privacy/security-relevant event
                       (vision start/stop/pause, exclusion changes, auth
                       failures) -- deliberately does NOT include screen
                       content, only the fact that an event occurred.

`RedactingFilter` defends against a key accidentally ending up in a log
message (e.g. via an exception's `str(exc)` containing a URL with the key
in a query string) by masking any substring that matches a configured
secret value.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path


class RedactingFilter(logging.Filter):
    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for secret in self._secrets:
            if secret in msg:
                msg = msg.replace(secret, "***REDACTED***")
        record.msg = msg
        record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(
    log_dir: str,
    level: str,
    max_bytes: int,
    backup_count: int,
    secrets_to_redact: list[str],
) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("era_vision")
    root.setLevel(level)
    root.handlers.clear()

    redactor = RedactingFilter(secrets_to_redact)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    stream_handler.addFilter(redactor)
    root.addHandler(stream_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "vision_agent.log", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    file_handler.addFilter(redactor)
    root.addHandler(file_handler)

    audit_logger = logging.getLogger("era_vision.audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False
    audit_handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "audit.log", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    audit_handler.setFormatter(JsonFormatter())
    audit_handler.addFilter(redactor)
    audit_logger.addHandler(audit_handler)
    audit_logger.addHandler(stream_handler)

    return root
