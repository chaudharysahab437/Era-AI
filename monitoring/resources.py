"""CPU/RAM sampling for this process, with a simple throttle signal.

`psutil` is an optional-but-recommended dependency; if unavailable, this
module reports readings as `None` rather than crashing the agent (the
resource check is a nice-to-have, not on the critical path).
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger("era_vision.resources")

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


class ResourceMonitor:
    def __init__(self, max_cpu_percent: float, max_rss_mb: float, check_interval_seconds: float = 5.0) -> None:
        self._max_cpu = max_cpu_percent
        self._max_rss_mb = max_rss_mb
        self._interval = check_interval_seconds
        self._process = psutil.Process(os.getpid()) if psutil else None
        self._lock = threading.Lock()
        self._last_cpu: float | None = None
        self._last_rss_mb: float | None = None
        self._last_checked = 0.0
        self._throttled = False
        if self._process is not None:
            self._process.cpu_percent(interval=None)  # prime the sampler

    @property
    def available(self) -> bool:
        return self._process is not None

    def sample(self, force: bool = False) -> tuple[float | None, float | None]:
        if self._process is None:
            return None, None
        now = time.time()
        with self._lock:
            if not force and (now - self._last_checked) < self._interval:
                return self._last_cpu, self._last_rss_mb
            try:
                cpu = self._process.cpu_percent(interval=None)
                rss_mb = self._process.memory_info().rss / (1024 * 1024)
            except Exception:
                logger.exception("Resource sampling failed")
                return self._last_cpu, self._last_rss_mb
            self._last_cpu = cpu
            self._last_rss_mb = rss_mb
            self._last_checked = now
            self._throttled = cpu > self._max_cpu or rss_mb > self._max_rss_mb
            if self._throttled:
                logger.warning("Resource usage above configured limits: cpu=%.1f%% rss=%.1fMB", cpu, rss_mb)
            return cpu, rss_mb

    @property
    def should_throttle(self) -> bool:
        return self._throttled
