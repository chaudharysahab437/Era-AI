"""Decides WHEN to call the (expensive) vision provider.

This is deliberately separated from capture/change-detection so the
"should we analyze right now" logic is unit-testable without any image
data, and so future phases can swap in a smarter policy without touching
capture or the provider.

Rules, in order:
  1. Hard hourly budget (`max_gemini_calls_per_hour`) is never exceeded.
  2. A minimum cooldown between analyses is always respected.
  3. A window/app change always qualifies once the cooldown has passed.
  4. A pixel change only qualifies if the screen isn't currently "volatile"
     (see ChangeDetector.is_volatile) -- for volatile screens we fall back
     to the idle cadence rather than analyzing on every sample.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class PolicyDecision:
    should_analyze: bool
    reason: str


class AnalysisPolicy:
    def __init__(self, max_calls_per_hour: int, min_seconds_between_analysis: float) -> None:
        self.max_calls_per_hour = max_calls_per_hour
        self.min_seconds_between_analysis = min_seconds_between_analysis
        self._call_timestamps: deque[float] = deque()
        self._last_analysis_at: float | None = None

    def _calls_in_last_hour(self, now: float) -> int:
        cutoff = now - 3600.0
        while self._call_timestamps and self._call_timestamps[0] < cutoff:
            self._call_timestamps.popleft()
        return len(self._call_timestamps)

    def calls_this_hour(self, now: float | None = None) -> int:
        return self._calls_in_last_hour(now if now is not None else time.time())

    def decide(
        self,
        *,
        window_changed: bool,
        pixel_changed: bool,
        is_volatile: bool,
        now: float | None = None,
    ) -> PolicyDecision:
        now = now if now is not None else time.time()

        if self._calls_in_last_hour(now) >= self.max_calls_per_hour:
            return PolicyDecision(False, "hourly_budget_exhausted")

        if self._last_analysis_at is not None and (now - self._last_analysis_at) < self.min_seconds_between_analysis:
            return PolicyDecision(False, "cooldown_active")

        if window_changed:
            return PolicyDecision(True, "window_changed")

        if pixel_changed and not is_volatile:
            return PolicyDecision(True, "pixel_change")

        if pixel_changed and is_volatile:
            return PolicyDecision(False, "volatile_backoff")

        return PolicyDecision(False, "no_change")

    def record_call(self, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        self._call_timestamps.append(now)
        self._last_analysis_at = now
