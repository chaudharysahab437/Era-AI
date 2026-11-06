"""Change detection.

Compares the current frame against the previous one using a coarse grid of
grayscale cell averages (see `Frame.to_small_grayscale_array`). This is
deliberately cheap -- it never runs a vision model just to decide whether
to run a vision model.

Also tracks "volatility": windows that change on nearly every sample (e.g.
a video, an animation, a live chart) are detected so the policy layer can
back off rather than calling the vision model every second forever.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from vision.capture import Frame
from vision.models import ChangeResult


class ChangeDetector:
    def __init__(self, cols: int, rows: int, change_threshold: float, volatility_window: int = 10) -> None:
        self.cols = cols
        self.rows = rows
        self.change_threshold = change_threshold
        self._prev_array: np.ndarray | None = None
        self._prev_frame_size: tuple[int, int] | None = None
        self._recent_changed: deque[bool] = deque(maxlen=volatility_window)

    def reset(self) -> None:
        self._prev_array = None
        self._prev_frame_size = None
        self._recent_changed.clear()

    def detect(self, frame: Frame) -> ChangeResult:
        arr = frame.to_small_grayscale_array(self.cols, self.rows)

        if self._prev_array is None or self._prev_frame_size != (frame.width, frame.height):
            # First frame, or resolution changed (e.g. display reconfigured).
            self._prev_array = arr
            self._prev_frame_size = (frame.width, frame.height)
            self._recent_changed.append(True)
            return ChangeResult(changed=True, changed_fraction=1.0, reason="initial_or_resolution_change")

        diff = np.abs(arr - self._prev_array)
        # Normalize 0-255 grayscale difference; a cell counts as "changed"
        # once its average brightness shifts more than this amount.
        cell_changed = diff > 18.0
        changed_fraction = float(cell_changed.mean())

        self._prev_array = arr
        changed = changed_fraction >= self.change_threshold
        self._recent_changed.append(changed)

        bbox = None
        if changed and cell_changed.any():
            ys, xs = np.where(cell_changed)
            cell_w = frame.width / self.cols
            cell_h = frame.height / self.rows
            x0 = int(xs.min() * cell_w)
            x1 = int((xs.max() + 1) * cell_w)
            y0 = int(ys.min() * cell_h)
            y1 = int((ys.max() + 1) * cell_h)
            bbox = (x0, y0, x1, y1)

        reason = "pixel_change" if changed else "no_meaningful_change"
        return ChangeResult(changed=changed, changed_fraction=changed_fraction, changed_bbox=bbox, reason=reason)

    @property
    def is_volatile(self) -> bool:
        """True when the screen has changed on nearly every recent sample
        (video, animation, live-updating chart). Used by the policy layer
        to fall back to a slower, budget-conserving cadence instead of
        treating constant motion as constant "meaningful change"."""
        if len(self._recent_changed) < self._recent_changed.maxlen:
            return False
        return sum(self._recent_changed) >= self._recent_changed.maxlen - 1
