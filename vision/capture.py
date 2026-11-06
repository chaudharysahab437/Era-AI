"""Screen capture.

Uses `mss` for fast, cross-platform-capable screen grabs, and Pillow for
encoding/resizing/cropping. All capture calls happen on a single dedicated
thread (`mss` instances are not thread-safe / not safe to share across
threads on Windows), via `CaptureWorker`.

On Windows, some window types (DRM-protected video, some elevated windows,
some games using exclusive fullscreen) will return black or garbage frames.
`is_probably_blank()` gives a best-effort detector for that so the rest of
the pipeline can report it rather than silently analyzing a black frame.
"""

from __future__ import annotations

import io
import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    import mss
except ImportError:  # pragma: no cover - exercised only when mss truly missing
    mss = None  # type: ignore[assignment]

logger = logging.getLogger("era_vision.capture")


class CaptureError(RuntimeError):
    pass


@dataclass
class Frame:
    image: Image.Image          # RGB Pillow image, full captured frame
    width: int
    height: int
    captured_at: float

    def to_jpeg_bytes(self, quality: int = 70, max_dim: int = 1280) -> bytes:
        img = self.image
        if max(img.size) > max_dim:
            scale = max_dim / max(img.size)
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def crop(self, bbox: tuple[int, int, int, int]) -> "Frame":
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(x0, self.width - 1))
        y0 = max(0, min(y0, self.height - 1))
        x1 = max(x0 + 1, min(x1, self.width))
        y1 = max(y0 + 1, min(y1, self.height))
        cropped = self.image.crop((x0, y0, x1, y1))
        return Frame(image=cropped, width=cropped.width, height=cropped.height, captured_at=self.captured_at)

    def to_small_grayscale_array(self, cols: int, rows: int) -> np.ndarray:
        """Downscaled grayscale array used by the change detector. Cheap and
        deliberately low-resolution -- this is not meant to be visually
        meaningful, only to detect that *something* changed."""
        small = self.image.convert("L").resize((cols, rows), Image.BILINEAR)
        return np.asarray(small, dtype=np.float32)


def is_probably_blank(frame: Frame) -> bool:
    """Heuristic for capture returning a black/near-uniform frame, which
    happens for some DRM-protected or exclusive-fullscreen windows."""
    arr = np.asarray(frame.image.convert("L"), dtype=np.uint8)
    if arr.size == 0:
        return True
    std = float(arr.std())
    mean = float(arr.mean())
    # Near-zero variance and near-zero (or near-max, e.g. pure white) mean
    # is very unlikely for a real desktop frame.
    return std < 1.5 and (mean < 5.0 or mean > 250.0)


class CaptureWorker:
    """Owns the single thread that talks to `mss`.

    All public methods are safe to call from any thread; work is dispatched
    to the capture thread via a queue and results returned through a Future-
    like mechanism (simple queue-based request/response).
    """

    def __init__(self) -> None:
        if mss is None:
            raise CaptureError(
                "The 'mss' package is not installed. Install it with "
                "'pip install mss' (see requirements.txt)."
            )
        self._req_q: "queue.Queue[tuple[str, queue.Queue]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vision-capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            sct = mss.mss()
        except Exception as exc:  # noqa: BLE001
            # mss failed to initialize at all (no display attached here in
            # dev/CI; on Windows this can happen with no active desktop
            # session, e.g. a locked/disconnected RDP session). Drain the
            # request queue for the lifetime of this thread, answering
            # every grab request with the init error, so callers get a
            # prompt CaptureError instead of a timeout, and close() does
            # not hang waiting for a reply that would otherwise never come.
            logger.error("Failed to initialize screen capture backend: %s", exc)
            init_error = exc
            while not self._stop.is_set():
                try:
                    action, reply_q = self._req_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if action == "stop":
                    reply_q.put(None)
                    return
                reply_q.put(("error", init_error))
            return

        with sct:
            while not self._stop.is_set():
                try:
                    action, reply_q = self._req_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if action == "stop":
                    reply_q.put(None)
                    break
                try:
                    monitor = sct.monitors[0]  # virtual bounding box of all monitors
                    shot = sct.grab(monitor)
                    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                    frame = Frame(image=img, width=img.width, height=img.height, captured_at=time.time())
                    reply_q.put(("ok", frame))
                except Exception as exc:  # noqa: BLE001 - reported to caller, not swallowed
                    logger.exception("Screen capture failed")
                    reply_q.put(("error", exc))

    def grab(self, timeout: float = 5.0) -> Frame:
        reply_q: "queue.Queue" = queue.Queue(maxsize=1)
        self._req_q.put(("grab", reply_q))
        try:
            result = reply_q.get(timeout=timeout)
        except queue.Empty as exc:
            raise CaptureError("Screen capture timed out") from exc
        status, payload = result
        if status == "error":
            raise CaptureError(f"Screen capture failed: {payload}") from payload
        return payload

    def close(self) -> None:
        self._stop.set()
        reply_q: "queue.Queue" = queue.Queue(maxsize=1)
        self._req_q.put(("stop", reply_q))
        try:
            reply_q.get(timeout=5.0)
        except queue.Empty:
            logger.warning("Capture worker did not acknowledge stop within 5s; thread may already be dead.")
        self._thread.join(timeout=5.0)
