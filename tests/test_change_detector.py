from __future__ import annotations

from PIL import Image

from tests.conftest import make_frame
from vision.change_detector import ChangeDetector


def test_first_frame_always_counts_as_changed():
    det = ChangeDetector(cols=16, rows=9, change_threshold=0.02)
    result = det.detect(make_frame(fill=(10, 10, 10)))
    assert result.changed is True
    assert result.reason == "initial_or_resolution_change"


def test_identical_frames_are_not_changed():
    det = ChangeDetector(cols=16, rows=9, change_threshold=0.02)
    det.detect(make_frame(fill=(50, 50, 50)))
    result = det.detect(make_frame(fill=(50, 50, 50)))
    assert result.changed is False
    assert result.changed_fraction == 0.0


def test_large_color_change_is_detected_with_bbox():
    det = ChangeDetector(cols=16, rows=9, change_threshold=0.02)
    det.detect(make_frame(fill=(10, 10, 10)))
    result = det.detect(make_frame(fill=(240, 240, 240)))
    assert result.changed is True
    assert result.changed_fraction > 0.5
    assert result.changed_bbox is not None


def test_partial_region_change_produces_tight_bbox():
    det = ChangeDetector(cols=16, rows=9, change_threshold=0.01)
    base = Image.new("RGB", (320, 180), color=(0, 0, 0))
    from vision.capture import Frame
    import time

    det.detect(Frame(image=base, width=320, height=180, captured_at=time.time()))

    changed_img = base.copy()
    # Paint only the right-hand quarter of the image white.
    for x in range(240, 320):
        for y in range(0, 180):
            changed_img.putpixel((x, y), (255, 255, 255))
    result = det.detect(Frame(image=changed_img, width=320, height=180, captured_at=time.time()))

    assert result.changed is True
    x0, y0, x1, y1 = result.changed_bbox
    assert x0 >= 200  # bbox should be confined to the right portion
    assert x1 <= 320


def test_resolution_change_resets_and_counts_as_changed():
    det = ChangeDetector(cols=16, rows=9, change_threshold=0.02)
    det.detect(make_frame(width=320, height=180))
    result = det.detect(make_frame(width=640, height=360))
    assert result.changed is True
    assert result.reason == "initial_or_resolution_change"


def test_volatility_detected_after_many_consecutive_changes():
    det = ChangeDetector(cols=8, rows=8, change_threshold=0.01, volatility_window=5)
    assert det.is_volatile is False
    colors = [(0, 0, 0), (255, 255, 255)]
    for i in range(10):
        det.detect(make_frame(fill=colors[i % 2]))
    assert det.is_volatile is True


def test_reset_clears_state():
    det = ChangeDetector(cols=8, rows=8, change_threshold=0.02)
    det.detect(make_frame(fill=(10, 10, 10)))
    det.detect(make_frame(fill=(10, 10, 10)))
    det.reset()
    result = det.detect(make_frame(fill=(10, 10, 10)))
    assert result.reason == "initial_or_resolution_change"
