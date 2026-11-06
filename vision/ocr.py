"""Optional OCR support.

Uses `pytesseract` (a wrapper around the Tesseract OCR engine) if it is
installed and the Tesseract binary is available on PATH. This is entirely
optional: if unavailable, `OcrEngine.available` is False and callers should
degrade gracefully (the Gemini provider does its own text reading from the
image directly and does not depend on this module).

Windows setup note (see README): pytesseract does NOT bundle the Tesseract
engine. Install the Tesseract-OCR Windows build separately and either add
it to PATH or set `TESSERACT_CMD` to its full path.
"""

from __future__ import annotations

import logging
import os

from PIL import Image

logger = logging.getLogger("era_vision.ocr")

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]


class OcrEngine:
    def __init__(self) -> None:
        self._available = False
        if pytesseract is None:
            logger.info("pytesseract not installed; OCR disabled (optional feature).")
            return

        cmd = os.environ.get("TESSERACT_CMD", "").strip()
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

        try:
            pytesseract.get_tesseract_version()
            self._available = True
        except Exception:
            logger.info(
                "Tesseract engine not found on PATH (and TESSERACT_CMD not set); "
                "OCR disabled (optional feature)."
            )

    @property
    def available(self) -> bool:
        return self._available

    def extract_text(self, image: Image.Image) -> str:
        if not self._available:
            return ""
        try:
            return pytesseract.image_to_string(image).strip()
        except Exception:
            logger.exception("OCR extraction failed")
            return ""
