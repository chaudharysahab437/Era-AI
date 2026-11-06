"""ERA Vision Agent entry point.

Usage (Windows, from inside the project's virtual environment):

    python main.py

Vision starts OFF. Use the REST API (see README.md) to turn it on.
"""

from __future__ import annotations

import sys

import uvicorn

from app import create_app
from config import load_settings
from vision.windows import enable_dpi_awareness


def main() -> None:
    enable_dpi_awareness()  # must happen before any window/monitor query
    settings = load_settings()

    if not settings.vision_key_configured:
        print(
            "ERROR: VISION_AGENT_API_KEY is not set. Generate one with "
            "'python scripts/store_gemini_key.py --help' or set it directly "
            "in your .env file. Refusing to start with no auth configured.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not settings.gemini_key_configured:
        print(
            "WARNING: GEMINI_API_KEY is not set. The agent will start and "
            "perception (capture, change detection, active-window "
            "detection) will work, but screen analysis will fail until a "
            "key is configured. See README.md.",
            file=sys.stderr,
        )

    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None)


if __name__ == "__main__":
    main()
