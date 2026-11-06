"""Windowless launcher for double-click / Task Scheduler use on Windows.

Runs with the `pythonw.exe` interpreter (no console window). Uvicorn's
default logging setup writes to a stream that does not exist under
pythonw, which can crash the process on startup -- main.py already passes
`log_config=None` to uvicorn.run() to avoid this, and our own
configure_logging() sets up file + stream handlers independently, so this
file is a thin wrapper around main.py with nothing extra to configure.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import main  # noqa: E402

if __name__ == "__main__":
    main()
