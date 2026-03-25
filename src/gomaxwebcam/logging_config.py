"""
logging_config.py -- Logging setup for GoMaxWebcam v2.

Configures Python logging with:
  - Console handler (brief format, INFO level)
  - Rotating file handler (detailed format with timestamps)
  - Log directory via platformdirs user_log_dir
  - Auto-rotation: 7 days OR 50MB total cap (whichever hits first)
  - Debug mode toggle for verbose output
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

# Log format strings
CONSOLE_FMT = "%(levelname)-5s | %(name)s | %(message)s"
FILE_FMT = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Rotation limits
MAX_BYTES_PER_FILE = 10 * 1024 * 1024  # 10 MB per file
MAX_BACKUP_COUNT = 4                    # 5 files total = ~50 MB cap
LOG_FILENAME = "gomaxwebcam.log"

# App name for platformdirs
APP_NAME = "GoMaxWebcam-v2"


def get_log_dir() -> Path:
    """Return the platform-appropriate log directory.

    Uses platformdirs.user_log_dir if available, otherwise falls
    back to a reasonable default.
    """
    try:
        from platformdirs import user_log_dir
        return Path(user_log_dir(APP_NAME, ensure_exists=True))
    except ImportError:
        # Fallback: <home>/.gomaxwebcam/logs
        fallback = Path.home() / ".gomaxwebcam" / "logs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def setup_logging(
    debug: bool = False,
    log_dir: Optional[Path] = None,
    console: bool = True,
) -> None:
    """Configure logging for the application.

    Args:
        debug: If True, set root level to DEBUG and file handler to DEBUG.
               Otherwise, root is INFO and file handler is INFO.
        log_dir: Override log directory. None uses platformdirs default.
        console: If True, add a console (stdout) handler.
    """
    root = logging.getLogger("gomaxwebcam")
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    # Avoid duplicate handlers on repeated calls
    root.handlers.clear()

    # Console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if debug else logging.INFO)
        ch.setFormatter(logging.Formatter(CONSOLE_FMT))
        root.addHandler(ch)

    # File handler with rotation
    directory = log_dir or get_log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / LOG_FILENAME

    try:
        fh = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=MAX_BYTES_PER_FILE,
            backupCount=MAX_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG if debug else logging.INFO)
        fh.setFormatter(logging.Formatter(FILE_FMT, datefmt=DATE_FMT))
        root.addHandler(fh)
    except (OSError, PermissionError) as e:
        # If we can't write logs to disk, just log to console
        root.warning("Could not create log file at %s: %s", log_path, e)

    root.info(
        "Logging configured (level=%s, dir=%s)",
        "DEBUG" if debug else "INFO",
        directory,
    )
