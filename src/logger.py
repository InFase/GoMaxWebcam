"""
logger.py — Centralized structured logging for GoPro Bridge

All log output goes to both console and a session-based log file stored
in %APPDATA%/GoProBridge/logs/.

Rotation policy (enforced on startup):
  - Keep at most N session files (default 5)
  - Keep total log size under a cap (default 50 MB)
  - Whichever limit is hit first triggers cleanup of oldest files

Modules obtain loggers via:
    from logger import get_logger
    log = get_logger(__name__)

Structured event logging convention:
    log.info("[EVENT:discovery_start] Searching for GoPro on USB")
    log.info("[EVENT:state_change] %s -> %s", old, new)
    log.warning("[EVENT:reconnect_attempt] attempt=%d delay=%.1fs", n, delay)

Event tag categories:
    startup, shutdown, config,
    discovery_start, discovery_found, discovery_failed,
    connection, disconnection, state_change,
    stream_start, stream_stop, stream_error,
    keepalive_ok, keepalive_fail,
    reconnect_attempt, recovery_success, recovery_failed,
    battery, camera_info,
    firewall, port_check,
    ffmpeg_start, ffmpeg_stop, ffmpeg_error,
    vcam_start, vcam_stop, vcam_error,
    freeze_frame, frame_recovery
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


_logger_initialized = False


def _get_session_log_files(log_dir: Path) -> list[Path]:
    """Return session log files sorted oldest-first by modification time.

    Session logs match the pattern gopro_bridge_*.log (not .log.1 etc,
    which are RotatingFileHandler's own rotated chunks).
    """
    files = sorted(
        log_dir.glob("gopro_bridge_*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    return files


def _get_all_log_files(log_dir: Path) -> list[Path]:
    """Return ALL log-related files (session + rotated chunks), oldest first.

    Includes both session files (*.log) and RotatingFileHandler backups
    (*.log.1, *.log.2, etc.).
    """
    files = sorted(
        list(log_dir.glob("gopro_bridge_*.log"))
        + list(log_dir.glob("gopro_bridge_*.log.*")),
        key=lambda p: p.stat().st_mtime,
    )
    return files


def _total_log_size(log_dir: Path) -> int:
    """Return total bytes consumed by all log files in the directory."""
    return sum(f.stat().st_size for f in _get_all_log_files(log_dir))


def _cleanup_by_session_count(log_dir: Path, max_sessions: int) -> int:
    """Delete oldest session files (and their rotated chunks) until
    at most *max_sessions* session files remain.

    Returns the number of files removed.
    """
    removed = 0
    session_files = _get_session_log_files(log_dir)

    while len(session_files) > max_sessions:
        oldest = session_files.pop(0)
        # Remove the session file and any rotated chunks (e.g. .log.1, .log.2)
        stem = oldest.name  # e.g. gopro_bridge_20250101_120000.log
        for chunk in log_dir.glob(f"{stem}.*"):
            try:
                chunk.unlink(missing_ok=True)
                removed += 1
            except PermissionError:
                pass  # File locked by another process
        try:
            oldest.unlink(missing_ok=True)
            removed += 1
        except PermissionError:
            pass  # File locked by another process

    return removed


def _cleanup_by_total_size(log_dir: Path, max_bytes: int) -> int:
    """Delete oldest log files until total size is under *max_bytes*.

    Removes files one at a time, oldest first, re-checking total after
    each deletion.

    Returns the number of files removed.
    """
    removed = 0

    while _total_log_size(log_dir) > max_bytes:
        all_files = _get_all_log_files(log_dir)
        if not all_files:
            break
        oldest = all_files[0]
        oldest.unlink(missing_ok=True)
        removed += 1

    return removed


def cleanup_logs(
    log_dir: Path,
    max_sessions: int = 5,
    max_total_bytes: int = 50 * 1024 * 1024,
) -> int:
    """Run the full cleanup policy: session-count cap then total-size cap.

    This is called automatically during logger setup but can also be
    invoked manually or from tests.

    Returns total number of files removed.
    """
    if not log_dir.exists():
        return 0

    removed = _cleanup_by_session_count(log_dir, max_sessions)
    removed += _cleanup_by_total_size(log_dir, max_total_bytes)
    return removed


def setup_logger(
    log_dir: Path,
    level: str = "INFO",
    max_session_files: int = 5,
    max_total_bytes: int = 50 * 1024 * 1024,
    max_file_bytes: int = 10 * 1024 * 1024,
) -> logging.Logger:
    """Set up the app-wide logger with console + file output.

    Each call to setup_logger creates a new session log file.  Before
    writing, old files are cleaned up according to the rotation policy:

      1. Keep at most ``max_session_files`` session logs.
      2. Keep total log directory size under ``max_total_bytes``.

    Within a single session file, ``RotatingFileHandler`` will rotate at
    ``max_file_bytes`` (creating .log.1, .log.2, etc.) so no individual
    session can grow unbounded.

    Args:
        log_dir: Directory for log files (created if missing).
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        max_session_files: Max number of session log files to keep.
        max_total_bytes: Total cap across all log files.
        max_file_bytes: Per-file rotation threshold.

    Returns:
        The root 'gopro_bridge' logger that all modules should use.
    """
    global _logger_initialized
    logger = logging.getLogger("gopro_bridge")

    if _logger_initialized:
        # Allow updating the level even if already initialized
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # File format: includes milliseconds for precise timing of events
    file_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console format: shorter timestamp for readability
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler — show INFO+ to the terminal
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(console_formatter)
    logger.addHandler(console)

    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Run cleanup BEFORE creating the new session file
    cleanup_logs(log_dir, max_sessions=max_session_files, max_total_bytes=max_total_bytes)

    # Session log file — unique per app launch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"gopro_bridge_{timestamp}.log"

    # RotatingFileHandler handles per-file size rotation within this session
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_file_bytes,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # File always captures DEBUG
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    _logger_initialized = True
    logger.info("[EVENT:startup] Logger initialized — log file: %s", log_file)
    logger.info("[EVENT:startup] Log level: %s", level.upper())
    logger.info(
        "[EVENT:startup] Log rotation policy: max %d session files, %d MB total cap",
        max_session_files,
        max_total_bytes // (1024 * 1024),
    )
    return logger


def reset_logger() -> None:
    """Reset logger state so setup_logger can be called again.

    Intended for testing only.
    """
    global _logger_initialized
    logger = logging.getLogger("gopro_bridge")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    _logger_initialized = False


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the 'gopro_bridge' namespace.

    Usage in any module:
        from logger import get_logger
        log = get_logger(__name__)
        log.info("something happened")

    The name is automatically prefixed with 'gopro_bridge.' so all
    module loggers inherit the root logger's handlers and level.
    """
    # Strip 'src.' prefix if present (when running from parent directory)
    if name.startswith("src."):
        name = name[4:]
    return logging.getLogger(f"gopro_bridge.{name}")
