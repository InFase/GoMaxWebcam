"""
stderr_ring_buffer.py — Thread-safe ring buffer for ffmpeg stderr output

Captures ffmpeg stderr diagnostic output in a bounded ring buffer so that
recent log lines are always available for error diagnosis without unbounded
memory growth.

Limits:
  - Max 1000 lines (oldest lines evicted when full)
  - Max 256 KB total byte size (oldest lines evicted when exceeded)

Thread safety:
  - All public methods acquire an internal lock, so the buffer can be
    safely written from a background stderr-reader thread and read from
    the main/GUI thread concurrently.

Usage:
    buf = StderrRingBuffer()
    buf.write_line("frame=  100 fps=30 ...")
    buf.write_line("[error] Something went wrong")

    # Get recent output for diagnostics
    recent = buf.get_lines()          # list of str
    full_text = buf.get_text()        # joined with newlines
    errors = buf.get_error_lines()    # lines containing error keywords
"""

import threading
from collections import deque
from typing import List

from logger import get_logger

log = get_logger("stderr_ring_buffer")

# Buffer limits
MAX_LINES = 1000
MAX_BYTES = 256 * 1024  # 256 KB

# Keywords that indicate ffmpeg errors worth flagging
_ERROR_KEYWORDS = frozenset({"error", "fatal", "invalid", "failed", "abort"})


class StderrRingBuffer:
    """Thread-safe ring buffer for ffmpeg stderr lines.

    Stores up to MAX_LINES lines with a total byte cap of MAX_BYTES.
    When either limit is exceeded, the oldest lines are evicted.

    Attributes:
        max_lines: Maximum number of lines to retain.
        max_bytes: Maximum total byte size of stored lines.
    """

    def __init__(
        self,
        max_lines: int = MAX_LINES,
        max_bytes: int = MAX_BYTES,
    ):
        """Initialize the ring buffer.

        Args:
            max_lines: Maximum number of lines to retain (default 1000).
            max_bytes: Maximum total bytes across all lines (default 256KB).
        """
        self.max_lines = max_lines
        self.max_bytes = max_bytes

        self._lock = threading.Lock()
        self._lines: deque[str] = deque()
        self._total_bytes: int = 0

    def write_line(self, line: str) -> None:
        """Add a single line to the buffer.

        If the line itself exceeds max_bytes, it is truncated to fit.
        After insertion, oldest lines are evicted until both the line
        count and byte size limits are satisfied.

        Args:
            line: A single line of stderr output (newline stripped).
        """
        # Truncate excessively long lines to prevent a single line
        # from blowing the byte budget
        if len(line.encode("utf-8", errors="replace")) > self.max_bytes:
            line = line[:self.max_bytes]

        line_bytes = len(line.encode("utf-8", errors="replace"))

        with self._lock:
            self._lines.append(line)
            self._total_bytes += line_bytes
            self._evict()

    def write_chunk(self, chunk: str) -> None:
        """Parse a raw stderr chunk into lines and add them all.

        ffmpeg stderr uses \\r for progress updates and \\n for real
        log lines. This method normalizes both into individual lines,
        skipping empty segments.

        Args:
            chunk: Raw text chunk from ffmpeg stderr.
        """
        for segment in chunk.replace("\r", "\n").split("\n"):
            segment = segment.strip()
            if segment:
                self.write_line(segment)

    def get_lines(self) -> List[str]:
        """Return a copy of all buffered lines, oldest first.

        Returns:
            List of stderr lines currently in the buffer.
        """
        with self._lock:
            return list(self._lines)

    def get_text(self) -> str:
        """Return all buffered lines joined with newlines.

        Returns:
            Single string of all buffered stderr output.
        """
        with self._lock:
            return "\n".join(self._lines)

    def get_last_line(self) -> str:
        """Return the most recently added line, or empty string.

        Returns:
            The newest line in the buffer, or '' if empty.
        """
        with self._lock:
            if self._lines:
                return self._lines[-1]
            return ""

    def get_error_lines(self) -> List[str]:
        """Return only lines that contain error-related keywords.

        Scans all buffered lines for keywords like 'error', 'fatal',
        'invalid', 'failed', 'abort' (case-insensitive).

        Returns:
            List of lines containing error keywords.
        """
        with self._lock:
            return [
                line for line in self._lines
                if any(kw in line.lower() for kw in _ERROR_KEYWORDS)
            ]

    def clear(self) -> None:
        """Remove all lines from the buffer."""
        with self._lock:
            self._lines.clear()
            self._total_bytes = 0

    @property
    def line_count(self) -> int:
        """Number of lines currently in the buffer."""
        with self._lock:
            return len(self._lines)

    @property
    def total_bytes(self) -> int:
        """Total byte size of all stored lines."""
        with self._lock:
            return self._total_bytes

    def _evict(self) -> None:
        """Remove oldest lines until both limits are satisfied.

        Must be called with self._lock held.
        """
        while len(self._lines) > self.max_lines:
            removed = self._lines.popleft()
            self._total_bytes -= len(removed.encode("utf-8", errors="replace"))

        while self._total_bytes > self.max_bytes and self._lines:
            removed = self._lines.popleft()
            self._total_bytes -= len(removed.encode("utf-8", errors="replace"))
