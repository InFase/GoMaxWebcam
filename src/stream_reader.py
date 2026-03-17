"""
stream_reader.py — ffmpeg subprocess for decoding GoPro MPEG-TS UDP stream

Launches ffmpeg to read MPEG-TS over UDP port 8554 and output raw BGR24
frames to stdout. Each frame is width × height × 3 bytes of contiguous
BGR pixel data.

Thread safety:
  - start() and stop() should be called from the same thread (the pipeline
    controller). read_frame() is called in a tight loop from the pipeline
    thread.
  - The ffmpeg process stdout is only read by one thread.

Usage:
    reader = StreamReader(config)
    reader.start()
    while running:
        frame = reader.read_frame()
        if frame is not None:
            push_to_virtual_camera(frame)
    reader.stop()
"""

import collections
import subprocess
import threading
import time
from typing import Optional

import numpy as np

from logger import get_logger

log = get_logger("stream_reader")


class StreamReader:
    """Reads raw video frames from the GoPro MPEG-TS UDP stream via ffmpeg.

    ffmpeg decodes H.264 in MPEG-TS from UDP and outputs raw BGR24 frames
    to its stdout pipe. We read exactly (width × height × 3) bytes per frame.

    Attributes:
        width: Output frame width in pixels.
        height: Output frame height in pixels.
        fps: Target frame rate (used for ffmpeg output rate limiting).
        is_running: True while the ffmpeg process is alive.
    """

    def __init__(self, config):
        """Initialize the stream reader from app config.

        Args:
            config: Config object with stream_width, stream_height,
                    stream_fps, udp_port, ffmpeg_path.
        """
        self.width: int = config.stream_width
        self.height: int = config.stream_height
        self.fps: int = config.stream_fps
        self.udp_port: int = config.udp_port
        self.ffmpeg_path: str = config.ffmpeg_path

        self._process: Optional[subprocess.Popen] = None
        self._frame_size: int = self.width * self.height * 3  # BGR24
        self._is_running: bool = False
        self._stderr_thread: Optional[threading.Thread] = None
        self._last_stderr: str = ""
        self._stderr_buffer: collections.deque = collections.deque(maxlen=50)
        self._stderr_lock: threading.Lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """True if the ffmpeg process is alive and producing frames."""
        if self._process is None:
            return False
        if self._process.poll() is not None:
            # Process has exited
            self._is_running = False
            return False
        return self._is_running

    @property
    def frame_size_bytes(self) -> int:
        """Number of bytes per raw BGR24 frame."""
        return self._frame_size

    def start(self) -> bool:
        """Launch the ffmpeg subprocess to decode the UDP stream.

        ffmpeg command:
            ffmpeg -f mpegts -i udp://@:{port}
                   -pix_fmt bgr24 -s {w}x{h}
                   -f rawvideo -an -sn
                   pipe:1

        Returns:
            True if ffmpeg launched successfully.
        """
        if self._process is not None:
            log.warning("[EVENT:ffmpeg_start] ffmpeg already running, stopping first")
            self.stop()

        cmd = self._build_ffmpeg_command()
        log.info(
            "[EVENT:ffmpeg_start] Launching ffmpeg: %s",
            " ".join(cmd),
        )

        try:
            # Launch ffmpeg with stdout pipe for raw frames.
            # stderr is piped and drained by a background thread to
            # prevent the 4KB Windows pipe buffer from filling up and
            # deadlocking stdout. The drain thread buffers lines for
            # diagnostics accessible via get_stderr_lines().
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # On Windows, CREATE_NO_WINDOW prevents a console flash
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._is_running = True

            # Start background thread to drain stderr
            self._stderr_thread = threading.Thread(
                target=self._read_stderr,
                name="ffmpeg-stderr-drain",
                daemon=True,
            )
            self._stderr_thread.start()

            log.info(
                "[EVENT:ffmpeg_start] ffmpeg started (PID %d) — "
                "reading UDP port %d, output %dx%d@%dfps BGR24",
                self._process.pid, self.udp_port,
                self.width, self.height, self.fps,
            )
            return True

        except FileNotFoundError:
            log.error(
                "[EVENT:ffmpeg_error] ffmpeg not found at '%s'. "
                "Ensure ffmpeg is installed and on PATH.",
                self.ffmpeg_path,
            )
            self._is_running = False
            return False
        except OSError as e:
            log.error("[EVENT:ffmpeg_error] Failed to launch ffmpeg: %s", e)
            self._is_running = False
            return False

    def _build_ffmpeg_command(self) -> list[str]:
        """Build the ffmpeg command line for ultra-low-latency MPEG-TS UDP decoding.

        Target: < 40ms end-to-end latency from GoPro sensor to virtual camera.

        Key optimizations:
          -fflags nobuffer+discardcorrupt: zero input buffering, skip bad packets
          -flags low_delay: frame-level decode (no reordering delay)
          -avioflags direct: bypass ffmpeg's I/O buffering layer
          -reorder_queue_size 0: disable MPEG-TS packet reordering queue
          -max_delay 0: zero mux/demux buffering
          -probesize/analyzeduration minimal: GoPro stream format is known
          -threads 1: single-thread decode avoids inter-thread sync overhead
          -pix_fmt bgr24: Unity Capture accepts BGR natively, avoids channel swap
          buffer_size=65536: small UDP socket buffer to prevent stale frames
        """
        return [
            self.ffmpeg_path,
            # Low-latency options
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "500000",
            "-probesize", "500000",
            # Input source
            "-f", "mpegts",
            "-i", f"udp://0.0.0.0:{self.udp_port}?overrun_nonfatal=1",
            # Decode options — single thread avoids sync overhead
            "-threads", "1",
            # Output options — BGR24 for Unity Capture, no FPS rate limiting
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-f", "rawvideo",
            "-an",
            "-sn",
            "pipe:1",
        ]

    def read_frame(self) -> Optional[np.ndarray]:
        """Read exactly one raw BGR24 frame from ffmpeg stdout.

        Blocks until a full frame is available or the pipe closes.

        Returns:
            numpy array of shape (height, width, 3) dtype uint8,
            or None if the stream has ended or an error occurred.
        """
        if self._process is None or self._process.stdout is None:
            return None

        try:
            raw = self._process.stdout.read(self._frame_size)
        except (ValueError, OSError):
            # Pipe closed or process terminated
            log.debug("[EVENT:stream_error] stdout read failed (pipe closed)")
            return None

        if len(raw) == 0:
            # EOF — ffmpeg has exited or stream ended
            log.debug("[EVENT:stream_stop] ffmpeg stdout EOF")
            self._is_running = False
            return None

        if len(raw) != self._frame_size:
            # Partial frame — stream interrupted mid-frame
            log.warning(
                "[EVENT:stream_error] Partial frame: got %d/%d bytes",
                len(raw), self._frame_size,
            )
            self._is_running = False
            return None

        # Reshape raw bytes into (H, W, 3) BGR array
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )
        return frame

    def stop(self):
        """Terminate the ffmpeg subprocess and clean up.

        Sends SIGTERM first, waits briefly, then SIGKILL if needed.
        """
        self._is_running = False

        if self._process is None:
            return

        pid = self._process.pid
        log.info("[EVENT:ffmpeg_stop] Stopping ffmpeg (PID %d)", pid)

        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=3.0)
                log.debug("ffmpeg terminated gracefully")
            except subprocess.TimeoutExpired:
                log.warning("ffmpeg did not exit after terminate, killing")
                self._process.kill()
                self._process.wait(timeout=2.0)
        except OSError:
            log.debug("ffmpeg process already gone")

        # Close pipes
        for pipe in (self._process.stdout, self._process.stderr):
            if pipe:
                try:
                    pipe.close()
                except OSError:
                    pass

        # Wait for stderr drain thread to finish
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
            self._stderr_thread = None

        self._process = None
        log.info("[EVENT:ffmpeg_stop] ffmpeg stopped")

    def _read_stderr(self):
        """Background thread: drain ffmpeg stderr to prevent pipe deadlock.

        ffmpeg writes progress updates to stderr using \\r (carriage return)
        without newline. readline() blocks waiting for \\n, the stderr pipe
        fills up (4KB on Windows), and ffmpeg blocks on its stderr write,
        which in turn blocks stdout — causing the frame pipeline to stall.

        We read in fixed chunks to prevent this deadlock, and scan for
        error keywords in each chunk. All non-empty lines are buffered
        (up to 50 most recent) for diagnostic retrieval via get_stderr_lines().
        """
        if self._process is None or self._process.stderr is None:
            return

        try:
            while True:
                chunk = self._process.stderr.read(4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                # Extract meaningful lines for diagnostics
                for segment in text.replace("\r", "\n").split("\n"):
                    segment = segment.strip()
                    if segment:
                        with self._stderr_lock:
                            self._last_stderr = segment
                            self._stderr_buffer.append(segment)
                        if any(kw in segment.lower() for kw in ("error", "fatal", "invalid")):
                            log.warning("[EVENT:ffmpeg_error] ffmpeg: %s", segment)
        except (ValueError, OSError):
            pass  # Pipe closed

    @property
    def last_error(self) -> str:
        """Last line from ffmpeg stderr — useful for error diagnostics."""
        with self._stderr_lock:
            return self._last_stderr

    def get_stderr_lines(self) -> list[str]:
        """Return the buffered stderr lines from ffmpeg for diagnostics.

        Returns up to the 50 most recent non-empty lines captured from
        ffmpeg's stderr output. Thread-safe — may be called from any thread.

        Returns:
            List of stderr line strings, oldest first.
        """
        with self._stderr_lock:
            return list(self._stderr_buffer)

    @property
    def exit_code(self) -> Optional[int]:
        """ffmpeg process exit code, or None if still running."""
        if self._process is None:
            return None
        return self._process.poll()
