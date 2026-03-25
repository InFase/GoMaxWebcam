"""
decode.py — PyAV-based UDP MPEG-TS decode pipeline

Receives H.264/H.265 video from the GoPro's UDP preview stream and decodes
frames into numpy arrays (BGR24, shape H×W×3, dtype uint8) suitable for
pyvirtualcam output via Unity Capture.

Architecture:
  - UDPDecoder runs a blocking decode loop on a dedicated thread
  - Frames are pushed to a callback (typically the frame buffer)
  - The decode thread is the ONLY thread that touches PyAV objects
  - Thread-safe stop via threading.Event

Design choices vs v1:
  - PyAV in-process decode replaces ffmpeg subprocess (no pipe overhead,
    no orphan process cleanup, no Windows pipe deadlocks)
  - Direct container.open() on UDP URL with low-latency options
  - Single-threaded decode (GoPro preview is low-bitrate, ~8 Mbps)
  - BGR24 output via VideoReformatter (hardware-accelerated swscale)
    matching v1's ffmpeg -pix_fmt bgr24 for Unity Capture native format
  - Bounded error recovery: transient UDP packet loss is tolerated,
    persistent errors trigger clean shutdown with callback notification

Usage:
    def on_frame(frame: np.ndarray):
        buffer.update(frame)

    decoder = UDPDecoder(
        udp_port=8554,
        width=1920,
        height=1080,
        on_frame=on_frame,
        on_error=lambda e: log.error("decode error: %s", e),
    )
    decoder.start()
    # ... later ...
    decoder.stop()

Thread model:
    Caller thread:  start() / stop() / property access
    Decode thread:  _run_decode_loop() — owns all PyAV objects
"""

import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

log = logging.getLogger("gomaxwebcam.pipeline.decode")


@dataclass(frozen=True)
class FrameMetadata:
    """Immutable metadata attached to each decoded frame.

    Tracks per-frame provenance information for diagnostics, dashboard
    display, and v1 parity validation. Created by the decoder on the
    decode thread and passed alongside the numpy frame data.

    Attributes:
        timestamp: Monotonic timestamp (time.monotonic()) when the frame
            was decoded and delivered. Used for inter-frame interval
            measurement and freeze-frame age tracking.
        sequence: Zero-based frame counter within the current decode
            session. Resets on decoder restart. Useful for detecting
            dropped frames and ordering.
        height: Frame height in pixels (e.g. 1080).
        width: Frame width in pixels (e.g. 1920).
        channels: Number of color channels (always 3 for BGR24/RGB24).
        nbytes: Total frame size in bytes (height * width * channels).
            Used for memory accounting and bandwidth estimation.
        pixel_format: Output pixel format string (e.g. "bgr24").
            Matches the decoder's configured format for v1 parity checks.
    """
    timestamp: float
    sequence: int
    height: int
    width: int
    channels: int
    nbytes: int
    pixel_format: str = "bgr24"


class FrameCallback(Protocol):
    """Protocol for frame delivery callbacks."""
    def __call__(self, frame: np.ndarray) -> None: ...


class FrameWithMetadataCallback(Protocol):
    """Protocol for frame delivery callbacks that include metadata."""
    def __call__(self, frame: np.ndarray, metadata: FrameMetadata) -> None: ...


class ErrorCallback(Protocol):
    """Protocol for error notification callbacks."""
    def __call__(self, error: Exception) -> None: ...


# -- Constants ---------------------------------------------------------------

# Maximum consecutive decode errors before giving up.
# GoPro UDP streams occasionally drop packets — we tolerate transient errors
# but shut down if the stream is fundamentally broken.
_MAX_CONSECUTIVE_ERRORS = 30

# Timeout for opening the UDP stream container (seconds).
# GoPro takes 1-3 seconds to start the preview stream after the API call.
_OPEN_TIMEOUT_SEC = 10.0

# PyAV container options for low-latency UDP MPEG-TS decoding.
# These mirror the v1 ffmpeg flags but as libav option dicts.
_CONTAINER_OPTIONS = {
    # Minimal probing — GoPro stream format is known (H.264 or H.265 in MPEG-TS)
    "analyzeduration": "500000",
    "probesize": "500000",
    # Low-latency flags
    "fflags": "nobuffer+discardcorrupt",
    "flags": "low_delay",
    # UDP socket buffer — small to prevent stale frames
    "buffer_size": "65536",
    # Don't fail on overrun (UDP packet loss is expected)
    "overrun_nonfatal": "1",
    # Reorder queue disabled — GoPro sends packets in order
    "reorder_queue_size": "0",
    # Zero mux/demux delay
    "max_delay": "0",
}


class DecodeStats:
    """Thread-safe decode statistics for dashboard display.

    All fields are updated atomically from the decode thread and read
    from the dashboard/API thread.

    Tracks per-frame decode latency (time from packet decode start to
    frame delivery), rolling FPS, inter-frame jitter, and cumulative
    error/drop counters.  These metrics are used by the dashboard for
    real-time diagnostics charts and by ``PipelineHealth`` for health
    assessment against v1 performance baselines.
    """

    #: v1 performance baselines for health assessment
    BASELINE_FPS_TARGET: float = 30.0
    BASELINE_FPS_MIN: float = 25.0          # Below this → degraded
    BASELINE_LATENCY_WARN_MS: float = 50.0  # Above this → degraded
    BASELINE_LATENCY_CRIT_MS: float = 100.0 # Above this → unhealthy
    BASELINE_DROP_RATE_WARN: float = 0.01   # 1% drop rate → degraded
    BASELINE_DROP_RATE_CRIT: float = 0.05   # 5% drop rate → unhealthy

    def __init__(self):
        self._lock = threading.Lock()
        self.frames_decoded: int = 0
        self.frames_dropped: int = 0
        self.errors: int = 0
        self.last_frame_time: Optional[float] = None
        self.decode_fps: float = 0.0
        self.codec_name: str = ""
        self._fps_samples: list[float] = []
        self._fps_window: float = 1.0  # 1-second rolling window

        # Decode latency tracking (per-frame decode time in seconds)
        self._decode_start: Optional[float] = None
        self._latency_samples: list[float] = []  # rolling window of latencies (seconds)
        self._latency_window: float = 5.0  # 5-second rolling window for averaging
        self.last_decode_ms: float = 0.0   # Most recent frame decode time in ms

        # Inter-frame jitter tracking
        self._prev_frame_time: Optional[float] = None
        self._jitter_samples: list[float] = []  # rolling window of inter-frame intervals
        self.avg_jitter_ms: float = 0.0  # Average deviation from ideal 33.3ms interval

    def begin_decode(self) -> None:
        """Mark the start of a frame decode operation.

        Call this just before decoding a packet. The elapsed time between
        ``begin_decode()`` and the subsequent ``record_frame()`` is the
        per-frame decode latency reported to the dashboard.
        """
        self._decode_start = time.monotonic()

    def record_frame(self) -> None:
        """Record a successfully decoded frame with latency measurement."""
        now = time.monotonic()
        with self._lock:
            self.frames_decoded += 1
            self.last_frame_time = now
            self._fps_samples.append(now)
            # Trim to rolling window
            cutoff = now - self._fps_window
            self._fps_samples = [t for t in self._fps_samples if t > cutoff]
            self.decode_fps = len(self._fps_samples) / self._fps_window

            # Decode latency
            if self._decode_start is not None:
                latency = now - self._decode_start
                self.last_decode_ms = latency * 1000.0
                self._latency_samples.append((now, latency))
                # Trim latency samples to rolling window
                lat_cutoff = now - self._latency_window
                self._latency_samples = [
                    (t, l) for t, l in self._latency_samples if t > lat_cutoff
                ]
                self._decode_start = None

            # Inter-frame jitter
            if self._prev_frame_time is not None:
                interval = now - self._prev_frame_time
                ideal_interval = 1.0 / 30.0  # 33.3ms for 30fps
                jitter = abs(interval - ideal_interval)
                self._jitter_samples.append((now, jitter))
                # Trim jitter samples to rolling window
                jit_cutoff = now - self._latency_window
                self._jitter_samples = [
                    (t, j) for t, j in self._jitter_samples if t > jit_cutoff
                ]
                if self._jitter_samples:
                    self.avg_jitter_ms = (
                        sum(j for _, j in self._jitter_samples)
                        / len(self._jitter_samples)
                        * 1000.0
                    )
            self._prev_frame_time = now

    def record_error(self) -> None:
        """Record a decode error."""
        with self._lock:
            self.errors += 1

    def record_drop(self) -> None:
        """Record a dropped frame (e.g., wrong dimensions)."""
        with self._lock:
            self.frames_dropped += 1

    def set_codec(self, name: str) -> None:
        """Set the codec name (e.g., 'h264', 'hevc')."""
        with self._lock:
            self.codec_name = name

    def snapshot(self) -> dict:
        """Return a snapshot of stats for the dashboard.

        Includes decode latency, jitter, and drop rate metrics alongside
        the basic frame counters and FPS measurement.
        """
        with self._lock:
            age = None
            if self.last_frame_time is not None:
                age = round(time.monotonic() - self.last_frame_time, 3)

            # Compute average decode latency over the rolling window
            avg_decode_ms = 0.0
            max_decode_ms = 0.0
            if self._latency_samples:
                latencies = [l for _, l in self._latency_samples]
                avg_decode_ms = (sum(latencies) / len(latencies)) * 1000.0
                max_decode_ms = max(latencies) * 1000.0

            # Compute drop rate
            total = self.frames_decoded + self.frames_dropped
            drop_rate = (self.frames_dropped / total) if total > 0 else 0.0

            return {
                "frames_decoded": self.frames_decoded,
                "frames_dropped": self.frames_dropped,
                "errors": self.errors,
                "decode_fps": round(self.decode_fps, 1),
                "codec": self.codec_name,
                "last_frame_age": age,
                "last_decode_ms": round(self.last_decode_ms, 2),
                "avg_decode_ms": round(avg_decode_ms, 2),
                "max_decode_ms": round(max_decode_ms, 2),
                "avg_jitter_ms": round(self.avg_jitter_ms, 2),
                "drop_rate": round(drop_rate, 4),
            }


class UDPDecoder:
    """PyAV-based UDP MPEG-TS decoder for GoPro preview streams.

    Opens a UDP MPEG-TS stream, finds the video track, and decodes each
    frame into an RGB24 numpy array. Decoded frames are delivered via
    the on_frame callback.

    The decoder runs on a dedicated thread and is fully self-contained —
    all PyAV objects are created and destroyed on that thread.

    Attributes:
        udp_port: UDP port to listen on (default 8554).
        width: Expected output frame width.
        height: Expected output frame height.
        is_running: True while the decode thread is active.
        stats: DecodeStats instance with live metrics.
    """

    #: Default pixel format — BGR24 matches v1's ffmpeg -pix_fmt bgr24 output
    #: which Unity Capture accepts natively without channel swapping.
    DEFAULT_PIXEL_FORMAT = "bgr24"

    def __init__(
        self,
        udp_port: int = 8554,
        width: int = 1920,
        height: int = 1080,
        pixel_format: str = "bgr24",
        on_frame: Optional[FrameCallback] = None,
        on_error: Optional[ErrorCallback] = None,
        on_stopped: Optional[Callable[[], None]] = None,
    ):
        """Initialize the UDP decoder.

        Args:
            udp_port: UDP port for the MPEG-TS stream.
            width: Expected frame width (frames not matching are dropped).
            height: Expected frame height (frames not matching are dropped).
            pixel_format: Output pixel format for decoded frames. Default
                "bgr24" matches v1's ffmpeg output and Unity Capture's native
                format. Use "rgb24" if downstream expects RGB order.
            on_frame: Callback receiving decoded frames as numpy arrays
                      of shape (height, width, 3) dtype uint8.
            on_error: Callback for fatal decode errors (stream unrecoverable).
            on_stopped: Callback when the decode loop exits (normal or error).
        """
        self.udp_port = udp_port
        self.width = width
        self.height = height
        self.pixel_format = pixel_format

        self._on_frame = on_frame
        self._on_error = on_error
        self._on_stopped = on_stopped

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_running = False

        self.stats = DecodeStats()

    @property
    def is_running(self) -> bool:
        """True if the decode thread is actively running."""
        return self._is_running and self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start the decode thread.

        Returns:
            True if the thread was started (or was already running).
        """
        if self.is_running:
            log.warning("[EVENT:decoder_start] Decoder already running")
            return True

        self._stop_event.clear()
        self._is_running = True

        self._thread = threading.Thread(
            target=self._run_decode_loop,
            name="pyav-decode",
            daemon=True,
        )
        self._thread.start()

        log.info(
            "[EVENT:decoder_start] Decode thread started — "
            "UDP port %d, target %dx%d",
            self.udp_port, self.width, self.height,
        )
        return True

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the decode thread and release resources.

        Args:
            timeout: Maximum seconds to wait for the thread to exit.
        """
        log.info("[EVENT:decoder_stop] Stopping decoder...")
        self._stop_event.set()
        self._is_running = False

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                log.warning(
                    "[EVENT:decoder_stop] Decode thread did not exit "
                    "within %.1fs — it will be cleaned up on process exit",
                    timeout,
                )
            self._thread = None

        log.info("[EVENT:decoder_stop] Decoder stopped")

    def _run_decode_loop(self) -> None:
        """Main decode loop — runs on the dedicated decode thread.

        Tries PyAV direct UDP first. If that fails (common on Windows with
        certain PyAV/ffmpeg builds), falls back to ffmpeg subprocess piping
        raw BGR24 frames — the same proven approach v1 uses.
        """
        try:
            self._run_pyav_decode()
        except Exception as pyav_err:
            log.warning(
                "[EVENT:decoder_fallback] PyAV UDP failed (%s), "
                "trying ffmpeg subprocess...", pyav_err,
            )
            if self._stop_event.is_set():
                self._is_running = False
                return
            try:
                self._run_ffmpeg_subprocess_decode()
            except Exception as ffmpeg_err:
                log.error("[EVENT:decoder_error] Both PyAV and ffmpeg failed: %s", ffmpeg_err)
                if self._on_error is not None:
                    self._on_error(ffmpeg_err)

        self._is_running = False

        if self._on_stopped is not None:
            try:
                self._on_stopped()
            except Exception:
                log.exception("[EVENT:decoder_stop] on_stopped callback error")

        log.info(
            "[EVENT:decoder_exit] Decode loop exited — "
            "decoded=%d, dropped=%d, errors=%d",
            self.stats.frames_decoded,
            self.stats.frames_dropped,
            self.stats.errors,
        )

    def _run_pyav_decode(self) -> None:
        """Decode via PyAV direct UDP — preferred when it works."""
        import av

        udp_url = f"udp://0.0.0.0:{self.udp_port}"
        log.info(
            "[EVENT:decoder_open] Opening stream via PyAV: %s (timeout=%.0fs)",
            udp_url, _OPEN_TIMEOUT_SEC,
        )

        container = av.open(
            udp_url,
            format="mpegts",
            options=_CONTAINER_OPTIONS,
            timeout=_OPEN_TIMEOUT_SEC,
        )

        try:
            video_streams = [s for s in container.streams if s.type == "video"]
            if not video_streams:
                raise RuntimeError(
                    f"No video stream found in UDP MPEG-TS on port {self.udp_port}"
                )

            video_stream = video_streams[0]
            video_stream.thread_type = "NONE"
            video_stream.thread_count = 1

            codec_name = video_stream.codec_context.name
            self.stats.set_codec(codec_name)
            log.info(
                "[EVENT:decoder_open] PyAV stream opened — codec=%s, %dx%d",
                codec_name,
                video_stream.codec_context.width,
                video_stream.codec_context.height,
            )

            consecutive_errors = 0
            for packet in container.demux(video_stream):
                if self._stop_event.is_set():
                    break

                try:
                    self.stats.begin_decode()
                    for av_frame in packet.decode():
                        if self._stop_event.is_set():
                            break

                        rgb_frame = av_frame.to_ndarray(format=self.pixel_format)

                        if rgb_frame.shape[0] != self.height or rgb_frame.shape[1] != self.width:
                            rgb_frame = self._scale_frame(rgb_frame, av_frame, av)
                            if rgb_frame is None:
                                self.stats.record_drop()
                                continue

                        if self._on_frame is not None:
                            self._on_frame(rgb_frame)

                        self.stats.record_frame()
                        consecutive_errors = 0

                except av.error.InvalidDataError as e:
                    consecutive_errors += 1
                    self.stats.record_error()
                    if consecutive_errors <= 3 or consecutive_errors % 10 == 0:
                        log.debug(
                            "[EVENT:decode_error] Invalid data (%d/%d): %s",
                            consecutive_errors, _MAX_CONSECUTIVE_ERRORS, e,
                        )
                    if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                        raise RuntimeError(
                            f"Too many consecutive decode errors ({consecutive_errors})"
                        ) from e

                except av.error.EOFError:
                    log.info("[EVENT:decoder_eof] Stream EOF")
                    break

                except av.AVError as e:
                    consecutive_errors += 1
                    self.stats.record_error()
                    if consecutive_errors <= 3:
                        log.warning(
                            "[EVENT:decode_error] AVError (%d/%d): %s",
                            consecutive_errors, _MAX_CONSECUTIVE_ERRORS, e,
                        )
                    if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                        raise RuntimeError(
                            f"Too many consecutive decode errors ({consecutive_errors})"
                        ) from e
        finally:
            try:
                container.close()
            except Exception:
                pass

    def _run_ffmpeg_subprocess_decode(self) -> None:
        """Decode via ffmpeg subprocess — fallback matching v1's proven approach.

        Spawns ffmpeg to receive UDP MPEG-TS and pipe raw BGR24 frames to
        stdout. This works on all platforms including Windows where PyAV's
        direct UDP open can fail.
        """
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg not found in PATH — required for decode fallback")

        frame_size = self.width * self.height * 3  # BGR24

        cmd = [
            ffmpeg_path,
            "-loglevel", "warning",
            # Low-latency options (same as v1)
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "500000",
            "-probesize", "500000",
            # Input
            "-f", "mpegts",
            "-i", f"udp://0.0.0.0:{self.udp_port}?overrun_nonfatal=1",
            # Decode
            "-threads", "1",
            # Output — raw BGR24 frames to stdout
            "-pix_fmt", self.pixel_format,
            "-s", f"{self.width}x{self.height}",
            "-f", "rawvideo",
            "-an", "-sn",
            "pipe:1",
        ]

        log.info(
            "[EVENT:decoder_open] Starting ffmpeg subprocess on port %d (%dx%d %s)",
            self.udp_port, self.width, self.height, self.pixel_format,
        )

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=frame_size * 2,
        )

        self.stats.set_codec("ffmpeg-subprocess")

        try:
            while not self._stop_event.is_set():
                self.stats.begin_decode()
                raw = proc.stdout.read(frame_size)
                if len(raw) == 0:
                    log.info("[EVENT:decoder_eof] ffmpeg stdout EOF")
                    break
                if len(raw) < frame_size:
                    self.stats.record_drop()
                    continue

                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (self.height, self.width, 3)
                )

                if self._on_frame is not None:
                    self._on_frame(frame)

                self.stats.record_frame()

        finally:
            proc.stdout.close()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.info("[EVENT:decoder_close] ffmpeg subprocess terminated")

    def _scale_frame(
        self,
        rgb_frame: np.ndarray,
        av_frame,  # av.VideoFrame
        av_module,  # the av module
    ) -> Optional[np.ndarray]:
        """Scale a frame to the expected output dimensions using PyAV reformatter.

        Uses libswscale via PyAV for hardware-optimized scaling, avoiding
        numpy-based nearest-neighbor which is slower for large frames.

        Args:
            rgb_frame: The decoded frame as numpy array (may be wrong size).
            av_frame: The original av.VideoFrame (used for reformatting).
            av_module: The av module reference.

        Returns:
            Scaled numpy array (height, width, 3) or None if scaling fails.
        """
        try:
            # Use PyAV's built-in reformatter (backed by libswscale)
            reformatted = av_frame.reformat(
                width=self.width,
                height=self.height,
                format=self.pixel_format,
            )
            result = reformatted.to_ndarray()

            if result.shape != (self.height, self.width, 3):
                log.warning(
                    "[EVENT:decode_scale] Unexpected shape after reformat: %s",
                    result.shape,
                )
                return None

            return result

        except Exception as e:
            log.warning("[EVENT:decode_scale] Frame scaling failed: %s", e)
            return None
