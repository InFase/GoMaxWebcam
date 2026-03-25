"""
pipeline/frame_pipeline.py — Integration wiring: Transport → Decoder → VirtualCamera.

The FramePipeline v2 is the central orchestration component that connects:
  1. Transport (USB/COHN/WiFi AP) — provides stream info (UDP port, resolution)
  2. UDPDecoder (PyAV) — decodes MPEG-TS UDP stream to BGR24 frames on decode thread
  3. VirtualCameraSink — pushes frames to the virtual camera device on vcam thread

Architecture:

    Transport.start_stream()
         ↓
    [UDP MPEG-TS stream over network]
         ↓
    UDPDecoder (dedicated decode thread)
         ↓  (callback-based frame delivery)
    VirtualCameraSink (dedicated consumer thread, bounded queue)
         ↓
    pyvirtualcam device → Zoom/Teams/OBS

Thread model:
    - Asyncio thread: transport control, orchestration, pipeline lifecycle
    - Decode thread: UDPDecoder._run_decode_loop() — owns all PyAV objects
    - VCam thread: VirtualCameraSink._consumer_loop() — owns pyvirtualcam
    - No thread touches another's resources — queue is the only boundary

Backpressure:
    The VirtualCameraSink has an internal bounded queue (max 2 frames).
    When the decode thread produces frames faster than the vcam thread
    consumes them (unlikely at 30fps, but possible during burst decode):
    - submit_frame() drops the oldest queued frame
    - Memory stays bounded regardless of decode speed
    - Decode thread never blocks on a full queue

Freeze-frame:
    When the decoder stops producing frames (stream lost):
    - VCam thread detects empty queue and re-sends last good frame
    - Virtual camera device stays visible to downstream apps
    - On stream recovery, fresh frames flow through automatically

Frame timing validation:
    The pipeline monitors frame timing to ensure 1080p 30fps output:
    - Inter-frame interval target: 33.3ms ±10%
    - FPS measurement over rolling 1-second windows
    - Dashboard reports actual vs target FPS
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Callable

import numpy as np

from gomaxwebcam.pipeline.decode import UDPDecoder, DecodeStats
from gomaxwebcam.pipeline.virtual_camera_sink import VirtualCameraSink
from gomaxwebcam.transport.base import Transport, TransportState, StreamInfo

log = logging.getLogger("gomaxwebcam.pipeline")


class PipelineState(Enum):
    """Pipeline lifecycle state."""
    STOPPED = auto()       # Nothing running
    STARTING = auto()      # Decoder/VCam starting up
    STREAMING = auto()     # Normal operation: live frames flowing
    FREEZE_FRAME = auto()  # Stream lost, pushing last good frame
    STOPPING = auto()      # Shutting down
    ERROR = auto()         # Unrecoverable error



#: Codecs supported by the UDPDecoder / PyAV MPEG-TS pipeline.
#: GoPro Hero 13 sends H.264 or HEVC depending on model/firmware;
#: MJPEG is accepted for forward-compatibility but arrives as MPEG-TS too.
SUPPORTED_CODECS = frozenset({"h264", "hevc", "h265", "mjpeg", "mpeg4"})


@dataclass
class PipelineConfig:
    """Configuration for the frame pipeline."""
    width: int = 1920
    height: int = 1080
    fps: int = 30
    device_name: str = "GoMaxWebcam"
    # Decoder settings
    udp_port: int = 8554
    decoder_connect_timeout: float = 10.0
    # Pixel format: "bgr24" matches v1's ffmpeg output and Unity Capture native format
    pixel_format: str = "bgr24"
    # Freeze-frame threshold: consecutive empty queue polls before declaring freeze
    freeze_threshold_frames: int = 10

    @classmethod
    def from_stream_info(
        cls,
        stream_info: "StreamInfo",
        device_name: str = "GoMaxWebcam",
        pixel_format: str = "bgr24",
    ) -> "PipelineConfig":
        """Create a PipelineConfig from a transport's StreamInfo.

        Maps stream dimensions, port, and FPS from the transport into
        a pipeline config suitable for creating a FramePipeline.  The
        codec from StreamInfo is validated against SUPPORTED_CODECS.

        Args:
            stream_info: StreamInfo from ``Transport.start_stream()``.
            device_name: Virtual camera device name for downstream apps.
            pixel_format: Output pixel format (default ``"bgr24"`` for
                v1 parity with Unity Capture).

        Returns:
            A PipelineConfig pre-populated from the stream metadata.

        Raises:
            ValueError: If the stream codec is not in SUPPORTED_CODECS.
        """
        codec = (stream_info.codec or "h264").lower()
        if codec not in SUPPORTED_CODECS:
            raise ValueError(
                f"Unsupported codec '{codec}' — "
                f"expected one of {sorted(SUPPORTED_CODECS)}"
            )

        return cls(
            width=stream_info.width,
            height=stream_info.height,
            fps=stream_info.fps,
            device_name=device_name,
            udp_port=stream_info.port,
            pixel_format=pixel_format,
        )


class HealthStatus(Enum):
    """Pipeline health assessment level.

    Based on v1 performance baselines:
      - HEALTHY: FPS >= 25, latency < 50ms, drop rate < 1%
      - DEGRADED: FPS 15-25, or latency 50-100ms, or drop rate 1-5%
      - UNHEALTHY: FPS < 15, or latency > 100ms, or drop rate > 5%
      - UNKNOWN: Pipeline not running or insufficient data
    """
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class PipelineHealth:
    """Computed health assessment for the pipeline.

    Evaluates pipeline performance against v1 baselines and produces
    a single ``status`` plus per-metric details. Used by the dashboard
    health indicator and ``/api/pipeline/health`` endpoint.
    """
    status: str = "unknown"          # HealthStatus value
    fps_ok: bool = True
    latency_ok: bool = True
    drop_rate_ok: bool = True
    details: str = ""

    # Per-metric values (for dashboard display)
    decode_fps: float = 0.0
    vcam_fps: float = 0.0
    avg_decode_ms: float = 0.0
    max_decode_ms: float = 0.0
    avg_jitter_ms: float = 0.0
    drop_rate: float = 0.0
    frames_decoded: int = 0
    frames_dropped: int = 0
    uptime_s: float = 0.0


@dataclass
class PipelineStats:
    """Runtime statistics for the frame pipeline.

    Extended with per-frame latency, jitter, drop rate, and health
    assessment to match v1 performance baselines for dashboard reporting.
    """
    state: str = "STOPPED"
    decoder_frames: int = 0
    decoder_dropped: int = 0
    decoder_fps: float = 0.0
    decoder_errors: int = 0
    vcam_frames: int = 0
    vcam_freeze_frames: int = 0
    vcam_fps: float = 0.0
    vcam_queue_depth: int = 0
    uptime_s: float = 0.0
    is_frozen: bool = False

    # Latency and timing metrics
    last_decode_ms: float = 0.0     # Most recent frame decode time
    avg_decode_ms: float = 0.0      # Rolling average decode time
    max_decode_ms: float = 0.0      # Rolling max decode time
    avg_jitter_ms: float = 0.0      # Average inter-frame jitter
    drop_rate: float = 0.0          # Fraction of frames dropped

    # Health assessment
    health: str = "unknown"         # HealthStatus value


class FramePipeline:
    """Connects Transport → UDPDecoder → VirtualCameraSink with backpressure.

    The pipeline manages the lifecycle of the decoder and virtual camera
    sink, wiring them together. The UDPDecoder delivers frames via callback
    to VirtualCameraSink.submit_frame(), which queues them for the vcam
    consumer thread.

    Usage:
        pipeline = FramePipeline(config)

        # Start from stream info (transport already started the stream)
        await pipeline.start(stream_info)

        # Monitor
        stats = pipeline.get_stats()

        # Stop
        await pipeline.stop()

    Or, start from a transport:
        await pipeline.start_from_transport(transport)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self._config = config or PipelineConfig()
        self._decoder: Optional[UDPDecoder] = None
        self._vcam_sink: Optional[VirtualCameraSink] = None
        self._transport: Optional[Transport] = None

        # State
        self._state = PipelineState.STOPPED
        self._state_lock = threading.Lock()
        self._start_time: Optional[float] = None

        # Freeze detection: tracks whether the vcam sink is in freeze mode
        self._was_frozen = False

        # Callbacks
        self.on_state_change: Optional[Callable[[PipelineState], None]] = None
        self.on_freeze: Optional[Callable[[], None]] = None
        self.on_unfreeze: Optional[Callable[[], None]] = None

    # -- Properties --

    @property
    def state(self) -> PipelineState:
        with self._state_lock:
            return self._state

    @property
    def is_running(self) -> bool:
        return self._state in (
            PipelineState.STARTING,
            PipelineState.STREAMING,
            PipelineState.FREEZE_FRAME,
        )

    @property
    def is_streaming(self) -> bool:
        return self._state == PipelineState.STREAMING

    @property
    def is_frozen(self) -> bool:
        return self._state == PipelineState.FREEZE_FRAME

    @property
    def decoder(self) -> Optional[UDPDecoder]:
        return self._decoder

    @property
    def vcam_sink(self) -> Optional[VirtualCameraSink]:
        return self._vcam_sink

    # -- Lifecycle --

    async def start(self, stream_info: StreamInfo) -> bool:
        """Start the pipeline from stream info.

        Creates and starts the decoder and virtual camera sink, wiring
        them together via submit_frame callback.

        The wiring:
          UDPDecoder.on_frame → VirtualCameraSink.submit_frame

        Args:
            stream_info: StreamInfo from a transport's start_stream().

        Returns:
            True if pipeline started successfully.
        """
        if self.is_running:
            log.warning("Pipeline already running")
            return True

        self._set_state(PipelineState.STARTING)
        self._start_time = time.monotonic()
        self._was_frozen = False

        try:
            # Create the virtual camera sink (1080p 30fps, "GoMaxWebcam")
            self._vcam_sink = VirtualCameraSink(
                device_name=self._config.device_name,
            )

            # Create the decoder, wired to push frames into the vcam sink
            # pixel_format defaults to "bgr24" matching v1's ffmpeg output
            self._decoder = UDPDecoder(
                udp_port=stream_info.port,
                width=stream_info.width,
                height=stream_info.height,
                pixel_format=self._config.pixel_format,
                on_frame=self._on_decoded_frame,
                on_error=self._on_decoder_error,
                on_stopped=self._on_decoder_stopped,
            )

            # Start virtual camera sink first (opens device, starts consumer thread)
            if not self._vcam_sink.start():
                log.error("Failed to start virtual camera sink")
                self._set_state(PipelineState.ERROR)
                return False

            # Start decoder (opens UDP input, starts decode thread)
            if not self._decoder.start():
                log.error("Failed to start decoder")
                self._vcam_sink.stop()
                self._set_state(PipelineState.ERROR)
                return False

            self._set_state(PipelineState.STREAMING)
            log.info(
                "Pipeline started: UDP:%d → decoder → vcam (%dx%d@%dfps, device='%s')",
                stream_info.port,
                self._config.width,
                self._config.height,
                self._config.fps,
                self._config.device_name,
            )
            return True

        except Exception:
            log.exception("Failed to start pipeline")
            await self._cleanup()
            self._set_state(PipelineState.ERROR)
            return False

    async def start_from_transport(self, transport: Transport) -> bool:
        """Start the pipeline by first starting the transport stream.

        Convenience method that calls transport.start_stream() and
        then wires the resulting StreamInfo into the pipeline.

        Args:
            transport: A connected transport (state must be CONNECTED).

        Returns:
            True if the transport stream and pipeline started.
        """
        if not transport.is_connected:
            log.error(
                "Cannot start pipeline: transport not connected (state=%s)",
                transport.state.name,
            )
            return False

        self._transport = transport

        stream_info = await transport.start_stream()
        if stream_info is None:
            log.error("Transport failed to start stream")
            return False

        return await self.start(stream_info)

    async def stop(self) -> None:
        """Stop the pipeline, decoder, and virtual camera sink."""
        if self._state == PipelineState.STOPPED:
            return

        log.info("Stopping pipeline...")
        self._set_state(PipelineState.STOPPING)
        await self._cleanup()
        self._set_state(PipelineState.STOPPED)

        uptime = time.monotonic() - self._start_time if self._start_time else 0
        log.info("Pipeline stopped (uptime=%.1fs)", uptime)

    async def restart(self, stream_info: StreamInfo) -> bool:
        """Restart the pipeline with new stream info.

        Stops the current pipeline and starts a new one.
        Used after transport reconnection.
        """
        await self.stop()
        return await self.start(stream_info)

    async def switch_stream(self, stream_info: StreamInfo) -> bool:
        """Switch to a new stream source while keeping the vcam sink alive.

        Used during failover to avoid virtual camera device disconnection.
        The vcam sink continues re-sending the last good frame (freeze-frame)
        during the decoder swap, so downstream apps (Zoom, Teams, NVIDIA
        Broadcast) see an uninterrupted video feed.

        Flow:
          1. Stop the old decoder (if any)
          2. Create a new decoder pointed at the new stream's UDP port
          3. Wire the new decoder's output to the existing vcam sink
          4. Start the new decoder — first decoded frame will trigger
             FREEZE_FRAME → STREAMING transition via _on_decoded_frame

        Args:
            stream_info: StreamInfo from the new transport's start_stream().

        Returns:
            True if the decoder was successfully swapped and started.
        """
        if self._vcam_sink is None or not self._vcam_sink.is_running:
            # No vcam sink to preserve — fall back to full restart
            log.info("switch_stream: no active vcam sink, falling back to full restart")
            return await self.restart(stream_info)

        log.info(
            "switch_stream: swapping decoder to UDP:%d (vcam sink stays alive)",
            stream_info.port,
        )

        # Step 1: Stop old decoder (produces no more frames, vcam freezes)
        if self._decoder is not None:
            self._decoder.stop()
            self._decoder = None

        # Step 2: Create new decoder wired to existing vcam sink
        try:
            self._decoder = UDPDecoder(
                udp_port=stream_info.port,
                width=stream_info.width,
                height=stream_info.height,
                pixel_format=self._config.pixel_format,
                on_frame=self._on_decoded_frame,
                on_error=self._on_decoder_error,
                on_stopped=self._on_decoder_stopped,
            )

            # Step 3: Start new decoder
            if not self._decoder.start():
                log.error("switch_stream: failed to start new decoder")
                self._set_state(PipelineState.ERROR)
                return False

            log.info(
                "switch_stream: new decoder started on UDP:%d, "
                "waiting for first frame to unfreeze",
                stream_info.port,
            )
            return True

        except Exception:
            log.exception("switch_stream: failed to create new decoder")
            self._set_state(PipelineState.ERROR)
            return False

    async def _cleanup(self) -> None:
        """Clean up decoder and virtual camera resources.

        Stop order matters: decoder first (stop producing), then sink.
        """
        if self._decoder is not None:
            self._decoder.stop()
            self._decoder = None

        if self._vcam_sink is not None:
            self._vcam_sink.stop()
            self._vcam_sink = None

    # -- Frame delivery callback (called from decode thread) --

    def _on_decoded_frame(self, frame: np.ndarray) -> None:
        """Callback from UDPDecoder when a frame is decoded.

        Receives a BGR24 numpy array (H, W, 3) dtype uint8 and submits it
        to the VirtualCameraSink's bounded queue. If the queue is full,
        the sink drops the oldest frame (newest-wins backpressure).

        This runs on the decode thread — must be fast and non-blocking.
        """
        if self._vcam_sink is not None:
            self._vcam_sink.submit_frame(frame)

            # Track freeze→unfreeze transitions
            if self._was_frozen:
                self._was_frozen = False
                self._set_state(PipelineState.STREAMING)
                log.info("Pipeline: live frames resumed after freeze")
                if self.on_unfreeze:
                    try:
                        self.on_unfreeze()
                    except Exception:
                        log.exception("Error in on_unfreeze callback")

    def _on_decoder_error(self, error: Exception) -> None:
        """Callback from UDPDecoder on fatal decode error.

        The decoder has given up — the stream is unrecoverable without
        reconnection. The vcam sink continues pushing freeze-frames.
        """
        log.error("Decoder fatal error: %s", error)
        if not self._was_frozen and self._state == PipelineState.STREAMING:
            self._was_frozen = True
            self._set_state(PipelineState.FREEZE_FRAME)
            log.info("Pipeline entered freeze-frame mode (decoder error)")
            if self.on_freeze:
                try:
                    self.on_freeze()
                except Exception:
                    log.exception("Error in on_freeze callback")

    def _on_decoder_stopped(self) -> None:
        """Callback from UDPDecoder when the decode loop exits.

        The decoder thread has exited (stream EOF, error, or stop signal).
        If not already stopping, enter freeze-frame mode.
        """
        if self._state in (PipelineState.STOPPING, PipelineState.STOPPED):
            return

        if not self._was_frozen:
            self._was_frozen = True
            self._set_state(PipelineState.FREEZE_FRAME)
            log.info("Pipeline entered freeze-frame mode (decoder stopped)")
            if self.on_freeze:
                try:
                    self.on_freeze()
                except Exception:
                    log.exception("Error in on_freeze callback")

    # -- Explicit freeze/unfreeze (called by orchestrator during failover) --

    def freeze(self) -> None:
        """Explicitly enter freeze-frame mode.

        Called by the orchestrator when the TransportManager detects a
        disconnect and initiates failover.  This forces an immediate
        transition to FREEZE_FRAME without waiting for the decoder's
        frame-timeout to fire, giving sub-second visual continuity.

        Safe to call from any thread (state lock is used internally).
        No-op if already frozen or not running.
        """
        if self._state not in (PipelineState.STREAMING, PipelineState.STARTING):
            return

        self._was_frozen = True
        self._set_state(PipelineState.FREEZE_FRAME)
        log.info("Pipeline: explicit freeze (failover in progress)")
        if self.on_freeze:
            try:
                self.on_freeze()
            except Exception:
                log.exception("Error in on_freeze callback")

    def unfreeze(self) -> None:
        """Signal that failover completed and fresh frames are expected.

        Called by the orchestrator after the TransportManager successfully
        connects a new transport.  The pipeline stays in FREEZE_FRAME
        state (the vcam sink keeps re-sending the last good frame) until
        the first decoded frame actually arrives from the new transport,
        at which point ``_on_decoded_frame`` transitions back to STREAMING.

        This method resets the ``_was_frozen`` flag so the next decoded
        frame will trigger the FREEZE_FRAME → STREAMING transition and
        fire the ``on_unfreeze`` callback.

        Safe to call from any thread.  No-op if not frozen.
        """
        if self._state != PipelineState.FREEZE_FRAME:
            return

        log.info(
            "Pipeline: unfreeze signalled — will resume on first decoded frame"
        )
        # NOTE: we do NOT change state here.  The vcam sink continues
        # re-sending freeze-frames.  State transitions to STREAMING
        # only when _on_decoded_frame receives a real frame, which
        # resets _was_frozen and fires on_unfreeze.

    # -- State --

    def _set_state(self, new_state: PipelineState) -> None:
        with self._state_lock:
            old = self._state
            if old == new_state:
                return
            self._state = new_state
        log.info("Pipeline state: %s -> %s", old.name, new_state.name)
        if self.on_state_change:
            try:
                self.on_state_change(new_state)
            except Exception:
                log.exception("Error in pipeline state callback")

    # -- Statistics --

    def get_stats(self) -> PipelineStats:
        """Return pipeline statistics for the dashboard.

        Includes per-frame decode latency, inter-frame jitter, drop rate,
        and a computed health assessment against v1 performance baselines.
        """
        uptime = time.monotonic() - self._start_time if self._start_time else 0.0

        stats = PipelineStats(
            state=self._state.name,
            uptime_s=round(uptime, 1),
            is_frozen=self.is_frozen,
        )

        if self._decoder is not None:
            snap = self._decoder.stats.snapshot()
            stats.decoder_frames = snap["frames_decoded"]
            stats.decoder_dropped = snap["frames_dropped"]
            stats.decoder_fps = snap["decode_fps"]
            stats.decoder_errors = snap["errors"]
            # Latency and timing metrics from enhanced DecodeStats
            stats.last_decode_ms = snap.get("last_decode_ms", 0.0)
            stats.avg_decode_ms = snap.get("avg_decode_ms", 0.0)
            stats.max_decode_ms = snap.get("max_decode_ms", 0.0)
            stats.avg_jitter_ms = snap.get("avg_jitter_ms", 0.0)
            stats.drop_rate = snap.get("drop_rate", 0.0)

        if self._vcam_sink is not None:
            stats.vcam_frames = self._vcam_sink.frames_sent
            stats.vcam_freeze_frames = self._vcam_sink.freeze_frames_sent
            stats.vcam_fps = self._vcam_sink.fps_actual
            stats.vcam_queue_depth = self._vcam_sink.get_stats()["queue_size"]

        # Compute health assessment
        stats.health = self._assess_health(stats).status

        return stats

    def get_health(self) -> PipelineHealth:
        """Return a computed health assessment for the pipeline.

        Evaluates current pipeline metrics against v1 performance baselines:
          - FPS: target 30fps, warn below 25, critical below 15
          - Decode latency: warn above 50ms, critical above 100ms
          - Drop rate: warn above 1%, critical above 5%

        Returns:
            PipelineHealth with status (healthy/degraded/unhealthy/unknown)
            and per-metric breakdown for dashboard display.
        """
        stats = self.get_stats()
        return self._assess_health(stats)

    def _assess_health(self, stats: PipelineStats) -> PipelineHealth:
        """Assess pipeline health from stats against v1 baselines."""
        from gomaxwebcam.pipeline.decode import DecodeStats

        health = PipelineHealth(
            decode_fps=stats.decoder_fps,
            vcam_fps=stats.vcam_fps,
            avg_decode_ms=stats.avg_decode_ms,
            max_decode_ms=stats.max_decode_ms,
            avg_jitter_ms=stats.avg_jitter_ms,
            drop_rate=stats.drop_rate,
            frames_decoded=stats.decoder_frames,
            frames_dropped=stats.decoder_dropped,
            uptime_s=stats.uptime_s,
        )

        # Not running or no data → unknown
        if self._state in (PipelineState.STOPPED, PipelineState.ERROR):
            health.status = HealthStatus.UNKNOWN.value
            health.details = "Pipeline not running"
            return health

        if stats.decoder_frames < 10:
            health.status = HealthStatus.UNKNOWN.value
            health.details = "Insufficient data (< 10 frames decoded)"
            return health

        issues: list[str] = []
        worst = HealthStatus.HEALTHY

        # FPS check
        if stats.decoder_fps < 15.0:
            health.fps_ok = False
            worst = HealthStatus.UNHEALTHY
            issues.append(f"FPS critically low: {stats.decoder_fps:.1f}")
        elif stats.decoder_fps < DecodeStats.BASELINE_FPS_MIN:
            health.fps_ok = False
            if worst != HealthStatus.UNHEALTHY:
                worst = HealthStatus.DEGRADED
            issues.append(f"FPS below baseline: {stats.decoder_fps:.1f}")

        # Latency check
        if stats.avg_decode_ms > DecodeStats.BASELINE_LATENCY_CRIT_MS:
            health.latency_ok = False
            worst = HealthStatus.UNHEALTHY
            issues.append(f"Decode latency critical: {stats.avg_decode_ms:.1f}ms")
        elif stats.avg_decode_ms > DecodeStats.BASELINE_LATENCY_WARN_MS:
            health.latency_ok = False
            if worst != HealthStatus.UNHEALTHY:
                worst = HealthStatus.DEGRADED
            issues.append(f"Decode latency high: {stats.avg_decode_ms:.1f}ms")

        # Drop rate check
        if stats.drop_rate > DecodeStats.BASELINE_DROP_RATE_CRIT:
            health.drop_rate_ok = False
            worst = HealthStatus.UNHEALTHY
            issues.append(f"Drop rate critical: {stats.drop_rate:.1%}")
        elif stats.drop_rate > DecodeStats.BASELINE_DROP_RATE_WARN:
            health.drop_rate_ok = False
            if worst != HealthStatus.UNHEALTHY:
                worst = HealthStatus.DEGRADED
            issues.append(f"Drop rate elevated: {stats.drop_rate:.1%}")

        health.status = worst.value
        health.details = "; ".join(issues) if issues else "All metrics within v1 baselines"
        return health

    def get_detailed_stats(self) -> dict:
        """Return detailed stats including sub-component stats and health."""
        base: dict = {
            "pipeline": {
                "state": self._state.name,
                "uptime_s": (
                    round(time.monotonic() - self._start_time, 1)
                    if self._start_time
                    else 0.0
                ),
                "is_frozen": self.is_frozen,
            },
        }
        if self._decoder:
            base["decoder"] = self._decoder.stats.snapshot()
        if self._vcam_sink:
            base["vcam"] = self._vcam_sink.get_stats()

        # Include health assessment
        from dataclasses import asdict
        health = self.get_health()
        base["health"] = asdict(health)

        return base


def validate_1080p_30fps(
    frame: np.ndarray,
    fps: float,
    tolerance: float = 0.1,
) -> dict:
    """Validate that a frame and FPS match 1080p 30fps spec.

    The frame should be BGR24 (matching v1 format) but this function
    only checks shape and dtype — it does not verify channel order.

    Args:
        frame: numpy array to validate (BGR24 or RGB24, H×W×3, uint8).
        fps: Measured FPS to validate.
        tolerance: Allowed FPS deviation as fraction (default 10%).

    Returns:
        Dict with validation results:
          - resolution_ok: bool
          - fps_ok: bool
          - width: int
          - height: int
          - channels: int
          - fps_actual: float
          - fps_target: int
          - errors: list[str]
          - valid: bool (all checks pass)
    """
    errors: list[str] = []
    h, w = frame.shape[:2] if frame.ndim >= 2 else (0, 0)
    channels = frame.shape[2] if frame.ndim == 3 else 0

    resolution_ok = w == 1920 and h == 1080 and channels == 3
    if not resolution_ok:
        errors.append(
            f"Resolution mismatch: got {w}x{h}x{channels}, expected 1920x1080x3"
        )

    fps_target = 30
    fps_ok = abs(fps - fps_target) <= fps_target * tolerance
    if not fps_ok:
        errors.append(
            f"FPS mismatch: got {fps:.1f}, expected {fps_target} "
            f"(±{tolerance * 100:.0f}%)"
        )

    dtype_ok = frame.dtype == np.uint8
    if not dtype_ok:
        errors.append(f"Dtype mismatch: got {frame.dtype}, expected uint8")

    return {
        "resolution_ok": resolution_ok,
        "fps_ok": fps_ok,
        "dtype_ok": dtype_ok,
        "width": w,
        "height": h,
        "channels": channels,
        "fps_actual": fps,
        "fps_target": fps_target,
        "errors": errors,
        "valid": resolution_ok and fps_ok and dtype_ok,
    }
