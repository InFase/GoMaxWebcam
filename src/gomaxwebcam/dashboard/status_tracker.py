"""
dashboard/status_tracker.py — Camera status aggregation for SSE events.

CameraStatusTracker aggregates state from the Transport and FramePipeline
into a single CameraStatus dataclass. It runs periodic polling and notifies
SSE subscribers whenever the status changes.

The tracker is decoupled from the Transport/Pipeline via a simple polling
interface — no subclassing or callbacks required. This makes it easy to
test with mocks.

Thread safety:
    - _current is protected by _lock (read from SSE endpoint, written by poll)
    - _subscribers is an asyncio.Queue per listener, safe for async iteration
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import json

log = logging.getLogger("gomaxwebcam.dashboard.status")


@dataclass(frozen=True)
class CameraStatus:
    """Snapshot of camera status for SSE broadcast.

    All fields are JSON-serializable primitives.
    """
    # Transport
    transport_type: str = "none"           # "USB", "COHN", "WiFi AP", "none"
    connection_state: str = "DISCONNECTED" # TransportState name
    camera_model: str = ""                 # e.g. "GoPro@172.20.10.1"
    camera_serial: str = ""

    # Battery (from camera status queries; -1 = unknown)
    battery_level: int = -1

    # Pipeline
    pipeline_state: str = "STOPPED"        # PipelineState name
    is_frozen: bool = False

    # Stats
    decoder_fps: float = 0.0
    vcam_fps: float = 0.0
    decoder_frames: int = 0
    vcam_frames: int = 0
    uptime_s: float = 0.0
    keepalives_sent: int = 0
    keepalives_failed: int = 0
    last_error: str = ""

    # Frame metrics (v1 parity: latency, jitter, drop rate, health)
    last_decode_ms: float = 0.0     # Most recent frame decode time
    avg_decode_ms: float = 0.0      # Rolling average decode time
    avg_jitter_ms: float = 0.0      # Average inter-frame jitter
    drop_rate: float = 0.0          # Fraction of frames dropped
    pipeline_health: str = "unknown" # HealthStatus: healthy/degraded/unhealthy/unknown

    # Timestamp (monotonic, for change detection)
    timestamp: float = 0.0

    def to_sse_data(self) -> str:
        """Serialize to JSON string for SSE data field."""
        return json.dumps(asdict(self), default=str)

    def has_changed(self, other: CameraStatus) -> bool:
        """Check if status has meaningfully changed (ignoring timestamp)."""
        # Compare all fields except timestamp
        d1 = asdict(self)
        d2 = asdict(other)
        d1.pop("timestamp", None)
        d2.pop("timestamp", None)
        return d1 != d2


class CameraStatusTracker:
    """Aggregates transport + pipeline status and broadcasts via SSE.

    Usage:
        tracker = CameraStatusTracker()
        tracker.set_transport(transport)
        tracker.set_pipeline(pipeline)

        # Start polling (call from asyncio loop)
        await tracker.start(poll_interval=1.0)

        # In SSE endpoint:
        async for status in tracker.subscribe():
            yield status.to_sse_data()

        # Shutdown
        await tracker.stop()

    For testing without real transport/pipeline, use update_status() directly.
    """

    def __init__(self) -> None:
        self._transport: Any = None
        self._pipeline: Any = None
        self._current = CameraStatus()
        self._lock = asyncio.Lock()
        self._subscribers: list[asyncio.Queue[CameraStatus]] = []
        self._poll_task: Optional[asyncio.Task] = None
        self._battery_level: int = -1
        self._running = False

    # -- Configuration --

    def set_transport(self, transport: Any) -> None:
        """Set the active transport for status polling."""
        self._transport = transport

    def set_pipeline(self, pipeline: Any) -> None:
        """Set the active pipeline for status polling."""
        self._pipeline = pipeline

    def set_battery_level(self, level: int) -> None:
        """Update battery level from external source (e.g. BLE notification)."""
        self._battery_level = level

    # -- Current status --

    @property
    def current(self) -> CameraStatus:
        """Return the latest status snapshot (non-blocking read)."""
        return self._current

    async def update_status(self, status: CameraStatus) -> None:
        """Manually set the current status and notify subscribers.

        Useful for testing or when status comes from external sources.
        """
        old = self._current
        self._current = status

        if status.has_changed(old):
            await self._notify_subscribers(status)

    # -- Polling lifecycle --

    async def start(self, poll_interval: float = 1.0) -> None:
        """Start periodic status polling.

        Args:
            poll_interval: Seconds between polls (default 1.0).
        """
        if self._running:
            return
        self._running = True
        self._poll_task = asyncio.create_task(
            self._poll_loop(poll_interval),
            name="status-tracker-poll",
        )
        log.info("Status tracker started (poll_interval=%.1fs)", poll_interval)

    async def stop(self) -> None:
        """Stop polling and clean up subscribers."""
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        # Signal all subscribers to stop
        for q in self._subscribers:
            try:
                q.put_nowait(None)  # type: ignore[arg-type]
            except asyncio.QueueFull:
                pass
        self._subscribers.clear()
        log.info("Status tracker stopped")

    # -- SSE subscription --

    def subscribe(self) -> _StatusSubscription:
        """Create a new SSE subscription.

        Returns an async iterator that yields CameraStatus objects.
        The subscription is automatically cleaned up when the iterator exits.

        Usage:
            async for status in tracker.subscribe():
                yield f"data: {status.to_sse_data()}\\n\\n"
        """
        q: asyncio.Queue[CameraStatus | None] = asyncio.Queue(maxsize=16)
        self._subscribers.append(q)

        # Send current status immediately so the client doesn't start blank
        try:
            q.put_nowait(self._current)
        except asyncio.QueueFull:
            pass

        return _StatusSubscription(q, self._subscribers)

    @property
    def subscriber_count(self) -> int:
        """Number of active SSE subscribers."""
        return len(self._subscribers)

    # -- Internal --

    async def _poll_loop(self, interval: float) -> None:
        """Periodically poll transport/pipeline and broadcast changes."""
        while self._running:
            try:
                status = self._collect_status()
                old = self._current
                self._current = status

                if status.has_changed(old):
                    await self._notify_subscribers(status)

            except Exception:
                log.exception("Error in status poll")

            await asyncio.sleep(interval)

    def _collect_status(self) -> CameraStatus:
        """Build a CameraStatus from current transport + pipeline state."""
        transport_type = "none"
        connection_state = "DISCONNECTED"
        camera_model = ""
        camera_serial = ""
        keepalives_sent = 0
        keepalives_failed = 0
        last_error = ""

        if self._transport is not None:
            transport_type = getattr(self._transport, "name", "unknown")
            state = getattr(self._transport, "state", None)
            connection_state = state.name if state else "UNKNOWN"
            stats = getattr(self._transport, "stats", None)
            if stats:
                camera_model = getattr(stats, "camera_model", "") or ""
                camera_serial = getattr(stats, "camera_serial", "") or ""
                keepalives_sent = getattr(stats, "keepalives_sent", 0)
                keepalives_failed = getattr(stats, "keepalives_failed", 0)
                last_error = getattr(stats, "last_error", "") or ""

        pipeline_state = "STOPPED"
        is_frozen = False
        decoder_fps = 0.0
        vcam_fps = 0.0
        decoder_frames = 0
        vcam_frames = 0
        uptime_s = 0.0
        last_decode_ms = 0.0
        avg_decode_ms = 0.0
        avg_jitter_ms = 0.0
        drop_rate = 0.0
        pipeline_health = "unknown"

        if self._pipeline is not None:
            p_state = getattr(self._pipeline, "state", None)
            pipeline_state = p_state.name if p_state else "UNKNOWN"
            is_frozen = getattr(self._pipeline, "is_frozen", False)

            get_stats = getattr(self._pipeline, "get_stats", None)
            if get_stats:
                try:
                    ps = get_stats()
                    decoder_fps = getattr(ps, "decoder_fps", 0.0)
                    vcam_fps = getattr(ps, "vcam_fps", 0.0)
                    decoder_frames = getattr(ps, "decoder_frames", 0)
                    vcam_frames = getattr(ps, "vcam_frames", 0)
                    uptime_s = getattr(ps, "uptime_s", 0.0)
                    # Frame metrics
                    last_decode_ms = getattr(ps, "last_decode_ms", 0.0)
                    avg_decode_ms = getattr(ps, "avg_decode_ms", 0.0)
                    avg_jitter_ms = getattr(ps, "avg_jitter_ms", 0.0)
                    drop_rate = getattr(ps, "drop_rate", 0.0)
                    pipeline_health = getattr(ps, "health", "unknown")
                except Exception:
                    pass

        return CameraStatus(
            transport_type=transport_type,
            connection_state=connection_state,
            camera_model=camera_model,
            camera_serial=camera_serial,
            battery_level=self._battery_level,
            pipeline_state=pipeline_state,
            is_frozen=is_frozen,
            decoder_fps=round(decoder_fps, 1),
            vcam_fps=round(vcam_fps, 1),
            decoder_frames=decoder_frames,
            vcam_frames=vcam_frames,
            uptime_s=round(uptime_s, 1),
            keepalives_sent=keepalives_sent,
            keepalives_failed=keepalives_failed,
            last_error=last_error,
            last_decode_ms=round(last_decode_ms, 2),
            avg_decode_ms=round(avg_decode_ms, 2),
            avg_jitter_ms=round(avg_jitter_ms, 2),
            drop_rate=round(drop_rate, 4),
            pipeline_health=pipeline_health,
            timestamp=time.monotonic(),
        )

    async def _notify_subscribers(self, status: CameraStatus) -> None:
        """Push status to all active subscriber queues."""
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            try:
                # Non-blocking: drop oldest if full (backpressure)
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(status)
            except Exception:
                dead.append(q)

        for q in dead:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass


class _StatusSubscription:
    """Async iterator for SSE status events.

    Automatically removes itself from the subscriber list on exit.
    """

    def __init__(
        self,
        queue: asyncio.Queue[CameraStatus | None],
        subscribers: list[asyncio.Queue],
    ):
        self._queue = queue
        self._subscribers = subscribers

    def __aiter__(self):
        return self

    async def __anext__(self) -> CameraStatus:
        item = await self._queue.get()
        if item is None:
            # Sentinel: tracker stopped or subscription cancelled
            self._cleanup()
            raise StopAsyncIteration
        return item

    def _cleanup(self) -> None:
        try:
            self._subscribers.remove(self._queue)
        except ValueError:
            pass
