"""
transport/base.py — Abstract base class for GoPro camera transports.

A Transport encapsulates one way of connecting to a GoPro camera:
  - USB (NCM adapter, wired)
  - COHN (Camera On Home Network, WiFi station mode)
  - WiFi AP (GoPro's own access point)

Each transport handles:
  1. Discovery / enumeration of the camera on its medium
  2. Connection setup (HTTP API handshake, authentication)
  3. Starting the preview stream (camera sends MPEG-TS over UDP)
  4. Keep-alive pings
  5. Clean shutdown / stream stop

The Transport ABC provides the contract that the pipeline and
orchestrator depend on. Concrete implementations live in their
own modules (usb.py, cohn.py, wifi_ap.py).

All methods are async — transports run on the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Optional

log = logging.getLogger("gomaxwebcam.transport")


class TransportState(Enum):
    """Lifecycle state of a transport connection."""
    DISCONNECTED = auto()    # No camera connected
    DISCOVERING = auto()     # Searching for camera on this medium
    CONNECTING = auto()      # Handshaking / authenticating
    CONNECTED = auto()       # Camera connected, ready to start stream
    STREAMING = auto()       # Preview stream active (MPEG-TS flowing)
    ERROR = auto()           # Unrecoverable error (must reconnect)


@dataclass(frozen=True)
class StreamInfo:
    """Metadata about an active preview stream.

    Returned by start_stream() to tell the decoder where to listen.
    """
    protocol: str = "udp"            # "udp" or "tcp"
    host: str = "0.0.0.0"           # Listen address
    port: int = 8554                 # UDP port for MPEG-TS
    width: int = 1920               # Expected stream width
    height: int = 1080              # Expected stream height
    fps: int = 30                   # Expected frame rate
    codec: str = "h264"             # Video codec in the stream


@dataclass
class TransportStats:
    """Runtime statistics for a transport connection."""
    keepalives_sent: int = 0
    keepalives_failed: int = 0
    connection_uptime_s: float = 0.0
    last_error: Optional[str] = None
    camera_model: Optional[str] = None
    camera_serial: Optional[str] = None
    transport_type: str = ""


class Transport(ABC):
    """Abstract base class for GoPro camera transports.

    Subclasses must implement all abstract methods. The orchestrator
    calls these in order:

        discover() → connect() → start_stream() → [keep_alive loop] → stop_stream() → disconnect()

    On errors, the orchestrator calls disconnect() and retries.

    Properties:
        state: Current TransportState.
        name: Human-readable transport name (e.g. "USB", "COHN").
        stream_info: StreamInfo if streaming, else None.
    """

    def __init__(self, name: str = "Transport"):
        self._name = name
        self._state = TransportState.DISCONNECTED
        self._stream_info: Optional[StreamInfo] = None
        self._state_listeners: list = []
        self._stats = TransportStats(transport_type=name)
        self._disconnect_listeners: list = []

    # -- Properties --

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> TransportState:
        return self._state

    @property
    def stream_info(self) -> Optional[StreamInfo]:
        return self._stream_info

    @property
    def stats(self) -> TransportStats:
        return self._stats

    @property
    def is_connected(self) -> bool:
        return self._state in (TransportState.CONNECTED, TransportState.STREAMING)

    @property
    def is_streaming(self) -> bool:
        return self._state == TransportState.STREAMING

    # -- State management --

    def _set_state(self, new_state: TransportState) -> None:
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        log.info("[%s] State: %s -> %s", self._name, old.name, new_state.name)
        for listener in self._state_listeners:
            try:
                listener(old, new_state)
            except Exception:
                log.exception("Error in transport state listener")

    def add_state_listener(self, callback) -> None:
        """Register a callback(old_state, new_state) for state transitions."""
        self._state_listeners.append(callback)

    def add_disconnect_listener(self, callback) -> None:
        """Register a callback(transport, reason) for unexpected disconnections.

        Called when the transport detects the camera has been lost (e.g.
        USB cable unplugged, health-check failures exceed threshold).
        This is distinct from state listeners — it fires only for
        *unexpected* disconnections, not intentional disconnect() calls.
        """
        self._disconnect_listeners.append(callback)

    def _emit_disconnect(self, reason: str) -> None:
        """Notify all disconnect listeners of an unexpected disconnection.

        Args:
            reason: Human-readable reason (e.g. "health_check_failed",
                    "keepalive_timeout", "usb_unplugged").
        """
        log.warning("[%s] Unexpected disconnect: %s", self._name, reason)
        for listener in self._disconnect_listeners:
            try:
                listener(self, reason)
            except Exception:
                log.exception("Error in disconnect listener")

    # -- Abstract interface --

    @abstractmethod
    async def discover(self, timeout: float = 10.0) -> bool:
        """Search for a GoPro camera on this transport.

        Returns True if a camera was found and is reachable.
        Sets state to DISCOVERING during search.
        """
        ...

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the discovered camera.

        Performs HTTP API handshake, sets webcam mode, etc.
        Returns True on success. Sets state to CONNECTED.
        """
        ...

    @abstractmethod
    async def start_stream(self) -> Optional[StreamInfo]:
        """Start the MPEG-TS preview stream from the camera.

        Returns StreamInfo telling the decoder where to listen,
        or None on failure. Sets state to STREAMING.
        """
        ...

    @abstractmethod
    async def stop_stream(self) -> None:
        """Stop the preview stream.

        Camera stops sending MPEG-TS. State returns to CONNECTED.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Fully disconnect from the camera.

        Clean up all resources. State returns to DISCONNECTED.
        """
        ...

    @abstractmethod
    async def keep_alive(self) -> bool:
        """Send a keep-alive ping to the camera.

        Returns True if the camera responded. The orchestrator calls
        this periodically (e.g. every 2.5s) while connected.
        """
        ...

    # -- Optional overrides --

    async def health_check(self) -> bool:
        """Check if the transport connection is healthy.

        Default implementation delegates to keep_alive().
        Subclasses can override for more thorough checks.
        """
        return await self.keep_alive()

    # -- Frame iteration --

    async def iter_frames(
        self,
        queue_size: int = 4,
        decode_timeout: float = 10.0,
        pixel_format: str = "bgr24",
    ) -> AsyncIterator:
        """Async generator yielding decoded BGR24 numpy frames from this transport.

        Encapsulates the full flow: start_stream() → UDPDecoder → yield frames.
        The transport must be in CONNECTED state before calling this method.

        Frames are numpy arrays of shape (height, width, 3) dtype uint8 in BGR24
        (matching v1's ffmpeg output and Unity Capture's native format).

        The generator handles cleanup on exit (StopIteration, GeneratorExit, or
        break): the decoder is stopped and resources released. The transport
        stream is NOT stopped — the caller is responsible for stop_stream().

        Args:
            queue_size: Max buffered frames between decode thread and async
                        generator. Small values (2-4) keep latency low.
            decode_timeout: Seconds to wait for the first frame before raising.
            pixel_format: Output pixel format, default "bgr24" for v1 parity.

        Yields:
            numpy.ndarray: BGR24 frames (H, W, 3) dtype uint8.

        Raises:
            RuntimeError: If the transport is not streaming or decoder fails.
            asyncio.TimeoutError: If no frame arrives within decode_timeout.

        Example::

            async for frame in transport.iter_frames():
                vcam_sink.submit_frame(frame)
        """
        import numpy as np
        from gomaxwebcam.pipeline.decode import UDPDecoder

        if self._state != TransportState.STREAMING:
            # Attempt to start the stream if connected
            if self._state == TransportState.CONNECTED:
                info = await self.start_stream()
                if info is None:
                    raise RuntimeError(
                        f"Failed to start stream on {self._name} transport"
                    )
            else:
                raise RuntimeError(
                    f"Cannot iterate frames: transport {self._name} is in "
                    f"state {self._state.name}, expected CONNECTED or STREAMING"
                )

        stream_info = self._stream_info
        if stream_info is None:
            raise RuntimeError("No stream info available after start_stream()")

        loop = asyncio.get_running_loop()
        frame_queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue(
            maxsize=queue_size,
        )
        error_holder: list[Optional[Exception]] = [None]

        def _on_frame(frame: np.ndarray) -> None:
            """Decode-thread callback: put frame into async queue."""
            def _enqueue() -> None:
                # Drop oldest if full (newest-wins backpressure)
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    frame_queue.put_nowait(frame)
                except asyncio.QueueFull:
                    pass  # Another frame beat us; drop this one
            try:
                loop.call_soon_threadsafe(_enqueue)
            except Exception:
                pass  # Queue closed or loop stopped

        def _on_error(error: Exception) -> None:
            """Decode-thread callback: report fatal error."""
            error_holder[0] = error
            try:
                loop.call_soon_threadsafe(frame_queue.put_nowait, None)
            except Exception:
                pass

        def _on_stopped() -> None:
            """Decode-thread callback: signal end of stream."""
            try:
                loop.call_soon_threadsafe(frame_queue.put_nowait, None)
            except Exception:
                pass

        decoder = UDPDecoder(
            udp_port=stream_info.port,
            width=stream_info.width,
            height=stream_info.height,
            pixel_format=pixel_format,
            on_frame=_on_frame,
            on_error=_on_error,
            on_stopped=_on_stopped,
        )

        if not decoder.start():
            raise RuntimeError("Failed to start UDPDecoder")

        try:
            # Wait for first frame with timeout
            first_frame = await asyncio.wait_for(
                frame_queue.get(), timeout=decode_timeout,
            )
            if first_frame is None:
                err = error_holder[0]
                raise RuntimeError(
                    f"Decoder stopped before first frame: {err}"
                )
            yield first_frame

            # Stream remaining frames
            while True:
                frame = await frame_queue.get()
                if frame is None:
                    # Decoder stopped or fatal error
                    err = error_holder[0]
                    if err is not None:
                        log.warning(
                            "[%s] iter_frames: decoder error: %s",
                            self._name, err,
                        )
                    return
                yield frame

        finally:
            decoder.stop(timeout=3.0)
            log.debug("[%s] iter_frames: decoder cleaned up", self._name)
