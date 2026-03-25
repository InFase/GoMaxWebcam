"""
virtual_camera_sink.py — Virtual camera output sink for GoMaxWebcam v2

Takes decoded BGR24 frames and writes them to a virtual camera device via
pyvirtualcam at fixed 1080p 30fps. The device appears as "GoMaxWebcam" in
downstream apps (Zoom, Teams, OBS, NVIDIA Broadcast, etc.).

Color format:
    BGR24 is used throughout the v2 pipeline, matching v1's ffmpeg -pix_fmt
    bgr24 output. Unity Capture (the primary Windows backend) accepts BGR
    natively, avoiding an extra channel swap.

Architecture:
    A dedicated consumer thread reads frames from a bounded queue and sends
    them to pyvirtualcam at the target cadence (30fps). The queue decouples
    the decode thread from the virtual camera output, preventing decode
    stalls from blocking the camera feed.

    Decode thread  ──(queue)──>  VCam consumer thread  ──>  pyvirtualcam

Freeze-frame resilience:
    When no new frames arrive (stream lost, reconnecting), the consumer
    thread keeps re-sending the last good frame at 30fps. Downstream apps
    see a frozen image instead of a black/missing feed. This is the v2
    equivalent of v1's send_last_frame() pattern.

Cross-platform:
    - Windows: Unity Capture or OBS VirtualCam backend
    - macOS: OBS VirtualCam backend
    - Linux: v4l2loopback backend

Thread safety:
    - submit_frame() is called from the decode thread (producer)
    - The consumer thread is the ONLY thread touching pyvirtualcam
    - start/stop use threading.Event for coordination
    - Stats are read from any thread via atomic-ish Python operations

Usage:
    sink = VirtualCameraSink()
    sink.start()                        # Opens device, starts consumer thread
    sink.submit_frame(rgb24_array)      # Called from decode thread
    sink.stop()                         # Closes device on app exit
"""

from __future__ import annotations

import logging
import platform
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger("gomaxwebcam.pipeline.virtual_camera_sink")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fixed output resolution — all frames are scaled to 1080p for the virtual cam
WIDTH = 1920
HEIGHT = 1080
FPS = 30

#: Virtual camera device name visible to downstream apps
DEVICE_NAME = "GoMaxWebcam"

#: Frame interval for 30fps timing
_FRAME_INTERVAL = 1.0 / FPS

#: Maximum frames buffered between decode and vcam threads.
#: At 30fps, 2 frames = ~67ms of buffering. Keeps latency low while
#: absorbing minor decode jitter.
_QUEUE_MAX_SIZE = 2

#: Dark gray placeholder shown before first real frame arrives
_PLACEHOLDER_COLOR = (40, 40, 40)

#: Timeout for consumer thread join on stop() — seconds
_THREAD_JOIN_TIMEOUT = 5.0

#: Timeout for queue.get() in consumer loop — allows periodic stop checks
_QUEUE_GET_TIMEOUT = 0.05  # 50ms


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _detect_backend() -> Optional[str]:
    """Auto-detect the best available pyvirtualcam backend for this platform.

    Returns:
        Backend name string for pyvirtualcam, or None if no backend found.
    """
    system = platform.system()

    if system == "Windows":
        # Prefer Unity Capture (supports custom device names), fall back to OBS
        for backend in ("unitycapture", "obs"):
            if _backend_available(backend):
                return backend
    elif system == "Darwin":
        if _backend_available("obs"):
            return "obs"
    elif system == "Linux":
        if _backend_available("v4l2loopback"):
            return "v4l2loopback"

    return None


def _backend_available(backend: str) -> bool:
    """Check if a pyvirtualcam backend is available without opening a device.

    Args:
        backend: Backend name ('unitycapture', 'obs', 'v4l2loopback').

    Returns:
        True if the backend appears to be installed/available.
    """
    try:
        import pyvirtualcam
        # pyvirtualcam doesn't expose a "check" API — we rely on the import
        # succeeding and the backend being compiled in. The actual device open
        # happens in start(). This is a best-effort check.
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# VirtualCameraSink
# ---------------------------------------------------------------------------

class VirtualCameraSink:
    """Consumes decoded frames and writes them to a virtual camera device.

    Fixed at 1080p 30fps with device name "GoMaxWebcam". Frames that don't
    match 1920x1080 are resized automatically.

    The consumer thread runs independently, pulling frames from a bounded
    queue and sending them to pyvirtualcam. If the queue is empty (no new
    frames), the last frame is re-sent to maintain the 30fps cadence
    (freeze-frame behavior).

    Attributes:
        width: Output width (always 1920).
        height: Output height (always 1080).
        fps: Output frame rate (always 30).
        device_name: Virtual camera device name.
        is_running: True when the consumer thread is active and device is open.
    """

    def __init__(
        self,
        device_name: str = DEVICE_NAME,
        backend: Optional[str] = None,
    ) -> None:
        """Initialize the virtual camera sink.

        Does NOT open the device — call start() to begin.

        Args:
            device_name: Virtual camera device name for downstream apps.
            backend: Force a specific pyvirtualcam backend, or None to
                auto-detect. One of 'unitycapture', 'obs', 'v4l2loopback'.
        """
        self.width: int = WIDTH
        self.height: int = HEIGHT
        self.fps: int = FPS
        self.device_name: str = device_name

        self._requested_backend: Optional[str] = backend
        self._active_backend: Optional[str] = None

        # pyvirtualcam.Camera instance (only touched by consumer thread)
        self._cam = None

        # Frame queue: decode thread submits, consumer thread reads
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=_QUEUE_MAX_SIZE,
        )

        # Last frame sent — used for freeze-frame when queue is empty
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_lock = threading.Lock()

        # Consumer thread
        self._consumer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Statistics (updated by consumer thread, read from any thread)
        self._frames_sent: int = 0
        self._freeze_frames_sent: int = 0
        self._frames_dropped: int = 0
        self._start_time: Optional[float] = None
        self._last_frame_time: Optional[float] = None
        self._fps_counter: int = 0
        self._fps_window_start: float = 0.0
        self._fps_actual: float = 0.0

    # -- Properties ----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True when the consumer thread is active and device is open."""
        return (
            not self._stop_event.is_set()
            and self._consumer_thread is not None
            and self._consumer_thread.is_alive()
        )

    @property
    def backend(self) -> Optional[str]:
        """The active pyvirtualcam backend name, or None if not started."""
        return self._active_backend

    @property
    def frames_sent(self) -> int:
        """Total frames sent to the virtual camera (live + freeze)."""
        return self._frames_sent

    @property
    def freeze_frames_sent(self) -> int:
        """Total freeze-frames re-sent when no new data was available."""
        return self._freeze_frames_sent

    @property
    def frames_dropped(self) -> int:
        """Frames dropped because the queue was full (decode too fast)."""
        return self._frames_dropped

    @property
    def fps_actual(self) -> float:
        """Measured output FPS over the last 1-second window."""
        return self._fps_actual

    @property
    def last_frame(self) -> Optional[np.ndarray]:
        """The last frame sent to the virtual camera (for external use)."""
        with self._last_frame_lock:
            return self._last_frame

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> bool:
        """Open the virtual camera device and start the consumer thread.

        After this returns True, the device appears in downstream app camera
        lists. The consumer thread begins sending placeholder frames at 30fps
        until real frames are submitted.

        Returns:
            True if the device was opened and consumer thread started.
        """
        if self.is_running:
            log.warning("Virtual camera sink already running")
            return True

        # Detect or validate backend
        backend = self._requested_backend or _detect_backend()
        if backend is None:
            log.error(
                "No virtual camera backend available. Install one of: "
                "Unity Capture (Windows), OBS VirtualCam (Windows/macOS), "
                "v4l2loopback (Linux)"
            )
            return False

        # Open the pyvirtualcam device
        try:
            import pyvirtualcam

            cam_kwargs: dict = {
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "fmt": pyvirtualcam.PixelFormat.BGR,
                "backend": backend,
            }

            # Unity Capture supports custom device names
            if backend == "unitycapture":
                cam_kwargs["device"] = self.device_name

            self._cam = pyvirtualcam.Camera(**cam_kwargs)
            self._active_backend = backend

            log.info(
                "Virtual camera opened: device='%s', backend=%s, "
                "resolution=%dx%d@%dfps, actual_device='%s'",
                self.device_name,
                backend,
                self.width,
                self.height,
                self.fps,
                self._cam.device,
            )

        except ImportError:
            log.error("pyvirtualcam not installed. Run: pip install pyvirtualcam")
            return False
        except RuntimeError as exc:
            log.error(
                "Failed to open virtual camera with backend '%s': %s",
                backend,
                exc,
            )
            return False

        # Reset state
        self._stop_event.clear()
        self._frames_sent = 0
        self._freeze_frames_sent = 0
        self._frames_dropped = 0
        self._start_time = time.monotonic()
        self._last_frame_time = None
        self._fps_counter = 0
        self._fps_window_start = time.monotonic()
        self._fps_actual = 0.0

        # Clear any stale frames in queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        with self._last_frame_lock:
            self._last_frame = None

        # Start consumer thread
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop,
            name="vcam-consumer",
            daemon=True,
        )
        self._consumer_thread.start()

        log.info("Virtual camera consumer thread started")
        return True

    def stop(self) -> None:
        """Stop the consumer thread and close the virtual camera device.

        After this, the device disappears from downstream app camera lists.
        Only call on app exit — during normal operation, keep the device
        open and rely on freeze-frame for resilience.
        """
        if self._stop_event.is_set() and self._consumer_thread is None:
            return

        log.info("Stopping virtual camera sink...")
        self._stop_event.set()

        # Wait for consumer thread to finish
        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=_THREAD_JOIN_TIMEOUT)
            if self._consumer_thread.is_alive():
                log.warning(
                    "Virtual camera consumer thread did not exit in %.1fs",
                    _THREAD_JOIN_TIMEOUT,
                )

        self._consumer_thread = None

        # Close pyvirtualcam device
        cam = self._cam
        self._cam = None
        if cam is not None:
            try:
                cam.close()
                log.info(
                    "Virtual camera closed after %d frames "
                    "(%d live, %d freeze)",
                    self._frames_sent,
                    self._frames_sent - self._freeze_frames_sent,
                    self._freeze_frames_sent,
                )
            except Exception as exc:
                log.warning("Error closing virtual camera: %s", exc)

        self._active_backend = None

    # -- Frame submission (called from decode thread) ------------------------

    def submit_frame(self, frame: np.ndarray) -> bool:
        """Submit a decoded frame for output to the virtual camera.

        Called from the decode thread. The frame is placed in a bounded
        queue for the consumer thread. If the queue is full (consumer is
        behind), the oldest frame is dropped to keep latency bounded.

        Args:
            frame: BGR24 numpy array of shape (H, W, 3) with dtype uint8.
                If not 1920x1080, it will be resized by the consumer.

        Returns:
            True if the frame was queued, False if the sink is stopped.
        """
        if self._stop_event.is_set():
            return False

        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            # Drop the oldest frame to make room — keeps latency bounded
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                self._frames_dropped += 1
                return False
            self._frames_dropped += 1

        return True

    # -- Consumer loop (dedicated thread) ------------------------------------

    def _consumer_loop(self) -> None:
        """Main consumer loop running on the dedicated vcam thread.

        Reads frames from the queue and sends them to pyvirtualcam.
        When the queue is empty, re-sends the last frame (freeze-frame)
        to maintain the 30fps cadence.
        """
        log.debug("Consumer loop started")

        # Send initial placeholder so downstream apps see valid video immediately
        self._send_placeholder()

        while not self._stop_event.is_set():
            try:
                frame = self._get_next_frame()
                if frame is not None:
                    self._send_to_device(frame, is_freeze=False)
                else:
                    # No new frame — send freeze-frame
                    self._send_freeze_frame()

                # FPS pacing: use pyvirtualcam's built-in timing
                self._sleep_until_next_frame()

                # Update FPS stats
                self._update_fps_stats()

            except Exception:
                log.exception("Error in virtual camera consumer loop")
                # Don't crash — try to keep going
                time.sleep(_FRAME_INTERVAL)

        log.debug("Consumer loop exited")

    def _get_next_frame(self) -> Optional[np.ndarray]:
        """Try to get the next frame from the queue.

        Returns:
            The next RGB24 frame, or None if the queue is empty.
        """
        try:
            return self._frame_queue.get(timeout=_QUEUE_GET_TIMEOUT)
        except queue.Empty:
            return None

    def _send_to_device(self, frame: np.ndarray, is_freeze: bool = False) -> bool:
        """Send a frame to the pyvirtualcam device.

        Handles resize if needed and updates statistics.

        Args:
            frame: RGB24 numpy array.
            is_freeze: True if this is a re-sent freeze-frame.

        Returns:
            True if the frame was sent successfully.
        """
        if self._cam is None:
            return False

        try:
            # Resize if dimensions don't match
            h, w = frame.shape[:2]
            if h != self.height or w != self.width:
                frame = self._resize_frame(frame)

            self._cam.send(frame)

            # Update last frame for freeze-frame
            with self._last_frame_lock:
                self._last_frame = frame

            # Update statistics
            self._frames_sent += 1
            if is_freeze:
                self._freeze_frames_sent += 1
            self._last_frame_time = time.monotonic()
            self._fps_counter += 1

            return True

        except Exception as exc:
            log.error("Failed to send frame to virtual camera: %s", exc)
            return False

    def _send_freeze_frame(self) -> bool:
        """Re-send the last good frame to maintain 30fps output.

        Called when no new frames are available from the decode thread.

        Returns:
            True if a freeze-frame was sent.
        """
        with self._last_frame_lock:
            frame = self._last_frame

        if frame is not None:
            return self._send_to_device(frame, is_freeze=True)

        # No frame ever received — send placeholder
        self._send_placeholder()
        return True

    def _send_placeholder(self) -> None:
        """Send a dark gray placeholder frame.

        Used on startup and when no real frames have been received.
        """
        if self._cam is None:
            return

        placeholder = np.full(
            (self.height, self.width, 3),
            _PLACEHOLDER_COLOR,
            dtype=np.uint8,
        )

        try:
            self._cam.send(placeholder)
            self._frames_sent += 1
            self._fps_counter += 1
            log.debug("Sent placeholder frame (%dx%d)", self.width, self.height)
        except Exception as exc:
            log.warning("Failed to send placeholder frame: %s", exc)

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame to the target 1920x1080 resolution.

        Uses numpy-based nearest-neighbor resize to avoid OpenCV dependency.
        For production quality, this could use Pillow or a more sophisticated
        algorithm, but nearest-neighbor is fast and sufficient for webcam use.

        Args:
            frame: RGB24 numpy array of any resolution.

        Returns:
            Resized RGB24 numpy array at 1920x1080.
        """
        src_h, src_w = frame.shape[:2]

        if src_h == self.height and src_w == self.width:
            return frame

        # Nearest-neighbor resize using numpy index mapping
        row_indices = (np.arange(self.height) * src_h / self.height).astype(int)
        col_indices = (np.arange(self.width) * src_w / self.width).astype(int)

        # Clamp to valid range
        row_indices = np.clip(row_indices, 0, src_h - 1)
        col_indices = np.clip(col_indices, 0, src_w - 1)

        return frame[row_indices][:, col_indices]

    def _sleep_until_next_frame(self) -> None:
        """Sleep until the next frame should be sent (maintains 30fps).

        Uses pyvirtualcam's built-in timing for accurate FPS pacing.
        """
        if self._cam is not None:
            try:
                self._cam.sleep_until_next_frame()
            except Exception:
                # Camera may have been closed — fall back to manual sleep
                time.sleep(_FRAME_INTERVAL)
        else:
            time.sleep(_FRAME_INTERVAL)

    def _update_fps_stats(self) -> None:
        """Update the rolling FPS measurement over 1-second windows."""
        now = time.monotonic()
        elapsed = now - self._fps_window_start

        if elapsed >= 1.0:
            self._fps_actual = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_window_start = now

    # -- Diagnostics ---------------------------------------------------------

    def get_stats(self) -> dict:
        """Return diagnostic statistics for the dashboard.

        Returns:
            Dict with runtime stats including frame counts, FPS, backend info.
        """
        now = time.monotonic()
        with self._last_frame_lock:
            has_frame = self._last_frame is not None

        return {
            "running": self.is_running,
            "backend": self._active_backend,
            "device_name": self.device_name,
            "resolution": f"{self.width}x{self.height}",
            "fps_target": self.fps,
            "fps_actual": round(self._fps_actual, 1),
            "frames_sent": self._frames_sent,
            "freeze_frames_sent": self._freeze_frames_sent,
            "frames_dropped": self._frames_dropped,
            "has_frame": has_frame,
            "uptime_seconds": round(now - self._start_time, 1) if self._start_time else 0.0,
            "seconds_since_last_frame": (
                round(now - self._last_frame_time, 2)
                if self._last_frame_time
                else None
            ),
            "queue_size": self._frame_queue.qsize(),
            "queue_max": _QUEUE_MAX_SIZE,
        }

    def __repr__(self) -> str:
        status = "running" if self.is_running else "stopped"
        return (
            f"<VirtualCameraSink {status} "
            f"{self.width}x{self.height}@{self.fps}fps "
            f"device='{self.device_name}' backend={self._active_backend}>"
        )
