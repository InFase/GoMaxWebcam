"""
frame_buffer.py — Thread-safe frame buffer with freeze-frame support

Stores the last-good video frame and serves it to the virtual camera when
the live stream is interrupted (USB disconnect, ffmpeg crash, etc.).

Key design goals:
  - Virtual camera device NEVER disappears from downstream apps (Zoom,
    NVIDIA Broadcast, Teams) during any failure scenario
  - During disconnects, downstream apps see a frozen image (the last
    good frame before the stream dropped)
  - When no frame has ever been received, a dark-gray placeholder is
    served so apps see valid video output from the moment the virtual
    camera opens
  - All operations are thread-safe (multiple producers/consumers OK)

Architecture:
  The FrameBuffer sits between the FramePipeline (producer) and the
  VirtualCamera (consumer). It decouples frame storage from the
  pipeline's streaming state:

    StreamReader → FramePipeline → FrameBuffer → VirtualCamera
                                        ↑
                            freeze-frame loop reads from here

  When the pipeline enters freeze-frame mode, it reads from the buffer
  instead of from the StreamReader. The buffer always has a valid frame
  (either the last live frame or the placeholder).

Frame lifecycle:
  1. Pipeline reads frame from StreamReader (live mode)
  2. Frame is pushed to FrameBuffer.update(frame)
  3. FrameBuffer stores a copy and timestamps it
  4. Pipeline reads from FrameBuffer.get_frame() and sends to VirtualCamera
  5. On disconnect: pipeline stops reading from StreamReader, keeps
     calling FrameBuffer.get_frame() which returns the frozen frame
  6. On reconnect: new StreamReader starts, pipeline resumes calling
     FrameBuffer.update() with fresh frames

Usage:
    buffer = FrameBuffer(width=1920, height=1080)
    buffer.start()          # Initializes with placeholder frame

    # Live streaming
    buffer.update(frame)    # Store new frame from StreamReader
    frame = buffer.get_frame()  # Get current frame for VirtualCamera

    # During disconnect (freeze-frame)
    frame = buffer.get_frame()  # Returns last good frame
    assert buffer.is_stale      # True when no fresh frames arriving

    # Stats for GUI
    stats = buffer.get_stats()  # age, frame_count, is_stale, etc.
"""

import threading
import time
from typing import Optional, Tuple

import numpy as np

from logger import get_logger

log = get_logger("frame_buffer")


# Default placeholder color: dark gray so apps see valid (but clearly
# "no signal") output. Matches VirtualCamera._PLACEHOLDER_COLOR.
_PLACEHOLDER_COLOR = (40, 40, 40)

# Frame is considered "stale" if no update received for this many seconds.
# This threshold triggers the "frozen" indicator in the GUI but does NOT
# affect the virtual camera output (which keeps serving the frozen frame).
_DEFAULT_STALE_THRESHOLD = 2.0


class FrameBuffer:
    """Thread-safe frame buffer for virtual camera freeze-frame support.

    Stores the most recent video frame and provides it on demand. When
    the live stream stops (USB disconnect, ffmpeg crash), the buffer
    continues returning the last good frame indefinitely.

    The buffer always has a valid frame available after start() is called
    — either a real frame from the stream or a placeholder. This ensures
    the virtual camera can always push pixels to downstream apps.

    Thread safety:
      - update() and get_frame() can be called from different threads
      - All frame access is protected by a threading.Lock
      - Frame copies are made on update (not on read) to minimize lock
        hold time during the frequent get_frame() calls

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        is_stale: True if no fresh frame has been received recently.
        frame_age: Seconds since the last frame update, or None if never updated.
        total_updates: Total number of frame updates received.
        total_reads: Total number of frame reads served.
        freeze_frame_reads: Number of reads served from the frozen buffer
            (i.e., reads after the stream stopped providing new frames).
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        stale_threshold: float = _DEFAULT_STALE_THRESHOLD,
        placeholder_color: Tuple[int, int, int] = _PLACEHOLDER_COLOR,
    ):
        """Initialize the frame buffer.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            stale_threshold: Seconds without an update before the frame
                is considered stale (triggers GUI "frozen" indicator).
            placeholder_color: RGB tuple for the initial placeholder frame.
        """
        self.width = width
        self.height = height
        self.stale_threshold = stale_threshold
        self._placeholder_color = placeholder_color

        # The current frame — always valid after start()
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        # Tracking
        self._last_update_time: Optional[float] = None
        self._total_updates: int = 0
        self._total_reads: int = 0
        self._freeze_reads: int = 0
        self._started: bool = False
        self._has_live_frame: bool = False  # True after first real frame

        # The frame that was live before stream stopped — used to detect
        # transition from live → frozen
        self._last_live_update_time: Optional[float] = None
        self._freeze_start_time: Optional[float] = None

    # -- Lifecycle -----------------------------------------------------------

    def start(self):
        """Initialize the buffer with a placeholder frame.

        Must be called before get_frame(). After this, the buffer always
        has a valid frame available.
        """
        placeholder = self._make_placeholder()
        with self._lock:
            self._frame = placeholder
            self._started = True
            self._has_live_frame = False
            self._total_updates = 0
            self._total_reads = 0
            self._freeze_reads = 0
            self._last_update_time = None
            self._last_live_update_time = None
            self._freeze_start_time = None

        log.info(
            "[EVENT:frame_buffer] Frame buffer started (%dx%d, "
            "stale threshold=%.1fs)",
            self.width, self.height, self.stale_threshold,
        )

    def reset(self):
        """Reset the buffer to placeholder state.

        Clears the stored frame and all tracking stats. Used when the
        virtual camera resolution changes or on full restart.
        """
        with self._lock:
            self._frame = self._make_placeholder()
            self._has_live_frame = False
            self._total_updates = 0
            self._total_reads = 0
            self._freeze_reads = 0
            self._last_update_time = None
            self._last_live_update_time = None
            self._freeze_start_time = None

        log.debug("[EVENT:frame_buffer] Frame buffer reset to placeholder")

    def resize(self, width: int, height: int):
        """Resize the buffer to new dimensions and reset to placeholder.

        Used when the stream resolution changes (e.g., user switches from
        1080p to 720p). The buffer is reset to a placeholder at the new
        dimensions. Any stored frame from the old resolution is discarded.

        Args:
            width: New frame width in pixels.
            height: New frame height in pixels.
        """
        old_w, old_h = self.width, self.height
        if old_w == width and old_h == height:
            log.debug("[EVENT:frame_buffer] Resize skipped — dimensions unchanged (%dx%d)", width, height)
            return

        log.info(
            "[EVENT:frame_buffer] Resizing buffer: %dx%d → %dx%d",
            old_w, old_h, width, height,
        )
        self.width = width
        self.height = height
        was_started = self._started
        self.reset()
        if was_started:
            self.start()
        log.info(
            "[EVENT:frame_buffer] Buffer resized to %dx%d (reset to placeholder)",
            width, height,
        )

    # -- Frame operations ----------------------------------------------------

    def update(self, frame: np.ndarray) -> bool:
        """Store a new frame from the live stream.

        Makes a copy of the frame to avoid aliasing issues with the
        caller's buffer. The copy is made inside the lock to ensure
        atomic update of frame + timestamp.

        Args:
            frame: numpy array of shape (height, width, 3) dtype uint8,
                   in RGB format. Must match the buffer dimensions.

        Returns:
            True if the frame was accepted and stored.
            False if the frame has wrong dimensions or the buffer isn't started.
        """
        if not self._started:
            log.warning("[EVENT:frame_buffer] update() called before start()")
            return False

        # Validate dimensions
        expected_shape = (self.height, self.width, 3)
        if frame.shape != expected_shape:
            log.warning(
                "[EVENT:frame_buffer] Frame shape mismatch: got %s, "
                "expected %s — dropping frame",
                frame.shape, expected_shape,
            )
            return False

        now = time.monotonic()

        with self._lock:
            # No copy needed — read_frame() creates a fresh numpy array
            # each time via np.frombuffer, so there's no aliasing risk.
            # Avoiding the copy saves ~6MB/frame of allocation at 30fps.
            self._frame = frame
            self._last_update_time = now
            self._last_live_update_time = now
            self._total_updates += 1
            self._has_live_frame = True

            # If we were frozen, we're no longer frozen
            if self._freeze_start_time is not None:
                freeze_dur = now - self._freeze_start_time
                log.info(
                    "[EVENT:frame_recovery] Live frames resumed after "
                    "%.1fs freeze (%d freeze-frame reads served)",
                    freeze_dur, self._freeze_reads,
                )
                self._freeze_start_time = None
                self._freeze_reads = 0

        return True

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame for the virtual camera.

        Returns the most recent frame (live or frozen). If no frame has
        ever been stored (start() not called), returns None.

        The returned frame is the buffer's internal copy — do NOT modify
        it. If you need to modify the frame, make your own copy.

        This method is designed for high-frequency calling (30+ FPS)
        with minimal lock contention.

        Returns:
            numpy array (H, W, 3) uint8 RGB, or None if not started.
        """
        with self._lock:
            if self._frame is None:
                return None

            self._total_reads += 1

            # Track freeze-frame reads
            if self._has_live_frame and self.is_stale_unlocked:
                self._freeze_reads += 1

                # Detect transition to frozen state
                if self._freeze_start_time is None:
                    self._freeze_start_time = time.monotonic()
                    log.info(
                        "[EVENT:freeze_frame] Stream appears frozen — "
                        "serving last good frame to downstream apps"
                    )

            return self._frame

    # -- Properties ----------------------------------------------------------

    @property
    def is_stale(self) -> bool:
        """True if no frame update received within the stale threshold.

        This indicates the stream has likely stopped (USB disconnect,
        ffmpeg crash, etc.) and the buffer is serving a frozen frame.
        Used by the GUI to show "Frozen" status.
        """
        with self._lock:
            return self.is_stale_unlocked

    @property
    def is_stale_unlocked(self) -> bool:
        """Internal stale check (caller must hold self._lock)."""
        if self._last_update_time is None:
            return True  # Never received a frame
        elapsed = time.monotonic() - self._last_update_time
        return elapsed > self.stale_threshold

    @property
    def has_live_frame(self) -> bool:
        """True if at least one real frame has been received (not just placeholder)."""
        with self._lock:
            return self._has_live_frame

    @property
    def frame_age(self) -> Optional[float]:
        """Seconds since the last frame update, or None if never updated.

        During normal streaming this is ~0.033s (at 30 FPS).
        During freeze-frame this grows continuously.
        """
        with self._lock:
            if self._last_update_time is None:
                return None
            return time.monotonic() - self._last_update_time

    @property
    def total_updates(self) -> int:
        """Total number of frames received from the live stream."""
        with self._lock:
            return self._total_updates

    @property
    def total_reads(self) -> int:
        """Total number of frames served to the virtual camera."""
        with self._lock:
            return self._total_reads

    @property
    def freeze_frame_reads(self) -> int:
        """Number of reads served from the frozen buffer."""
        with self._lock:
            return self._freeze_reads

    @property
    def is_frozen(self) -> bool:
        """True if currently in freeze-frame state (stream stopped, serving frozen frame)."""
        with self._lock:
            return self._freeze_start_time is not None

    @property
    def freeze_duration(self) -> float:
        """Seconds spent in current freeze-frame period, or 0 if not frozen."""
        with self._lock:
            if self._freeze_start_time is None:
                return 0.0
            return time.monotonic() - self._freeze_start_time

    @property
    def has_frame(self) -> bool:
        """True if a frame is available (placeholder or real)."""
        with self._lock:
            return self._frame is not None

    # -- Stats ---------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return buffer statistics for the GUI dashboard.

        Returns dict with:
          - started: bool
          - has_live_frame: bool (has received at least one real frame)
          - is_stale: bool (no recent updates)
          - is_frozen: bool (actively serving frozen frames)
          - frame_age: float or None (seconds since last update)
          - freeze_duration: float (seconds in freeze state, or 0)
          - total_updates: int (frames received from stream)
          - total_reads: int (frames served to virtual camera)
          - freeze_reads: int (reads during current freeze period)
          - resolution: str (e.g., '1920x1080')
        """
        with self._lock:
            age = None
            if self._last_update_time is not None:
                age = round(time.monotonic() - self._last_update_time, 2)

            freeze_dur = 0.0
            if self._freeze_start_time is not None:
                freeze_dur = round(
                    time.monotonic() - self._freeze_start_time, 1
                )

            return {
                "started": self._started,
                "has_live_frame": self._has_live_frame,
                "is_stale": self.is_stale_unlocked,
                "is_frozen": self._freeze_start_time is not None,
                "frame_age": age,
                "freeze_duration": freeze_dur,
                "total_updates": self._total_updates,
                "total_reads": self._total_reads,
                "freeze_reads": self._freeze_reads,
                "resolution": f"{self.width}x{self.height}",
            }

    # -- Resolution transition -----------------------------------------------

    def change_resolution(self, new_width: int, new_height: int) -> bool:
        """Change the buffer resolution while preserving the last good frame.

        During a resolution transition, the last good frame is scaled to the
        new dimensions using nearest-neighbor interpolation. This ensures the
        virtual camera always has a valid frame to push — no black frames,
        no gaps — even while ffmpeg restarts with the new resolution.

        After calling this, the buffer serves the scaled freeze-frame until
        update() is called with a frame at the new resolution.

        Args:
            new_width: New frame width in pixels.
            new_height: New frame height in pixels.

        Returns:
            True if the resolution was changed successfully.
        """
        if new_width == self.width and new_height == self.height:
            log.debug("[EVENT:frame_buffer] Resolution unchanged (%dx%d)", new_width, new_height)
            return True

        old_width, old_height = self.width, self.height

        with self._lock:
            old_frame = self._frame
            had_live = self._has_live_frame

            # Update dimensions
            self.width = new_width
            self.height = new_height

            if old_frame is not None and had_live:
                # Scale the last good frame to new dimensions
                self._frame = self._scale_frame(old_frame, new_width, new_height)
                log.info(
                    "[EVENT:frame_buffer] Resolution changed %dx%d → %dx%d "
                    "(last good frame scaled for freeze-frame)",
                    old_width, old_height, new_width, new_height,
                )
            else:
                # No live frame yet — create a new placeholder at the new resolution
                self._frame = self._make_placeholder()
                log.info(
                    "[EVENT:frame_buffer] Resolution changed %dx%d → %dx%d "
                    "(placeholder frame created)",
                    old_width, old_height, new_width, new_height,
                )

            # Reset stats for the new resolution period but keep freeze state
            self._total_updates = 0
            self._total_reads = 0

        return True

    @staticmethod
    def _scale_frame(frame: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
        """Scale a frame to new dimensions using nearest-neighbor interpolation.

        Uses numpy index mapping for fast, dependency-free scaling. The result
        is a contiguous array suitable for pyvirtualcam.

        Args:
            frame: Source frame (H, W, 3) uint8.
            new_width: Target width.
            new_height: Target height.

        Returns:
            Scaled frame (new_height, new_width, 3) uint8.
        """
        src_h, src_w = frame.shape[:2]
        if src_h == new_height and src_w == new_width:
            return frame.copy()

        row_indices = (np.arange(new_height) * src_h // new_height).astype(int)
        col_indices = (np.arange(new_width) * src_w // new_width).astype(int)
        scaled = frame[row_indices][:, col_indices]
        return np.ascontiguousarray(scaled)

    # -- Internal helpers ----------------------------------------------------

    def _make_placeholder(self) -> np.ndarray:
        """Create a solid-color placeholder frame.

        Returns a dark gray frame that downstream apps will see when
        no live frames have been received yet. This is intentionally
        different from black (0,0,0) so it's distinguishable from a
        truly blank/broken feed.
        """
        return np.full(
            (self.height, self.width, 3),
            self._placeholder_color,
            dtype=np.uint8,
        )
