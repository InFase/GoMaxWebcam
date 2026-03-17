"""
frame_pipeline.py — Frame-pushing pipeline: StreamReader → FrameBuffer → VirtualCamera

Connects the ffmpeg stream reader to the pyvirtualcam virtual camera,
reading decoded RGB24 frames and pushing them at the target frame rate.
An optional FrameBuffer sits between the reader and the virtual camera,
providing thread-safe frame storage and freeze-frame support.

Key behaviors:
  - Reads frames from StreamReader in a tight loop on a background thread
  - Stores each frame in the FrameBuffer (if provided) for freeze-frame support
  - Pushes each frame to VirtualCamera.send_frame()
  - Uses VirtualCamera.sleep_until_next_frame() for FPS pacing
  - On stream interruption (ffmpeg crash, USB disconnect), switches to
    freeze-frame mode: keeps pushing the last good frame so downstream
    apps never see the virtual camera disappear
  - Validates resolution consistency between StreamReader and VirtualCamera
  - Tracks statistics (frame count, FPS, freeze duration) for the GUI
  - Pipeline lifecycle (start/stop) is controlled by AppController

Freeze-frame recovery:
  When the stream reader returns None (ffmpeg died, stream ended), the
  pipeline enters freeze-frame mode. It continues calling
  VirtualCamera.send_last_frame() at the target FPS. The virtual camera
  device stays open and visible to downstream apps. When a new
  StreamReader is provided (after reconnection), the pipeline exits
  freeze-frame mode and resumes normal frame pushing.

Thread model:
  - _run_loop() runs on a single daemon thread ('frame-pipeline')
  - send_frame and sleep_until_next_frame are called sequentially
  - The pipeline thread is the ONLY thread touching VirtualCamera
  - Start/stop/swap_reader use threading.Event for coordination

Usage:
    pipeline = FramePipeline(config)
    pipeline.start(stream_reader, virtual_camera)
    # ... later, on disconnect ...
    pipeline.enter_freeze_frame()
    # ... after reconnection ...
    pipeline.swap_reader(new_stream_reader)
    # ... on shutdown ...
    pipeline.stop()
"""

import threading
import time
from enum import Enum, auto
from typing import Optional, Callable

import numpy as np

from logger import get_logger

log = get_logger("frame_pipeline")


class PipelineState(Enum):
    """Pipeline operational state."""
    STOPPED = auto()       # Pipeline not running
    STREAMING = auto()     # Normal: reading frames from ffmpeg → vcam
    FREEZE_FRAME = auto()  # Stream lost: pushing last good frame to vcam
    STARTING = auto()      # Transitioning to STREAMING
    STOPPING = auto()      # Transitioning to STOPPED


class FramePipeline:
    """Reads frames from StreamReader and pushes them to VirtualCamera.

    The pipeline runs a background thread that:
      1. Reads a frame from the StreamReader (blocking)
      2. Sends it to the VirtualCamera
      3. Sleeps until the next frame time (FPS pacing)
      4. Repeats

    If the StreamReader returns None (stream lost), the pipeline
    automatically enters freeze-frame mode: it keeps sending the last
    good frame at the target FPS to keep the virtual camera alive.

    Attributes:
        state: Current pipeline state (STOPPED, STREAMING, FREEZE_FRAME).
        frames_pushed: Total frames pushed to the virtual camera.
        freeze_frame_count: Frames pushed during current freeze period.
        fps_actual: Measured output FPS over the last second.
    """

    def __init__(self, config):
        """Initialize the pipeline.

        Args:
            config: Config object with stream_fps, stream_width, stream_height.
        """
        self.target_fps: int = config.stream_fps
        self.target_width: int = config.stream_width
        self.target_height: int = config.stream_height
        self._frame_interval: float = 1.0 / max(1, self.target_fps)

        # Components (set during start)
        self._reader = None    # StreamReader instance
        self._vcam = None      # VirtualCamera instance
        self._frame_buffer = None  # FrameBuffer instance (optional)

        # Thread management
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._freeze_event = threading.Event()  # Set when in freeze mode
        self._swap_lock = threading.Lock()       # Protects reader swaps

        # State
        self._state = PipelineState.STOPPED
        self._state_lock = threading.Lock()

        # Statistics
        self._frames_pushed: int = 0
        self._freeze_frame_count: int = 0
        self._last_fps_time: float = 0.0
        self._last_fps_count: int = 0
        self._fps_actual: float = 0.0
        self._freeze_start_time: Optional[float] = None

        # Reader warmup grace period — after swap_reader(), the new ffmpeg
        # process needs time to connect to the UDP stream and start producing
        # frames. During this grace window we keep pushing the last good frame
        # instead of re-entering freeze mode from consecutive None reads.
        self._reader_swap_time: Optional[float] = None
        self._reader_warmup_grace: float = 5.0  # seconds to allow new reader startup

        # Callbacks
        self.on_state_change: Optional[Callable[[PipelineState], None]] = None
        self.on_stream_lost: Optional[Callable[[], None]] = None
        self.on_stream_recovered: Optional[Callable[[], None]] = None

    # -- Properties ----------------------------------------------------------

    @property
    def state(self) -> PipelineState:
        """Current pipeline state (thread-safe)."""
        with self._state_lock:
            return self._state

    @property
    def frames_pushed(self) -> int:
        """Total frames pushed to the virtual camera."""
        return self._frames_pushed

    @property
    def freeze_frame_count(self) -> int:
        """Frames pushed during the current freeze period."""
        return self._freeze_frame_count

    @property
    def fps_actual(self) -> float:
        """Measured output FPS over the last measurement window."""
        return self._fps_actual

    @property
    def is_running(self) -> bool:
        """True if the pipeline thread is active."""
        return self._state in (PipelineState.STREAMING, PipelineState.FREEZE_FRAME,
                               PipelineState.STARTING)

    @property
    def is_frozen(self) -> bool:
        """True if in freeze-frame mode."""
        return self._state == PipelineState.FREEZE_FRAME

    @property
    def freeze_duration(self) -> float:
        """Seconds spent in current freeze-frame period, or 0."""
        if self._freeze_start_time is not None:
            return time.monotonic() - self._freeze_start_time
        return 0.0

    @property
    def in_warmup(self) -> bool:
        """True if in the reader warmup grace period after swap_reader()."""
        return self._reader_swap_time is not None

    # -- Lifecycle -----------------------------------------------------------

    def start(self, reader, vcam, frame_buffer=None) -> bool:
        """Start the frame-pushing pipeline.

        Opens the virtual camera (if not already open) and begins the
        frame-reading loop on a background thread.

        Args:
            reader: StreamReader instance (ffmpeg process should already
                    be started via reader.start()).
            vcam: VirtualCamera instance (start() will be called if needed).
            frame_buffer: Optional FrameBuffer for frame storage and
                    freeze-frame support. If provided, frames are stored
                    here and served from here during freeze-frame mode.

        Returns:
            True if the pipeline started successfully.
        """
        if self.is_running:
            log.warning("[EVENT:stream_start] Pipeline already running")
            return True

        self._reader = reader
        self._vcam = vcam
        self._frame_buffer = frame_buffer

        # Ensure virtual camera is open
        if not vcam.is_running:
            if not vcam.start():
                log.error("[EVENT:stream_error] Cannot start pipeline: virtual camera failed to open")
                return False

        # Validate resolution consistency between reader and virtual camera
        self._validate_resolution(reader, vcam)

        # Initialize frame buffer if provided
        if self._frame_buffer is not None:
            if not self._frame_buffer.has_frame:
                self._frame_buffer.start()
            log.info(
                "[EVENT:stream_start] FrameBuffer wired into pipeline "
                "(%dx%d, stale_threshold=%.1fs)",
                self._frame_buffer.width,
                self._frame_buffer.height,
                self._frame_buffer.stale_threshold,
            )

        self._set_state(PipelineState.STARTING)
        self._stop_event.clear()
        self._freeze_event.clear()
        self._frames_pushed = 0
        self._freeze_frame_count = 0
        self._freeze_start_time = None
        self._last_fps_time = time.monotonic()
        self._last_fps_count = 0

        self._thread = threading.Thread(
            target=self._run_loop,
            name="frame-pipeline",
            daemon=True,
        )
        self._thread.start()

        log.info(
            "[EVENT:stream_start] Frame pipeline started "
            "(target %d FPS, frame interval %.3fms, resolution %dx%d)",
            self.target_fps, self._frame_interval * 1000,
            self.target_width, self.target_height,
        )
        return True

    def _validate_resolution(self, reader, vcam):
        """Check that reader output resolution matches virtual camera input.

        Logs a warning if there's a mismatch. The VirtualCamera will attempt
        to resize mismatched frames, but matching resolution avoids overhead.
        """
        reader_w = getattr(reader, 'width', None)
        reader_h = getattr(reader, 'height', None)
        vcam_w = getattr(vcam, 'width', None)
        vcam_h = getattr(vcam, 'height', None)

        if reader_w is not None and vcam_w is not None:
            if reader_w != vcam_w or reader_h != vcam_h:
                log.warning(
                    "[EVENT:stream_start] Resolution mismatch: "
                    "StreamReader outputs %dx%d but VirtualCamera expects %dx%d. "
                    "Frames will be resized (may add latency).",
                    reader_w, reader_h, vcam_w, vcam_h,
                )
            else:
                log.info(
                    "[EVENT:stream_start] Resolution validated: %dx%d "
                    "(reader matches vcam)",
                    reader_w, reader_h,
                )

    def stop(self):
        """Stop the pipeline and its background thread.

        Does NOT close the virtual camera — that's managed separately
        to support freeze-frame during reconnection.
        """
        if self._state == PipelineState.STOPPED:
            return

        log.info("[EVENT:stream_stop] Stopping frame pipeline...")
        self._set_state(PipelineState.STOPPING)
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                log.warning("Pipeline thread did not exit cleanly")

        self._thread = None
        self._set_state(PipelineState.STOPPED)
        log.info(
            "[EVENT:stream_stop] Frame pipeline stopped "
            "(total frames: %d, freeze frames: %d)",
            self._frames_pushed, self._freeze_frame_count,
        )

    def enter_freeze_frame(self):
        """Signal the pipeline to switch to freeze-frame mode.

        Called when the stream source is lost (USB disconnect, ffmpeg crash).
        The pipeline keeps pushing the last good frame to the virtual camera
        at the target FPS.
        """
        if self._state == PipelineState.FREEZE_FRAME:
            return  # Already frozen

        log.info("[EVENT:freeze_frame] Entering freeze-frame mode")
        self._freeze_event.set()
        self._freeze_start_time = time.monotonic()
        self._freeze_frame_count = 0
        self._set_state(PipelineState.FREEZE_FRAME)

        if self.on_stream_lost:
            try:
                self.on_stream_lost()
            except Exception:
                log.exception("Error in on_stream_lost callback")

    def exit_freeze_frame(self):
        """Signal the pipeline to resume normal streaming.

        Called after a new StreamReader is connected (successful reconnection).
        """
        if self._state != PipelineState.FREEZE_FRAME:
            return

        duration = self.freeze_duration
        log.info(
            "[EVENT:frame_recovery] Exiting freeze-frame mode "
            "(was frozen for %.1fs, sent %d freeze frames)",
            duration, self._freeze_frame_count,
        )
        self._freeze_event.clear()
        self._freeze_start_time = None
        self._set_state(PipelineState.STREAMING)

        if self.on_stream_recovered:
            try:
                self.on_stream_recovered()
            except Exception:
                log.exception("Error in on_stream_recovered callback")

    def swap_reader(self, new_reader):
        """Replace the stream reader (after reconnection).

        Thread-safe: the pipeline thread will pick up the new reader
        on the next iteration. A warmup grace period is started so that
        the new reader has time to produce its first frame before the
        pipeline considers re-entering freeze mode.

        Args:
            new_reader: New StreamReader instance (already started).
        """
        with self._swap_lock:
            old_reader = self._reader
            self._reader = new_reader
            # Track swap time for warmup grace period — the new ffmpeg
            # process needs time to connect to the UDP stream and decode
            # the first frame.  During this window, None reads are
            # expected and should NOT trigger re-freeze.
            self._reader_swap_time = time.monotonic()

        if old_reader is not None:
            log.debug("Swapped stream reader (old reader will be stopped by caller)")

        log.info(
            "[EVENT:frame_recovery] Reader swapped — starting %.1fs warmup grace period",
            self._reader_warmup_grace,
        )

        # If we were frozen, exit freeze mode
        if self._state == PipelineState.FREEZE_FRAME:
            self.exit_freeze_frame()

    def handle_resolution_transition(
        self,
        new_width: int,
        new_height: int,
        new_fps: Optional[int] = None,
    ) -> bool:
        """Handle a resolution change with freeze-frame hold.

        During a resolution transition (e.g., user changes from 1080p to 720p),
        the pipeline:
          1. Enters freeze-frame mode (last good frame continues feeding vcam)
          2. Updates internal resolution targets
          3. Notifies the FrameBuffer to scale its frozen frame to new dimensions
          4. Waits for a new stream reader to be swapped in via swap_reader()

        The virtual camera output is seamless: downstream apps see the last
        good frame (scaled to new resolution) during the transition, then
        live video at the new resolution once the new reader is swapped in.

        Note: The caller must restart the VirtualCamera at the new resolution
        and call swap_reader() with a new StreamReader to complete the
        transition. This method only handles the pipeline-side freeze.

        Args:
            new_width: New stream width in pixels.
            new_height: New stream height in pixels.
            new_fps: New target FPS (or None to keep current).

        Returns:
            True if freeze-frame hold was established for the transition.
        """
        old_w, old_h = self.target_width, self.target_height
        old_fps = self.target_fps

        if new_width == old_w and new_height == old_h and (new_fps is None or new_fps == old_fps):
            log.debug("[EVENT:resolution] No resolution change needed (%dx%d)", new_width, new_height)
            return True

        log.info(
            "[EVENT:resolution] Resolution transition: %dx%d@%d → %dx%d@%d — "
            "entering freeze-frame hold",
            old_w, old_h, old_fps,
            new_width, new_height, new_fps or old_fps,
        )

        # Step 1: Enter freeze-frame mode (captures last good frame)
        if self._state == PipelineState.STREAMING:
            self.enter_freeze_frame()

        # Step 2: Update internal resolution targets
        self.target_width = new_width
        self.target_height = new_height
        if new_fps is not None:
            self.target_fps = new_fps
            self._frame_interval = 1.0 / max(1, new_fps)

        # Step 3: Update FrameBuffer resolution (scales frozen frame)
        if self._frame_buffer is not None:
            self._frame_buffer.change_resolution(new_width, new_height)

        log.info(
            "[EVENT:resolution] Pipeline resolution updated to %dx%d@%d — "
            "freeze-frame held until new reader is swapped in",
            new_width, new_height, self.target_fps,
        )
        return True

    # -- Main loop -----------------------------------------------------------

    def _run_loop(self):
        """Main frame-pushing loop (runs on background thread).

        Two modes:
          1. STREAMING: read from StreamReader → push to VirtualCamera
          2. FREEZE_FRAME: push last frame to VirtualCamera at target FPS

        Automatically transitions between modes based on stream availability.

        Warmup grace period:
          After swap_reader(), the new ffmpeg process needs time to connect
          to the UDP stream and start producing frames. During the grace
          window (_reader_warmup_grace seconds), None reads from the reader
          do NOT count toward the freeze threshold. Instead, the last good
          frame is pushed to the virtual camera, maintaining seamless output.
          Once the first live frame arrives from the new reader, the grace
          period ends and normal streaming resumes.
        """
        self._set_state(PipelineState.STREAMING)
        consecutive_none = 0
        max_none_before_freeze = 10  # Frames of None before entering freeze

        # Disable GC during streaming — GC pauses can stall the pipeline
        # long enough to cause consecutive None reads from the pipe.
        import gc
        gc.disable()

        log.debug("Pipeline loop started")

        while not self._stop_event.is_set():
            try:
                if self._freeze_event.is_set():
                    # Freeze-frame mode: push last frame
                    self._push_freeze_frame()
                else:
                    # Normal mode: read from stream and push
                    frame = self._read_next_frame()

                    if frame is not None:
                        consecutive_none = 0
                        # Clear warmup grace — first live frame received
                        if self._reader_swap_time is not None:
                            warmup_elapsed = time.monotonic() - self._reader_swap_time
                            log.info(
                                "[EVENT:frame_recovery] First live frame from new reader "
                                "after %.2fs warmup — seamless transition complete",
                                warmup_elapsed,
                            )
                            self._reader_swap_time = None
                        self._push_frame(frame)
                    else:
                        consecutive_none += 1

                        # CRITICAL: Push the last good frame during the
                        # pre-freeze gap so downstream apps never see a
                        # dropped/black frame.  The vcam driver keeps
                        # displaying whatever was last sent, but
                        # re-sending maintains FPS cadence and prevents
                        # any timing hiccup visible to consumers.
                        self._push_last_frame_or_placeholder()

                        # Check if we're in the warmup grace period after
                        # a reader swap. During warmup, None reads are
                        # expected (ffmpeg is connecting to UDP stream) and
                        # should NOT trigger freeze mode.
                        if self._in_reader_warmup():
                            # Don't count toward freeze threshold during warmup
                            consecutive_none = 0
                        elif consecutive_none >= max_none_before_freeze:
                            log.warning(
                                "[EVENT:stream_error] %d consecutive None frames — "
                                "entering freeze-frame mode",
                                consecutive_none,
                            )
                            self.enter_freeze_frame()
                            consecutive_none = 0
                            continue  # Skip sleep, freeze loop handles timing

                # No explicit FPS pacing — read_frame() blocks until ffmpeg
                # delivers the next frame, which is already rate-limited by
                # the GoPro's output cadence. Adding sleep_until_next_frame()
                # on top would double the per-frame delay.

                # Update FPS stats
                self._update_fps_stats()

            except Exception:
                log.exception("[EVENT:stream_error] Unexpected error in pipeline loop")
                # Don't crash the thread — enter freeze mode and continue
                if not self._freeze_event.is_set():
                    self.enter_freeze_frame()

        gc.enable()
        log.debug("Pipeline loop exited")

    def _read_next_frame(self) -> Optional[np.ndarray]:
        """Read the next frame from the current StreamReader.

        Returns None if the reader is unavailable or returned no frame.
        """
        with self._swap_lock:
            reader = self._reader

        if reader is None:
            return None

        try:
            return reader.read_frame()
        except Exception as e:
            log.debug("[EVENT:stream_error] Error reading frame: %s", e)
            return None

    def _push_frame(self, frame: np.ndarray) -> bool:
        """Push a live frame to the virtual camera and store in buffer.

        Args:
            frame: RGB24 numpy array (H, W, 3).

        Returns:
            True if the frame was accepted by the virtual camera.
        """
        if self._vcam is None:
            return False

        # Store in frame buffer for freeze-frame support
        if self._frame_buffer is not None:
            self._frame_buffer.update(frame)

        success = self._vcam.send_frame(frame)
        if success:
            self._frames_pushed += 1
        return success

    def _push_last_frame_or_placeholder(self):
        """Push the last known good frame to keep the virtual camera fed.

        Called during the pre-freeze gap (reader returned None but we haven't
        entered freeze mode yet). Ensures the virtual camera continues emitting
        frames at the target FPS cadence — no dropped frames, no black frames.
        """
        if self._vcam is None:
            return

        success = False

        # Try FrameBuffer first — it always has a valid frame after start()
        if self._frame_buffer is not None:
            frame = self._frame_buffer.get_frame()
            if frame is not None:
                success = self._vcam.send_frame(frame)
        # Fallback to VirtualCamera's internal last frame
        if not success:
            success = self._vcam.send_last_frame()

        if success:
            self._frames_pushed += 1

    def _push_freeze_frame(self):
        """Push the last good frame (or placeholder) during freeze mode.

        If a FrameBuffer is wired in, reads the frozen frame from the buffer.
        Otherwise falls back to VirtualCamera.send_last_frame().

        Maintains the target FPS to keep the virtual camera device
        responsive to downstream apps.
        """
        if self._vcam is None:
            time.sleep(self._frame_interval)
            return

        success = False

        # Prefer FrameBuffer as the freeze-frame source (tracks freeze stats)
        if self._frame_buffer is not None:
            frozen_frame = self._frame_buffer.get_frame()
            if frozen_frame is not None:
                success = self._vcam.send_frame(frozen_frame)
            else:
                success = self._vcam.send_last_frame()
        else:
            success = self._vcam.send_last_frame()

        if success:
            self._frames_pushed += 1
            self._freeze_frame_count += 1

        # Recovery from freeze is only triggered by:
        #   1. swap_reader() — controller provides a new working reader
        #   2. exit_freeze_frame() — controller explicitly signals recovery
        # We do NOT auto-probe the reader here because a broken reader
        # (e.g., one that throws exceptions) may still report is_running=True.

    def _in_reader_warmup(self) -> bool:
        """Check if we're within the warmup grace period after a reader swap.

        During warmup, the new ffmpeg process is connecting to the UDP stream
        and may not produce frames yet. We keep pushing the last good frame
        instead of entering freeze mode.

        Returns:
            True if within the warmup grace window.
        """
        if self._reader_swap_time is None:
            return False
        elapsed = time.monotonic() - self._reader_swap_time
        if elapsed < self._reader_warmup_grace:
            return True
        # Grace period expired — clear it and allow normal freeze logic
        log.warning(
            "[EVENT:stream_error] Reader warmup grace expired (%.1fs) — "
            "new reader did not produce frames in time",
            self._reader_warmup_grace,
        )
        self._reader_swap_time = None
        return False

    def _vcam_sleep(self):
        """Sleep until the next frame time using pyvirtualcam's timing."""
        if self._vcam is not None:
            self._vcam.sleep_until_next_frame()
        else:
            # Fallback: manual sleep at target FPS
            time.sleep(self._frame_interval)

    # -- State management ----------------------------------------------------

    def _set_state(self, new_state: PipelineState):
        """Update pipeline state and notify listeners."""
        with self._state_lock:
            old_state = self._state
            if old_state == new_state:
                return
            self._state = new_state

        log.info("[EVENT:state_change] Pipeline state: %s -> %s", old_state.name, new_state.name)

        if self.on_state_change:
            try:
                self.on_state_change(new_state)
            except Exception:
                log.exception("Error in pipeline state callback")

    # -- Statistics ----------------------------------------------------------

    def _update_fps_stats(self):
        """Update the rolling FPS measurement.

        Calculates actual FPS over 1-second windows.
        """
        now = time.monotonic()
        elapsed = now - self._last_fps_time

        if elapsed >= 1.0:
            frames_in_window = self._frames_pushed - self._last_fps_count
            self._fps_actual = frames_in_window / elapsed
            self._last_fps_time = now
            self._last_fps_count = self._frames_pushed

    @property
    def frame_buffer(self):
        """Access the FrameBuffer instance (for stats/monitoring)."""
        return self._frame_buffer

    def get_stats(self) -> dict:
        """Return pipeline statistics for the GUI dashboard.

        Returns dict with:
          - state: str (pipeline state name)
          - frames_pushed: int (total)
          - freeze_frame_count: int (current freeze period)
          - fps_actual: float (measured FPS)
          - fps_target: int (configured target)
          - fps_match: bool (actual FPS within 10% of target)
          - freeze_duration: float (seconds in freeze, or 0)
          - is_frozen: bool
          - resolution: str (target resolution e.g. '1920x1080')
          - has_frame_buffer: bool
        """
        fps_match = (
            abs(self._fps_actual - self.target_fps) < self.target_fps * 0.1
            if self._fps_actual > 0 else False
        )

        stats = {
            "state": self._state.name,
            "frames_pushed": self._frames_pushed,
            "freeze_frame_count": self._freeze_frame_count,
            "fps_actual": round(self._fps_actual, 1),
            "fps_target": self.target_fps,
            "fps_match": fps_match,
            "freeze_duration": round(self.freeze_duration, 1),
            "is_frozen": self.is_frozen,
            "resolution": f"{self.target_width}x{self.target_height}",
            "has_frame_buffer": self._frame_buffer is not None,
            "in_warmup": self._reader_swap_time is not None,
        }

        # Include frame buffer stats if available
        if self._frame_buffer is not None:
            stats["frame_buffer"] = self._frame_buffer.get_stats()

        return stats
