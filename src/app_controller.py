"""
app_controller.py — Central application controller for GoPro Bridge

Orchestrates the full lifecycle:
  1. Load config
  2. Check prerequisites (ffmpeg)
  3. Discover GoPro on USB
  4. Connect and start webcam mode
  5. Launch streaming pipeline (StreamReader → FramePipeline → VirtualCamera)
  6. Keep-alive monitoring with disconnect detection
  7. Seamless auto-recovery: freeze-frame → re-discover → reconnect → swap reader

The virtual camera NEVER shuts down during recovery. When the GoPro disconnects,
the pipeline enters freeze-frame mode (keeps pushing the last good frame to the
virtual camera device). When the GoPro reconnects, a new StreamReader is created
and swapped into the running pipeline, seamlessly replacing the freeze-frame with
live video.

Recovery flow:
  STREAMING → keep-alive fails → RECONNECTING → pipeline enters freeze-frame
  → USB reconnection detected or poll-based rediscovery → re-verify HTTP API
  → start webcam mode → create new ffmpeg StreamReader → swap into pipeline
  → pipeline exits freeze-frame → STREAMING (seamless transition)

Thread model:
  - _startup_flow runs on a background thread ("StartupFlow")
  - _keepalive_loop runs on a daemon thread ("KeepAlive")
  - _recovery_loop runs on a daemon thread ("RecoveryLoop")
  - _staleness_monitor_loop runs on a daemon thread ("StalenessMonitor")
  - FramePipeline runs on its own daemon thread ("frame-pipeline")
  - USBEventListener runs on its own daemon thread ("USBEventListener")
  - All callbacks fire from background threads; GUI must use thread-safe dispatch

This runs on background threads so the GUI stays responsive.
All status updates are pushed via callbacks to the GUI layer.
"""

import threading
import time
from enum import Enum, auto
from typing import Optional, Callable, List

from config import Config
from gopro_connection import GoProConnection, WebcamStatus, ConnectionState
from logger import get_logger, setup_logger

log = get_logger(__name__)


class AppState(Enum):
    """High-level application states shown in the GUI."""
    INITIALIZING = auto()
    CHECKING_PREREQUISITES = auto()
    DISCOVERING = auto()
    CONNECTING = auto()
    STREAMING = auto()
    PAUSED = auto()
    CHARGE_MODE = auto()
    RECONNECTING = auto()
    DISCONNECTED = auto()
    ERROR = auto()
    STOPPED = auto()


class StatusMessage:
    """A status update with message text and severity level."""

    def __init__(self, text: str, level: str = "info", state: Optional[AppState] = None):
        self.text = text
        self.level = level       # "info", "success", "warning", "error"
        self.state = state
        self.timestamp = time.time()

    def __repr__(self):
        return f"[{self.level.upper()}] {self.text}"


class AppController:
    """Manages the full lifecycle of GoPro Bridge including streaming pipeline.

    Provides callbacks for GUI integration:
      - on_state_change(state: AppState) — app state transitions
      - on_status(message: str, level: str) — human-readable status updates
      - on_camera_info(info: dict) — battery level, etc.

    The controller manages three key components for seamless recovery:
      - VirtualCamera: Opened once, stays open through all disconnects
      - FramePipeline: Runs continuously, switches between live/freeze-frame
      - StreamReader: Created per connection, swapped into pipeline on recovery

    Usage:
        controller = AppController()
        controller.on_state_change = lambda state: update_gui(state)
        controller.on_status = lambda msg, lvl: show_in_log(msg, lvl)
        controller.start()  # kicks off auto-discovery in background
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load()
        self.gopro = GoProConnection(self.config)
        self.state = AppState.INITIALIZING

        # Status history for GUI log display
        self.status_history: List[StatusMessage] = []

        # Callbacks — set these before calling start()
        self.on_state_change: Optional[Callable[[AppState], None]] = None
        self.on_status: Optional[Callable[[str, str], None]] = None
        self.on_camera_info: Optional[Callable[[dict], None]] = None
        self.on_webcam_mode_failed: Optional[Callable[[str], None]] = None
        self.on_active_port: Optional[Callable[[int], None]] = None

        # Thread management
        self._running = False
        self._startup_thread: Optional[threading.Thread] = None
        self._keepalive_thread: Optional[threading.Thread] = None
        self._recovery_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Streaming pipeline components (lazy-initialized)
        self._virtual_camera = None   # VirtualCamera — stays open across reconnects
        self._frame_pipeline = None   # FramePipeline — runs continuously
        self._stream_reader = None    # StreamReader — recreated per connection
        self._frame_buffer = None     # FrameBuffer — stores last good frame for freeze-frame

        # Disconnect detector — bridges USB detach events to freeze-frame
        self._disconnect_detector = None

        # USB event listener for fast reconnection detection
        self._usb_listener = None
        self._usb_reconnect_event = threading.Event()

        # USB device poller — reliable fallback for detecting GoPro reconnection
        # when the event-driven USBEventListener misses events or fails to start
        self._usb_poller = None

        # Staleness monitor — polls FrameBuffer.is_stale at config.stale_poll_interval
        # to detect stream freezes independently of USB events and ffmpeg health
        self._staleness_thread: Optional[threading.Thread] = None
        self._staleness_interval: float = self.config.stale_poll_interval
        self._staleness_stop_event = threading.Event()  # Dedicated stop for staleness monitor
        self._staleness_was_stale: bool = True  # Start True to avoid false trigger before first frame arrives

        # Recovery state tracking
        self._recovery_count = 0
        self._last_recovery_time: Optional[float] = None
        self._is_recovering = False
        self._recovery_lock = threading.Lock()

        # Wire up the GoProConnection status callback to our unified handler
        self.gopro.on_status_change = self._handle_gopro_status

    # ── Public API ─────────────────────────────────────────────────

    def start(self):
        """Begin the auto-startup flow in a background thread.

        This is non-blocking — call from the GUI thread.
        The flow: prerequisites → discover → connect → stream → keep-alive
        """
        if self._running:
            log.warning("[EVENT:startup] start() called but already running")
            return

        log.info("[EVENT:startup] Starting GoPro Bridge auto-connect flow")
        self._running = True
        self._stop_event.clear()
        self._usb_reconnect_event.clear()
        # Reset recovery flag — if a previous session's recovery thread didn't
        # finish its finally block before stop() timed out, this flag may be
        # stuck True, which would block all recovery in the new session.
        with self._recovery_lock:
            self._is_recovering = False
        self._startup_thread = threading.Thread(
            target=self._startup_flow,
            name="StartupFlow",
            daemon=True,
        )
        self._startup_thread.start()

    def stop(self):
        """Gracefully stop everything: streaming, keep-alive, connection.

        Shuts down in reverse order: keep-alive → pipeline → stream reader
        → virtual camera → GoPro webcam mode → USB listener.
        The virtual camera is the LAST thing to close so downstream apps
        see a valid feed until the very end.
        """
        log.info("[EVENT:shutdown] Initiating graceful shutdown")
        self._emit_status("Shutting down...", "info")
        self._running = False
        self._stop_event.set()
        self._usb_reconnect_event.set()  # Unblock any waiting recovery thread

        # Stop staleness monitor
        self._stop_staleness_monitor()

        # Stop disconnect detector (before USB listener, since it wraps one)
        self._stop_disconnect_detector()

        # Stop USB device poller
        self._stop_usb_poller()

        # Stop USB event listener
        self._stop_usb_listener()

        # Fully exit GoPro webcam mode (stop + exit)
        if self.gopro.is_connected:
            try:
                self.gopro.stop_webcam()
            except Exception:
                log.debug("Failed to stop webcam during shutdown (camera may be gone)")
            try:
                self.gopro.exit_webcam()
            except Exception:
                log.debug("Failed to exit webcam during shutdown")
            try:
                self.gopro.disable_wired_usb_control()
            except Exception:
                log.debug("Failed to disable USB control during shutdown")

        # Stop streaming pipeline (but NOT the virtual camera yet)
        if self._frame_pipeline is not None:
            try:
                self._frame_pipeline.stop()
            except Exception:
                log.debug("Failed to stop pipeline during shutdown")

        # Stop stream reader (kills ffmpeg)
        if self._stream_reader is not None:
            try:
                self._stream_reader.stop()
            except Exception:
                log.debug("Failed to stop stream reader during shutdown")

        # NOW stop the virtual camera (last, so downstream apps lose it last)
        if self._virtual_camera is not None:
            try:
                self._virtual_camera.stop()
            except Exception:
                log.debug("Failed to stop virtual camera during shutdown")

        # Wait for threads to finish
        for thread in (self._keepalive_thread, self._startup_thread, self._recovery_thread, self._staleness_thread):
            if thread and thread.is_alive():
                thread.join(timeout=5)

        self._set_state(AppState.STOPPED)
        log.info("[EVENT:shutdown] GoPro Bridge stopped")
        self._emit_status("GoPro Bridge stopped", "info")

    def retry_connection(self):
        """Manually trigger a reconnection attempt."""
        if self._running:
            self.stop()

        self.start()

    def pause_webcam(self):
        """Pause streaming — GoPro stops streaming, downstream apps see freeze-frame.

        The camera's red webcam icon turns off (camera enters IDLE state).
        The virtual camera stays open showing the last good frame so
        downstream apps (Zoom, Teams) never lose the camera device.

        On resume, the GoPro stream is restarted and a new ffmpeg process
        is swapped into the pipeline.
        """
        if self.state != AppState.STREAMING:
            log.warning("[EVENT:pause] Cannot pause — not streaming (state=%s)", self.state.name)
            return

        log.info("[EVENT:pause] Pausing webcam stream")
        self._emit_status("Pausing webcam...", "info")

        # Set state FIRST so monitoring callbacks see PAUSED immediately
        self._set_state(AppState.PAUSED)

        # Stop monitors before killing ffmpeg to prevent false recovery
        self._stop_staleness_monitor()
        self._stop_disconnect_detector()

        # Enter freeze-frame (downstream apps keep seeing last good frame)
        if self._frame_pipeline:
            self._frame_pipeline.enter_freeze_frame()

        # Stop ffmpeg
        if self._stream_reader:
            try:
                self._stream_reader.stop()
            except Exception:
                log.debug("Failed to stop stream reader during pause")

        # Tell GoPro to stop streaming (camera enters IDLE, red icon turns off)
        try:
            self.gopro.stop_webcam()
        except Exception:
            log.debug("Failed to stop webcam during pause")

        self._emit_status("Webcam paused", "info")

    def resume_webcam(self):
        """Resume streaming after a pause or charge mode.

        From PAUSED: restarts GoPro stream and ffmpeg. The IDLE workaround
        is skipped (intentional stop flag) for faster restart.
        From CHARGE_MODE: full restart with 2s settle time (exit_webcam
        was called, camera needs time to re-enter webcam mode).
        """
        if self.state not in (AppState.PAUSED, AppState.CHARGE_MODE):
            log.warning("[EVENT:resume] Cannot resume — not paused (state=%s)", self.state.name)
            return

        log.info("[EVENT:resume] Resuming webcam stream (from %s)", self.state.name)
        self._emit_status("Resuming webcam...", "info")

        if self.state == AppState.CHARGE_MODE:
            # Charge mode did exit_webcam — needs more settle time
            time.sleep(max(self.config.idle_reset_delay, 2.0))

        if not self.gopro.start_webcam():
            self._emit_status("Failed to restart webcam — try again", "error")
            return

        try:
            if not self._create_and_swap_stream_reader():
                self._emit_status("Failed to create stream reader for resume", "error")
                return
        except Exception as e:
            log.error("[EVENT:resume] Failed to create stream reader: %s", e)
            self._emit_status(f"Failed to resume stream: {e}", "error")
            return

        # Set state BEFORE restarting monitors
        self._set_state(AppState.STREAMING)

        # Restart monitoring (stopped during charge mode)
        self._start_disconnect_detector()
        self._start_staleness_monitor()

        self._emit_status("Live video restored!", "success")

    def enter_charge_mode(self):
        """Enter charge mode — exit webcam to maximize USB charging.

        Stops everything except the USB connection monitoring, so the
        camera can charge faster over USB.
        """
        if self.state not in (AppState.STREAMING, AppState.PAUSED):
            log.warning("[EVENT:charge_mode] Cannot enter charge mode (state=%s)", self.state.name)
            return

        log.info("[EVENT:charge_mode] Entering charge mode")
        self._emit_status("Entering charge mode...", "info")

        # Set state FIRST so all monitoring callbacks see CHARGE_MODE
        # immediately and don't trigger false recovery during shutdown.
        self._set_state(AppState.CHARGE_MODE)

        # Stop monitoring — after state change so callbacks are guarded
        self._stop_staleness_monitor()
        self._stop_disconnect_detector()

        # Enter freeze-frame first
        if self._frame_pipeline:
            self._frame_pipeline.enter_freeze_frame()

        # Stop streaming pipeline
        if self._stream_reader:
            try:
                self._stream_reader.stop()
            except Exception:
                log.debug("Failed to stop stream reader for charge mode")

        # Fully exit webcam mode (not just stop — exit frees more power)
        try:
            self.gopro.stop_webcam()
        except Exception:
            log.debug("Failed to stop webcam for charge mode")
        try:
            self.gopro.exit_webcam()
        except Exception:
            log.debug("Failed to exit webcam for charge mode")

        self._emit_status("Charge mode active — webcam stopped to maximize charging", "info")

    # ── Pipeline Access (for GUI stats) ────────────────────────────

    @property
    def virtual_camera(self):
        """Access the VirtualCamera instance (for stats/monitoring)."""
        return self._virtual_camera

    @property
    def frame_pipeline(self):
        """Access the FramePipeline instance (for stats/monitoring)."""
        return self._frame_pipeline

    @property
    def stream_reader(self):
        """Access the current StreamReader instance."""
        return self._stream_reader

    @property
    def frame_buffer(self):
        """Access the FrameBuffer instance (for stats/monitoring)."""
        return self._frame_buffer

    @property
    def disconnect_detector(self):
        """Access the DisconnectDetector instance (for stats/monitoring)."""
        return self._disconnect_detector

    @property
    def usb_poller(self):
        """Access the USBDevicePoller instance (for stats/monitoring)."""
        return self._usb_poller

    @property
    def recovery_count(self) -> int:
        """Number of successful recoveries since app start."""
        return self._recovery_count

    # ── State Management ───────────────────────────────────────────

    def _set_state(self, new_state: AppState):
        """Update the app state and notify listeners."""
        old_state = self.state
        self.state = new_state
        if old_state != new_state:
            log.info("[EVENT:state_change] %s -> %s", old_state.name, new_state.name)
            if self.on_state_change:
                try:
                    self.on_state_change(new_state)
                except Exception as e:
                    log.error("State change callback error: %s", e)

    def _emit_status(self, message: str, level: str = "info"):
        """Push a status message to history and notify listeners."""
        msg = StatusMessage(message, level, self.state)
        self.status_history.append(msg)

        # Keep history bounded (last 200 messages)
        if len(self.status_history) > 200:
            self.status_history = self.status_history[-200:]

        log.info("[STATUS] %s", message) if level != "error" else log.error("[STATUS] %s", message)

        if self.on_status:
            try:
                self.on_status(message, level)
            except Exception as e:
                log.error("Status callback error: %s", e)

    def _emit_active_port(self, port: int):
        """Notify the GUI of the active UDP port (may differ from config after auto-selection)."""
        if self.on_active_port:
            try:
                self.on_active_port(port)
            except Exception as e:
                log.error("Active port callback error: %s", e)

    def _handle_gopro_status(self, message: str, level: str):
        """Forward GoProConnection status updates through our unified handler."""
        self._emit_status(message, level)

    # ── USB Event Listener ─────────────────────────────────────────

    def _start_usb_listener(self):
        """Start the single USB event listener that fans out to all consumers.

        Creates ONE USBEventListener instance whose attach/detach callbacks
        fan out to both:
          - AppController: sets _usb_reconnect_event for fast recovery
          - DisconnectDetector: triggers proactive freeze-frame on detach

        This avoids creating multiple Windows message-only windows and
        duplicate device notification registrations.
        """
        try:
            from usb_event_listener import USBEventListener
        except ImportError:
            log.debug("[EVENT:usb_listener] USBEventListener not available")
            return

        if self._usb_listener is not None and self._usb_listener.is_running:
            return

        def on_gopro_attached(device_id: str):
            log.info(
                "[EVENT:usb_listener] GoPro USB device attached: %s",
                device_id[:80],
            )
            # Fan out to AppController: signal recovery loop
            self._usb_reconnect_event.set()
            # Fan out to DisconnectDetector: notify of reconnection
            if self._disconnect_detector is not None:
                try:
                    self._disconnect_detector.handle_usb_attach(device_id)
                except Exception:
                    log.exception("[EVENT:usb_listener] Error forwarding attach to disconnect detector")

        def on_gopro_detached(device_id: str):
            log.info(
                "[EVENT:usb_listener] GoPro USB device detached: %s",
                device_id[:80],
            )
            # Fan out to DisconnectDetector: trigger proactive freeze-frame
            if self._disconnect_detector is not None:
                try:
                    self._disconnect_detector.handle_usb_detach(device_id)
                except Exception:
                    log.exception("[EVENT:usb_listener] Error forwarding detach to disconnect detector")

        self._usb_listener = USBEventListener(
            on_attach=on_gopro_attached,
            on_detach=on_gopro_detached,
        )
        try:
            if self._usb_listener.start(timeout=3.0):
                log.info("[EVENT:usb_listener] USB event listener started (fans out to controller + detector)")
            else:
                log.debug("[EVENT:usb_listener] USB event listener failed to start (will use polling)")
                self._usb_listener = None
        except Exception as e:
            log.debug("[EVENT:usb_listener] USB event listener error: %s (will use polling)", e)
            self._usb_listener = None

    def _stop_usb_listener(self):
        """Stop the USB event listener."""
        if self._usb_listener is not None:
            try:
                self._usb_listener.stop()
            except Exception:
                log.debug("Error stopping USB listener")
            self._usb_listener = None

    # ── USB Device Poller ─────────────────────────────────────────

    def _start_usb_poller(self):
        """Start the USB device poller for reliable reconnection detection.

        The poller periodically enumerates USB devices to detect GoPro
        presence/absence. This provides guaranteed reconnection detection
        as a fallback when the event-driven USBEventListener misses events
        or fails to start.

        The poller fires _usb_reconnect_event when the GoPro appears,
        waking up the recovery loop immediately instead of waiting for
        the poll-based discovery timeout.
        """
        try:
            from usb_device_poller import USBDevicePoller
        except ImportError:
            log.debug("[EVENT:usb_poller] USBDevicePoller not available")
            return

        if self._usb_poller is not None and self._usb_poller.is_running:
            return

        def on_gopro_appeared(devices):
            device_descs = []
            for d in devices:
                desc = getattr(d, 'description', str(d))
                device_descs.append(desc)
            log.info(
                "[EVENT:usb_poller] GoPro USB device appeared (poll-based): %s",
                ", ".join(device_descs),
            )
            # Signal the recovery loop that the GoPro is back
            self._usb_reconnect_event.set()
            self._emit_status("GoPro USB device detected (polling)", "info")

        def on_gopro_disappeared():
            log.info("[EVENT:usb_poller] GoPro USB device disappeared (poll-based)")
            self._emit_status("GoPro USB device disconnected", "warning")

        self._usb_poller = USBDevicePoller(
            on_device_appeared=on_gopro_appeared,
            on_device_disappeared=on_gopro_disappeared,
            poll_interval=self.config.reconnect_delay,
            settling_time=0.0,  # Don't add extra delay; recovery loop handles settling
        )

        try:
            if self._usb_poller.start():
                log.info(
                    "[EVENT:usb_poller] USB device poller started "
                    "(interval=%.1fs) for reliable reconnect detection",
                    self.config.reconnect_delay,
                )
            else:
                log.debug("[EVENT:usb_poller] USB device poller failed to start")
                self._usb_poller = None
        except Exception as e:
            log.debug("[EVENT:usb_poller] USB device poller error: %s", e)
            self._usb_poller = None

    def _stop_usb_poller(self):
        """Stop the USB device poller."""
        if self._usb_poller is not None:
            try:
                self._usb_poller.stop()
            except Exception:
                log.debug("Error stopping USB device poller")
            self._usb_poller = None

    # ── Disconnect Detector ──────────────────────────────────────────

    def _start_disconnect_detector(self):
        """Start the disconnect detector for proactive freeze-frame on USB yank.

        The disconnect detector bridges two fast signals to the pipeline:
          1. USB detach event (instant) → pipeline.enter_freeze_frame()
          2. Stream health monitor (sub-second) → pipeline.enter_freeze_frame()

        This ensures the virtual camera captures the last good frame BEFORE
        ffmpeg starts producing corrupted frames from its dying buffers.

        NOTE: When AppController's USB listener is active, the detector receives
        USB events via handle_usb_attach()/handle_usb_detach() from the shared
        listener — it does NOT create its own USBEventListener instance.
        """
        from disconnect_detector import DisconnectDetector

        if self._disconnect_detector is not None and self._disconnect_detector.is_running:
            log.debug("[EVENT:disconnect_detector] Already running, skipping start")
            return

        self._disconnect_detector = DisconnectDetector(
            pipeline=self._frame_pipeline,
            connection=self.gopro,
            stream_reader=self._stream_reader,
            config=self.config,
            usb_listener=self._usb_listener,  # Share the single listener
            startup_grace=self.config.stream_startup_timeout,  # Delay first health check
        )

        # Wire callbacks: USB detach triggers recovery, USB attach speeds it up,
        # ffmpeg crash triggers lightweight recovery (just restart ffmpeg)
        self._disconnect_detector.on_disconnect = self._on_detector_disconnect
        self._disconnect_detector.on_reconnect_ready = self._on_detector_reconnect_ready
        self._disconnect_detector.on_ffmpeg_crash = self._on_ffmpeg_crash

        if self._disconnect_detector.start():
            log.info(
                "[EVENT:disconnect_detector] Disconnect detector started — "
                "USB detach will trigger proactive freeze-frame"
            )
        else:
            log.warning(
                "[EVENT:disconnect_detector] Disconnect detector failed to start — "
                "relying on keep-alive for disconnect detection"
            )

    def _stop_disconnect_detector(self):
        """Stop the disconnect detector and clean up."""
        if self._disconnect_detector is not None:
            try:
                self._disconnect_detector.stop()
            except Exception:
                log.debug("Error stopping disconnect detector")
            self._disconnect_detector = None

    def _on_detector_disconnect(self):
        """Called by DisconnectDetector when USB disconnect is detected.

        This fires INSTANTLY when the USB cable is yanked — much faster than
        the keep-alive polling loop (which takes 3 x 2.5s = 7.5s to detect).

        The disconnect detector has already triggered freeze-frame on the
        pipeline. Here we:
          1. Log the event
          2. Emit a status update for the GUI
          3. Trigger recovery if not already in progress
        """
        # Don't trigger recovery if we intentionally stopped streaming
        if self.state in (AppState.PAUSED, AppState.CHARGE_MODE, AppState.STOPPED):
            log.debug("[EVENT:disconnection] Ignoring disconnect in %s state", self.state.name)
            return

        log.warning(
            "[EVENT:disconnection] USB disconnect detected by disconnect detector — "
            "freeze-frame activated, starting recovery"
        )
        self._emit_status(
            "USB disconnect detected — freeze-frame active, reconnecting...",
            "warning"
        )

        # Trigger auto-recovery (skip if already recovering)
        with self._recovery_lock:
            already_recovering = self._is_recovering
        if not already_recovering:
            self._auto_recover()

    def _on_detector_reconnect_ready(self, device_id: str):
        """Called by DisconnectDetector when USB reconnection is detected.

        Signals the recovery loop via the USB reconnect event so it can
        skip the poll timeout and immediately attempt reconnection.
        """
        log.info(
            "[EVENT:usb_listener] USB reconnection detected via disconnect detector: %s",
            device_id[:80] if device_id else "unknown",
        )
        self._usb_reconnect_event.set()

    def _on_ffmpeg_crash(self):
        """Called by DisconnectDetector when ffmpeg dies without USB disconnect.

        This triggers a lightweight recovery path: since the GoPro is likely
        still connected via USB, we skip the full USB reconnect dance and
        just verify the API is reachable, then restart ffmpeg.

        This is MUCH faster than the full recovery (~1-2s vs 10-30s).
        """
        # Don't trigger recovery if we intentionally stopped ffmpeg
        if self.state in (AppState.PAUSED, AppState.CHARGE_MODE, AppState.STOPPED):
            log.debug("[EVENT:ffmpeg_crash] Ignoring ffmpeg exit in %s state (intentional)", self.state.name)
            return

        log.warning(
            "[EVENT:ffmpeg_crash] ffmpeg process crashed — "
            "GoPro likely still connected, attempting quick recovery"
        )
        self._emit_status(
            "ffmpeg crashed — freeze-frame active, restarting stream...",
            "warning"
        )

        # Skip if already recovering (USB disconnect recovery takes precedence)
        with self._recovery_lock:
            already_recovering = self._is_recovering
        if already_recovering:
            log.debug("[EVENT:ffmpeg_crash] Full recovery already in progress, skipping ffmpeg recovery")
            return

        # Launch lightweight recovery on a background thread
        thread = threading.Thread(
            target=self._recover_ffmpeg_crash,
            name="FFmpegRecovery",
            daemon=True,
        )
        thread.start()

    def _recover_ffmpeg_crash(self):
        """Lightweight recovery: restart ffmpeg without full USB reconnect.

        Since the GoPro is likely still connected and in webcam mode:
          1. Enter freeze-frame (if not already)
          2. Verify GoPro API is still reachable
          3. If reachable: just restart ffmpeg and swap reader
          4. If not reachable: fall back to full _auto_recover()

        The pipeline and virtual camera are NEVER stopped — freeze-frame
        continues throughout.
        """
        with self._recovery_lock:
            if self._is_recovering:
                log.debug("[EVENT:ffmpeg_crash] Recovery already in progress")
                return
            self._is_recovering = True

        try:
            # Step 1: Ensure pipeline is in freeze-frame mode
            if self._frame_pipeline is not None and self._frame_pipeline.is_running:
                self._frame_pipeline.enter_freeze_frame()

            # Step 2: Check if GoPro API is still reachable
            api_reachable = False
            try:
                api_reachable = self.gopro.keep_alive()
            except Exception:
                log.debug("[EVENT:ffmpeg_crash] keep_alive check failed")

            if api_reachable:
                # GoPro still connected — just restart ffmpeg
                log.info(
                    "[EVENT:ffmpeg_crash] GoPro API still reachable — "
                    "restarting ffmpeg only (fast recovery)"
                )
                self._emit_status(
                    "GoPro still connected — restarting video stream...",
                    "info"
                )

                if self._create_and_swap_stream_reader():
                    self._recovery_count += 1
                    self._last_recovery_time = time.monotonic()
                    log.info(
                        "[EVENT:recovery_success] ffmpeg crash recovery #%d — "
                        "live video restored (fast path)",
                        self._recovery_count,
                    )
                    self._emit_status(
                        f"Stream restored after ffmpeg crash (recovery #{self._recovery_count})",
                        "success"
                    )
                    self._set_state(AppState.STREAMING)
                    return
                else:
                    log.warning(
                        "[EVENT:ffmpeg_crash] Fast recovery failed — "
                        "falling back to full reconnection"
                    )

            # API not reachable or fast recovery failed — fall through to full recovery
            log.info(
                "[EVENT:ffmpeg_crash] GoPro not reachable — "
                "falling back to full USB reconnect recovery"
            )
            # Reset recovery lock so _auto_recover can acquire it
            with self._recovery_lock:
                self._is_recovering = False
            self._auto_recover()

        except Exception as e:
            log.exception("[EVENT:ffmpeg_crash] ffmpeg crash recovery failed: %s", e)
            self._emit_status(f"Recovery error: {e}", "error")
            with self._recovery_lock:
                self._is_recovering = False
            # Fall back to full recovery
            self._auto_recover()

    # ── Staleness Monitor ─────────────────────────────────────────

    def _start_staleness_monitor(self):
        """Start the staleness monitor thread.

        Polls FrameBuffer.is_stale every 500ms to detect stream freezes
        independently of USB events and ffmpeg health checks. When the
        buffer transitions from fresh to stale, the monitor:
          1. Triggers freeze-frame on the pipeline (if not already frozen)
          2. Emits a status update for the GUI
          3. Logs the staleness detection

        This provides a third layer of disconnect detection (complementing
        USB detach events and stream health monitoring) that works even when
        USB events are missed and ffmpeg is still technically running but
        not receiving data.

        The monitor uses edge detection: it only triggers on the fresh→stale
        transition, not on every poll while stale. When fresh frames resume,
        the edge resets so the next stale transition will trigger again.
        """
        if self._staleness_thread is not None and self._staleness_thread.is_alive():
            log.debug("[EVENT:staleness_monitor] Already running, skipping start")
            return

        # Start with stale=True so the first poll doesn't trigger a
        # fresh->stale edge. The frame buffer is naturally stale when
        # no frames have arrived yet (cold start, resume from charge).
        # The first real fresh->stale transition will fire after frames
        # start flowing and then stop.
        self._staleness_was_stale = True
        self._staleness_stop_event.clear()
        self._staleness_thread = threading.Thread(
            target=self._staleness_monitor_loop,
            name="StalenessMonitor",
            daemon=True,
        )
        self._staleness_thread.start()
        log.info(
            "[EVENT:staleness_monitor] Staleness monitor started "
            "(interval=%.1fs)",
            self._staleness_interval,
        )

    def _stop_staleness_monitor(self):
        """Stop the staleness monitor thread.

        Uses a dedicated stop event (not the shared _stop_event) so the
        monitor can be stopped independently during pause/charge mode
        without affecting other threads.
        """
        self._staleness_stop_event.set()
        if self._staleness_thread is not None and self._staleness_thread.is_alive():
            self._staleness_thread.join(timeout=3.0)
        self._staleness_thread = None

    def _staleness_monitor_loop(self):
        """Background loop polling FrameBuffer.is_stale every 500ms.

        Detects the fresh→stale edge transition and triggers freeze-frame
        on the pipeline. Does NOT trigger on every stale poll — only on
        the transition from fresh to stale. When the buffer receives fresh
        frames again (stale→fresh), the edge resets.

        The monitor is a safety net: if USB detach events are missed and
        the stream health monitor hasn't caught the failure yet, the
        staleness monitor will notice within 500ms of the stale threshold
        being exceeded.
        """
        log.debug("[EVENT:staleness_monitor] Monitor loop started")

        while not self._stop_event.is_set() and not self._staleness_stop_event.is_set():
            if self._staleness_stop_event.wait(timeout=self._staleness_interval):
                break
            if self._stop_event.is_set():
                break

            buffer = self._frame_buffer
            if buffer is None:
                continue

            is_stale = buffer.is_stale

            if is_stale and not self._staleness_was_stale:
                # Fresh → stale transition detected
                self._staleness_was_stale = True
                log.warning(
                    "[EVENT:staleness_monitor] Frame buffer stale — "
                    "no fresh frames for >%.1fs",
                    buffer.stale_threshold,
                )

                # Trigger freeze-frame on the pipeline if not already frozen
                pipeline = self._frame_pipeline
                if pipeline is not None and pipeline.is_running and not pipeline.is_frozen:
                    pipeline.enter_freeze_frame()
                    log.info(
                        "[EVENT:staleness_monitor] Freeze-frame triggered "
                        "by staleness detection"
                    )

                self._emit_status(
                    "Stream stale — no fresh frames (freeze-frame active)",
                    "warning",
                )

                # Trigger recovery if not already in progress and not
                # intentionally paused/charging (monitors should have been
                # stopped, but guard against race conditions)
                if self.state in (AppState.PAUSED, AppState.CHARGE_MODE, AppState.STOPPED):
                    log.debug("[EVENT:staleness_monitor] Ignoring stale in %s state", self.state.name)
                else:
                    with self._recovery_lock:
                        already_recovering = self._is_recovering
                    if not already_recovering:
                        self._auto_recover()

            elif not is_stale and self._staleness_was_stale:
                # Stale → fresh transition: reset edge detection
                self._staleness_was_stale = False
                log.info(
                    "[EVENT:staleness_monitor] Frame buffer receiving fresh "
                    "frames again"
                )

        log.debug("[EVENT:staleness_monitor] Monitor loop exited")

    # ── Startup Flow ───────────────────────────────────────────────

    def _startup_flow(self):
        """The main startup sequence. Runs on a background thread.

        Steps:
          1. Check prerequisites (ffmpeg available?)
          2. Discover GoPro on USB with retries
          3. Connect and start webcam mode
          4. Initialize streaming pipeline (virtual camera + frame pipeline + stream reader)
          5. Launch keep-alive thread
          6. Start USB event listener for fast reconnection detection

        If the connection drops at any point, the keep-alive loop
        triggers _auto_recover() which preserves the virtual camera
        and pipeline, only swapping in a new stream reader.
        """
        try:
            # Step 1: Check prerequisites
            self._set_state(AppState.CHECKING_PREREQUISITES)
            if not self._check_prerequisites():
                self._set_state(AppState.ERROR)
                return

            # Step 2: Discover GoPro with retries
            if not self._discover_with_retries():
                self._set_state(AppState.ERROR)
                return

            # Step 3: Start webcam mode
            self._set_state(AppState.CONNECTING)
            if not self._connect_and_start():
                self._set_state(AppState.ERROR)
                return

            # Step 4: Initialize streaming pipeline
            if not self._start_streaming_pipeline():
                self._emit_status(
                    "Webcam mode started but streaming pipeline failed. "
                    "Video feed may not be available.",
                    "warning"
                )
                # Don't fail entirely — webcam mode is working, just pipeline failed

            # Step 5: We're streaming! Start keep-alive monitoring
            self._set_state(AppState.STREAMING)
            self._start_keepalive()

            # Step 6: Start USB listener + disconnect detector
            # The single USB listener fans out to both AppController (reconnect)
            # and DisconnectDetector (freeze-frame). Must start listener FIRST
            # so the detector can receive forwarded events instead of creating
            # its own duplicate listener.
            # NOTE: USB device poller is NOT started here — it spawns PowerShell
            # every 2 seconds which causes I/O stalls that kill ffmpeg's stdout pipe.
            # The poller is only started during recovery (RECONNECTING state).
            self._start_usb_listener()
            self._start_disconnect_detector()

            # Step 7: Start staleness monitor (polls FrameBuffer.is_stale every 500ms)
            self._start_staleness_monitor()

            # Step 9: Fetch camera info (battery, etc.) for the GUI
            self._fetch_camera_info()

        except Exception as e:
            log.exception("[EVENT:startup] Startup flow failed with unexpected error")
            self._emit_status(f"Unexpected error: {e}", "error")
            self._set_state(AppState.ERROR)

    def _kill_orphaned_ffmpeg(self):
        """Kill any orphaned ffmpeg processes that are binding our UDP port.

        When GoProBridge crashes or is killed by Task Manager, the ffmpeg
        subprocess may keep running and hold the UDP port. This prevents
        the next launch from binding to the same port, forcing auto-selection
        to a different port number.

        Only kills ffmpeg processes whose command line includes our UDP port.
        """
        try:
            import psutil
            target_port = str(self.config.udp_port)
            killed = 0
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if proc.info["name"] and "ffmpeg" in proc.info["name"].lower():
                        cmdline = proc.info.get("cmdline") or []
                        cmd_str = " ".join(cmdline)
                        if target_port in cmd_str and "udp" in cmd_str.lower():
                            log.info(
                                "[EVENT:startup] Killing orphaned ffmpeg (PID %d) "
                                "holding UDP port %s",
                                proc.pid, target_port,
                            )
                            proc.kill()
                            proc.wait(timeout=3)
                            killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            if killed:
                self._emit_status(
                    f"Cleaned up {killed} orphaned ffmpeg process(es)", "info"
                )
                # Brief wait for port release
                time.sleep(0.5)
        except Exception as e:
            log.debug("Orphaned ffmpeg cleanup failed: %s", e)

    def _check_prerequisites(self) -> bool:
        """Verify that ffmpeg is available, firewall rule is in place, and port is free.

        Checks:
          1. ffmpeg on PATH (required for MPEG-TS decoding)
          2. Windows Firewall rule for UDP port 8554 (required to receive
             the GoPro's video stream). On first run this triggers a UAC
             elevation prompt to create the rule persistently.
          3. UDP port 8554 is not already in use by another process
        """
        import shutil
        from firewall import ensure_firewall_rule
        from port_checker import check_udp_port_available

        self._emit_status("Checking prerequisites...", "info")

        # Check 1: ffmpeg
        ffmpeg_path = self.config.ffmpeg_path
        found = shutil.which(ffmpeg_path)

        if found:
            log.info("[EVENT:startup] ffmpeg found at: %s", found)
            self._emit_status(f"ffmpeg found: {found}", "success")
        else:
            log.error("[EVENT:startup] ffmpeg not found on PATH")
            self._emit_status(
                f"ffmpeg not found! Please install ffmpeg and add it to PATH.\n"
                f"Download from: https://ffmpeg.org/download.html",
                "error"
            )
            return False

        # Check 1b: Kill orphaned ffmpeg processes from previous crashes
        # These hold the UDP port and prevent the new ffmpeg from binding.
        self._kill_orphaned_ffmpeg()

        # Check 2: Firewall rule for UDP 8554
        self._emit_status("Checking firewall rule for UDP port 8554...", "info")
        if ensure_firewall_rule():
            self._emit_status("Firewall rule OK — UDP port 8554 is allowed", "success")
        else:
            log.warning("[EVENT:startup] Firewall rule could not be created")
            self._emit_status(
                "Warning: Could not create firewall rule for UDP port 8554. "
                "The GoPro video stream may be blocked. You can allow it manually "
                "in Windows Firewall settings or re-run as administrator.",
                "warning"
            )
            # Don't fail startup — the stream might still work if the user
            # already has a permissive firewall or the rule exists but
            # netsh check failed. Log a warning and proceed.

        # Check 3: Virtual camera backend detection (non-blocking warning)
        from virtual_camera import detect_backend
        backend_info = detect_backend()
        if backend_info["backend"] is not None:
            if backend_info["is_recommended"]:
                self._emit_status(
                    f"Virtual camera backend: Unity Capture (recommended)",
                    "success"
                )
            else:
                # OBS fallback — warn but don't block startup
                self._emit_status(backend_info["warning"], "warning")
        else:
            # No backend at all — warn but don't block startup
            self._emit_status(backend_info["warning"], "warning")

        # Check 4: UDP port availability (with auto-selection fallback)
        from port_checker import find_available_port, PortInUseError
        # Restore the user's preferred port if a previous run auto-selected
        # a different one (ephemeral selection should not persist between runs)
        if self.config._preferred_udp_port:
            self.config.udp_port = self.config._preferred_udp_port
            self.config._preferred_udp_port = 0
        preferred_port = self.config.udp_port
        self._emit_status(f"Checking UDP port {preferred_port} availability...", "info")
        try:
            selected_port = find_available_port(preferred_port)
        except PortInUseError as e:
            log.error(
                "[EVENT:startup] No available UDP port in range %d-%d: %s",
                preferred_port, preferred_port + 9,
                e.conflict.user_message.replace('\n', ' '),
            )
            self._emit_status(e.conflict.user_message, "error")
            return False

        if selected_port != preferred_port:
            log.info(
                "[EVENT:startup] Preferred UDP port %d busy, auto-selected port %d",
                preferred_port, selected_port,
            )
            self._emit_status(
                f"Port {preferred_port} busy — auto-selected port {selected_port}",
                "warning",
            )
            # Update runtime config for this session only — save the user's
            # preferred port so config.save() never persists the auto-selected one.
            self.config._preferred_udp_port = preferred_port
            self.config.udp_port = selected_port
        else:
            log.info("[EVENT:startup] UDP port %d is available", selected_port)
            self._emit_status(f"UDP port {selected_port} is available", "success")

        # Notify GUI of the active port (may differ from config spinner value)
        self._emit_active_port(selected_port)

        return True

    def _discover_with_retries(self) -> bool:
        """Try to find the GoPro, retrying up to config.discovery_max_retries times."""
        self._set_state(AppState.DISCOVERING)
        max_retries = self.config.discovery_max_retries
        retry_interval = self.config.discovery_retry_interval

        for attempt in range(1, max_retries + 1):
            if self._stop_event.is_set():
                return False

            log.info(
                "[EVENT:discovery_start] Discovery attempt %d/%d",
                attempt, max_retries,
            )
            self._emit_status(
                f"Discovery attempt {attempt}/{max_retries}...",
                "info"
            )

            if self.gopro.discover():
                log.info("[EVENT:discovery_found] GoPro discovered on attempt %d", attempt)
                return True

            if attempt < max_retries:
                log.info(
                    "[EVENT:reconnect_attempt] GoPro not found, retrying in %.0fs (attempt %d/%d)",
                    retry_interval, attempt, max_retries,
                )
                self._emit_status(
                    f"GoPro not found. Retrying in {retry_interval:.0f}s...",
                    "warning"
                )
                # Wait but check stop event periodically
                if self._stop_event.wait(timeout=retry_interval):
                    return False

        log.error(
            "[EVENT:discovery_failed] Could not find GoPro after %d attempts",
            max_retries,
        )
        self._emit_status(
            "Could not find GoPro after all retries. "
            "Make sure it's connected via USB-C and powered on.",
            "error"
        )
        return False

    def _connect_and_start(self) -> bool:
        """Open the control connection and start webcam mode on the GoPro.

        This wires the full discovery-connect-initialize pipeline:
          1. open_connection() — API verification, wired USB control, IDLE workaround
          2. start_webcam() — enter webcam mode with resolution/FOV settings

        If discovery found a device with an IP but open_connection() hasn't been
        called yet, it runs here. If webcam mode fails but the camera is still
        reachable, triggers the on_webcam_mode_failed callback so the GUI can
        show manual setup instructions as a fallback.
        """
        # Step 1: Open control connection if not already done by discover()
        # discover() verifies HTTP reachability but does NOT:
        #   - enable wired USB control
        #   - run the IDLE state workaround
        #   - start keep-alive on the GoProConnection level
        # open_connection() does all of those, making the camera ready for webcam start.
        if self.gopro.device_info and self.gopro.state != ConnectionState.CONNECTED:
            self._emit_status("Opening control connection to GoPro...", "info")
            if not self.gopro.open_connection(self.gopro.device_info):
                self._emit_status("Failed to open control connection", "error")
                return False

        # Apply anti-flicker setting before starting webcam
        self._apply_anti_flicker()

        self._emit_status("Starting webcam mode on GoPro...", "info")

        success = self.gopro.start_webcam(
            resolution=self.config.resolution,
            fov=self.config.fov,
        )

        if success:
            self._emit_status(
                f"GoPro webcam streaming at {self.config.stream_width}x{self.config.stream_height} "
                f"on UDP port {self.config.udp_port}",
                "success"
            )
        else:
            self._emit_status("Failed to start GoPro webcam mode", "error")
            self._notify_webcam_mode_failed(
                "Automatic webcam mode switching failed. The GoPro is connected\n"
                "but would not enter webcam mode. Follow the steps below to\n"
                "manually configure your camera, then click Retry:"
            )

        return success

    def _apply_anti_flicker(self):
        """Apply the anti-flicker setting (50Hz PAL / 60Hz NTSC) to the camera.

        GoPro setting 134: 0=NTSC (60Hz), 1=PAL (50Hz).
        Should match the local AC power frequency to avoid light flicker.
        """
        value = self.config.anti_flicker
        label = "50Hz (PAL)" if value == 1 else "60Hz (NTSC)"
        result = self.gopro._api_get(
            f"/gopro/camera/setting?setting=134&option={value}",
        )
        if result is not None:
            log.info("[EVENT:config] Anti-flicker set to %s", label)
        else:
            log.warning("[EVENT:config] Failed to set anti-flicker to %s", label)

    def _notify_webcam_mode_failed(self, reason: str):
        """Notify listeners that programmatic webcam mode switching failed.

        This triggers the manual mode guide in the GUI so the user can
        follow step-by-step instructions for manual camera setup.

        Args:
            reason: Human-readable explanation of the failure.
        """
        log.warning("[EVENT:webcam_mode_failed] %s", reason.replace('\n', ' '))
        if self.on_webcam_mode_failed:
            try:
                self.on_webcam_mode_failed(reason)
            except Exception as e:
                log.error("Webcam mode failed callback error: %s", e)

    # ── Streaming Pipeline Management ──────────────────────────────

    def _start_streaming_pipeline(self) -> bool:
        """Initialize and start the full streaming pipeline.

        Creates (if needed):
          1. VirtualCamera — opened once, stays open across reconnects
          2. FrameBuffer — thread-safe frame storage for freeze-frame support
          3. StreamReader — ffmpeg subprocess for MPEG-TS decoding
          4. FramePipeline — connects reader → buffer → vcam, handles freeze-frame

        The virtual camera is the first thing created and the last thing
        destroyed. It NEVER closes during recovery — this ensures downstream
        apps always see a valid camera device.

        Returns:
            True if all components started successfully.
        """
        from stream_reader import StreamReader
        from frame_pipeline import FramePipeline
        from frame_buffer import FrameBuffer
        from virtual_camera import VirtualCamera, verify_device_visible

        self._emit_status("Starting streaming pipeline...", "info")

        # Step 1: Open virtual camera (if not already open from a previous session)
        if self._virtual_camera is None or not self._virtual_camera.is_running:
            self._virtual_camera = VirtualCamera(self.config)
            if not self._virtual_camera.start():
                self._emit_status(
                    "Failed to open virtual camera. Install Unity Capture or OBS VirtualCam.",
                    "error"
                )
                log.error("[EVENT:vcam_error] Failed to start virtual camera")
                return False
            log.info("[EVENT:vcam_start] Virtual camera opened: %s", self._virtual_camera.device_name)

            # Step 1b: Verify the virtual camera is visible in DirectShow device list
            # This confirms apps like Zoom and Teams will see 'GoPro Webcam' in their
            # camera dropdown. Non-blocking — if verification fails, we log a warning
            # but continue (the camera may still work in some apps).
            device_name = self._virtual_camera.device_name
            if verify_device_visible(device_name, timeout=5.0):
                self._emit_status(
                    f"'{device_name}' verified in system camera list — "
                    f"visible to Zoom, Teams, and other apps",
                    "success"
                )
            else:
                log.warning(
                    "[EVENT:vcam_error] '%s' not found in DirectShow device list. "
                    "Apps may not see the virtual camera.",
                    device_name,
                )
                self._emit_status(
                    f"Warning: '{device_name}' not detected in system camera list. "
                    f"It may still work — try selecting it in Zoom or Teams.",
                    "warning"
                )
        else:
            log.info("[EVENT:vcam_start] Virtual camera already running (reusing)")

        # Step 2: Start stream reader (ffmpeg)
        self._stream_reader = StreamReader(self.config)
        if not self._stream_reader.start():
            self._emit_status("Failed to start ffmpeg stream reader", "error")
            log.error("[EVENT:ffmpeg_error] Failed to start StreamReader")
            return False
        log.info("[EVENT:ffmpeg_start] Stream reader started (PID %s)", self._stream_reader._process.pid if self._stream_reader._process else "?")

        # Step 3: Create FrameBuffer for freeze-frame support
        if self._frame_buffer is None:
            self._frame_buffer = FrameBuffer(
                width=self.config.stream_width,
                height=self.config.stream_height,
            )
            log.info(
                "[EVENT:stream_start] FrameBuffer created (%dx%d)",
                self.config.stream_width, self.config.stream_height,
            )

        # Step 4: Start frame pipeline (connects reader → buffer → vcam)
        if self._frame_pipeline is None or not self._frame_pipeline.is_running:
            self._frame_pipeline = FramePipeline(self.config)
            # Wire callbacks for recovery coordination
            self._frame_pipeline.on_stream_lost = self._on_pipeline_stream_lost
            self._frame_pipeline.on_stream_recovered = self._on_pipeline_stream_recovered
        else:
            # Pipeline already running (from a previous recovery) — just swap the reader
            log.info("[EVENT:stream_start] Pipeline already running, swapping reader")
            self._frame_pipeline.swap_reader(self._stream_reader)
            return True

        if not self._frame_pipeline.start(
            self._stream_reader, self._virtual_camera, self._frame_buffer
        ):
            self._emit_status("Failed to start frame pipeline", "error")
            log.error("[EVENT:stream_error] Failed to start FramePipeline")
            return False

        self._emit_status("Streaming pipeline active", "success")
        log.info("[EVENT:stream_start] Full streaming pipeline started")
        return True

    def _create_and_swap_stream_reader(self) -> bool:
        """Create a new StreamReader and swap it into the running pipeline.

        This is the core of seamless recovery -- the virtual camera stays
        open, the pipeline keeps running (in freeze-frame mode), and we
        just swap in a new ffmpeg process with fresh stream data.

        Steps:
          1. Stop the old ffmpeg process (release UDP port)
          2. Wait briefly for port release (OS may hold the socket)
          3. Create and start a new ffmpeg process
          4. Verify the new stream produces at least one frame (stabilization)
          5. Swap the new reader into the running pipeline
          6. Update the disconnect detector with the new reader reference

        Returns:
            True if the new reader was created, verified, and swapped successfully.
        """
        from stream_reader import StreamReader

        # Pause health monitor during the entire swap to prevent false
        # disconnect triggers when we intentionally stop the old reader.
        if self._disconnect_detector is not None:
            self._disconnect_detector.pause_health_monitor()

        try:
            # Step 1: Stop the old reader (kill stale ffmpeg process)
            if self._stream_reader is not None:
                log.info("[EVENT:ffmpeg_stop] Stopping old stream reader before re-init")
                try:
                    self._stream_reader.stop()
                except Exception:
                    log.debug("Error stopping old stream reader (may already be dead)")

            # Step 2: Wait for UDP port release
            # When ffmpeg exits, the OS may hold the UDP socket briefly.
            # Without this delay, the new ffmpeg may fail to bind or miss packets.
            port_release_delay = self.config.ffmpeg_port_release_delay
            if port_release_delay > 0:
                log.debug(
                    "[EVENT:stream_reinit] Waiting %.1fs for UDP port release...",
                    port_release_delay,
                )
                if self._stop_event.wait(timeout=port_release_delay):
                    return False  # App shutting down

            # Step 3: Create and start a new reader
            self._stream_reader = StreamReader(self.config)
            if not self._stream_reader.start():
                log.error("[EVENT:ffmpeg_error] Failed to start new StreamReader for recovery")
                return False

            log.info(
                "[EVENT:ffmpeg_start] New stream reader started for recovery "
                "(PID %s)",
                self._stream_reader._process.pid if self._stream_reader._process else "?",
            )

            # Step 4: Verify the new stream is producing frames (stabilization)
            if not self._wait_for_first_frame(self._stream_reader):
                log.warning(
                    "[EVENT:stream_reinit] New stream reader did not produce frames "
                    "within timeout -- swapping anyway (GoPro may need more time)"
                )
                # Don't fail here -- the pipeline will enter freeze-frame if
                # frames aren't flowing, and recovery will retry.
                # Some cameras take extra time to start the MPEG-TS stream.

            # Step 5: Update the disconnect detector's stream reader reference
            if self._disconnect_detector is not None:
                self._disconnect_detector.update_stream_reader(self._stream_reader)
                self._disconnect_detector.mark_connected()

            # Step 6: Swap into the running pipeline (this exits freeze-frame mode)
            if self._frame_pipeline is not None and self._frame_pipeline.is_running:
                self._frame_pipeline.swap_reader(self._stream_reader)
                log.info("[EVENT:frame_recovery] New reader swapped into pipeline -- freeze-frame replaced with live video")
                return True
            else:
                # Pipeline died during recovery -- try a full pipeline restart
                log.warning("[EVENT:stream_error] Pipeline not running -- attempting full pipeline restart")
                return self._start_streaming_pipeline()
        finally:
            # Always resume health monitor, even if swap failed or raised
            if self._disconnect_detector is not None:
                self._disconnect_detector.resume_health_monitor()

    def _wait_for_first_frame(self, reader) -> bool:
        """Wait for the new StreamReader to produce at least one frame.

        This validates that:
          - ffmpeg successfully bound to the UDP port
          - The GoPro is actually sending MPEG-TS packets
          - The stream is decodable (valid H.264)

        The frame is read and discarded -- the pipeline will read subsequent
        frames. This is purely a health check.

        Args:
            reader: StreamReader instance (already started).

        Returns:
            True if a frame was received within the timeout.
        """
        timeout = self.config.stream_startup_timeout
        poll_interval = 0.25  # Check every 250ms
        elapsed = 0.0

        log.debug(
            "[EVENT:stream_reinit] Waiting up to %.1fs for first frame from new stream reader...",
            timeout,
        )

        while elapsed < timeout and not self._stop_event.is_set():
            if not reader.is_running:
                log.warning(
                    "[EVENT:stream_reinit] Stream reader died during stabilization wait "
                    "(exit_code=%s, last_error=%s)",
                    reader.exit_code, reader.last_error,
                )
                return False

            frame = reader.read_frame()
            if frame is not None:
                log.info(
                    "[EVENT:stream_reinit] First frame received after %.1fs -- "
                    "stream is stable (frame shape: %s)",
                    elapsed, frame.shape,
                )
                return True

            if self._stop_event.wait(timeout=poll_interval):
                return False  # App shutting down
            elapsed += poll_interval

        log.warning(
            "[EVENT:stream_reinit] No frame received within %.1fs timeout",
            timeout,
        )
        return False

    def _on_pipeline_stream_lost(self):
        """Called by FramePipeline when the stream drops (entering freeze-frame).

        Suppressed during intentional pause/charge — user already knows.
        """
        if self.state in (AppState.PAUSED, AppState.CHARGE_MODE):
            log.debug("[EVENT:freeze_frame] Freeze-frame in %s state (suppressed)", self.state.name)
            return

        log.info(
            "[EVENT:freeze_frame] Pipeline entered freeze-frame mode — "
            "virtual camera continues showing last good frame"
        )
        self._emit_status(
            "Stream interrupted — showing freeze-frame while reconnecting",
            "warning"
        )

    def _on_pipeline_stream_recovered(self):
        """Called by FramePipeline when exiting freeze-frame (live video restored)."""
        log.info("[EVENT:frame_recovery] Pipeline exited freeze-frame — live video restored")
        self._emit_status("Live video restored!", "success")

    # ── Keep-Alive & Health Monitoring ─────────────────────────────

    def _start_keepalive(self):
        """Launch the keep-alive thread that prevents camera sleep and detects disconnects."""
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            return

        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop,
            name="KeepAlive",
            daemon=True,
        )
        self._keepalive_thread.start()
        log.info("[EVENT:keepalive_ok] Keep-alive thread started (interval: %.1fs)", self.config.keepalive_interval)

    def _keepalive_loop(self):
        """Periodically ping the GoPro to keep it awake and detect disconnects.

        If the camera stops responding, triggers auto-recovery which:
          1. Enters freeze-frame mode on the pipeline
          2. Waits for USB reconnection (event-driven or polled)
          3. Re-discovers the camera and restarts webcam mode
          4. Swaps a new stream reader into the pipeline
          5. Exits freeze-frame → live video restored
        """
        consecutive_failures = 0
        max_failures_before_reconnect = 3
        # Enforce minimum keep-alive interval to avoid spamming the camera
        keepalive_interval = max(self.config.keepalive_interval, 1.0)

        while self._running and not self._stop_event.is_set():
            if self._stop_event.wait(timeout=keepalive_interval):
                break

            # Skip keep-alive when intentionally paused/charging
            if self.state in (AppState.PAUSED, AppState.CHARGE_MODE):
                consecutive_failures = 0
                continue

            alive = self.gopro.keep_alive()

            if alive:
                consecutive_failures = 0
                # Optionally refresh camera info periodically
            else:
                consecutive_failures += 1
                log.warning(
                    "[EVENT:keepalive_fail] Keep-alive failed (%d/%d)",
                    consecutive_failures, max_failures_before_reconnect,
                )
                self._emit_status(
                    f"Keep-alive failed ({consecutive_failures}/{max_failures_before_reconnect})",
                    "warning"
                )

                if consecutive_failures >= max_failures_before_reconnect:
                    log.warning(
                        "[EVENT:disconnection] GoPro connection lost after %d consecutive keep-alive failures",
                        consecutive_failures,
                    )
                    self._emit_status("GoPro connection lost — starting auto-recovery", "warning")
                    self._auto_recover()
                    break  # _auto_recover handles the recovery loop

    # ── Auto-Recovery ──────────────────────────────────────────────

    def _auto_recover(self):
        """Attempt seamless reconnection after a disconnect.

        This is the core recovery logic. It:
          1. Enters freeze-frame mode on the pipeline (virtual camera keeps showing
             the last good frame — downstream apps never see it disappear)
          2. Resets the GoProConnection state
          3. Starts the recovery loop that waits for USB reconnection and
             re-establishes the stream

        Key design: The virtual camera and frame pipeline are NEVER stopped.
        Only the stream reader (ffmpeg process) is killed and recreated.
        """
        with self._recovery_lock:
            if self._is_recovering:
                log.debug("[EVENT:reconnect_attempt] Recovery already in progress, skipping")
                return
            self._is_recovering = True

        self._set_state(AppState.RECONNECTING)

        # Step 1: Enter freeze-frame mode on the pipeline
        if self._frame_pipeline is not None and self._frame_pipeline.is_running:
            self._frame_pipeline.enter_freeze_frame()
            log.info(
                "[EVENT:freeze_frame] Pipeline in freeze-frame mode — "
                "virtual camera alive, showing last good frame"
            )
        else:
            log.warning("[EVENT:freeze_frame] No active pipeline to freeze (vcam may show placeholder)")

        # Step 2: Stop the old stream reader (kill stale ffmpeg)
        if self._stream_reader is not None:
            try:
                self._stream_reader.stop()
            except Exception:
                log.debug("Error stopping old stream reader during recovery")

        # Step 3: Reset GoPro connection state (but DON'T disconnect — just clear tracking)
        self.gopro.reset_for_recovery()

        self._emit_status("Waiting for GoPro to reconnect...", "info")

        # Step 4: Start the recovery loop on a separate thread
        self._usb_reconnect_event.clear()
        self._recovery_thread = threading.Thread(
            target=self._recovery_loop,
            name="RecoveryLoop",
            daemon=True,
        )
        self._recovery_thread.start()

    def _recovery_loop(self):
        """Background loop that waits for USB reconnection and restores the stream.

        Waits for either:
          a) USB event listener fires (instant notification of GoPro reconnect)
          b) Poll-based timeout (fallback if USB listener isn't working)

        Then re-discovers the camera, re-establishes the webcam connection,
        and swaps a new stream reader into the running pipeline.

        The virtual camera NEVER stops during this entire process.
        """
        reconnect_delay = self.config.reconnect_delay
        reconnect_max_delay = self.config.reconnect_max_delay
        max_retries = self.config.reconnect_max_retries  # 0 = infinite
        current_delay = reconnect_delay  # Grows with exponential backoff up to reconnect_max_delay

        log.info(
            "[EVENT:reconnect_attempt] Recovery loop started "
            "(delay=%.1fs, max_delay=%.1fs, max_retries=%s)",
            reconnect_delay, reconnect_max_delay,
            max_retries if max_retries > 0 else "infinite",
        )

        attempt = 0
        try:
            while self._running and not self._stop_event.is_set():
                attempt += 1

                if max_retries > 0 and attempt > max_retries:
                    log.error(
                        "[EVENT:recovery_failed] Max recovery attempts (%d) reached",
                        max_retries,
                    )
                    self._emit_status(
                        f"Could not recover after {max_retries} attempts. "
                        "Check USB connection and click Retry.",
                        "error"
                    )
                    self._set_state(AppState.ERROR)
                    return

                # Wait for USB reconnection event OR timeout
                self._emit_status(
                    f"Recovery attempt {attempt}: waiting for GoPro USB reconnection...",
                    "info"
                )

                # Use the USB reconnect event with a timeout for poll-based fallback
                # If USB listener detected reconnection, the event is already set
                usb_detected = self._usb_reconnect_event.wait(timeout=current_delay)

                if self._stop_event.is_set():
                    return

                if usb_detected:
                    self._usb_reconnect_event.clear()
                    ncm_wait = self.config.ncm_adapter_wait
                    log.info(
                        "[EVENT:reconnect_attempt] USB reconnection detected (event-driven) "
                        "— polling for NCM network adapter (timeout=%.1fs)...",
                        ncm_wait,
                    )
                    self._emit_status(
                        "USB reconnection detected! Waiting for network...",
                        "info"
                    )
                    # Poll for the NCM network adapter instead of blind sleep —
                    # returns as soon as the interface is up, or None on timeout
                    ncm_ip = self.gopro.wait_for_network_interface(timeout=ncm_wait)
                    if self._stop_event.is_set():
                        return
                    if ncm_ip:
                        log.info(
                            "[EVENT:reconnect_attempt] NCM adapter ready at %s",
                            ncm_ip,
                        )
                else:
                    log.debug(
                        "[EVENT:reconnect_attempt] No USB event — trying poll-based rediscovery "
                        "(attempt %d)",
                        attempt,
                    )

                # Try to re-discover and reconnect
                if self._try_reconnect():
                    # Success! Stream is restored
                    self._recovery_count += 1
                    self._last_recovery_time = time.monotonic()
                    log.info(
                        "[EVENT:recovery_success] Recovery #%d successful — "
                        "live video restored (attempt %d)",
                        self._recovery_count, attempt,
                    )
                    self._emit_status(
                        f"Reconnected to GoPro! (recovery #{self._recovery_count})",
                        "success"
                    )
                    self._set_state(AppState.STREAMING)

                    # Restart keep-alive monitoring
                    self._start_keepalive()

                    # Fetch updated camera info
                    self._fetch_camera_info()
                    return

                # Recovery attempt failed — apply exponential backoff up to reconnect_max_delay
                log.info(
                    "[EVENT:reconnect_attempt] Recovery attempt %d failed, "
                    "retrying in %.1fs...",
                    attempt, current_delay,
                )
                self._emit_status(
                    f"Reconnection attempt {attempt} failed, retrying...",
                    "warning"
                )
                # Exponential backoff: double the delay, capped at reconnect_max_delay
                current_delay = min(current_delay * 2, reconnect_max_delay)

        except Exception as e:
            log.exception("[EVENT:recovery_failed] Recovery loop crashed: %s", e)
            self._emit_status(f"Recovery error: {e}", "error")
            self._set_state(AppState.ERROR)
        finally:
            with self._recovery_lock:
                self._is_recovering = False

    def _try_reconnect(self) -> bool:
        """Attempt a single reconnection: discover → connect → swap reader.

        Does NOT restart the virtual camera or pipeline — only creates
        a new stream reader and swaps it in.

        Returns:
            True if reconnection and stream restoration succeeded.
        """
        # Step 1: Re-discover the GoPro
        log.info("[EVENT:reconnect_attempt] Attempting GoPro rediscovery...")
        if not self.gopro.discover():
            log.debug("[EVENT:reconnect_attempt] Rediscovery failed — camera not found")
            return False

        # Step 2: Open the control connection (API verification, USB control, IDLE workaround)
        log.info("[EVENT:reconnect_attempt] GoPro found, opening control connection...")
        if not self.gopro.open_connection(self.gopro.device_info):
            log.debug("[EVENT:reconnect_attempt] Control connection failed")
            return False

        # Step 3: Start webcam mode
        log.info("[EVENT:reconnect_attempt] Control connected, starting webcam mode...")
        if not self.gopro.start_webcam(
            resolution=self.config.resolution,
            fov=self.config.fov,
        ):
            log.debug("[EVENT:reconnect_attempt] Webcam start failed")
            return False

        # Step 4: Create a new stream reader and swap it into the pipeline
        log.info("[EVENT:reconnect_attempt] Webcam started, creating new stream reader...")
        if not self._create_and_swap_stream_reader():
            log.debug("[EVENT:reconnect_attempt] Stream reader swap failed")
            return False

        log.info(
            "[EVENT:recovery_success] Full reconnection complete — "
            "live video seamlessly restored"
        )
        return True

    # ── Resolution Change ─────────────────────────────────────────

    # GoPro webcam resolution codes -> stream pixel dimensions
    # Resolution code -> (width, height). Must match gopro_connection.RESOLUTION_MAP.
    # Codes vary by model; start_webcam() auto-remaps if the camera rejects a code.
    _RESOLUTION_MAP = {
        4:  (1920, 1080),  # 1080p
        7:  (1280, 720),   # 720p
        12: (1920, 1080),  # 1080p (alternate code)
    }

    def change_resolution(self, resolution: int, fov=None) -> bool:
        """Change the GoPro webcam resolution with seamless freeze-frame transition.

        Flow:
          1. Validate the requested resolution
          2. Enter freeze-frame mode on the pipeline (last good frame keeps playing)
          3. Stop the old ffmpeg stream reader
          4. Send webcam/stop then webcam/start with new resolution
          5. Update config and pipeline dimensions
          6. Create new StreamReader and swap into pipeline
          7. Pipeline exits freeze-frame -> live video at new resolution

        If the resolution change fails at any step, attempts to fall back to
        the previous resolution so the stream is never permanently broken.

        Args:
            resolution: GoPro resolution code (7=720p, 12=1080p).
            fov: Optional FOV code (0=wide, 2=narrow, 3=superview, 4=linear).
                 If None, keeps current config value.

        Returns:
            True if the resolution change succeeded and live video resumed.
        """
        if resolution not in self._RESOLUTION_MAP:
            log.warning(
                "[EVENT:resolution_change] Invalid resolution code %d -- "
                "valid codes: %s",
                resolution, list(self._RESOLUTION_MAP.keys()),
            )
            self._emit_status(
                f"Invalid resolution code {resolution}. "
                f"Use 7 (720p) or 12 (1080p).",
                "error",
            )
            return False

        new_width, new_height = self._RESOLUTION_MAP[resolution]
        # Compare against what the CAMERA is currently running, not the config
        # (the GUI updates config before calling us, so config already has new values)
        old_resolution = self.gopro.current_resolution
        old_fov = self.gopro.current_fov
        # Get old dimensions from the running stream reader (not config, which is already mutated)
        old_width = self._stream_reader.width if self._stream_reader else 1920
        old_height = self._stream_reader.height if self._stream_reader else 1080
        new_fov = fov if fov is not None else self.config.fov

        # No-op if the camera is already running at these settings
        if old_resolution is not None and resolution == old_resolution and new_fov == old_fov:
            log.info(
                "[EVENT:resolution_change] Camera already at resolution %d fov %d "
                "-- no change needed",
                resolution, new_fov,
            )
            self._emit_status("Resolution unchanged", "info")
            return True

        log.info(
            "[EVENT:resolution_change] Changing resolution: %dx%d -> %dx%d "
            "(res=%d->%d, fov=%d->%d)",
            old_width, old_height, new_width, new_height,
            old_resolution, resolution, old_fov, new_fov,
        )
        self._emit_status(
            f"Changing resolution to {new_width}x{new_height}...",
            "info",
        )

        # Step 0: Stop monitoring BEFORE killing ffmpeg — prevents false
        # recovery from staleness monitor and disconnect detector seeing the
        # intentional shutdown during resolution change.
        self._stop_staleness_monitor()
        self._stop_disconnect_detector()

        # Step 1: Enter freeze-frame so downstream apps keep seeing valid video
        if self._frame_pipeline is not None and self._frame_pipeline.is_running:
            self._frame_pipeline.enter_freeze_frame()
            log.info(
                "[EVENT:freeze_frame] Freeze-frame activated for resolution change"
            )

        # Step 2: Stop old stream reader
        if self._stream_reader is not None:
            try:
                self._stream_reader.stop()
            except Exception:
                log.debug("Error stopping stream reader during resolution change")

        # Step 3: Send webcam stop + start with new settings
        success = self._apply_resolution_on_camera(resolution, new_fov)

        if not success:
            log.warning(
                "[EVENT:resolution_change] Resolution change failed on camera -- "
                "falling back to previous resolution %dx%d",
                old_width, old_height,
            )
            self._emit_status(
                f"Resolution change failed -- reverting to {old_width}x{old_height}",
                "warning",
            )
            # Attempt fallback: restore previous settings
            fallback_ok = self._apply_resolution_on_camera(old_resolution, old_fov)
            if not fallback_ok:
                log.error(
                    "[EVENT:resolution_change] Fallback to previous resolution "
                    "also failed!"
                )
                self._emit_status(
                    "Resolution change and fallback both failed. "
                    "Try reconnecting the camera.",
                    "error",
                )
                # Try to at least restore the stream at old settings
                return self._resume_stream_after_resolution_change(
                    old_width, old_height, old_resolution, old_fov
                )

            # Fallback succeeded -- resume at old settings
            return self._resume_stream_after_resolution_change(
                old_width, old_height, old_resolution, old_fov
            )

        # Step 4: Update config and component dimensions for new resolution
        return self._resume_stream_after_resolution_change(
            new_width, new_height, resolution, new_fov
        )

    def _apply_resolution_on_camera(self, resolution, fov):
        """Send webcam stop + start with the given resolution/fov to the GoPro.

        Returns True if the camera accepted the new settings and entered
        STREAMING or READY state.
        """
        if not self.gopro.is_connected:
            log.error(
                "[EVENT:resolution_change] Cannot change resolution -- "
                "camera not connected"
            )
            return False

        # Stop webcam stream (not full exit — stop+start is sufficient
        # for resolution changes per official GoPro API docs)
        try:
            self.gopro.stop_webcam()
        except Exception as e:
            log.warning(
                "[EVENT:resolution_change] Error stopping webcam: %s", e
            )

        # Brief settle time — 1.0s is sufficient for stop+start
        time.sleep(max(self.config.idle_reset_delay, 1.0))

        # Start webcam with new resolution
        return self.gopro.start_webcam(resolution=resolution, fov=fov)

    def _resume_stream_after_resolution_change(
        self, width: int, height: int, resolution: int, fov: int
    ) -> bool:
        """Tear down and re-initialize the pipeline at new resolution.

        Full pipeline teardown and re-initialization:
          1. Stop disconnect detector
          2. Stop frame pipeline completely
          3. Update config
          4. Reconfigure virtual camera at new dimensions
          5. Resize frame buffer
          6. Create new stream reader
          7. Create and start new frame pipeline
          8. Restart disconnect detector

        Returns True if the stream resumed successfully.
        """
        from stream_reader import StreamReader
        from frame_pipeline import FramePipeline

        log.info(
            "[EVENT:resolution_change] Pipeline teardown and reinit at %dx%d",
            width, height,
        )

        # Step 1: Stop disconnect detector (depends on pipeline/reader)
        self._stop_disconnect_detector()

        # Step 2: Stop frame pipeline completely
        if self._frame_pipeline is not None:
            self._frame_pipeline.stop()
            self._frame_pipeline = None

        # Step 3: Update config
        self.config.stream_width = width
        self.config.stream_height = height
        self.config.resolution = resolution
        self.config.fov = fov

        # Step 4: Reconfigure virtual camera at new dimensions
        if self._virtual_camera is not None:
            if not self._virtual_camera.reconfigure(width, height):
                log.error(
                    "[EVENT:resolution_change] Virtual camera reconfigure to "
                    "%dx%d failed",
                    width, height,
                )
                self._emit_status(
                    f"Failed to reconfigure virtual camera to {width}x{height}",
                    "error",
                )
                return False

        # Step 5: Resize frame buffer
        if self._frame_buffer is not None:
            self._frame_buffer.resize(width, height)
            log.info(
                "[EVENT:resolution_change] FrameBuffer resized to %dx%d",
                width, height,
            )

        # Step 6: Create new stream reader and wait for first frame
        self._stream_reader = StreamReader(self.config)
        if not self._stream_reader.start():
            log.error(
                "[EVENT:resolution_change] Failed to start stream reader "
                "at new resolution"
            )
            self._emit_status(
                "Resolution changed on camera but stream reader failed "
                "to start",
                "error",
            )
            return False

        # Wait for ffmpeg to connect and produce the first frame.
        # Without this, the pipeline immediately gets None reads and
        # enters freeze-frame mode.
        self._emit_status("Waiting for new stream...", "info")
        import time as _time
        first_frame = None
        for _ in range(int(self.config.stream_startup_timeout * 2)):
            first_frame = self._stream_reader.read_frame()
            if first_frame is not None:
                log.info("[EVENT:resolution_change] First frame received from new reader")
                break
            _time.sleep(0.5)

        if first_frame is None:
            log.warning("[EVENT:resolution_change] No frames from new reader after timeout")
            self._emit_status("Stream reader started but no frames yet", "warning")

        # Step 7: Create and start new frame pipeline
        self._frame_pipeline = FramePipeline(self.config)
        self._frame_pipeline.on_stream_lost = self._on_pipeline_stream_lost
        self._frame_pipeline.on_stream_recovered = self._on_pipeline_stream_recovered
        if not self._frame_pipeline.start(
            self._stream_reader, self._virtual_camera, self._frame_buffer
        ):
            log.error("[EVENT:resolution_change] Failed to start new frame pipeline")
            self._emit_status(
                "Failed to start frame pipeline at new resolution", "error"
            )
            return False

        # Step 8: Restart disconnect detector
        self._start_disconnect_detector()

        self._emit_status(
            f"Resolution changed to {width}x{height} - live video resumed",
            "success",
        )
        log.info(
            "[EVENT:resolution_change] Resolution change complete: %dx%d - "
            "pipeline fully re-initialized",
            width, height,
        )

        # Save updated config to disk
        try:
            self.config.save()
        except Exception as e:
            log.debug("Failed to save config after resolution change: %s", e)

        return True

    # ── Camera Info ──────────────────────────────────────────────

    def _fetch_camera_info(self):
        """Get battery level and other camera info, push to GUI."""
        log.debug("[EVENT:camera_info] Fetching camera info...")
        info = self.gopro.get_camera_info()
        if info and self.on_camera_info:
            try:
                self.on_camera_info(info)
            except Exception as e:
                log.error("[EVENT:camera_info] Camera info callback error: %s", e)

        if info:
            battery = info.get("battery_level")
            if battery is not None:
                log.info("[EVENT:battery] Camera battery level: %s%%", battery)
                self._emit_status(f"Camera battery: {battery}%", "info")

    # ── Resolution Change Detection ───────────────────────────────

    def request_resolution_change(self, resolution: int, fov: Optional[int] = None) -> bool:
        """Request a mid-session resolution change on the GoPro stream.

        This is the public entry point for resolution changes. It detects
        whether the requested resolution differs from the current active
        settings and delegates to change_resolution() if a change is needed.

        Detection logic:
          - If the GoProConnection reports the same resolution is active,
            this is a no-op (returns True immediately).
          - If the resolution differs (or we're not yet tracking it),
            delegates to change_resolution() for the full stop/restart cycle.

        Args:
            resolution: GoPro resolution code (7=720p, 12=1080p).
            fov: GoPro FOV code, or None to keep current.

        Returns:
            True if the resolution was changed (or already at target).
            False if the change failed.
        """
        from gopro_connection import RESOLUTION_MAP, VALID_RESOLUTIONS

        if resolution not in VALID_RESOLUTIONS:
            log.error(
                "[EVENT:resolution_change] Invalid resolution code %d", resolution
            )
            self._emit_status(f"Invalid resolution: {resolution}", "error")
            return False

        # Check if the GoPro connection layer detects a change is needed
        if not self.gopro.needs_resolution_change(resolution, fov):
            new_label = RESOLUTION_MAP[resolution][2]
            log.info(
                "[EVENT:resolution_change] Already at %s — no change needed",
                new_label,
            )
            self._emit_status(f"Already streaming at {new_label}", "info")
            return True

        # Delegate to the full change_resolution flow
        return self.change_resolution(resolution, fov)
