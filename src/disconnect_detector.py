"""
disconnect_detector.py — USB disconnect detection with proactive freeze-frame trigger

Detects GoPro USB disconnects and immediately triggers freeze-frame capture
of the last good decoded video frame BEFORE the stream degrades.

Detection signals (fastest to slowest):
  1. USB device removal event (via USBEventListener on_detach) — ~instant
  2. ffmpeg process exit / stream EOF (StreamReader returns None) — 0.5-3s
  3. Keep-alive HTTP failures (GoProConnection) — 5-7.5s (3 failures × 2.5s)

This module bridges signal #1 (instant USB detach) to the FramePipeline's
freeze-frame mechanism, ensuring the virtual camera captures the last good
frame before corrupted/partial frames arrive from the dying ffmpeg process.

Architecture:
  - Wraps USBEventListener for USB attach/detach events
  - On USB detach: immediately signals FramePipeline.enter_freeze_frame()
  - On USB attach: notifies app controller to begin reconnection
  - Also monitors StreamReader health as a backup signal
  - Thread-safe: callbacks fire from the USB listener thread; freeze-frame
    trigger uses the pipeline's existing thread-safe enter_freeze_frame()

Why proactive freeze is critical:
  When USB is yanked, the GoPro stops sending MPEG-TS packets. ffmpeg may
  continue for 0.5-3 seconds reading buffered data, but the last few frames
  can be corrupted (partial reads, decode artifacts). By freezing immediately
  on USB detach, we guarantee the last frame shown to downstream apps is a
  clean, fully-decoded frame — not a glitchy partial decode.

Usage:
    from disconnect_detector import DisconnectDetector

    detector = DisconnectDetector(pipeline, connection)
    detector.on_disconnect = lambda: print("GoPro disconnected!")
    detector.on_reconnect_ready = lambda dev_id: print("GoPro reconnected!")
    detector.start()
    # ... later ...
    detector.stop()
"""

import threading
import time
from typing import Optional, Callable

from logger import get_logger

log = get_logger("disconnect_detector")


class DisconnectDetector:
    """Monitors for GoPro USB disconnects and triggers proactive freeze-frame.

    Combines fast USB event detection with stream health monitoring to ensure
    the virtual camera freezes on the last good frame immediately when the
    GoPro is unplugged.

    The detector maintains three layers of disconnect detection:
      1. USB event listener (instant) — primary signal
      2. Stream health monitor (sub-second) — backup for non-USB disconnects
      3. Keep-alive failures (seconds) — handled by GoProConnection, not here

    Attributes:
        is_running: True if the detector is actively monitoring.
        is_disconnected: True if a USB disconnect has been detected.
        last_disconnect_time: Monotonic timestamp of the last disconnect, or None.
        last_attach_time: Monotonic timestamp of the last USB attach, or None.
    """

    def __init__(
        self,
        pipeline=None,
        connection=None,
        stream_reader=None,
        config=None,
        usb_listener=None,
        startup_grace: float = 0.0,
    ):
        """Initialize the disconnect detector.

        Args:
            pipeline: FramePipeline instance — enter_freeze_frame() called on disconnect.
            connection: GoProConnection instance — state checked for disconnect confirmation.
            stream_reader: StreamReader instance — monitored for process death.
            config: Config instance — for timing parameters.
            usb_listener: Optional external USBEventListener instance. When provided,
                the detector will NOT create its own listener — USB events should be
                forwarded via handle_usb_attach()/handle_usb_detach() from the owner.
                This enables a single USBEventListener to fan out to multiple consumers.
        """
        self._pipeline = pipeline
        self._connection = connection
        self._stream_reader = stream_reader
        self._config = config
        self._startup_grace = startup_grace

        # USB event listener — either externally owned or created internally
        self._usb_listener = None
        self._external_usb_listener = usb_listener is not None

        # State
        self._running = False
        self._disconnected = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Stream health monitor thread
        self._health_thread: Optional[threading.Thread] = None
        self._health_check_interval: float = 0.5  # Check ffmpeg health every 500ms
        self._health_paused = threading.Event()  # When set, health monitor is paused

        # Timestamps
        self._last_disconnect_time: Optional[float] = None
        self._last_attach_time: Optional[float] = None
        self._last_freeze_trigger_time: Optional[float] = None

        # Debounce: ignore rapid attach/detach within this window (seconds)
        self._debounce_interval: float = 1.0

        # Callbacks
        self.on_disconnect: Optional[Callable[[], None]] = None
        self.on_reconnect_ready: Optional[Callable[[str], None]] = None
        self.on_usb_attach: Optional[Callable[[str], None]] = None
        self.on_usb_detach: Optional[Callable[[str], None]] = None
        self.on_ffmpeg_crash: Optional[Callable[[], None]] = None

    # -- Properties ----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True if the detector is actively monitoring."""
        with self._lock:
            return self._running

    @property
    def is_disconnected(self) -> bool:
        """True if a USB disconnect has been detected and not yet recovered."""
        with self._lock:
            return self._disconnected

    @property
    def last_disconnect_time(self) -> Optional[float]:
        """Monotonic timestamp of the last disconnect event."""
        return self._last_disconnect_time

    @property
    def last_attach_time(self) -> Optional[float]:
        """Monotonic timestamp of the last USB attach event."""
        return self._last_attach_time

    @property
    def disconnect_duration(self) -> float:
        """Seconds since last disconnect, or 0.0 if connected."""
        if self._last_disconnect_time is None or not self._disconnected:
            return 0.0
        return time.monotonic() - self._last_disconnect_time

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> bool:
        """Start monitoring for USB disconnects.

        Initializes the USB event listener and starts the stream health
        monitor thread.

        Returns:
            True if monitoring started successfully.
        """
        with self._lock:
            if self._running:
                log.warning("[EVENT:disconnect_detector] Already running")
                return True

        log.info("[EVENT:disconnect_detector] Starting disconnect detector")
        self._stop_event.clear()

        # Start USB event listener
        usb_started = self._start_usb_listener()

        # Start stream health monitor
        self._start_health_monitor()

        with self._lock:
            self._running = True
            self._disconnected = False

        if usb_started:
            log.info(
                "[EVENT:disconnect_detector] Disconnect detector started "
                "(USB events + stream health monitoring)"
            )
        else:
            log.warning(
                "[EVENT:disconnect_detector] Disconnect detector started "
                "(stream health monitoring only — USB events unavailable)"
            )

        return True

    def stop(self):
        """Stop monitoring and clean up resources."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        log.info("[EVENT:disconnect_detector] Stopping disconnect detector")
        self._stop_event.set()

        # Stop USB listener
        self._stop_usb_listener()

        # Stop health monitor thread
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=3.0)

        log.info("[EVENT:disconnect_detector] Disconnect detector stopped")

    # -- USB Event Listener --------------------------------------------------

    def _start_usb_listener(self) -> bool:
        """Initialize and start the USB event listener.

        If an external USB listener was provided at construction time,
        this is a no-op — the owner is responsible for starting it and
        forwarding events via handle_usb_attach()/handle_usb_detach().

        Returns:
            True if USB events will be received (either external or internal).
        """
        if self._external_usb_listener:
            log.debug(
                "[EVENT:disconnect_detector] Using external USB listener "
                "(events forwarded via handle_usb_attach/handle_usb_detach)"
            )
            return True

        try:
            from usb_event_listener import USBEventListener

            self._usb_listener = USBEventListener(
                on_attach=self._on_usb_attach,
                on_detach=self._on_usb_detach,
            )
            return self._usb_listener.start(timeout=5.0)
        except ImportError:
            log.warning(
                "[EVENT:disconnect_detector] USBEventListener not available "
                "(non-Windows platform?) — falling back to stream monitoring only"
            )
            return False
        except Exception as e:
            log.warning(
                "[EVENT:disconnect_detector] Failed to start USB listener: %s "
                "— falling back to stream monitoring only",
                e,
            )
            return False

    def _stop_usb_listener(self):
        """Stop the USB event listener if active.

        Only stops internally-created listeners. External listeners are
        owned by the caller (e.g. AppController) and not stopped here.
        """
        if self._external_usb_listener:
            log.debug("[EVENT:disconnect_detector] External USB listener — not stopping (owned by caller)")
            return

        if self._usb_listener is not None:
            try:
                self._usb_listener.stop()
            except Exception as e:
                log.debug("Error stopping USB listener: %s", e)
            self._usb_listener = None

    def _on_usb_detach(self, device_id: str):
        """Handle GoPro USB detach event — CRITICAL PATH.

        This is the fastest disconnect signal. We immediately:
          1. Capture the current frame state (already in VirtualCamera._last_frame)
          2. Trigger freeze-frame mode on the pipeline
          3. Mark ourselves as disconnected
          4. Fire the on_disconnect callback

        The freeze-frame trigger is the most time-critical operation here.
        It must happen BEFORE ffmpeg starts producing corrupted frames from
        its internal buffers.

        Args:
            device_id: The USB device path string from Windows.
        """
        now = time.monotonic()

        # Debounce: ignore if we just processed a detach
        if (self._last_disconnect_time is not None
                and (now - self._last_disconnect_time) < self._debounce_interval):
            log.debug(
                "[EVENT:usb_detach] Debounced duplicate detach event (%.2fs since last)",
                now - self._last_disconnect_time,
            )
            return

        log.warning(
            "[EVENT:usb_detach] GoPro USB disconnect detected: %s",
            device_id[:80] if device_id else "unknown",
        )

        # CRITICAL: Trigger freeze-frame IMMEDIATELY
        # This captures the last good frame before stream corruption
        self._trigger_freeze_frame("USB detach event")

        with self._lock:
            self._disconnected = True
            self._last_disconnect_time = now

        # Fire external callback
        if self.on_usb_detach:
            try:
                self.on_usb_detach(device_id)
            except Exception:
                log.exception("Error in on_usb_detach callback")

        if self.on_disconnect:
            try:
                self.on_disconnect()
            except Exception:
                log.exception("Error in on_disconnect callback")

    def _on_usb_attach(self, device_id: str):
        """Handle GoPro USB attach event.

        This fires when the GoPro is plugged back in. We don't immediately
        resume streaming — the app controller handles reconnection. We just
        signal that the device is back.

        Args:
            device_id: The USB device path string from Windows.
        """
        now = time.monotonic()

        # Debounce
        if (self._last_attach_time is not None
                and (now - self._last_attach_time) < self._debounce_interval):
            log.debug("[EVENT:usb_attach] Debounced duplicate attach event")
            return

        log.info(
            "[EVENT:usb_attach] GoPro USB reconnect detected: %s",
            device_id[:80] if device_id else "unknown",
        )

        self._last_attach_time = now

        # Fire callbacks
        if self.on_usb_attach:
            try:
                self.on_usb_attach(device_id)
            except Exception:
                log.exception("Error in on_usb_attach callback")

        if self.on_reconnect_ready:
            try:
                self.on_reconnect_ready(device_id)
            except Exception:
                log.exception("Error in on_reconnect_ready callback")

    # -- Public USB Event Forwarding -----------------------------------------

    def handle_usb_attach(self, device_id: str):
        """Public entry point for USB attach events from an external listener.

        When a single USBEventListener fans out to multiple consumers,
        the owner calls this method to forward attach events to this detector.

        Args:
            device_id: The USB device path string from Windows.
        """
        self._on_usb_attach(device_id)

    def handle_usb_detach(self, device_id: str):
        """Public entry point for USB detach events from an external listener.

        When a single USBEventListener fans out to multiple consumers,
        the owner calls this method to forward detach events to this detector.

        Args:
            device_id: The USB device path string from Windows.
        """
        self._on_usb_detach(device_id)

    # -- Freeze-Frame Trigger ------------------------------------------------

    def _trigger_freeze_frame(self, reason: str):
        """Immediately trigger freeze-frame mode on the pipeline.

        This is the core of proactive disconnect handling. By freezing
        before ffmpeg's buffers drain, we guarantee the virtual camera
        shows the last cleanly-decoded frame.

        Args:
            reason: Human-readable description of why freeze was triggered.
        """
        now = time.monotonic()

        # Prevent duplicate freeze triggers within a short window
        if (self._last_freeze_trigger_time is not None
                and (now - self._last_freeze_trigger_time) < self._debounce_interval):
            log.debug("Freeze trigger debounced (%.2fs since last)",
                      now - self._last_freeze_trigger_time)
            return

        self._last_freeze_trigger_time = now

        if self._pipeline is not None:
            log.info(
                "[EVENT:freeze_frame] Proactive freeze-frame triggered: %s",
                reason,
            )
            try:
                self._pipeline.enter_freeze_frame()
            except Exception:
                log.exception(
                    "[EVENT:freeze_frame] Failed to trigger freeze-frame"
                )
        else:
            log.debug(
                "Freeze trigger skipped (no pipeline): %s", reason
            )

    # -- Stream Health Monitor -----------------------------------------------

    def _start_health_monitor(self):
        """Start the background stream health monitor thread.

        This provides a backup disconnect signal by monitoring the
        StreamReader's ffmpeg process. If ffmpeg exits unexpectedly,
        we trigger freeze-frame (if not already triggered by USB events).
        """
        self._health_thread = threading.Thread(
            target=self._health_monitor_loop,
            name="stream-health-monitor",
            daemon=True,
        )
        self._health_thread.start()

    def _health_monitor_loop(self):
        """Background loop monitoring stream reader health.

        Checks whether the ffmpeg process is still alive. If it dies
        unexpectedly (not from a clean stop), triggers freeze-frame.

        Distinguishes between two failure modes:
          1. USB disconnect + stream death: USB detach already triggered freeze,
             health monitor fires on_disconnect as backup.
          2. Pure ffmpeg crash (USB still connected): fires on_ffmpeg_crash
             for lightweight recovery (just restart ffmpeg, skip USB reconnect).

        Detection heuristic: if _disconnected was already set by a USB detach
        event before we noticed stream death, it's a USB disconnect. Otherwise,
        it's a pure ffmpeg crash.
        """
        was_running = False

        if self._startup_grace > 0:
            log.debug(
                "[EVENT:disconnect_detector] Startup grace: %.1fs before first health check",
                self._startup_grace,
            )
            if self._stop_event.wait(timeout=self._startup_grace):
                return  # Stopped during grace period

        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=self._health_check_interval):
                break

            # Skip health checks while paused (e.g., during reader swap)
            if self._health_paused.is_set():
                was_running = False  # Reset so we don't false-trigger on resume
                continue

            reader = self._stream_reader
            if reader is None:
                continue

            is_running = False
            try:
                is_running = reader.is_running
            except Exception:
                pass

            if was_running and not is_running:
                # Stream just died — trigger freeze if not already frozen
                log.warning(
                    "[EVENT:stream_health] Stream reader stopped unexpectedly"
                )
                self._trigger_freeze_frame("ffmpeg process exited")

                # Determine cause: USB disconnect (already flagged) vs pure ffmpeg crash
                with self._lock:
                    was_usb_disconnect = self._disconnected
                    if not self._disconnected:
                        self._disconnected = True
                        self._last_disconnect_time = time.monotonic()

                if was_usb_disconnect:
                    # USB detach already handled this — fire on_disconnect as backup
                    log.info(
                        "[EVENT:stream_health] Stream death after USB disconnect "
                        "(backup signal — USB detach already triggered recovery)"
                    )
                    if self.on_disconnect:
                        try:
                            self.on_disconnect()
                        except Exception:
                            log.exception("Error in on_disconnect callback (health monitor)")
                else:
                    # Pure ffmpeg crash — GoPro likely still connected via USB
                    log.warning(
                        "[EVENT:ffmpeg_crash] ffmpeg process died without USB disconnect — "
                        "triggering lightweight ffmpeg recovery"
                    )
                    if self.on_ffmpeg_crash:
                        try:
                            self.on_ffmpeg_crash()
                        except Exception:
                            log.exception("Error in on_ffmpeg_crash callback")
                    elif self.on_disconnect:
                        # Fallback: use generic disconnect handler if no ffmpeg crash handler
                        try:
                            self.on_disconnect()
                        except Exception:
                            log.exception("Error in on_disconnect callback (ffmpeg crash fallback)")

            was_running = is_running

    # -- Health Monitor Pause/Resume -----------------------------------------

    def pause_health_monitor(self):
        """Pause the stream health monitor to prevent false triggers.

        Call this before swapping stream readers so that the monitor
        does not interpret the intentional reader stop as a disconnect.
        Always pair with resume_health_monitor() in a finally block.
        """
        self._health_paused.set()
        log.debug("[EVENT:disconnect_detector] Health monitor paused")

    def resume_health_monitor(self):
        """Resume the stream health monitor after a reader swap.

        Should be called in a finally block after pause_health_monitor().
        """
        self._health_paused.clear()
        log.debug("[EVENT:disconnect_detector] Health monitor resumed")

    # -- Component Updates ---------------------------------------------------

    def update_stream_reader(self, reader):
        """Update the stream reader reference (e.g., after reconnection).

        Args:
            reader: New StreamReader instance to monitor.
        """
        self._stream_reader = reader
        log.debug("[EVENT:disconnect_detector] Stream reader updated")

    def update_pipeline(self, pipeline):
        """Update the pipeline reference.

        Args:
            pipeline: New FramePipeline instance.
        """
        self._pipeline = pipeline
        log.debug("[EVENT:disconnect_detector] Pipeline updated")

    def mark_connected(self):
        """Mark the connection as recovered (called after successful reconnect).

        Resets the disconnected state so the detector is ready for the
        next disconnect event.
        """
        with self._lock:
            self._disconnected = False
        log.debug("[EVENT:disconnect_detector] Marked as connected")

    def get_status(self) -> dict:
        """Return detector status for diagnostics/GUI.

        Returns dict with:
          - running: bool
          - disconnected: bool
          - disconnect_duration: float (seconds)
          - usb_listener_active: bool
          - last_disconnect_time: float or None
          - last_attach_time: float or None
        """
        return {
            "running": self.is_running,
            "disconnected": self.is_disconnected,
            "disconnect_duration": round(self.disconnect_duration, 1),
            "usb_listener_active": (
                self._usb_listener is not None
                and hasattr(self._usb_listener, 'is_running')
                and self._usb_listener.is_running
            ),
            "last_disconnect_time": self._last_disconnect_time,
            "last_attach_time": self._last_attach_time,
        }
