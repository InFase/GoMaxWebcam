"""
usb_device_poller.py — USB device polling/detection loop for GoPro reconnection

Provides a poll-based USB device detection loop that complements the event-driven
USBEventListener. This acts as a reliable fallback when Windows device notification
events are missed or when the USBEventListener fails to start.

Detection strategy:
  - Polls `enumerate_usb_gopro_devices()` at a configurable interval
  - Tracks device presence transitions (absent → present = reconnection)
  - Fires callbacks on state changes (device_appeared, device_disappeared)
  - Debounces rapid transitions to avoid false positives during USB enumeration

The poller is designed to run alongside the event-driven listener:
  - USBEventListener provides instant notifications (~0ms latency)
  - USBDevicePoller provides guaranteed detection (~poll_interval latency)
  - Together they ensure no reconnection event is missed

Thread model:
  - Runs on a single daemon thread ("USBDevicePoller")
  - Callbacks fire from the polling thread
  - Thread-safe start/stop lifecycle

Usage:
    from usb_device_poller import USBDevicePoller

    poller = USBDevicePoller(
        on_device_appeared=lambda devs: print(f"GoPro appeared: {devs}"),
        on_device_disappeared=lambda: print("GoPro disappeared"),
        poll_interval=2.0,
    )
    poller.start()
    # ... later ...
    poller.stop()
"""

import threading
import time
from typing import Callable, List, Optional

from logger import get_logger

log = get_logger("usb_device_poller")


class USBDevicePoller:
    """Polls for GoPro USB device presence at regular intervals.

    Detects transitions between device-absent and device-present states,
    firing callbacks when the GoPro appears or disappears from the USB bus.

    This is a reliable fallback for detecting GoPro reconnection when the
    event-driven USBEventListener misses events or fails to start.

    Attributes:
        poll_interval: Seconds between USB enumeration polls.
        is_running: True if the polling loop is active.
        is_device_present: True if a GoPro was found on the last poll.
        last_seen_time: Monotonic timestamp of last successful detection.
        poll_count: Total number of polls completed.
    """

    def __init__(
        self,
        on_device_appeared: Optional[Callable[[list], None]] = None,
        on_device_disappeared: Optional[Callable[[], None]] = None,
        poll_interval: float = 2.0,
        settling_time: float = 1.0,
    ):
        """Initialize the USB device poller.

        Args:
            on_device_appeared: Called when GoPro USB device(s) appear after being absent.
                                Receives a list of GoProDevice objects.
            on_device_disappeared: Called when GoPro USB device(s) disappear after being present.
            poll_interval: Seconds between USB bus scans.
            settling_time: Seconds to wait after detecting a new device before
                           firing the callback. This allows the NCM network adapter
                           to initialize. Set to 0 for immediate notification.
        """
        self.on_device_appeared = on_device_appeared
        self.on_device_disappeared = on_device_disappeared
        self.poll_interval = poll_interval
        self.settling_time = settling_time

        # State
        self._running = False
        self._device_present = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Tracking
        self._last_seen_time: Optional[float] = None
        self._last_disappeared_time: Optional[float] = None
        self._poll_count = 0
        self._transition_count = 0

        # Debounce: require N consecutive polls confirming a state change
        self._consecutive_present = 0
        self._consecutive_absent = 0
        self._confirm_threshold = 1  # Require 1 confirmation poll (instant)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """True if the polling loop is active."""
        with self._lock:
            return self._running

    @property
    def is_device_present(self) -> bool:
        """True if a GoPro was found on the last poll."""
        with self._lock:
            return self._device_present

    @property
    def last_seen_time(self) -> Optional[float]:
        """Monotonic timestamp of the last successful detection."""
        return self._last_seen_time

    @property
    def poll_count(self) -> int:
        """Total number of polls completed."""
        return self._poll_count

    @property
    def transition_count(self) -> int:
        """Number of device state transitions detected."""
        return self._transition_count

    # ── Lifecycle ───────────────────────────────────────────────────

    def start(self) -> bool:
        """Start the polling loop on a background thread.

        Returns:
            True if the poller started successfully.
        """
        with self._lock:
            if self._running:
                log.debug("[EVENT:usb_poller] Already running")
                return True
            self._running = True

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="USBDevicePoller",
            daemon=True,
        )
        self._thread.start()

        log.info(
            "[EVENT:usb_poller] USB device poller started "
            "(interval=%.1fs, settling=%.1fs)",
            self.poll_interval, self.settling_time,
        )
        return True

    def stop(self):
        """Stop the polling loop and wait for the thread to finish."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                log.warning("[EVENT:usb_poller] Poller thread did not stop cleanly")

        log.info(
            "[EVENT:usb_poller] USB device poller stopped "
            "(polls=%d, transitions=%d)",
            self._poll_count, self._transition_count,
        )

    # ── Polling Loop ────────────────────────────────────────────────

    def _poll_loop(self):
        """Background loop that periodically scans for GoPro USB devices.

        Detects state transitions (device appeared/disappeared) and fires
        callbacks. Uses debounce counting to avoid false positives from
        momentary USB enumeration failures.
        """
        log.debug("[EVENT:usb_poller] Polling loop started")

        while not self._stop_event.is_set():
            try:
                devices = self._enumerate_devices()
                self._poll_count += 1
                found = len(devices) > 0

                if found:
                    self._last_seen_time = time.monotonic()
                    self._consecutive_present += 1
                    self._consecutive_absent = 0
                else:
                    self._consecutive_absent += 1
                    self._consecutive_present = 0

                # Check for state transitions
                was_present = self._device_present

                if not was_present and found and self._consecutive_present >= self._confirm_threshold:
                    # Device appeared (was absent, now present)
                    self._handle_device_appeared(devices)

                elif was_present and not found and self._consecutive_absent >= self._confirm_threshold:
                    # Device disappeared (was present, now absent)
                    self._handle_device_disappeared()

            except Exception as e:
                log.debug(
                    "[EVENT:usb_poller] Poll error (will retry): %s", e
                )

            # Wait for next poll interval (or stop signal)
            if self._stop_event.wait(timeout=self.poll_interval):
                break

        log.debug("[EVENT:usb_poller] Polling loop exited")

    def _enumerate_devices(self) -> list:
        """Scan the USB bus for GoPro devices.

        Uses the discovery module's enumerate function. Isolated in a method
        for easy mocking in tests.

        Returns:
            List of GoProDevice objects found.
        """
        from discovery import enumerate_usb_gopro_devices
        return enumerate_usb_gopro_devices()

    def _handle_device_appeared(self, devices: list):
        """Process a device-appeared transition.

        Optionally waits for settling_time before firing the callback,
        allowing the NCM network adapter to initialize.

        Args:
            devices: List of GoProDevice objects that were found.
        """
        with self._lock:
            self._device_present = True
        self._transition_count += 1
        self._last_disappeared_time = None

        device_descs = [
            f"{d.description} ({d.usb_id_str})"
            if hasattr(d, 'usb_id_str') else str(d)
            for d in devices
        ]
        log.info(
            "[EVENT:usb_poller] GoPro USB device appeared (poll #%d): %s",
            self._poll_count, ", ".join(device_descs),
        )

        # Wait for settling time if configured
        if self.settling_time > 0:
            log.debug(
                "[EVENT:usb_poller] Waiting %.1fs for device settling...",
                self.settling_time,
            )
            if self._stop_event.wait(timeout=self.settling_time):
                return  # Stop requested during settling

        # Fire callback
        if self.on_device_appeared:
            try:
                self.on_device_appeared(devices)
            except Exception:
                log.exception("[EVENT:usb_poller] Error in on_device_appeared callback")

    def _handle_device_disappeared(self):
        """Process a device-disappeared transition."""
        with self._lock:
            self._device_present = False
        self._transition_count += 1
        self._last_disappeared_time = time.monotonic()

        log.info(
            "[EVENT:usb_poller] GoPro USB device disappeared (poll #%d)",
            self._poll_count,
        )

        if self.on_device_disappeared:
            try:
                self.on_device_disappeared()
            except Exception:
                log.exception("[EVENT:usb_poller] Error in on_device_disappeared callback")

    # ── Status ──────────────────────────────────────────────────────

    def force_poll(self) -> bool:
        """Perform a single immediate poll outside the regular interval.

        Useful for checking device state on demand (e.g., after a manual
        retry button click).

        Returns:
            True if a GoPro device was found.
        """
        try:
            devices = self._enumerate_devices()
            found = len(devices) > 0
            self._poll_count += 1

            if found:
                self._last_seen_time = time.monotonic()

            log.debug(
                "[EVENT:usb_poller] Force poll: %s (total polls: %d)",
                "found" if found else "not found", self._poll_count,
            )
            return found
        except Exception as e:
            log.debug("[EVENT:usb_poller] Force poll error: %s", e)
            return False

    def get_status(self) -> dict:
        """Return poller status for diagnostics/GUI.

        Returns dict with:
          - running: bool
          - device_present: bool
          - poll_count: int
          - transition_count: int
          - poll_interval: float
          - last_seen_time: float or None
        """
        return {
            "running": self.is_running,
            "device_present": self.is_device_present,
            "poll_count": self._poll_count,
            "transition_count": self._transition_count,
            "poll_interval": self.poll_interval,
            "last_seen_time": self._last_seen_time,
        }
