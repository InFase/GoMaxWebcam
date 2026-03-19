"""
gopro_connection.py -- GoPro USB control connection and HTTP API management

Once a GoPro is detected on USB (by discovery.py), this module:
  1. Auto-connect: Open the HTTP control endpoint at {camera_ip}:8080
  2. API verification: Confirm the GoPro HTTP API responds correctly
  3. USB mode switch: Enable wired USB control via the Open GoPro API
  4. IDLE workaround: Reset the webcam state machine (start/stop cycle)
  5. Webcam control: Start/stop webcam mode with resolution + FOV settings
  6. Keep-alive: Background thread pings the camera to prevent sleep
  7. Disconnect detection: Detects when camera goes away, signals reconnect

Discovery of the GoPro device is delegated to discovery.py, which handles
USB enumeration (vendor ID 0x2672, GoPro Inc.) and network interface scanning.
This module takes the discovered IP and manages the HTTP API interaction.

Thread safety:
  - All state changes are protected by a threading lock.
  - The keep-alive runs on a daemon thread with Event-based shutdown.
  - Callbacks fire from whichever thread triggers the change; the GUI
    layer should use thread-safe dispatch.
"""

import threading
import time
from enum import IntEnum
from typing import Optional, Callable

import requests

from discovery import (
    GoProDevice,
    DiscoveryTimeout,
    full_discovery,
    timed_full_discovery,
    discover_gopro_ip,
)
from logger import get_logger

log = get_logger("connection")


# -- Constants ---------------------------------------------------------------

GOPRO_API_PORT = 8080
GOPRO_WEBCAM_UDP_PORT = 8554

# HTTP request timeouts (seconds)
_TIMEOUT_QUICK = 3.0    # Status checks and keep-alive
_TIMEOUT_NORMAL = 5.0   # Most API calls
_TIMEOUT_START = 10.0   # Webcam start (camera needs time to switch modes)

# GoPro resolution codes -> (width, height, label)
# Maps the 'res' parameter for /gopro/webcam/start to human-readable info.
# NOTE: Resolution codes vary across GoPro models and firmware versions.
# When the camera rejects a resolution code, start_webcam() auto-remaps
# using the supported_settings returned in the error response, so the app
# adapts dynamically to whatever the connected camera supports.
RESOLUTION_MAP = {
    7:  (1280, 720, "720p"),
    4:  (1920, 1080, "1080p"),
    12: (1920, 1080, "1080p"),
}

# Valid resolution codes accepted by the GoPro webcam API
VALID_RESOLUTIONS = set(RESOLUTION_MAP.keys())

# Maps a human-readable resolution name to all known codes that represent it
# Used to remap when the camera rejects a code (some models don't accept all codes)
_RESOLUTION_BY_NAME = {
    "720": 7,
    "720p": 7,
    "1080": 12,  # Default code; start_webcam remaps dynamically per camera
    "1080p": 12,
}
# Note: 4K webcam mode is not supported on most GoPro models (Hero 13 caps at 1080p)

# Maps a logical resolution (width) to preferred code per camera generation
_RESOLUTION_LABELS = {
    720: "720",
    1080: "1080",
    2160: "2160",
}


# -- Enums -------------------------------------------------------------------

class WebcamStatus(IntEnum):
    """GoPro webcam status codes from /gopro/webcam/status.

    Maps directly to the 'status' field in the JSON response.
    See: https://gopro.github.io/OpenGoPro/http#webcam-status
    """
    OFF = 0
    IDLE = 1
    READY = 2           # HIGH_POWER_PREVIEW — camera is actively streaming
    STREAMING = 3       # LOW_POWER_PREVIEW — preview mode
    UNAVAILABLE = 4     # Hero 13 may return this during normal streaming
    UNKNOWN = -1


class WebcamError(IntEnum):
    """GoPro webcam error codes from /gopro/webcam/start response.
    Per official GoPro Kotlin SDK (Webcam.kt).
    """
    SUCCESS = 0
    SET_PRESET = 1
    SET_WINDOW_SIZE = 2
    EXEC_STREAM = 3
    SHUTTER = 4
    COM_TIMEOUT = 5
    INVALID_PARAM = 6
    UNAVAILABLE = 7
    EXIT = 8


class CameraMode(IntEnum):
    """High-level GoPro operating mode detected via USB queries.

    A GoPro can be in one of several modes. This enum represents the mode
    as determined by combining /gopro/webcam/status (webcam state machine)
    with /gopro/camera/state (overall camera state, preset group).

    Open GoPro status IDs used:
      - Status 89 (0x59): Active preset group (0=video, 1=photo, 2=timelapse)
      - /gopro/webcam/status: Webcam-specific state (OFF/IDLE/READY/STREAMING)

    The detection logic:
      1. If webcam status is READY or STREAMING -> WEBCAM
      2. If webcam status is IDLE -> WEBCAM_IDLE (stale state, needs reset)
      3. If webcam status is OFF -> check preset group for VIDEO/PHOTO/TIMELAPSE
      4. If webcam status is UNAVAILABLE -> camera is in a mode that blocks webcam
      5. If queries fail -> UNKNOWN
    """
    UNKNOWN = -1
    VIDEO = 0           # Standard video recording mode (preset group 0)
    PHOTO = 1           # Photo capture mode (preset group 1)
    TIMELAPSE = 2       # Timelapse mode (preset group 2)
    WEBCAM = 10         # Webcam mode active (READY or STREAMING)
    WEBCAM_IDLE = 11    # Webcam IDLE state (stale after USB connect, needs reset)
    WEBCAM_STARTING = 12  # Webcam transitioning (between OFF and READY)
    UNAVAILABLE = 20    # Camera in a mode that blocks webcam (e.g., updating)

    @property
    def is_webcam_active(self) -> bool:
        """True if the camera is in any webcam-related state."""
        return self in (CameraMode.WEBCAM, CameraMode.WEBCAM_IDLE,
                        CameraMode.WEBCAM_STARTING)

    @property
    def is_ready_to_stream(self) -> bool:
        """True if the camera is actively streaming or ready to stream."""
        return self == CameraMode.WEBCAM

    @property
    def needs_webcam_start(self) -> bool:
        """True if webcam/start needs to be called to begin streaming."""
        return self in (CameraMode.VIDEO, CameraMode.PHOTO,
                        CameraMode.TIMELAPSE, CameraMode.WEBCAM_IDLE,
                        CameraMode.UNKNOWN)

    @property
    def label(self) -> str:
        """Human-readable label for GUI display."""
        labels = {
            CameraMode.UNKNOWN: "Unknown",
            CameraMode.VIDEO: "Video Mode",
            CameraMode.PHOTO: "Photo Mode",
            CameraMode.TIMELAPSE: "Timelapse Mode",
            CameraMode.WEBCAM: "Webcam Mode (Active)",
            CameraMode.WEBCAM_IDLE: "Webcam Mode (Idle)",
            CameraMode.WEBCAM_STARTING: "Webcam Mode (Starting)",
            CameraMode.UNAVAILABLE: "Unavailable",
        }
        return labels.get(self, "Unknown")


# Status ID for the active preset group in /gopro/camera/state response
# See: https://gopro.github.io/OpenGoPro/http#status-89
_STATUS_ID_PRESET_GROUP = "89"
_STATUS_ID_PRESET_GROUP_INT = 89


class ConnectionState(IntEnum):
    """High-level connection state for UI display and app logic.

    State machine:
        DISCONNECTED -> CONNECTING -> CONNECTED -> STREAMING
                     ^                              |
                     +-- RECONNECTING <-------------+
    """
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2       # API verified, webcam not started yet
    STREAMING = 3       # Webcam mode active, UDP stream flowing
    RECONNECTING = 4    # Lost contact, trying to reconnect


# -- Main class --------------------------------------------------------------

class GoProConnection:
    """Manages connection, control, and health monitoring of a GoPro over USB.

    Discovery is delegated to discovery.py (USB enumeration + network scan).
    This class takes the discovered device/IP and manages the webcam lifecycle.

    Two usage patterns:

    1. Step-by-step (used by app_controller.py):
        gopro = GoProConnection(config)
        if gopro.discover():
            gopro.start_webcam()

    2. Auto-connect (all-in-one):
        gopro = GoProConnection(config)
        if gopro.auto_connect():
            # Connected, IDLE workaround done, keep-alive running
            gopro.start_webcam()
    """

    def __init__(self, config):
        self.config = config
        self.ip: Optional[str] = None
        self.base_url: Optional[str] = None
        self._connected = False
        self.device_info: Optional[GoProDevice] = None

        # Thread safety for state changes
        self._lock = threading.Lock()
        self._state = ConnectionState.DISCONNECTED
        self._webcam_status = WebcamStatus.UNKNOWN

        # Keep-alive thread management
        self._keepalive_thread: Optional[threading.Thread] = None
        self._keepalive_stop = threading.Event()

        # Track the active resolution/FOV so we can detect change requests
        self._current_resolution: Optional[int] = None
        self._current_fov: Optional[int] = None

        # Phase 1.3: Track whether the last stop was user-initiated
        self._last_stop_was_intentional: bool = False

        # Callback for status updates: fn(message: str, level: str)
        # level is one of: "info", "warning", "error", "success"
        self.on_status_change: Optional[Callable[[str, str], None]] = None

        # Callback for connection state changes: fn(state: ConnectionState)
        self.on_connection_state: Optional[Callable[[ConnectionState], None]] = None

    # -- Properties ----------------------------------------------------------

    @property
    def state(self) -> ConnectionState:
        """Current connection state (thread-safe read)."""
        with self._lock:
            return self._state

    @property
    def is_connected(self) -> bool:
        """True if the HTTP API connection is verified and active."""
        with self._lock:
            return self._connected and self.ip is not None

    @property
    def is_streaming(self) -> bool:
        """True if webcam mode is active and the stream should be flowing."""
        with self._lock:
            return self._state == ConnectionState.STREAMING

    # -- State & notification helpers ----------------------------------------

    def _notify(self, message: str, level: str = "info"):
        """Send a status update to the callback (if set) and log it."""
        log_fn = getattr(log, level if level != "success" else "info")
        log_fn(message)
        if self.on_status_change:
            try:
                self.on_status_change(message, level)
            except Exception:
                pass  # Never let callback errors break the connection flow

    def _set_state(self, new_state: ConnectionState):
        """Update connection state and notify listeners (thread-safe)."""
        with self._lock:
            old = self._state
            if old == new_state:
                return
            self._state = new_state

        log.info("[EVENT:state_change] Connection state: %s -> %s", old.name, new_state.name)
        if self.on_connection_state:
            try:
                self.on_connection_state(new_state)
            except Exception:
                log.exception("Error in connection-state callback")

    # -- Auto-connect (all-in-one entry point) -------------------------------

    def auto_connect(self) -> bool:
        """Discover a GoPro on USB and automatically open the control connection.

        This all-in-one method:
          1. Runs timed_full_discovery() to find the GoPro on USB (vendor ID + IP)
             with an overall timeout from config.discovery_overall_timeout
          2. If USB device found but no IP within timeout, raises DiscoveryTimeout
          3. Opens the HTTP control connection and verifies the API
          4. Performs the IDLE-state workaround (start/stop cycle)
          5. Starts the keep-alive thread to prevent camera sleep

        After this returns True, call start_webcam() to begin streaming.

        Returns:
            True if the control connection was established successfully.
        """
        log.info("[EVENT:connection] Starting auto-connect sequence")
        self._set_state(ConnectionState.CONNECTING)
        self._notify("Searching for GoPro on USB...", "info")

        try:
            device = timed_full_discovery(
                overall_timeout=self.config.discovery_overall_timeout,
                probe_timeout=self.config.discovery_timeout,
                poll_interval=self.config.discovery_retry_interval,
            )
        except DiscoveryTimeout as e:
            self._notify(
                f"Discovery timed out after {e.elapsed:.0f}s: "
                f"GoPro USB device found ({e.device.usb_id_str}) but network "
                f"interface never came up. Try unplugging and replugging the USB cable.",
                "error"
            )
            log.warning(
                "[EVENT:discovery_timeout] %s", e,
            )
            self._set_state(ConnectionState.DISCONNECTED)
            return False

        if device is None:
            log.warning("[EVENT:discovery_failed] auto_connect: no GoPro found on USB bus")
            self._notify("No GoPro found on USB bus.", "error")
            self._set_state(ConnectionState.DISCONNECTED)
            return False

        log.info("[EVENT:discovery_found] auto_connect: device found, opening connection")
        return self.open_connection(device)

    def auto_connect_with_retries(self) -> bool:
        """Run auto_connect() with retries from config.

        Uses config.discovery_max_retries and config.discovery_retry_interval.
        """
        max_retries = self.config.discovery_max_retries
        interval = self.config.discovery_retry_interval

        for attempt in range(1, max_retries + 1):
            log.info("[EVENT:reconnect_attempt] Connection attempt %d/%d", attempt, max_retries)
            self._notify(f"Connection attempt {attempt}/{max_retries}...", "info")
            if self.auto_connect():
                log.info("[EVENT:connection] auto_connect_with_retries succeeded on attempt %d", attempt)
                return True
            if attempt < max_retries:
                log.info("[EVENT:reconnect_attempt] Attempt %d failed, retrying in %.0fs", attempt, interval)
                self._notify(f"Retrying in {interval:.0f}s...", "warning")
                time.sleep(interval)

        log.error("[EVENT:connection] Failed to connect after %d attempts", max_retries)
        self._notify(
            f"Could not connect after {max_retries} attempts. "
            "Is the GoPro connected via USB-C and powered on?",
            "error"
        )
        self._set_state(ConnectionState.DISCONNECTED)
        return False

    # -- open_connection (once device + IP are known) ------------------------

    def open_connection(self, device: GoProDevice) -> bool:
        """Open the HTTP control connection to a discovered GoPro.

        Called after discovery finds the device and resolves its IP.
        Steps: store info -> verify API (with retries) -> IDLE workaround -> keep-alive.

        Args:
            device: A GoProDevice with camera_ip set.

        Returns:
            True if the control connection was established.
        """
        if not device.camera_ip:
            self._notify("Cannot connect: device has no IP address", "error")
            self._set_state(ConnectionState.DISCONNECTED)
            return False

        self._set_state(ConnectionState.CONNECTING)
        self._notify(f"Opening control connection to GoPro at {device.camera_ip}...", "info")

        with self._lock:
            self.device_info = device
            self.ip = device.camera_ip
            self.base_url = f"http://{device.camera_ip}:{GOPRO_API_PORT}"

        if not self._verify_api_with_retries(retries=3):
            log.error("[EVENT:connection] HTTP API verification failed for %s after 3 retries", device.camera_ip)
            self._notify(
                f"GoPro at {device.camera_ip} not responding to HTTP API.",
                "error"
            )
            with self._lock:
                self.ip = None
                self.base_url = None
                self.device_info = None
            self._set_state(ConnectionState.DISCONNECTED)
            return False

        self._notify(f"HTTP API verified at {device.camera_ip}", "success")

        # Enable wired USB control before any webcam commands.
        # This tells the camera to accept commands over the USB connection.
        # Some firmware versions auto-enable it, but being explicit is more
        # reliable and avoids intermittent command failures.
        if not self.enable_wired_usb_control():
            self._notify(
                "Could not enable USB control — webcam commands may fail. "
                "Continuing anyway...",
                "warning",
            )

        # Verify the USB control connection is in a good state.
        # This confirms the webcam state machine is reachable after
        # enabling wired USB control — catches firmware timing issues.
        if not self.verify_usb_control_connection():
            self._notify(
                "USB control verify failed — camera may need a moment. "
                "Continuing anyway...",
                "warning",
            )

        # NOTE: We do NOT call reset_webcam_state() here. The IDLE workaround
        # is handled by start_webcam() when it detects IDLE state. Calling it
        # here AND in start_webcam() causes double start/stop cycling on the
        # camera, which can put it into UNAVAILABLE state.

        self._start_keepalive()

        with self._lock:
            self._connected = True
        self._set_state(ConnectionState.CONNECTED)
        log.info(
            "[EVENT:connection] GoPro control connection established — IP=%s, USB=%s, discovery=%s",
            device.camera_ip,
            device.usb_id_str,
            device.discovery_method.value if device.discovery_method else "unknown",
        )
        self._notify(f"Connected to GoPro at {device.camera_ip}", "success")
        return True

    # -- Backward-compatible discover() (used by app_controller.py) ----------

    def discover(self) -> bool:
        """Find the GoPro on USB and verify its HTTP API.

        Uses timed_full_discovery() from discovery.py which:
          1. Enumerates USB devices for GoPro vendor ID (0x2672)
          2. Scans network interfaces (ipconfig, psutil) for NCM adapter
          3. Polls for IP resolution with configurable overall timeout
          4. Probes candidate IPs on port 8080

        Returns True if camera found and HTTP API responds.
        """
        self._notify("Searching for GoPro on USB...", "info")

        try:
            device = timed_full_discovery(
                overall_timeout=self.config.discovery_overall_timeout,
                probe_timeout=self.config.discovery_timeout,
                poll_interval=self.config.discovery_retry_interval,
            )
        except DiscoveryTimeout as e:
            self._notify(
                f"Discovery timed out after {e.elapsed:.0f}s: "
                f"USB device {e.device.usb_id_str} found but no network. "
                f"Will retry...",
                "warning"
            )
            self.device_info = e.device
            return False

        if device is None:
            self._notify("GoPro not found on USB. Is it connected and powered on?", "error")
            return False

        self.device_info = device

        if device.camera_ip:
            self._notify(f"Found GoPro: {device.description} (USB {device.usb_id_str})", "info")

            # Some GoPro models (Hero 13+) require wired USB control to be
            # enabled before the webcam API responds.  Pre-set base_url so
            # _api_get works, then try to enable USB control before the full
            # verification check.
            with self._lock:
                self.ip = device.camera_ip
                self.base_url = f"http://{device.camera_ip}:{GOPRO_API_PORT}"
            self._try_enable_usb_control_early()

            if self._verify_connection(device.camera_ip):
                return True
            else:
                self._notify("GoPro found on USB but HTTP API not responding", "warning")
                # Clean up partial state so open_connection starts fresh
                with self._lock:
                    self.ip = None
                    self.base_url = None
                return False
        else:
            # Shouldn't reach here since timed_full_discovery raises on timeout,
            # but handle gracefully just in case
            self._notify(
                f"GoPro USB device detected ({device.usb_id_str}) but network not ready.",
                "warning"
            )
            return False

    # -- Connection helpers --------------------------------------------------

    def _try_enable_usb_control_early(self):
        """Best-effort attempt to enable wired USB control during discovery.

        Some GoPro models (Hero 13 and newer firmware) won't respond to
        /gopro/webcam/status until wired USB control is explicitly enabled.
        This sends the enable command early so the subsequent verification
        check succeeds.  Failures are silently ignored — the full
        open_connection() flow will retry later.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/gopro/camera/control/wired_usb?p=1",
                timeout=_TIMEOUT_NORMAL,
            )
            if resp.status_code == 200:
                log.info("[EVENT:discovery] Early USB control enable succeeded")
            else:
                log.debug("[EVENT:discovery] Early USB control enable returned %d", resp.status_code)
        except requests.RequestException as e:
            log.debug("[EVENT:discovery] Early USB control enable failed: %s", e)

    def _verify_connection(self, ip: str) -> bool:
        """Verify the camera at this IP responds to the HTTP API."""
        try:
            resp = requests.get(
                f"http://{ip}:{GOPRO_API_PORT}/gopro/webcam/status",
                timeout=self.config.discovery_timeout
            )
            if resp.status_code == 200:
                with self._lock:
                    self.ip = ip
                    self.base_url = f"http://{ip}:{GOPRO_API_PORT}"
                    self._connected = True
                self._notify(f"Connected to GoPro at {ip}", "success")
                return True
        except requests.RequestException as e:
            log.debug("Verification failed for %s: %s", ip, e)
        return False

    def _verify_api_with_retries(self, retries: int = 3) -> bool:
        """Verify the GoPro HTTP API with multiple attempts.

        Sends GET /gopro/webcam/status and checks for valid JSON
        with a 'status' field. Retries with 1-second delays.
        """
        for attempt in range(1, retries + 1):
            log.debug("[EVENT:connection] API verification attempt %d/%d", attempt, retries)
            data = self._api_get("/gopro/webcam/status", timeout=self.config.discovery_timeout)
            if data is not None and "status" in data:
                raw_status = data.get("status", -1)
                try:
                    wc_status = WebcamStatus(raw_status)
                except ValueError:
                    wc_status = WebcamStatus.UNKNOWN
                with self._lock:
                    self._webcam_status = wc_status
                log.info(
                    "[EVENT:connection] API verified on attempt %d — webcam status: %s (code %d)",
                    attempt, wc_status.name, raw_status,
                )
                return True
            if attempt < retries:
                log.debug("[EVENT:connection] API not ready, retrying in 1s (attempt %d/%d)", attempt, retries)
                time.sleep(1.0)
        log.warning("[EVENT:connection] API verification failed after %d attempts", retries)
        return False

    def enable_wired_usb_control(self) -> bool:
        """Enable wired USB control mode on the GoPro.

        Sends GET /gopro/camera/control/wired_usb?p=1 which tells the camera
        to accept commands over the USB-C connection. This is part of the
        Open GoPro API's USB webcam flow and should be called BEFORE
        webcam/start to ensure the camera's command interface is ready.

        The GoPro has two USB modes:
          - p=0: USB control disabled (camera ignores API commands)
          - p=1: USB control enabled (camera accepts API commands)

        After a fresh USB connection, the camera may not accept webcam
        commands until wired USB control is explicitly enabled. Some
        firmware versions auto-enable it, but sending this command
        explicitly ensures reliable behavior across all firmware versions.

        Returns:
            True if the command was accepted (error code 0), or if the
            camera is already in USB control mode. False if the camera
            didn't respond or returned an error.
        """
        self._notify("Enabling wired USB control...", "info")
        result = self._api_get(
            "/gopro/camera/control/wired_usb?p=1",
            timeout=_TIMEOUT_NORMAL,
        )

        if result is None:
            self._notify("Failed to enable wired USB control — camera not responding", "warning")
            log.warning("[EVENT:usb_mode_switch] Wired USB control command failed (no response)")
            return False

        error_code = result.get("error", -1)
        if error_code == 0:
            log.info("[EVENT:usb_mode_switch] Wired USB control enabled successfully")
            self._notify("Wired USB control enabled", "info")
            return True
        else:
            # Non-zero error code — log but don't fail hard. Some firmware
            # versions return errors when USB control is already active.
            log.warning(
                "[EVENT:usb_mode_switch] Wired USB control returned error %d "
                "(may already be enabled — continuing)",
                error_code,
            )
            self._notify(
                f"USB control command returned code {error_code} (continuing anyway)",
                "warning",
            )
            return True  # Treat non-fatal — proceed with connection

    def verify_usb_control_connection(self) -> bool:
        """Verify the USB control connection is in a known-good state.

        After enabling wired USB control, this method confirms the camera
        responds to status queries and is in a state where webcam commands
        will be accepted. This catches cases where:
          - enable_wired_usb_control() returned success but the control
            interface isn't actually ready (firmware timing issue)
          - The NCM network interface is up but the camera's internal
            command processor hasn't initialized yet

        The verification is lightweight: a single /gopro/webcam/status
        query that confirms the camera's webcam state machine is reachable.

        Returns:
            True if the camera's webcam state machine responds.
        """
        data = self._api_get("/gopro/webcam/status", timeout=_TIMEOUT_QUICK)
        if data is not None and "status" in data:
            raw_status = data.get("status", -1)
            try:
                wc_status = WebcamStatus(raw_status)
            except ValueError:
                wc_status = WebcamStatus.UNKNOWN
            log.info(
                "[EVENT:usb_control_verify] USB control connection verified — "
                "webcam state machine reachable (status: %s)",
                wc_status.name,
            )
            return True
        else:
            log.warning(
                "[EVENT:usb_control_verify] USB control verification failed — "
                "webcam state machine not responding after USB control enable"
            )
            return False

    def disable_wired_usb_control(self) -> bool:
        """Disable wired USB control mode on the GoPro.

        Sends GET /gopro/camera/control/wired_usb?p=0 to release control.
        Called during clean disconnect to leave the camera in a known state.

        Returns:
            True if the command succeeded.
        """
        result = self._api_get(
            "/gopro/camera/control/wired_usb?p=0",
            timeout=_TIMEOUT_QUICK,
        )

        if result is None:
            log.debug("Failed to disable wired USB control (camera may already be disconnected)")
            return False

        error_code = result.get("error", -1)
        if error_code == 0:
            log.info("[EVENT:usb_mode_switch] Wired USB control disabled")
        else:
            log.debug("Disable USB control returned error %d", error_code)
        return error_code == 0

    def wait_for_network_interface(self, timeout: float = 15.0) -> Optional[str]:
        """Wait for the GoPro NCM network interface to come up.

        After USB detection, the virtual network adapter may take
        several seconds to initialize. Polls discover_gopro_ip().

        Returns:
            Camera IP string, or None if timed out.
        """
        deadline = time.monotonic() + timeout
        poll_interval = 1.5
        while time.monotonic() < deadline:
            ip = discover_gopro_ip()
            if ip:
                log.info("Network interface came up -- camera IP: %s", ip)
                return ip
            remaining = deadline - time.monotonic()
            log.debug("Network interface not ready, %.0fs remaining...", max(0, remaining))
            time.sleep(poll_interval)
        log.warning("Network interface did not come up within %.0fs", timeout)
        return None

    # -- HTTP API Commands ---------------------------------------------------

    def _api_get(self, endpoint: str, timeout: float = _TIMEOUT_NORMAL) -> Optional[dict]:
        """Make a GET request to the GoPro HTTP API.

        All GoPro commands are simple GET requests -- the camera API
        is intentionally straightforward (no POST, no request body).

        Args:
            endpoint: API path like '/gopro/webcam/status'
            timeout: Request timeout in seconds

        Returns:
            Parsed JSON response, or None on failure.
        """
        if not self.base_url:
            return None

        url = f"{self.base_url}{endpoint}"
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            with self._lock:
                self._connected = True
            return resp.json()
        except requests.ConnectionError as e:
            log.warning("[EVENT:connection] Connection lost to GoPro: %s — %s", endpoint, e)
            with self._lock:
                self._connected = False
            return None
        except requests.Timeout:
            # Don't mark as disconnected on a single timeout
            log.warning("[EVENT:connection] API request timed out: %s (%.1fs)", endpoint, timeout)
            return None
        except requests.HTTPError as e:
            log.warning("[EVENT:connection] API HTTP error: %s — %s", endpoint, e)
            # Try to parse the error response body — GoPro returns useful
            # data in error responses (e.g. supported_settings on 400)
            try:
                return resp.json()
            except (ValueError, AttributeError):
                return None
        except ValueError:
            log.debug("API returned non-JSON response: %s", endpoint)
            return {}
        except Exception:
            log.exception("Unexpected error calling %s", endpoint)
            return None

    def webcam_status(self) -> WebcamStatus:
        """Get the current webcam status from the camera."""
        data = self._api_get("/gopro/webcam/status", timeout=_TIMEOUT_QUICK)
        if data is None:
            return WebcamStatus.UNKNOWN

        status_code = data.get("status", -1)
        try:
            status = WebcamStatus(status_code)
        except ValueError:
            log.warning("Unknown webcam status code: %s", status_code)
            status = WebcamStatus.UNKNOWN

        with self._lock:
            self._webcam_status = status
        return status

    def detect_camera_mode(self) -> CameraMode:
        """Query the GoPro over USB to determine its current operating mode.

        Combines two API queries to build a complete picture:

          1. GET /gopro/webcam/status  — Is webcam mode active?
             Returns status: OFF(0), IDLE(1), READY(2), STREAMING(3), UNAVAILABLE(4)

          2. GET /gopro/camera/state   — What preset group is active?
             Status ID 89 = active preset group: 0=video, 1=photo, 2=timelapse

        Decision logic:
          - STREAMING or READY  -> CameraMode.WEBCAM (already streaming/ready)
          - IDLE                -> CameraMode.WEBCAM_IDLE (stale, needs start/stop reset)
          - UNAVAILABLE         -> CameraMode.UNAVAILABLE (firmware update, etc.)
          - OFF                 -> Check preset group (VIDEO / PHOTO / TIMELAPSE)
          - API unreachable     -> CameraMode.UNKNOWN

        This method is used by the app controller to decide whether to:
          - Skip webcam start (already active)
          - Run IDLE workaround first (stale IDLE state)
          - Just call webcam/start (camera in video/photo/timelapse mode)

        Returns:
            CameraMode indicating the camera's current operating mode.
        """
        # Step 1: Check webcam-specific status
        wc_status = self.webcam_status()
        log.info(
            "[EVENT:mode_detect] Webcam status: %s (%d)",
            wc_status.name, wc_status.value,
        )

        if wc_status == WebcamStatus.UNKNOWN:
            # Camera not reachable — can't determine mode
            log.warning("[EVENT:mode_detect] Camera unreachable, mode unknown")
            return CameraMode.UNKNOWN

        if wc_status == WebcamStatus.STREAMING:
            log.info("[EVENT:mode_detect] Camera is in webcam STREAMING mode")
            return CameraMode.WEBCAM

        if wc_status == WebcamStatus.READY:
            log.info("[EVENT:mode_detect] Camera is in webcam READY mode")
            return CameraMode.WEBCAM

        if wc_status == WebcamStatus.IDLE:
            # IDLE is a known stale state after fresh USB connect
            log.info(
                "[EVENT:mode_detect] Camera reports webcam IDLE — "
                "likely stale state after USB connect"
            )
            return CameraMode.WEBCAM_IDLE

        if wc_status == WebcamStatus.UNAVAILABLE:
            log.warning(
                "[EVENT:mode_detect] Webcam mode unavailable — "
                "camera may be updating or in an incompatible mode"
            )
            return CameraMode.UNAVAILABLE

        # Step 2: Webcam is OFF — query camera state for the active preset group
        # This tells us what mode the camera is actually in (video, photo, timelapse)
        camera_state = self._api_get("/gopro/camera/state", timeout=_TIMEOUT_QUICK)

        if camera_state is None:
            log.warning(
                "[EVENT:mode_detect] Webcam OFF but camera state query failed"
            )
            # We know webcam is OFF but can't determine the specific mode
            # Default to VIDEO since that's the most common non-webcam state
            return CameraMode.VIDEO

        # Extract preset group from the status dict
        # The camera state response has a "status" dict with numeric keys
        status_dict = camera_state.get("status", {})
        preset_group = status_dict.get(
            _STATUS_ID_PRESET_GROUP,
            status_dict.get(_STATUS_ID_PRESET_GROUP_INT, None),
        )

        if preset_group is not None:
            try:
                preset_group = int(preset_group)
            except (ValueError, TypeError):
                log.warning(
                    "[EVENT:mode_detect] Invalid preset group value: %s",
                    preset_group,
                )
                preset_group = None

        if preset_group == 0:
            log.info("[EVENT:mode_detect] Camera is in VIDEO mode (preset group 0)")
            return CameraMode.VIDEO
        elif preset_group == 1:
            log.info("[EVENT:mode_detect] Camera is in PHOTO mode (preset group 1)")
            return CameraMode.PHOTO
        elif preset_group == 2:
            log.info(
                "[EVENT:mode_detect] Camera is in TIMELAPSE mode (preset group 2)"
            )
            return CameraMode.TIMELAPSE
        else:
            # Unknown preset group — still report webcam OFF with VIDEO fallback
            log.info(
                "[EVENT:mode_detect] Webcam OFF, unknown preset group: %s "
                "(defaulting to VIDEO mode)",
                preset_group,
            )
            return CameraMode.VIDEO

    def _remap_resolution(self, requested_res: int, supported_settings: list) -> Optional[int]:
        """Find the best matching resolution from the camera's supported_settings.

        When the camera rejects a resolution code (e.g., Hero 13 doesn't support
        res=4 for 1080p), it returns the valid options in supported_settings.
        This method finds the closest match to what was requested.

        Args:
            requested_res: The resolution code that was rejected.
            supported_settings: List of setting dicts from the error response,
                e.g. [{"display_name": "Res", "supported_options": [{"display_name": "1080", "id": 12}]}]

        Returns:
            A valid resolution code, or None if no match found.
        """
        # Find the "Res" setting
        for setting in supported_settings:
            if setting.get("display_name", "").lower() in ("res", "resolution"):
                options = setting.get("supported_options", [])
                if not options:
                    continue

                # Determine what the user wanted by label
                requested_info = RESOLUTION_MAP.get(requested_res)
                wanted_label = requested_info[2] if requested_info else None  # e.g. "1080p"

                # Try to find a matching option by label
                if wanted_label:
                    wanted_num = wanted_label.rstrip("p").upper()  # "1080" or "720"
                    for opt in options:
                        if opt.get("display_name", "").upper() == wanted_num:
                            new_id = opt["id"]
                            log.info(
                                "[EVENT:stream_start] Resolution remap: %s → id=%d (was id=%d)",
                                wanted_label, new_id, requested_res,
                            )
                            return new_id

                # No label match — pick the highest resolution available
                best = max(options, key=lambda o: o.get("id", 0))
                log.info(
                    "[EVENT:stream_start] Resolution remap: no label match, "
                    "using highest available: %s (id=%d)",
                    best.get("display_name"), best.get("id"),
                )
                return best.get("id")

        return None

    def reset_webcam_state(self) -> bool:
        """Perform the IDLE workaround: start then stop to reset the state machine.

        After a fresh USB connection, the GoPro may report IDLE instead of OFF.
        This is a known GoPro quirk documented in their official FAQ.

        The workaround:
          1. Send webcam/start (triggers internal state transition)
          2. Wait ~1 second (configurable via idle_reset_delay)
          3. Send webcam/stop (resets to clean OFF state)
          4. Wait ~1 second

        After this cycle, the camera's state machine is properly reset and
        the next webcam/start will work correctly.

        Returns True even if individual commands error — the important thing
        is that we "tickled" the state machine. Only returns False if the
        camera is completely unreachable.
        """
        self._notify("Resetting webcam state (IDLE workaround)...", "info")
        # Enforce a minimum delay — the GoPro state machine needs time to
        # process transitions. Values below 1.0s cause rapid cycling that
        # puts the camera into UNAVAILABLE state.
        delay = max(self.config.idle_reset_delay, 1.0)

        # Check if reset is needed
        current = self.webcam_status()
        log.info("[EVENT:state_change] Webcam status before IDLE reset: %s", current.name)

        if current == WebcamStatus.OFF:
            log.info("[EVENT:state_change] Webcam already in OFF state — skipping IDLE workaround")
            self._notify("Webcam state is clean (OFF)", "info")
            return True

        if current == WebcamStatus.UNAVAILABLE:
            log.warning("[EVENT:state_change] Camera is UNAVAILABLE — skipping IDLE workaround")
            self._notify("Camera unavailable — cannot reset webcam state", "warning")
            return False

        # Step 1: Send start to trigger state transition
        result = self._api_get("/gopro/webcam/start", timeout=_TIMEOUT_START)
        if result is None:
            self._notify("Start command failed during state reset", "warning")
        else:
            log.debug("IDLE workaround: start sent, error=%s", result.get("error", "N/A"))

        time.sleep(delay)

        # Step 2: Send stop to reset back to OFF
        result = self._api_get("/gopro/webcam/stop", timeout=_TIMEOUT_NORMAL)
        if result is None:
            self._notify("Stop command failed during state reset", "warning")
            return False
        else:
            log.debug("IDLE workaround: stop sent, error=%s", result.get("error", "N/A"))

        time.sleep(delay)

        # Verify we're back to a clean state
        final_status = self.webcam_status()
        if final_status == WebcamStatus.OFF:
            log.info("[EVENT:state_change] IDLE workaround complete — webcam state reset to OFF")
            self._notify("Webcam state reset complete (now OFF)", "info")
        else:
            # Try webcam/exit as a stronger reset
            log.warning(
                "[EVENT:state_change] IDLE workaround: status is %s (expected OFF) — trying webcam/exit",
                final_status.name,
            )
            self._api_get("/gopro/webcam/exit", timeout=_TIMEOUT_NORMAL)
            time.sleep(delay)
            final_status = self.webcam_status()
            log.info("[EVENT:state_change] Status after webcam/exit fallback: %s", final_status.name)

        return True

    def start_webcam(self, resolution: Optional[int] = None, fov: Optional[int] = None) -> bool:
        """Run the full GoPro-specific command sequence to initialize webcam mode.

        This handles ALL camera states automatically — no user interaction needed:

          Step 1: Query current webcam status
          Step 2: If UNAVAILABLE (firmware update, incompatible mode), bail early
          Step 3: If IDLE (stale state after fresh USB connect), apply the
                  start/stop workaround to reset the internal state machine
          Step 4: If already STREAMING or READY, stop first so we can apply
                  the desired resolution/FOV settings cleanly
          Step 5: Send webcam/start with resolution and FOV parameters
          Step 6: Poll status until READY or STREAMING is confirmed
                  (READY means the camera is ready to stream; STREAMING means
                  a client is actively reading the UDP feed)

        Args:
            resolution: 7=720p, 12=1080p (codes vary by model)
            fov: 0=wide, 2=narrow, 3=superview, 4=linear (default from config)

        Returns:
            True if webcam mode started successfully (READY or STREAMING).
        """
        res = resolution if resolution is not None else self.config.resolution
        field_of_view = fov if fov is not None else self.config.fov

        if res not in VALID_RESOLUTIONS:
            log.warning(
                "[EVENT:stream_error] Invalid resolution code %d — valid: %s",
                res, sorted(VALID_RESOLUTIONS),
            )
            self._notify(f"Invalid resolution code {res}", "error")
            return False

        self._notify(f"Starting webcam mode (res={res}, fov={field_of_view})...", "info")

        # Step 1: Check current status
        status = self.webcam_status()
        log.info("[EVENT:stream_start] Starting webcam — current status: %s (%d), res=%d, fov=%d",
                 status.name, status.value, res, field_of_view)

        # Step 2: Handle UNAVAILABLE (camera in a mode that blocks webcam)
        if status == WebcamStatus.UNAVAILABLE:
            log.warning(
                "[EVENT:stream_error] Camera reports UNAVAILABLE — webcam mode cannot be started "
                "(camera may be updating firmware or in an incompatible mode)"
            )
            self._notify(
                "Camera is unavailable for webcam mode. "
                "It may be updating firmware or in an incompatible mode.",
                "error"
            )
            return False

        # Step 3: Pre-emptive state reset — only for IDLE state (stale after
        # fresh USB connect). READY and STREAMING states are handled differently:
        # - READY: camera was just stopped intentionally (pause/resolution change)
        #   — skip reset, just send start with new params
        # - STREAMING: already streaming, send stop first then start
        # - IDLE: stale state from USB connect, needs the start/stop workaround
        if status == WebcamStatus.IDLE:
            # Phase 1.6: Skip IDLE workaround if the last stop was intentional
            if self._last_stop_was_intentional:
                log.info("[EVENT:state_change] Camera in IDLE but last stop was intentional — skipping reset")
                self._last_stop_was_intentional = False
            else:
                log.info("[EVENT:state_change] Camera in IDLE state — performing reset workaround")
                self.reset_webcam_state()
                status = self.webcam_status()
                log.info("[EVENT:state_change] Status after IDLE reset: %s", status.name)
        elif status == WebcamStatus.STREAMING:
            log.info("[EVENT:stream_stop] Camera is STREAMING — stopping first")
            self._notify("Stopping current stream...", "info")
            self._api_get("/gopro/webcam/stop", timeout=_TIMEOUT_NORMAL)
            time.sleep(max(self.config.idle_reset_delay, 1.0))
            status = self.webcam_status()
            log.info("[EVENT:state_change] Status after stop: %s", status.name)
        elif status == WebcamStatus.READY:
            log.info("[EVENT:stream_start] Camera is READY — proceeding to start directly")

        # Phase 1.8: Preview warmup — if status is OFF, call preview to prime the sensor
        if status == WebcamStatus.OFF:
            log.info("[EVENT:stream_start] Status is OFF — sending preview warmup")
            self._api_get("/gopro/webcam/preview", timeout=_TIMEOUT_NORMAL)

        # Step 5: Send webcam/start with parameters
        endpoint = f"/gopro/webcam/start?res={res}&fov={field_of_view}&port={self.config.udp_port}"
        result = self._api_get(endpoint, timeout=_TIMEOUT_START)

        if result is None:
            log.error("[EVENT:stream_error] Webcam start command got no response from camera")
            self._notify("Failed to start webcam mode -- camera not responding", "error")
            return False

        error_code = result.get("error", -1)

        # Handle error 6: unsupported resolution. The camera returns
        # supported_settings with valid resolution codes. This happens when
        # the config has a Hero 12 code (e.g. res=4 for 1080p) but the
        # camera is a Hero 13 (which uses res=12 for 1080p).
        if error_code != 0 and "supported_settings" in result:
            remapped_res = self._remap_resolution(res, result["supported_settings"])
            if remapped_res is not None and remapped_res != res:
                log.info(
                    "[EVENT:stream_start] Camera rejected res=%d, remapping to res=%d "
                    "based on supported_settings", res, remapped_res,
                )
                self._notify(f"Resolution remapped: {res} → {remapped_res} (camera preference)", "info")
                res = remapped_res
                # Update config to match the remapped resolution
                if res in RESOLUTION_MAP:
                    w, h, label = RESOLUTION_MAP[res]
                    self.config.resolution = res
                    self.config.stream_width = w
                    self.config.stream_height = h
                    log.info("[EVENT:stream_start] Config updated: resolution=%d, stream=%dx%d", res, w, h)
                endpoint = f"/gopro/webcam/start?res={res}&fov={field_of_view}&port={self.config.udp_port}"
                result = self._api_get(endpoint, timeout=_TIMEOUT_START)
                if result is None:
                    self._notify("Failed to start webcam after resolution remap", "error")
                    return False
                error_code = result.get("error", -1)

        if error_code != 0:
            log.warning("[EVENT:stream_error] Webcam start returned error code %d", error_code)
            self._notify(f"Webcam start returned error code {error_code}", "error")

            # Phase 1.5: Handle specific error codes using WebcamError enum
            if error_code == WebcamError.SHUTTER:
                # Error 4: shutter stuck — release it and retry once
                log.info("[EVENT:stream_start] SHUTTER error — releasing shutter and retrying")
                self._api_get("/gopro/camera/shutter?mode=0", timeout=_TIMEOUT_NORMAL)
                time.sleep(1.0)
                result = self._api_get(endpoint, timeout=_TIMEOUT_START)
                if result is None or result.get("error", -1) != 0:
                    self._notify("Webcam start failed after shutter release retry", "error")
                    return False
            elif error_code == WebcamError.UNAVAILABLE:
                # Error 7: camera truly unavailable — bail early
                self._notify("Camera reports UNAVAILABLE — cannot start webcam", "error")
                return False
            else:
                # Other errors: refresh status, then try IDLE workaround as fallback
                status = self.webcam_status()
                if status != WebcamStatus.IDLE:
                    log.info("[EVENT:stream_start] Trying IDLE workaround as fallback after error %d", error_code)
                    self.reset_webcam_state()
                    result = self._api_get(endpoint, timeout=_TIMEOUT_START)
                    if result is None or result.get("error", -1) != 0:
                        self._notify("Webcam start failed after retry", "error")
                        return False
                else:
                    return False

        # Step 6: Wait for READY or STREAMING
        for attempt in range(50):  # Up to 10 seconds of polling (0.2s intervals)
            time.sleep(0.2)
            status = self.webcam_status()

            # Phase 1.7: Accept READY and STREAMING immediately; also accept
            # UNAVAILABLE (status 4) as streaming for Hero 13 compatibility.
            if status == WebcamStatus.STREAMING:
                self._set_state(ConnectionState.STREAMING)
                self._current_resolution = res
                self._current_fov = field_of_view
                self._notify("GoPro webcam is streaming!", "success")
                self._last_stop_was_intentional = False  # Phase 1.9
                return True
            elif status == WebcamStatus.READY:
                self._set_state(ConnectionState.STREAMING)
                self._current_resolution = res
                self._current_fov = field_of_view
                self._notify("GoPro webcam mode active (READY)", "success")
                self._last_stop_was_intentional = False  # Phase 1.9
                return True
            elif status == WebcamStatus.UNAVAILABLE:
                # Hero 13 may return UNAVAILABLE while actually streaming
                log.info("[EVENT:stream_start] Hero 13 compat: accepting UNAVAILABLE as streaming")
                self._set_state(ConnectionState.STREAMING)
                self._current_resolution = res
                self._current_fov = field_of_view
                self._notify("GoPro webcam mode active (Hero 13 compat)", "success")
                self._last_stop_was_intentional = False  # Phase 1.9
                return True
            elif status == WebcamStatus.OFF:
                log.debug("Still OFF, processing start... (attempt %d)", attempt + 1)
            else:
                log.debug("Status %s during startup (attempt %d)", status.name, attempt + 1)

        final_status = self.webcam_status()
        if final_status in (WebcamStatus.READY, WebcamStatus.STREAMING, WebcamStatus.UNAVAILABLE):
            self._set_state(ConnectionState.STREAMING)
            self._current_resolution = res
            self._current_fov = field_of_view
            self._notify("GoPro webcam mode active", "success")
            self._last_stop_was_intentional = False  # Phase 1.9
            return True

        self._notify(
            f"Webcam did not reach streaming state (status={final_status.name}). "
            f"The camera may need a power cycle.",
            "warning"
        )
        return False

    def stop_webcam(self) -> bool:
        """Stop the GoPro webcam stream (camera stays in webcam context).

        This calls /gopro/webcam/stop which transitions to READY state,
        NOT fully out of webcam mode. Use exit_webcam() to fully leave.
        """
        self._last_stop_was_intentional = True  # Phase 1.4
        self._notify("Stopping webcam mode...", "info")
        result = self._api_get("/gopro/webcam/stop", timeout=_TIMEOUT_NORMAL)
        if result is None:
            self._notify("Failed to stop webcam (camera may be disconnected)", "warning")
            return False

        # /gopro/webcam/stop transitions camera to READY (not OFF).
        # Set the internal status to READY to match the actual camera state.
        # Previously this was set to OFF which caused start_webcam() to skip
        # the status check and trigger unnecessary IDLE reset cycles.
        with self._lock:
            self._webcam_status = WebcamStatus.READY
        if self._state == ConnectionState.STREAMING:
            self._set_state(ConnectionState.CONNECTED)

        self._notify("Webcam mode stopped", "info")
        return True

    def exit_webcam(self) -> bool:
        """Fully exit webcam mode so it can be cleanly re-entered.

        Calls /gopro/webcam/exit which transitions the camera from
        READY/IDLE back to DISABLED (normal camera mode). Without this,
        the camera stays in webcam context after stop and may refuse to
        re-enter webcam mode on the next start_webcam() call.

        The full clean sequence is: stop → exit → (optionally toggle USB control)
        """
        self._last_stop_was_intentional = True  # Phase 1.4
        log.info("[EVENT:stream_stop] Sending webcam/exit to fully leave webcam mode")
        result = self._api_get("/gopro/webcam/exit", timeout=_TIMEOUT_NORMAL)
        if result is None:
            log.warning("[EVENT:stream_stop] webcam/exit got no response (camera may be disconnected)")
            return False

        time.sleep(0.5)

        status = self.webcam_status()
        log.info("[EVENT:stream_stop] Status after webcam/exit: %s", status.name)

        with self._lock:
            self._webcam_status = status
        return True

    # -- Resolution change ---------------------------------------------------

    @property
    def current_resolution(self) -> Optional[int]:
        """The resolution code currently active on the camera, or None if not streaming."""
        with self._lock:
            return self._current_resolution

    @property
    def current_fov(self) -> Optional[int]:
        """The FOV code currently active on the camera, or None if not streaming."""
        with self._lock:
            return self._current_fov

    def needs_resolution_change(self, resolution: int, fov: Optional[int] = None) -> bool:
        """Check if the requested resolution/FOV differs from what's currently active.

        Args:
            resolution: Desired resolution code (7=720p, 12=1080p).
            fov: Desired FOV code, or None to keep current.

        Returns:
            True if a change is needed (different from active settings).
        """
        with self._lock:
            if self._current_resolution is None:
                # Not streaming yet — no change needed, start_webcam will set it
                return False
            if self._current_resolution != resolution:
                return True
            if fov is not None and self._current_fov != fov:
                return True
            return False

    def change_resolution(self, resolution: int, fov: Optional[int] = None) -> bool:
        """Change the GoPro stream resolution mid-session.

        This stops the current webcam session and restarts with the new
        resolution. The GoPro API requires a stop/start cycle to change
        resolution — there is no in-place resolution change command.

        The caller (AppController) is responsible for:
          - Putting the pipeline into freeze-frame mode before calling this
          - Creating a new StreamReader with updated config after this returns
          - Swapping the new reader into the pipeline

        Args:
            resolution: New resolution code (7=720p, 12=1080p).
            fov: New FOV code, or None to keep current FOV.

        Returns:
            True if the webcam was successfully restarted at the new resolution.
        """
        if resolution not in VALID_RESOLUTIONS:
            log.error(
                "[EVENT:resolution_change] Invalid resolution code %d — valid: %s",
                resolution, sorted(VALID_RESOLUTIONS),
            )
            self._notify(f"Invalid resolution code {resolution}", "error")
            return False

        actual_fov = fov if fov is not None else (self._current_fov or self.config.fov)
        old_res = self._current_resolution
        old_label = RESOLUTION_MAP.get(old_res, (0, 0, "unknown"))[2] if old_res else "none"
        new_label = RESOLUTION_MAP[resolution][2]

        log.info(
            "[EVENT:resolution_change] Changing resolution: %s (%s) → %s (%s), fov=%d",
            old_res, old_label, resolution, new_label, actual_fov,
        )
        self._notify(f"Changing resolution to {new_label}...", "info")

        # Step 1: Stop the current webcam session
        if not self.stop_webcam():
            log.warning(
                "[EVENT:resolution_change] Failed to stop webcam for resolution change"
            )
            # Try anyway — the camera might accept a fresh start
            time.sleep(self.config.idle_reset_delay)

        # Step 2: Brief pause for camera state machine to settle
        time.sleep(self.config.idle_reset_delay)

        # Step 3: Start webcam with new resolution
        success = self.start_webcam(resolution=resolution, fov=actual_fov)

        if success:
            log.info(
                "[EVENT:resolution_change] Resolution changed successfully to %s (%s)",
                resolution, new_label,
            )
            self._notify(f"Resolution changed to {new_label}", "success")
        else:
            log.error(
                "[EVENT:resolution_change] Failed to restart webcam at %s (%s)",
                resolution, new_label,
            )
            self._notify(f"Failed to change resolution to {new_label}", "error")

        return success

    # -- Camera info ---------------------------------------------------------

    def get_camera_info(self) -> Optional[dict]:
        """Fetch basic camera info (model, firmware, battery, etc.)."""
        log.debug("[EVENT:camera_info] Fetching camera state from API")
        data = self._api_get("/gopro/camera/state", timeout=_TIMEOUT_NORMAL)
        if data is None:
            log.debug("[EVENT:camera_info] Camera state query returned no data")
            return None

        # Status IDs from Open GoPro spec differ across models:
        #   Hero 12 and earlier: 2 = battery level (0-100), 70 = battery present (bool)
        #   Hero 13+:           2 = internal_battery_present (bool), 70 = battery level (0-100)
        # We detect which mapping to use based on the device model name.
        status = data.get("status", {})
        raw_2 = status.get("2", status.get(2, None))
        raw_70 = status.get("70", status.get(70, None))

        # Use device model name if available, fall back to value heuristic
        model_name = ""
        if self.device_info and self.device_info.description:
            model_name = self.device_info.description.upper()

        is_hero_13_plus = "HERO13" in model_name or "HERO14" in model_name or "HERO15" in model_name

        # If model name not available, fall back to value heuristic
        if not model_name and raw_2 is not None and raw_70 is not None:
            is_hero_13_plus = raw_2 in (0, 1) and raw_70 > 1

        if is_hero_13_plus:
            # Hero 13+ mapping: 2=present (bool), 70=level (0-100)
            info = {
                "battery_level": raw_70,
                "battery_present": bool(raw_2),
            }
        else:
            # Hero 12 and earlier mapping: 2=level (0-100), 70=present (bool)
            info = {
                "battery_level": raw_2,
                "battery_present": raw_70,
            }

        battery = info.get("battery_level")
        if battery is not None:
            if battery <= 10:
                log.warning("[EVENT:battery] Camera battery critically low: %s%%", battery)
            elif battery <= 20:
                log.warning("[EVENT:battery] Camera battery low: %s%%", battery)
            else:
                log.debug("[EVENT:battery] Camera battery level: %s%%", battery)

        return info

    # -- Keep-alive thread ---------------------------------------------------

    def _start_keepalive(self):
        """Start the background keep-alive thread.

        Periodically pings /gopro/webcam/status to:
          - Keep the camera awake (prevents auto-sleep)
          - Detect if the camera has disconnected
        """
        self._stop_keepalive()
        self._keepalive_stop.clear()

        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop,
            name="gopro-keepalive",
            daemon=True,
        )
        self._keepalive_thread.start()
        log.info("[EVENT:keepalive_ok] Keep-alive thread started (interval=%.1fs)", self.config.keepalive_interval)

    def _stop_keepalive(self):
        """Signal the keep-alive thread to stop and wait for it."""
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            self._keepalive_stop.set()
            self._keepalive_thread.join(timeout=5.0)
            log.debug("Keep-alive thread stopped")
        self._keepalive_stop.clear()

    def _keepalive_loop(self):
        """Background loop: ping the camera periodically.

        If 3 consecutive pings fail, declare disconnected and set state
        to RECONNECTING so the app controller can handle recovery.
        """
        consecutive_failures = 0
        max_failures = 3

        log.info("[EVENT:keepalive_ok] Keep-alive loop starting (interval=%.1fs)", self.config.keepalive_interval)
        while not self._keepalive_stop.wait(self.config.keepalive_interval):
            log.info("[EVENT:keepalive_ok] Sending keep-alive ping...")
            alive = self.keep_alive()

            if alive:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                self._notify(
                    f"Keep-alive failed ({consecutive_failures}/{max_failures})",
                    "warning"
                )
                if consecutive_failures >= max_failures:
                    log.error("[EVENT:disconnection] GoPro unreachable after %d consecutive keep-alive failures", max_failures)
                    with self._lock:
                        self._connected = False
                    self._set_state(ConnectionState.RECONNECTING)
                    self._notify("GoPro connection lost", "error")
                    break  # Exit -- app controller handles reconnection

    def keep_alive(self) -> bool:
        """Send a single keep-alive ping to the camera.

        Returns True if the camera responded.
        """
        data = self._api_get("/gopro/webcam/status", timeout=_TIMEOUT_QUICK)
        if data is not None:
            raw = data.get("status", -1)
            try:
                wc_status = WebcamStatus(raw)
            except ValueError:
                wc_status = WebcamStatus.UNKNOWN
            with self._lock:
                self._webcam_status = wc_status
                self._connected = True
            log.debug("[EVENT:keepalive_ok] Keep-alive OK — webcam status: %s", wc_status.name)
            return True
        else:
            log.debug("[EVENT:keepalive_fail] Keep-alive ping failed (no response)")
            with self._lock:
                self._connected = False
            return False

    # -- Disconnect ----------------------------------------------------------

    def reset_for_recovery(self):
        """Reset connection tracking state for seamless recovery.

        Unlike disconnect(), this does NOT attempt to stop webcam mode or
        disable USB control — the camera is already gone.  It simply clears
        the IP / URL / connected flag so the recovery loop can rediscover
        the camera from a clean slate.

        Called by AppController._auto_recover() instead of reaching into
        private fields directly.
        """
        log.info("[EVENT:reset_for_recovery] Clearing connection state for recovery")
        self._last_stop_was_intentional = False  # Phase 1.9
        with self._lock:
            self.ip = None
            self.base_url = None
            self._connected = False

    def disconnect(self):
        """Cleanly disconnect from the GoPro.

        Stops keep-alive, stops webcam mode (if active), resets all state.
        """
        log.info("[EVENT:disconnection] Initiating clean disconnect from GoPro (IP=%s)", self.ip)
        self._notify("Disconnecting from GoPro...", "info")

        # Stop keep-alive first so it does not trigger reconnect
        self._stop_keepalive()

        # Fully exit webcam mode, then release USB control.
        # The sequence stop → exit → disable_usb ensures the camera
        # returns to normal mode and can cleanly re-enter webcam on next connect.
        if self._connected:
            try:
                self.stop_webcam()
            except Exception:
                log.debug("Failed to stop webcam during disconnect (camera may be gone)")
            try:
                self.exit_webcam()
            except Exception:
                log.debug("Failed to exit webcam during disconnect")
            try:
                self.disable_wired_usb_control()
            except Exception:
                log.debug("Failed to disable USB control during disconnect")

        # Reset all state
        with self._lock:
            self.ip = None
            self.base_url = None
            self.device_info = None
            self._connected = False
            self._webcam_status = WebcamStatus.UNKNOWN

        self._set_state(ConnectionState.DISCONNECTED)
        self._notify("Disconnected from GoPro", "info")
