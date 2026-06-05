"""
transport/usb.py — USB transport for GoPro cameras.

Connects to a GoPro camera over USB-NCM (wired Ethernet-over-USB).
Thin wrapper over open-gopro SDK's WiredGoPro for camera control.

The USB transport:
  1. Discovers GoPro by serial (or auto-discovers via mDNS)
  2. Opens the WiredGoPro connection (HTTP handshake + USB control enable)
  3. Starts the MPEG-TS preview stream over UDP via webcam_start
  4. Sends periodic keep-alive pings via the SDK's maintain_state
  5. Stops stream, exits webcam, and closes the SDK connection on shutdown

For CI/testing, all open-gopro calls go through self._gopro which can
be replaced with a mock via _inject_gopro().
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from gomaxwebcam.transport.base import (
    Transport,
    TransportState,
    StreamInfo,
    TransportStats,
)

log = logging.getLogger("gomaxwebcam.transport.usb")

# GoPro USB vendor ID (all models)
GOPRO_VENDOR_ID = 0x2672

# Default ports
GOPRO_HTTP_PORT = 8080
GOPRO_UDP_PORT = 8554

# Webcam resolution codes (matches open_gopro.models.streaming.WebcamResolution)
RES_480 = 4
RES_720 = 7
RES_1080 = 12

# Webcam FOV codes (matches open_gopro.models.streaming.WebcamFOV)
FOV_WIDE = 0
FOV_NARROW = 2
FOV_SUPERVIEW = 3
FOV_LINEAR = 4

# Only expose 16:9 modes at 30fps
ALLOWED_RESOLUTIONS = {RES_1080, RES_720, RES_480}
ALLOWED_FOVS = {FOV_WIDE, FOV_NARROW, FOV_SUPERVIEW, FOV_LINEAR}

# Resolution → (width, height) mapping
_RES_DIMENSIONS: dict[int, tuple[int, int]] = {
    RES_1080: (1920, 1080),
    RES_720: (1280, 720),
    RES_480: (854, 480),
}

# Timeouts
CONNECT_TIMEOUT_S = 60
STREAM_START_TIMEOUT_S = 20
HEALTH_CHECK_TIMEOUT_S = 5
KEEPALIVE_INTERVAL_S = 2.5

# Disconnect detection
MAX_CONSECUTIVE_FAILURES = 3  # Consecutive failed health checks before disconnect
HEALTH_CHECK_INTERVAL_S = 2.5  # How often to run health checks (aligned with keepalive)

# Status polling
WEBCAM_STATUS_POLL_INTERVAL_S = 0.5
WEBCAM_STATUS_POLL_MAX_S = 15.0


class USBTransport(Transport):
    """USB-NCM transport for GoPro cameras.

    Wraps open-gopro SDK's WiredGoPro to provide the Transport ABC.

    Args:
        serial: Last 3 digits of GoPro serial number.  None = auto-discover.
        udp_port: Local UDP port for MPEG-TS stream (default 8554).
        resolution: Webcam resolution code (default RES_1080).
        fov: Webcam FOV code (default FOV_WIDE).
        keepalive_interval: Seconds between keep-alive pings.
    """

    def __init__(
        self,
        serial: str | None = None,
        udp_port: int = GOPRO_UDP_PORT,
        resolution: int = RES_1080,
        fov: int = FOV_WIDE,
        keepalive_interval: float = KEEPALIVE_INTERVAL_S,
        max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    ):
        super().__init__(name="USB")
        self._serial = serial
        self._udp_port = udp_port
        self._resolution = resolution
        self._fov = fov
        self._keepalive_interval = keepalive_interval
        self._max_consecutive_failures = max_consecutive_failures

        # open-gopro SDK handle — set during discover/connect, cleared on disconnect()
        self._gopro: Any = None  # WiredGoPro instance
        self._camera_ip: str | None = None
        self._connected_at: float = 0.0
        self._keepalive_task: asyncio.Task | None = None
        self._device_info: Any = None  # GoProDeviceInfo from USB enumeration
        self._consecutive_failures: int = 0  # Tracks consecutive health-check failures

    # ------------------------------------------------------------------
    # Transport ABC implementation
    # ------------------------------------------------------------------

    async def discover(self, timeout: float = 10.0) -> bool:
        """Discover a GoPro camera connected via USB-NCM.

        First enumerates USB devices to confirm a GoPro is physically
        connected (vendor ID 0x2672). If found, extracts the serial
        number and computes the camera's NCM IP address. Then creates
        a WiredGoPro SDK handle using the discovered serial.

        If no serial was provided at construction and USB enumeration
        finds one, it is used automatically.

        Returns True if a camera was found and a WiredGoPro handle created.
        """
        self._set_state(TransportState.DISCOVERING)
        log.info("Discovering GoPro on USB (serial=%s, timeout=%.1fs)", self._serial, timeout)

        # Step 1: Enumerate USB devices to confirm GoPro is connected
        try:
            from gomaxwebcam.discovery import find_gopro_device
            device_info = await find_gopro_device()
            if device_info is not None:
                log.info(
                    "USB enumeration found GoPro: %s (serial=%s, ip=%s)",
                    device_info.description,
                    device_info.serial_number or "unknown",
                    device_info.camera_ip or "unknown",
                )
                # Auto-fill serial from USB enumeration if not provided
                if self._serial is None and device_info.serial_suffix:
                    self._serial = device_info.serial_suffix
                    log.info("Auto-detected serial suffix: %s", self._serial)
                # Cache computed IP for diagnostics
                if device_info.camera_ip:
                    self._camera_ip = device_info.camera_ip
                # Store device info for stats
                self._stats.camera_serial = device_info.serial_number or self._serial or "unknown"
                self._stats.camera_model = device_info.description
                self._device_info = device_info
            else:
                log.info("USB enumeration found no GoPro — will try SDK mDNS discovery")
                self._device_info = None
        except Exception as e:
            log.debug("USB enumeration failed (non-fatal): %s", e)
            self._device_info = None

        # Step 2: Create WiredGoPro SDK handle
        try:
            from open_gopro import WiredGoPro
        except ImportError:
            log.error("open-gopro SDK not installed — cannot use USB transport")
            self._set_state(TransportState.ERROR)
            self._stats.last_error = "open-gopro SDK not installed"
            return False

        try:
            # Create SDK handle — serial=None means auto-discover first found
            self._gopro = WiredGoPro(serial=self._serial)
            log.info("WiredGoPro SDK handle created (serial=%s)", self._serial or "auto")
            return True

        except Exception as e:
            log.error("Failed to create WiredGoPro handle: %s", e)
            self._set_state(TransportState.ERROR)
            self._stats.last_error = str(e)
            self._gopro = None
            return False

    async def connect(self) -> bool:
        """Connect to the GoPro camera via the WiredGoPro SDK.

        This opens the SDK connection which performs:
          - mDNS discovery of the camera
          - USB-NCM interface setup
          - HTTP API handshake
          - Enable wired USB control
          - Set third-party client info

        Returns True on success. Sets state to CONNECTED.
        """
        if self._gopro is None:
            log.error("Cannot connect: no GoPro SDK handle (run discover() first)")
            return False

        self._set_state(TransportState.CONNECTING)
        log.info("Opening WiredGoPro connection...")

        try:
            await asyncio.wait_for(
                self._gopro.open(),
                timeout=CONNECT_TIMEOUT_S,
            )

            # Extract camera IP from the SDK handle
            self._camera_ip = getattr(self._gopro, "ip_address", None) or "unknown"
            self._stats.camera_serial = self._serial or "auto"
            self._stats.camera_model = f"GoPro@{self._camera_ip}"
            self._connected_at = time.monotonic()
            self._consecutive_failures = 0
            self._set_state(TransportState.CONNECTED)
            log.info(
                "USB connection established (camera=GoPro@%s)",
                self._camera_ip,
            )
            return True

        except asyncio.TimeoutError:
            log.error("USB connection timed out after %ds", CONNECT_TIMEOUT_S)
            self._set_state(TransportState.ERROR)
            self._stats.last_error = "Connection timeout"
            return False
        except Exception as e:
            log.error("USB connection failed: %s", e)
            self._set_state(TransportState.ERROR)
            self._stats.last_error = str(e)
            return False

    async def start_stream(self) -> Optional[StreamInfo]:
        """Start the MPEG-TS preview stream via WiredGoPro SDK.

        Sends webcam_start with resolution and FOV params via the SDK,
        then polls webcam_status until the camera reports HIGH_POWER_PREVIEW.

        Returns StreamInfo on success, None on failure.
        """
        if self._state != TransportState.CONNECTED:
            log.error("Cannot start stream: not connected (state=%s)", self._state.name)
            return None

        if self._gopro is None:
            log.error("Cannot start stream: no GoPro SDK handle")
            return None

        # Validate and clamp resolution/FOV
        if self._resolution not in ALLOWED_RESOLUTIONS:
            log.warning("Resolution %d not in allowed set, defaulting to 1080p", self._resolution)
            self._resolution = RES_1080

        if self._fov not in ALLOWED_FOVS:
            log.warning("FOV %d not in allowed set, defaulting to WIDE", self._fov)
            self._fov = FOV_WIDE

        log.info(
            "Starting webcam stream (res=%d, fov=%d, port=%d)",
            self._resolution, self._fov, self._udp_port,
        )

        try:
            # Import SDK streaming enums for the webcam_start call
            from open_gopro.models.streaming import (
                WebcamResolution,
                WebcamFOV,
                WebcamProtocol,
                WebcamError,
                WebcamStatus as SDKWebcamStatus,
            )

            # Build SDK enum values from our int codes
            res_enum = WebcamResolution(self._resolution)
            fov_enum = WebcamFOV(self._fov)

            # Send webcam/start via SDK
            start_resp = await self._gopro.http_command.webcam_start(
                resolution=res_enum,
                fov=fov_enum,
                port=self._udp_port,
                protocol=WebcamProtocol.TS,
            )

            if not start_resp.ok:
                log.error("webcam_start SDK call failed (ok=False)")
                self._stats.last_error = "webcam_start failed"
                return None

            # Check for webcam error in the response
            if hasattr(start_resp.data, "error") and start_resp.data.error is not None:
                error_val = start_resp.data.error
                # Compare by value — SDK error enum vs our mock
                is_success = (
                    error_val == WebcamError.SUCCESS
                    or (hasattr(error_val, "value") and error_val.value == 0)
                    or error_val == 0
                )
                if not is_success:
                    log.error("webcam_start returned error: %s", error_val)
                    self._stats.last_error = f"webcam error: {error_val}"
                    return None

            log.info("webcam_start sent successfully, polling for streaming status...")

            # Poll webcam_status until HIGH_POWER_PREVIEW or timeout
            deadline = time.monotonic() + WEBCAM_STATUS_POLL_MAX_S
            streaming = False

            while time.monotonic() < deadline:
                await asyncio.sleep(WEBCAM_STATUS_POLL_INTERVAL_S)
                try:
                    status_resp = await self._gopro.http_command.webcam_status()
                    if status_resp.ok and hasattr(status_resp.data, "status"):
                        st = status_resp.data.status
                        # Check for HIGH_POWER_PREVIEW (2) or LOW_POWER_PREVIEW (3)
                        st_val = getattr(st, "value", st)
                        if st_val in (2, 3) or st in (
                            getattr(SDKWebcamStatus, "HIGH_POWER_PREVIEW", 2),
                            getattr(SDKWebcamStatus, "LOW_POWER_PREVIEW", 3),
                        ):
                            streaming = True
                            break
                        log.debug("Webcam status poll: %s", st)
                except Exception as e:
                    log.debug("Status poll error: %s", e)

            if not streaming:
                log.error("Webcam did not reach STREAMING state within %.1fs", WEBCAM_STATUS_POLL_MAX_S)
                self._stats.last_error = "Webcam status timeout"
                # Try to clean up
                try:
                    await self._gopro.http_command.webcam_stop()
                except Exception:
                    pass
                return None

            # Build stream info from resolution
            w, h = _RES_DIMENSIONS.get(self._resolution, (1920, 1080))
            info = StreamInfo(
                protocol="udp",
                host="0.0.0.0",
                port=self._udp_port,
                width=w,
                height=h,
                fps=30,
                codec="h264",
            )
            self._stream_info = info
            self._set_state(TransportState.STREAMING)

            # Start keep-alive background task
            self._start_keepalive_task()

            log.info("Preview stream started: udp://0.0.0.0:%d (%dx%d)", self._udp_port, w, h)
            return info

        except asyncio.TimeoutError:
            log.error("Stream start timed out after %ds", STREAM_START_TIMEOUT_S)
            self._stats.last_error = "Stream start timeout"
            return None
        except Exception as e:
            log.error("Failed to start stream: %s", e, exc_info=True)
            self._stats.last_error = str(e)
            return None

    async def stop_stream(self) -> None:
        """Stop the MPEG-TS preview stream via WiredGoPro SDK."""
        if self._state != TransportState.STREAMING:
            return

        log.info("Stopping webcam stream")
        self._stop_keepalive_task()

        if self._gopro is not None:
            try:
                await self._gopro.http_command.webcam_stop()
            except Exception as e:
                log.warning("Error stopping webcam: %s", e)

            try:
                await self._gopro.http_command.webcam_exit()
            except Exception as e:
                log.warning("Error exiting webcam: %s", e)

        self._stream_info = None
        self._set_state(TransportState.CONNECTED)
        log.info("Webcam stream stopped")

    async def disconnect(self) -> None:
        """Disconnect from the GoPro camera.

        Stops any active stream, cancels keep-alive, closes the SDK
        connection, and resets all internal state.
        """
        if self._state == TransportState.STREAMING:
            await self.stop_stream()

        self._stop_keepalive_task()

        # Close the SDK connection
        if self._gopro is not None:
            try:
                await self._gopro.close()
            except Exception as e:
                log.warning("Error closing WiredGoPro: %s", e)

        self._gopro = None
        self._camera_ip = None
        self._stream_info = None
        self._consecutive_failures = 0
        self._update_uptime()
        self._set_state(TransportState.DISCONNECTED)
        log.info("USB transport disconnected")

    async def keep_alive(self) -> bool:
        """Send a keep-alive ping via the SDK's webcam_status.

        Returns True if the camera responded successfully.
        """
        if not self.is_connected or self._gopro is None:
            return False

        try:
            resp = await self._gopro.http_command.webcam_status()
            self._stats.keepalives_sent += 1
            if resp.ok:
                return True
            else:
                self._stats.keepalives_failed += 1
                return False
        except Exception as e:
            log.warning("Keep-alive failed: %s", e)
            self._stats.keepalives_failed += 1
            self._stats.last_error = str(e)
            return False

    async def health_check(self) -> bool:
        """Check if the camera connection is healthy.

        When streaming, also verifies the webcam status matches
        expected streaming state (HIGH_POWER_PREVIEW or LOW_POWER_PREVIEW).
        """
        if not self.is_connected or self._gopro is None:
            return False

        try:
            resp = await self._gopro.http_command.webcam_status()
            if not resp.ok:
                return False

            if self._state == TransportState.STREAMING:
                if hasattr(resp.data, "status"):
                    st = resp.data.status
                    st_val = getattr(st, "value", st)
                    if st_val not in (2, 3):
                        log.warning(
                            "Health check: expected streaming but got status %s", st
                        )
                        return False
            return True
        except Exception as e:
            log.warning("Health check failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Public configuration
    # ------------------------------------------------------------------

    @property
    def gopro_handle(self) -> Any:
        return self._gopro

    @property
    def serial(self) -> str | None:
        return self._serial

    @property
    def device_info(self) -> Any:
        """GoProDeviceInfo from USB enumeration, or None."""
        return self._device_info

    @property
    def udp_port(self) -> int:
        return self._udp_port

    @property
    def resolution(self) -> int:
        return self._resolution

    @resolution.setter
    def resolution(self, value: int) -> None:
        if value not in ALLOWED_RESOLUTIONS:
            raise ValueError(f"Resolution {value} not allowed. Use one of {ALLOWED_RESOLUTIONS}")
        self._resolution = value

    @property
    def fov(self) -> int:
        return self._fov

    @fov.setter
    def fov(self, value: int) -> None:
        if value not in ALLOWED_FOVS:
            raise ValueError(f"FOV {value} not allowed. Use one of {ALLOWED_FOVS}")
        self._fov = value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_keepalive_task(self) -> None:
        """Start background keep-alive + health-check loop.

        The loop sends periodic keep-alive pings and tracks consecutive
        failures. After ``max_consecutive_failures`` consecutive failures,
        it runs a confirming health_check(). If that also fails, the
        transport emits a disconnect event and transitions to ERROR state.
        """
        self._stop_keepalive_task()
        self._consecutive_failures = 0

        async def _keepalive_loop() -> None:
            while True:
                await asyncio.sleep(self._keepalive_interval)
                if not self.is_connected:
                    break

                ok = await self.keep_alive()
                if ok:
                    # Reset failure counter on any success
                    self._consecutive_failures = 0
                    continue

                self._consecutive_failures += 1
                log.warning(
                    "Keep-alive failed (%d/%d) — camera may be unreachable",
                    self._consecutive_failures,
                    self._max_consecutive_failures,
                )

                if self._consecutive_failures >= self._max_consecutive_failures:
                    # Confirm with a full health check before declaring dead
                    log.warning(
                        "Consecutive failure threshold reached (%d), "
                        "running confirming health check...",
                        self._consecutive_failures,
                    )
                    try:
                        healthy = await asyncio.wait_for(
                            self.health_check(),
                            timeout=HEALTH_CHECK_TIMEOUT_S,
                        )
                    except (asyncio.TimeoutError, Exception) as exc:
                        log.warning("Confirming health check error: %s", exc)
                        healthy = False

                    if healthy:
                        # False alarm — camera recovered
                        log.info("Health check passed after failures, resetting counter")
                        self._consecutive_failures = 0
                        continue

                    # Camera confirmed dead — emit disconnect event
                    log.error(
                        "USB camera confirmed disconnected after %d consecutive failures",
                        self._consecutive_failures,
                    )
                    self._stats.last_error = (
                        f"USB disconnect detected: {self._consecutive_failures} "
                        f"consecutive health-check failures"
                    )
                    self._set_state(TransportState.ERROR)
                    self._emit_disconnect("health_check_failed")
                    break

        self._keepalive_task = asyncio.ensure_future(_keepalive_loop())

    def _stop_keepalive_task(self) -> None:
        """Cancel the background keep-alive task."""
        if self._keepalive_task is not None and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            self._keepalive_task = None

    def _update_uptime(self) -> None:
        """Update connection uptime in stats."""
        if self._connected_at > 0:
            self._stats.connection_uptime_s = time.monotonic() - self._connected_at
            self._connected_at = 0.0

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def _inject_gopro(self, gopro: Any) -> None:
        """For testing: inject a mock WiredGoPro SDK handle.

        Bypasses discover() by directly setting the SDK handle and
        extracting the camera IP from it.
        """
        self._gopro = gopro
        self._camera_ip = getattr(gopro, "ip_address", "172.20.100.51")
        self._stats.camera_model = f"GoPro@{self._camera_ip}"
        log.debug("Injected GoPro SDK handle (ip=%s)", self._camera_ip)

    def _inject_camera_ip(self, ip: str) -> None:
        """For testing: inject a camera IP."""
        self._camera_ip = ip
        self._stats.camera_model = f"GoPro@{ip}"
        log.debug("Injected camera IP: %s", ip)
