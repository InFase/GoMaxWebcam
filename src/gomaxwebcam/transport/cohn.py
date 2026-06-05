"""
transport/cohn.py — COHN (Camera On Home Network) transport for GoPro cameras.

Wraps open-gopro SDK's WirelessGoPro for camera control over COHN (WiFi Station
mode with HTTPS). The camera must be pre-provisioned onto the home network
(via BLE provisioning or the GoPro app) before this transport can connect.

The COHN transport:
  1. Validates that COHN credentials (ip, username, password, certificate) are
     available — either passed directly or loaded by open-gopro from its cohn_db
  2. Opens a WirelessGoPro instance with Interface.COHN
  3. Starts the MPEG-TS preview stream over UDP via webcam_start
  4. Sends periodic keep-alive pings via webcam_status
  5. Stops stream, exits webcam, and closes the SDK connection on shutdown

Credential lifecycle:
  - open-gopro owns COHN credential persistence in its cohn_db.json
  - Credentials can also be passed as CohnInfo at construction time
  - No keyring duplication — open-gopro is the single source of truth

For CI/testing, the WirelessGoPro handle can be replaced with a mock
via _inject_gopro().
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

from gomaxwebcam.transport.base import (
    Transport,
    TransportState,
    StreamInfo,
    TransportStats,
)

log = logging.getLogger("gomaxwebcam.transport.cohn")

# Default UDP port for MPEG-TS stream (same as USB)
COHN_UDP_PORT = 8554

# Webcam resolution codes (match open_gopro.models.streaming.WebcamResolution)
RES_480 = 4
RES_720 = 7
RES_1080 = 12

# Webcam FOV codes (match open_gopro.models.streaming.WebcamFOV)
FOV_WIDE = 0
FOV_NARROW = 2
FOV_SUPERVIEW = 3
FOV_LINEAR = 4

# Only expose 16:9 modes at 30fps
ALLOWED_RESOLUTIONS = {RES_1080, RES_720, RES_480}
ALLOWED_FOVS = {FOV_WIDE, FOV_NARROW, FOV_SUPERVIEW, FOV_LINEAR}

# Timeouts
CONNECT_TIMEOUT_S = 30
STREAM_START_TIMEOUT_S = 20
HEALTH_CHECK_TIMEOUT_S = 5
KEEPALIVE_INTERVAL_S = 2.5

# Status polling
WEBCAM_STATUS_POLL_INTERVAL_S = 0.5
WEBCAM_STATUS_POLL_MAX_S = 15.0

# Resolution → (width, height) mapping
_RES_DIMENSIONS: dict[int, tuple[int, int]] = {
    RES_1080: (1920, 1080),
    RES_720:  (1280, 720),
    RES_480:  (854, 480),
}


class COHNTransport(Transport):
    """COHN (Camera On Home Network) transport wrapping open-gopro's WirelessGoPro.

    Connects to a GoPro camera provisioned on the home WiFi network.  All COHN
    credential management is delegated to the open-gopro SDK (CohnInfo / cohn_db).

    Args:
        ip_address: Camera IP on the local network (from CohnInfo or mDNS).
        username: COHN Basic Auth username.
        password: COHN Basic Auth password (from BLE provisioning).
        certificate: PEM certificate string from the camera.
        cohn_db: Path to open-gopro's COHN credential database.
        udp_port: Local UDP port for MPEG-TS stream (default 8554).
        resolution: Webcam resolution code (default RES_1080).
        fov: Webcam FOV code (default FOV_WIDE).
        keepalive_interval: Seconds between keep-alive pings.
    """

    def __init__(
        self,
        ip_address: str | None = None,
        username: str = "",
        password: str = "",
        certificate: str = "",
        cohn_db: Path | None = None,
        udp_port: int = COHN_UDP_PORT,
        resolution: int = RES_1080,
        fov: int = FOV_WIDE,
        keepalive_interval: float = KEEPALIVE_INTERVAL_S,
    ):
        super().__init__(name="COHN")
        self._ip_address = ip_address
        self._username = username
        self._password = password
        self._certificate = certificate
        self._cohn_db = cohn_db
        self._udp_port = udp_port
        self._resolution = resolution
        self._fov = fov
        self._keepalive_interval = keepalive_interval

        # open-gopro SDK handle — set during connect(), cleared on disconnect()
        self._gopro: Any = None  # WirelessGoPro instance
        self._connected_at: float = 0.0
        self._keepalive_task: asyncio.Task | None = None

        # Set to True when a connection/auth failure indicates stale credentials
        self._credentials_invalid: bool = False

    # ------------------------------------------------------------------
    # Transport ABC implementation
    # ------------------------------------------------------------------

    async def discover(self, timeout: float = 10.0) -> bool:
        """Validate that COHN credentials are available for connection.

        With pre-configured credentials (ip_address + password), this simply
        validates the credential set is complete.  Without credentials, attempts
        mDNS discovery to find a GoPro on the local network.

        Returns True if credentials are available and a camera can be reached.
        """
        self._set_state(TransportState.DISCOVERING)

        if self._ip_address and self._password:
            log.info(
                "COHN discovery: credentials pre-configured for %s",
                self._ip_address,
            )
            self._stats.camera_model = f"GoPro@{self._ip_address} (COHN)"
            return True

        # Try mDNS discovery as fallback
        log.info(
            "Discovering GoPro on local network via mDNS (timeout=%.1fs)",
            timeout,
        )
        try:
            from zeroconf import Zeroconf, ServiceBrowser

            found_ip: str | None = None
            found_event = asyncio.Event()
            loop = asyncio.get_event_loop()

            class _Listener:
                def add_service(self, zc: Any, type_: str, name: str) -> None:
                    nonlocal found_ip
                    import socket
                    info = zc.get_service_info(type_, name)
                    if info and info.addresses:
                        addr = socket.inet_ntoa(info.addresses[0])
                        log.info("mDNS: found GoPro at %s (%s)", addr, name)
                        found_ip = addr
                        loop.call_soon_threadsafe(found_event.set)

                def remove_service(self, *a: Any) -> None:
                    pass

                def update_service(self, *a: Any) -> None:
                    pass

            zc = Zeroconf()
            browser = ServiceBrowser(
                zc, "_gopro-web._tcp.local.", _Listener()
            )
            try:
                await asyncio.wait_for(found_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                log.warning("mDNS discovery timed out after %.1fs", timeout)
            finally:
                browser.cancel()
                zc.close()

            if found_ip:
                self._ip_address = found_ip
                self._stats.camera_model = f"GoPro@{found_ip} (COHN)"
                return True

            log.warning("No GoPro found via mDNS")
            self._set_state(TransportState.ERROR)
            self._stats.last_error = "No GoPro found via mDNS"
            return False

        except ImportError:
            log.error("zeroconf not installed — cannot use mDNS discovery")
            self._set_state(TransportState.ERROR)
            self._stats.last_error = "zeroconf not installed"
            return False
        except Exception as e:
            log.error("mDNS discovery failed: %s", e)
            self._set_state(TransportState.ERROR)
            self._stats.last_error = str(e)
            return False

    async def connect(self) -> bool:
        """Open a WirelessGoPro connection in COHN-only mode.

        Creates a WirelessGoPro instance with Interface.COHN, passing
        pre-provisioned CohnInfo credentials.  The SDK handles TLS
        certificate verification and HTTP Basic Auth internally.
        """
        if not self._ip_address:
            log.error("Cannot connect: no camera IP (run discover() first)")
            return False

        if not self._password:
            log.error("Cannot connect: no COHN password (credentials missing)")
            self._set_state(TransportState.ERROR)
            self._stats.last_error = "COHN password not configured"
            return False

        self._set_state(TransportState.CONNECTING)
        log.info("Connecting to GoPro via COHN at %s...", self._ip_address)

        try:
            from open_gopro import WirelessGoPro
            from open_gopro.models.general import CohnInfo

            # Build CohnInfo credentials for the SDK
            cohn_info = CohnInfo(
                ip_address=self._ip_address,
                username=self._username,
                password=self._password,
                certificate=self._certificate,
            )

            # Create WirelessGoPro in COHN-only mode (no BLE, no WiFi AP)
            kwargs: dict[str, Any] = {
                "interfaces": {WirelessGoPro.Interface.COHN},
                "cohn_credentials": cohn_info,
            }
            if self._cohn_db is not None:
                kwargs["cohn_db"] = self._cohn_db

            gopro = WirelessGoPro(**kwargs)

            # Open the connection (SDK handles TLS + auth)
            await asyncio.wait_for(
                gopro.open(),
                timeout=CONNECT_TIMEOUT_S,
            )

            if not gopro.is_http_connected:
                log.error("WirelessGoPro opened but HTTP not connected")
                await gopro.close()
                self._set_state(TransportState.ERROR)
                self._stats.last_error = "HTTP connection failed after open"
                return False

            self._gopro = gopro
            self._credentials_invalid = False  # Reset on successful connect

            # Validate connectivity with a webcam status check
            try:
                resp = await gopro.http_command.webcam_status()
                log.info("COHN webcam status check: ok=%s", resp.ok)
            except Exception as e:
                log.warning("Initial webcam status check failed: %s (continuing)", e)

            # Populate stats
            self._stats.camera_serial = self._username or "cohn"
            self._stats.camera_model = f"GoPro@{self._ip_address} (COHN)"
            self._connected_at = time.monotonic()
            self._set_state(TransportState.CONNECTED)
            log.info("COHN connection established to %s", self._ip_address)
            return True

        except asyncio.TimeoutError:
            log.error("COHN connection timed out after %ds", CONNECT_TIMEOUT_S)
            self._set_state(TransportState.ERROR)
            self._stats.last_error = "Connection timeout"
            return False
        except ImportError:
            log.error("open-gopro not installed — cannot use COHN transport")
            self._set_state(TransportState.ERROR)
            self._stats.last_error = "open-gopro not installed"
            return False
        except Exception as e:
            log.error("COHN connection failed: %s", e, exc_info=True)
            self._set_state(TransportState.ERROR)
            self._stats.last_error = str(e)
            # Detect credential/certificate errors → flag for re-provisioning
            if self._is_credential_error(e):
                self._mark_credentials_invalid()
            return False

    async def start_stream(self) -> Optional[StreamInfo]:
        """Start the MPEG-TS preview stream via open-gopro's webcam_start.

        Sends the webcam start command via the SDK, then polls webcam_status
        until the camera reports HIGH_POWER_PREVIEW.

        Returns StreamInfo on success, None on failure.
        """
        if self._state != TransportState.CONNECTED:
            log.error("Cannot start stream: not connected (state=%s)", self._state.name)
            return None

        if self._gopro is None:
            log.error("Cannot start stream: no WirelessGoPro instance")
            return None

        if self._resolution not in ALLOWED_RESOLUTIONS:
            log.warning("Resolution %d not in allowed set, defaulting to 1080p", self._resolution)
            self._resolution = RES_1080

        if self._fov not in ALLOWED_FOVS:
            log.warning("FOV %d not in allowed set, defaulting to WIDE", self._fov)
            self._fov = FOV_WIDE

        log.info(
            "Starting COHN webcam stream (res=%d, fov=%d, port=%d)",
            self._resolution, self._fov, self._udp_port,
        )

        try:
            from open_gopro.models.streaming import (
                WebcamResolution,
                WebcamFOV,
                WebcamProtocol,
                WebcamStatus,
            )

            # Handle IDLE state with reset workaround (same as USB)
            try:
                status_resp = await self._gopro.http_command.webcam_status()
                if status_resp.ok and status_resp.data.status is not None:
                    current = status_resp.data.status
                    log.info("Current webcam status: %s", current)

                    if current == WebcamStatus.IDLE:
                        log.info("Camera in IDLE — performing reset workaround")
                        await self._gopro.http_command.webcam_start()
                        await asyncio.sleep(1.0)
                        await self._gopro.http_command.webcam_stop()
                        await asyncio.sleep(1.0)
                    elif current in (
                        WebcamStatus.HIGH_POWER_PREVIEW,
                        WebcamStatus.LOW_POWER_PREVIEW,
                    ):
                        log.info("Camera already streaming — stopping first")
                        await self._gopro.http_command.webcam_stop()
                        await asyncio.sleep(1.0)
            except Exception as e:
                log.debug("Pre-start status check failed (continuing): %s", e)

            # Map our int codes to open-gopro enums
            res_enum = WebcamResolution(self._resolution)
            fov_enum = WebcamFOV(self._fov)

            # Send webcam_start
            resp = await self._gopro.http_command.webcam_start(
                resolution=res_enum,
                fov=fov_enum,
                port=self._udp_port,
                protocol=WebcamProtocol.TS,
            )

            if not resp.ok:
                log.error("webcam_start failed: %s", resp)
                self._stats.last_error = f"webcam_start failed: {resp}"
                return None

            # Check for error in response data
            if hasattr(resp.data, 'error') and resp.data.error is not None:
                error_val = resp.data.error.value if hasattr(resp.data.error, 'value') else resp.data.error
                if error_val != 0:
                    log.error("webcam_start returned error: %s", resp.data.error)
                    self._stats.last_error = f"webcam_start error: {resp.data.error}"
                    return None

            log.info("webcam_start sent successfully")

            # Poll until camera is streaming
            streaming = await self._poll_webcam_status()
            if not streaming:
                log.error("Webcam did not reach STREAMING state within timeout")
                self._stats.last_error = "Webcam status timeout"
                try:
                    await self._gopro.http_command.webcam_stop()
                except Exception:
                    pass
                return None

            # Build stream info
            width, height = _RES_DIMENSIONS.get(
                self._resolution, (1920, 1080)
            )
            info = StreamInfo(
                protocol="udp",
                host="0.0.0.0",
                port=self._udp_port,
                width=width,
                height=height,
                fps=30,
                codec="h264",
            )
            self._stream_info = info
            self._set_state(TransportState.STREAMING)

            # Start keep-alive background task
            self._start_keepalive_task()

            log.info("COHN preview stream started: udp://0.0.0.0:%d", self._udp_port)
            return info

        except asyncio.TimeoutError:
            log.error("Stream start timed out after %ds", STREAM_START_TIMEOUT_S)
            self._stats.last_error = "Stream start timeout"
            return None
        except Exception as e:
            log.error("Failed to start COHN stream: %s", e, exc_info=True)
            self._stats.last_error = str(e)
            return None

    async def stop_stream(self) -> None:
        """Stop the MPEG-TS preview stream via the SDK."""
        if self._state != TransportState.STREAMING:
            return

        log.info("Stopping COHN webcam stream")
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
        log.info("COHN webcam stream stopped")

    async def disconnect(self) -> None:
        """Close the WirelessGoPro connection and clean up."""
        if self._state == TransportState.STREAMING:
            await self.stop_stream()

        self._stop_keepalive_task()

        if self._gopro is not None:
            try:
                await self._gopro.close()
            except Exception as e:
                log.warning("Error closing WirelessGoPro: %s", e)
            self._gopro = None

        self._stream_info = None
        self._update_uptime()
        self._set_state(TransportState.DISCONNECTED)
        log.info("COHN transport disconnected")

    async def keep_alive(self) -> bool:
        """Send a keep-alive ping via webcam_status.

        Uses the SDK's webcam_status command as a lightweight ping that
        also lets us detect if the camera dropped the stream.
        """
        if not self.is_connected or self._gopro is None:
            return False

        try:
            resp = await self._gopro.http_command.webcam_status()
            self._stats.keepalives_sent += 1

            if resp.ok:
                return True
            else:
                log.warning("Keep-alive got non-OK response: %s", resp)
                self._stats.keepalives_failed += 1
                return False

        except asyncio.TimeoutError:
            log.warning("Keep-alive timed out")
            self._stats.keepalives_failed += 1
            self._stats.last_error = "Keep-alive timeout"
            return False
        except Exception as e:
            log.warning("Keep-alive failed: %s", e)
            self._stats.keepalives_failed += 1
            self._stats.last_error = str(e)
            return False

    async def health_check(self) -> bool:
        """Check if the camera connection is healthy.

        Verifies webcam status and, if streaming, confirms the camera
        reports an active preview state.
        """
        if not self.is_connected or self._gopro is None:
            return False

        try:
            resp = await self._gopro.http_command.webcam_status()
            if not resp.ok:
                return False

            # If we think we're streaming, verify camera agrees
            if self._state == TransportState.STREAMING:
                if resp.data.status is not None:
                    status_val = (
                        resp.data.status.value
                        if hasattr(resp.data.status, 'value')
                        else resp.data.status
                    )
                    # HIGH_POWER_PREVIEW = 2, LOW_POWER_PREVIEW = 3
                    if status_val not in (2, 3):
                        log.warning(
                            "Health check: expected streaming but camera reports %s",
                            resp.data.status,
                        )
                        return False

            return True

        except Exception as e:
            log.warning("Health check failed: %s", e)
            if self._is_credential_error(e):
                self._mark_credentials_invalid()
            return False

    # ------------------------------------------------------------------
    # Public configuration
    # ------------------------------------------------------------------

    @property
    def gopro_handle(self) -> Any:
        return self._gopro

    @property
    def ip_address(self) -> str | None:
        return self._ip_address

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

    @property
    def username(self) -> str:
        return self._username

    @property
    def has_credentials(self) -> bool:
        """True if COHN credentials are configured."""
        return bool(self._password)

    @property
    def credentials_invalid(self) -> bool:
        """True if the last connection attempt failed due to invalid credentials.

        Set when connect() or health_check() encounters an authentication
        or TLS certificate error, indicating that cached COHN credentials
        are stale and BLE re-provisioning is required.
        """
        return self._credentials_invalid

    # ------------------------------------------------------------------
    # Frame iteration
    # ------------------------------------------------------------------

    async def iter_frames(
        self,
        queue_size: int = 4,
        decode_timeout: float = 15.0,
        pixel_format: str = "bgr24",
    ):
        """Async generator yielding decoded BGR24 numpy frames over COHN.

        Overrides the base Transport.iter_frames() with COHN-specific defaults:
          - Longer decode_timeout (15s) to account for WiFi latency
          - Validates COHN connection health before starting iteration
          - Monitors connection via keep-alive during iteration

        The generator starts the stream if not already streaming, creates an
        internal UDPDecoder, and yields numpy BGR24 frames (H, W, 3) uint8.

        Args:
            queue_size: Max buffered frames (default 4 for WiFi jitter).
            decode_timeout: Seconds to wait for first frame (default 15s).
            pixel_format: Output pixel format, default "bgr24" for v1 parity.

        Yields:
            numpy.ndarray: BGR24 frames compatible with VirtualCameraSink.

        Raises:
            RuntimeError: If COHN connection is not established or stream fails.
            asyncio.TimeoutError: If no frame arrives within decode_timeout.
        """
        if self._gopro is None:
            raise RuntimeError(
                "Cannot iterate frames: no WirelessGoPro instance "
                "(call connect() first)"
            )

        # Pre-flight: verify COHN HTTP connectivity
        try:
            resp = await self._gopro.http_command.webcam_status()
            if not resp.ok:
                log.warning(
                    "COHN pre-flight check: webcam_status not OK, proceeding anyway"
                )
        except Exception as e:
            log.warning("COHN pre-flight check failed: %s (proceeding)", e)

        # Delegate to base implementation with COHN-tuned defaults.
        # Explicit try/finally ensures the inner generator is properly
        # closed when this generator is closed (nested async gen cleanup).
        inner = super().iter_frames(
            queue_size=queue_size,
            decode_timeout=decode_timeout,
            pixel_format=pixel_format,
        )
        try:
            async for frame in inner:
                yield frame
        finally:
            await inner.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _poll_webcam_status(self) -> bool:
        """Poll webcam_status until camera reports HIGH_POWER_PREVIEW or timeout.

        Returns True if the camera reached streaming state.
        """
        if self._gopro is None:
            return False

        deadline = time.monotonic() + WEBCAM_STATUS_POLL_MAX_S
        while time.monotonic() < deadline:
            try:
                resp = await self._gopro.http_command.webcam_status()
                if resp.ok and resp.data.status is not None:
                    status_val = (
                        resp.data.status.value
                        if hasattr(resp.data.status, 'value')
                        else resp.data.status
                    )
                    log.debug("Webcam status poll: %s (raw=%s)", resp.data.status, status_val)
                    # HIGH_POWER_PREVIEW = 2, LOW_POWER_PREVIEW = 3
                    if status_val in (2, 3):
                        return True
            except Exception as e:
                log.debug("Status poll error: %s", e)

            await asyncio.sleep(WEBCAM_STATUS_POLL_INTERVAL_S)

        return False

    def _start_keepalive_task(self) -> None:
        """Start background keep-alive pinging."""
        self._stop_keepalive_task()

        async def _keepalive_loop() -> None:
            while True:
                await asyncio.sleep(self._keepalive_interval)
                if not self.is_connected:
                    break
                ok = await self.keep_alive()
                if not ok:
                    log.warning("COHN keep-alive failed — camera may be unreachable")

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
    # Credential error detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_credential_error(exc: Exception) -> bool:
        """Determine if an exception indicates invalid/expired COHN credentials.

        Checks for HTTP 401/403, TLS certificate verification failures, and
        other authentication-related errors that signal the cached credentials
        are stale and BLE re-provisioning is needed.

        Args:
            exc: The exception to inspect.

        Returns:
            True if the error is caused by invalid credentials or certificates.
        """
        msg = str(exc).lower()

        # HTTP authentication failures
        if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg:
            return True

        # TLS/SSL certificate errors
        if any(kw in msg for kw in (
            "certificate", "ssl", "tls",
            "certificate_verify_failed", "ssl_error",
            "cert", "handshake",
        )):
            return True

        # open-gopro specific authentication errors
        if "authentication" in msg or "auth" in msg:
            return True

        return False

    def _mark_credentials_invalid(self) -> None:
        """Mark credentials as invalid and log the situation."""
        if not self._credentials_invalid:
            self._credentials_invalid = True
            log.warning(
                "COHN credentials marked as invalid — BLE re-provisioning required"
            )

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def _inject_gopro(self, mock_gopro: Any) -> None:
        """For testing: inject a mock WirelessGoPro instance."""
        self._gopro = mock_gopro
        log.debug("Injected mock WirelessGoPro")

    def _inject_camera_ip(self, ip: str) -> None:
        """For testing: inject a camera IP address."""
        self._ip_address = ip
        self._stats.camera_model = f"GoPro@{ip} (COHN)"
        log.debug("Injected camera IP: %s", ip)

    def _inject_credentials(self, username: str, password: str) -> None:
        """For testing: inject COHN credentials."""
        self._username = username
        self._password = password
        log.debug("Injected COHN credentials")
