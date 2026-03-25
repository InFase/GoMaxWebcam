"""
transport/wifi_ap.py — WiFi AP transport for GoPro cameras.

Connects to a GoPro camera via its built-in WiFi Access Point mode.
The GoPro acts as an AP and the host connects to its network.

In WiFi AP mode:
  - GoPro IP is always 10.5.5.9 (fixed by firmware)
  - HTTP API on port 8080 (same commands as USB)
  - MPEG-TS preview stream sent over UDP

The WiFi AP transport:
  1. Discovers GoPro by pinging 10.5.5.9:8080 (assumes host is on GoPro WiFi)
  2. Opens an HTTP session and verifies the GoPro API responds
  3. Starts the MPEG-TS preview stream over UDP via webcam API
  4. Sends periodic keep-alive pings via webcam_status
  5. Stops stream, exits webcam, and closes session on shutdown

For CI/testing, all HTTP calls go through self._session which can
be replaced with a mock via _inject_session().  A mock response
factory can be injected via _inject_response_factory().
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import aiohttp

from gomaxwebcam.transport.base import (
    Transport,
    TransportState,
    StreamInfo,
)
from gomaxwebcam.transport.wifi_manager import WiFiManager, WiFiNetwork

log = logging.getLogger("gomaxwebcam.transport.wifi_ap")

# GoPro WiFi AP fixed IP address (all models)
GOPRO_AP_IP = "10.5.5.9"

# Default ports
GOPRO_HTTP_PORT = 8080
GOPRO_UDP_PORT = 8554

# HTTP API endpoints (Open GoPro HTTP API)
EP_WEBCAM_START = "/gopro/webcam/start"
EP_WEBCAM_STOP = "/gopro/webcam/stop"
EP_WEBCAM_EXIT = "/gopro/webcam/exit"
EP_WEBCAM_STATUS = "/gopro/webcam/status"
EP_WEBCAM_VERSION = "/gopro/webcam/version"
EP_KEEP_ALIVE = "/gopro/camera/keep_alive"

# Webcam status values from Open GoPro API
WEBCAM_STATUS_OFF = 0
WEBCAM_STATUS_IDLE = 1
WEBCAM_STATUS_HIGH_POWER_PREVIEW = 2
WEBCAM_STATUS_LOW_POWER_PREVIEW = 3

# Webcam error values
WEBCAM_ERROR_SUCCESS = 0

# Resolution codes (matches Open GoPro webcam API)
RES_480 = 4
RES_720 = 7
RES_1080 = 12

# FOV codes
FOV_WIDE = 0
FOV_NARROW = 2
FOV_SUPERVIEW = 3
FOV_LINEAR = 4

# Only expose 16:9 modes at 30fps
ALLOWED_RESOLUTIONS = {RES_1080, RES_720, RES_480}
ALLOWED_FOVS = {FOV_WIDE, FOV_NARROW, FOV_SUPERVIEW, FOV_LINEAR}

# Timeouts
DISCOVER_TIMEOUT_S = 5.0
CONNECT_TIMEOUT_S = 10.0
STREAM_START_TIMEOUT_S = 20.0
HEALTH_CHECK_TIMEOUT_S = 5.0
KEEPALIVE_INTERVAL_S = 2.5
HTTP_TIMEOUT_S = 8.0

# Status polling
WEBCAM_STATUS_POLL_INTERVAL_S = 0.5
WEBCAM_STATUS_POLL_MAX_S = 15.0

# Resolution to dimensions mapping
_RES_DIMENSIONS = {
    RES_1080: (1920, 1080),
    RES_720: (1280, 720),
    RES_480: (854, 480),
}


class WiFiAPTransport(Transport):
    """WiFi AP transport for GoPro cameras.

    Connects to a GoPro in AP mode at its fixed IP (10.5.5.9:8080).
    Uses direct HTTP API calls (no open-gopro SDK required).

    Args:
        camera_ip: GoPro IP address (default 10.5.5.9).
        http_port: HTTP API port (default 8080).
        udp_port: Local UDP port for MPEG-TS stream (default 8554).
        resolution: Webcam resolution code (default RES_1080).
        fov: Webcam FOV code (default FOV_WIDE).
        keepalive_interval: Seconds between keep-alive pings.
        wifi_password: Password for GoPro WiFi AP (for auto-connect).
        auto_connect_wifi: If True, auto-scan and join GoPro AP on discover().
        wifi_manager: Optional pre-configured WiFiManager instance.
    """

    def __init__(
        self,
        camera_ip: str = GOPRO_AP_IP,
        http_port: int = GOPRO_HTTP_PORT,
        udp_port: int = GOPRO_UDP_PORT,
        resolution: int = RES_1080,
        fov: int = FOV_WIDE,
        keepalive_interval: float = KEEPALIVE_INTERVAL_S,
        wifi_password: str = "",
        auto_connect_wifi: bool = False,
        wifi_manager: Optional[WiFiManager] = None,
    ):
        super().__init__(name="WiFi AP")
        self._camera_ip = camera_ip
        self._http_port = http_port
        self._udp_port = udp_port
        self._resolution = resolution
        self._fov = fov
        self._keepalive_interval = keepalive_interval
        self._wifi_password = wifi_password
        self._auto_connect_wifi = auto_connect_wifi

        # WiFi manager for OS-level WiFi AP discovery and connection
        self._wifi_manager = wifi_manager
        self._wifi_managed_connection = False  # True if we joined the WiFi AP

        # HTTP session — created on connect, closed on disconnect
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected_at: float = 0.0
        self._keepalive_task: Optional[asyncio.Task] = None
        self._base_url: str = f"http://{camera_ip}:{http_port}"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def camera_ip(self) -> str:
        return self._camera_ip

    @property
    def http_port(self) -> int:
        return self._http_port

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
    # Transport ABC implementation
    # ------------------------------------------------------------------

    async def discover(self, timeout: float = DISCOVER_TIMEOUT_S) -> bool:
        """Discover a GoPro camera on its WiFi AP network.

        If auto_connect_wifi is True and a WiFiManager is available,
        first scans for GoPro SSIDs and joins the hotspot. Otherwise,
        assumes the host is already on the GoPro WiFi network.

        Then pings the GoPro's fixed IP at 10.5.5.9:8080 to verify
        it's reachable.

        Returns True if the GoPro HTTP API responds.
        """
        self._set_state(TransportState.DISCOVERING)
        log.info(
            "Discovering GoPro on WiFi AP (ip=%s, port=%d, timeout=%.1fs)",
            self._camera_ip, self._http_port, timeout,
        )

        # Step 1: Auto-connect to GoPro WiFi AP if configured
        if self._auto_connect_wifi and self._wifi_password:
            if self._wifi_manager is None:
                try:
                    self._wifi_manager = WiFiManager()
                except Exception as e:
                    log.warning("Could not create WiFiManager: %s", e)

            if self._wifi_manager is not None:
                try:
                    gopro_networks = await self._wifi_manager.scan_for_gopro(timeout=timeout)
                    if gopro_networks:
                        best = gopro_networks[0]  # Strongest signal
                        log.info("Auto-connecting to GoPro AP: %s (signal=%d)",
                                 best.ssid, best.signal_strength)
                        await self._wifi_manager.connect(best.ssid, self._wifi_password)
                        self._wifi_managed_connection = True
                        # Give the network a moment to stabilize
                        await asyncio.sleep(2.0)
                    else:
                        log.warning("No GoPro WiFi APs found during scan")
                except Exception as e:
                    log.warning("WiFi auto-connect failed: %s (will try direct ping)", e)

        # Step 2: Ping the GoPro HTTP API to verify reachability
        try:
            session = self._session or aiohttp.ClientSession()
            close_session = self._session is None

            try:
                url = f"{self._base_url}{EP_WEBCAM_STATUS}"
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 200:
                        log.info(
                            "GoPro found at %s:%d (WiFi AP)",
                            self._camera_ip, self._http_port,
                        )
                        return True
                    else:
                        log.warning(
                            "GoPro API returned status %d at %s",
                            resp.status, url,
                        )
                        self._set_state(TransportState.ERROR)
                        self._stats.last_error = f"HTTP {resp.status} from GoPro"
                        return False
            finally:
                if close_session:
                    await session.close()

        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            log.error("WiFi AP discovery failed: %s", e)
            self._set_state(TransportState.ERROR)
            self._stats.last_error = str(e)
            return False

    async def connect(self) -> bool:
        """Establish HTTP session with the GoPro over WiFi AP.

        Creates an aiohttp session and verifies the camera API is responsive
        by querying webcam_status.

        Returns True on success. Sets state to CONNECTED.
        """
        self._set_state(TransportState.CONNECTING)
        log.info("Connecting to GoPro at %s:%d (WiFi AP)", self._camera_ip, self._http_port)

        try:
            # Create HTTP session
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S),
            )

            # Verify API is reachable
            url = f"{self._base_url}{EP_WEBCAM_STATUS}"
            async with self._session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=CONNECT_TIMEOUT_S),
            ) as resp:
                if resp.status != 200:
                    log.error("GoPro API not responsive: HTTP %d", resp.status)
                    await self._close_session()
                    self._set_state(TransportState.ERROR)
                    self._stats.last_error = f"HTTP {resp.status} on connect"
                    return False

                # Parse status response
                data = await resp.json()
                log.debug("Webcam status on connect: %s", data)

            # Populate stats
            self._stats.camera_model = f"GoPro@{self._camera_ip} (WiFi AP)"
            self._stats.camera_serial = "wifi-ap"
            self._connected_at = time.monotonic()
            self._set_state(TransportState.CONNECTED)
            log.info("WiFi AP connection established (camera=%s)", self._stats.camera_model)
            return True

        except asyncio.TimeoutError:
            log.error("WiFi AP connection timed out after %ds", CONNECT_TIMEOUT_S)
            await self._close_session()
            self._set_state(TransportState.ERROR)
            self._stats.last_error = "Connection timeout"
            return False
        except (aiohttp.ClientError, OSError) as e:
            log.error("WiFi AP connection failed: %s", e)
            await self._close_session()
            self._set_state(TransportState.ERROR)
            self._stats.last_error = str(e)
            return False

    async def start_stream(self) -> Optional[StreamInfo]:
        """Start the MPEG-TS preview stream from the camera over UDP.

        Calls the GoPro HTTP webcam_start endpoint with resolution, FOV,
        and port parameters. Polls webcam_status until the camera reports
        HIGH_POWER_PREVIEW (actually streaming).

        Returns StreamInfo on success, None on failure.
        """
        if self._state != TransportState.CONNECTED:
            log.error("Cannot start stream: not connected (state=%s)", self._state.name)
            return None

        if self._session is None:
            log.error("Cannot start stream: no HTTP session")
            return None

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
            # Build query parameters for webcam_start
            params = {
                "res": str(self._resolution),
                "fov": str(self._fov),
                "port": str(self._udp_port),
                "protocol": "TS",
            }

            url = f"{self._base_url}{EP_WEBCAM_START}"
            async with self._session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=STREAM_START_TIMEOUT_S),
            ) as resp:
                if resp.status != 200:
                    log.error("webcam_start HTTP %d", resp.status)
                    self._stats.last_error = f"webcam_start HTTP {resp.status}"
                    return None

                data = await resp.json()
                log.debug("webcam_start response: %s", data)

                # Check for error in response
                error_code = data.get("error", WEBCAM_ERROR_SUCCESS)
                if error_code != WEBCAM_ERROR_SUCCESS:
                    log.error("webcam_start error code: %d", error_code)
                    self._stats.last_error = f"webcam_start error: {error_code}"
                    return None

            # Poll until camera is actually streaming
            streaming = await self._poll_webcam_status(WEBCAM_STATUS_HIGH_POWER_PREVIEW)
            if not streaming:
                log.error("Webcam did not reach STREAMING state within timeout")
                self._stats.last_error = "Webcam status timeout"
                # Try to clean up
                await self._http_get(EP_WEBCAM_STOP)
                return None

            # Build stream info
            width, height = _RES_DIMENSIONS.get(self._resolution, (1920, 1080))
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

            log.info("Preview stream started: udp://0.0.0.0:%d", self._udp_port)
            return info

        except asyncio.TimeoutError:
            log.error("Stream start timed out after %ds", STREAM_START_TIMEOUT_S)
            self._stats.last_error = "Stream start timeout"
            return None
        except (aiohttp.ClientError, OSError) as e:
            log.error("Failed to start stream: %s", e, exc_info=True)
            self._stats.last_error = str(e)
            return None

    async def stop_stream(self) -> None:
        """Stop the MPEG-TS preview stream."""
        if self._state != TransportState.STREAMING:
            return

        log.info("Stopping webcam stream")
        self._stop_keepalive_task()

        # Send stop + exit to camera
        await self._http_get(EP_WEBCAM_STOP, timeout=5.0)
        await self._http_get(EP_WEBCAM_EXIT, timeout=5.0)

        self._stream_info = None
        self._set_state(TransportState.CONNECTED)
        log.info("Webcam stream stopped")

    async def disconnect(self) -> None:
        """Disconnect from the GoPro camera.

        If we managed the WiFi connection (auto-connected to GoPro AP),
        restores the previous WiFi network.
        """
        if self._state == TransportState.STREAMING:
            await self.stop_stream()

        self._stop_keepalive_task()
        await self._close_session()

        # Restore previous WiFi if we managed the connection
        if self._wifi_managed_connection and self._wifi_manager is not None:
            log.info("Restoring previous WiFi connection...")
            try:
                await self._wifi_manager.disconnect(restore_previous=True)
            except Exception as e:
                log.warning("Failed to restore previous WiFi: %s", e)
            self._wifi_managed_connection = False

        self._stream_info = None
        self._update_uptime()
        self._set_state(TransportState.DISCONNECTED)
        log.info("WiFi AP transport disconnected")

    async def keep_alive(self) -> bool:
        """Send a keep-alive ping to the camera.

        Queries webcam_status as a lightweight ping. Also sends the
        dedicated keep_alive endpoint to prevent camera sleep.

        Returns True if the camera responded.
        """
        if not self.is_connected or self._session is None:
            return False

        try:
            # Use the dedicated keep-alive endpoint
            status, _ = await self._http_get(
                EP_KEEP_ALIVE,
                timeout=HEALTH_CHECK_TIMEOUT_S,
            )

            if status is not None and status == 200:
                self._stats.keepalives_sent += 1
                return True
            else:
                # Fallback: try webcam_status as keep-alive
                status2, _ = await self._http_get(
                    EP_WEBCAM_STATUS,
                    timeout=HEALTH_CHECK_TIMEOUT_S,
                )
                if status2 is not None and status2 == 200:
                    self._stats.keepalives_sent += 1
                    return True

                log.warning("Keep-alive got non-200 response")
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

        Verifies the webcam status endpoint responds and that the
        camera is in the expected state.
        """
        if not self.is_connected or self._session is None:
            return False

        try:
            status, data = await self._http_get(
                EP_WEBCAM_STATUS,
                timeout=HEALTH_CHECK_TIMEOUT_S,
            )
            if status is None or status != 200 or data is None:
                return False

            # If we think we're streaming, verify camera agrees
            if self._state == TransportState.STREAMING:
                cam_status = data.get("status", WEBCAM_STATUS_OFF)
                if cam_status not in (
                    WEBCAM_STATUS_HIGH_POWER_PREVIEW,
                    WEBCAM_STATUS_LOW_POWER_PREVIEW,
                ):
                    log.warning(
                        "Health check: expected streaming but camera reports status=%d",
                        cam_status,
                    )
                    return False

            return True

        except Exception as e:
            log.warning("Health check failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _http_get(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        timeout: float = HTTP_TIMEOUT_S,
    ) -> tuple[Optional[int], Optional[dict]]:
        """Make an HTTP GET to the GoPro API.

        Returns (status_code, json_body) or (None, None) on failure.
        """
        if self._session is None:
            return None, None

        try:
            url = f"{self._base_url}{endpoint}"
            async with self._session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                try:
                    data = await resp.json(content_type=None)
                except Exception:
                    data = None
                return resp.status, data

        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            log.debug("HTTP GET %s failed: %s", endpoint, e)
            return None, None

    async def _poll_webcam_status(self, target_status: int) -> bool:
        """Poll webcam_status until we see the target status or timeout."""
        deadline = time.monotonic() + WEBCAM_STATUS_POLL_MAX_S
        while time.monotonic() < deadline:
            status, data = await self._http_get(
                EP_WEBCAM_STATUS,
                timeout=HEALTH_CHECK_TIMEOUT_S,
            )
            if status == 200 and data is not None:
                current = data.get("status", WEBCAM_STATUS_OFF)
                log.debug("Webcam status poll: %d (target: %d)", current, target_status)
                if current == target_status:
                    return True

            await asyncio.sleep(WEBCAM_STATUS_POLL_INTERVAL_S)

        return False

    async def _close_session(self) -> None:
        """Close the aiohttp session if open."""
        if self._session is not None:
            try:
                await self._session.close()
            except Exception as e:
                log.debug("Error closing HTTP session: %s", e)
            self._session = None

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
                    log.warning("Keep-alive failed — camera may be unreachable")

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

    @property
    def wifi_manager(self) -> Optional[WiFiManager]:
        """The WiFi manager used for OS-level WiFi operations."""
        return self._wifi_manager

    def _inject_session(self, session: Any) -> None:
        """For testing: inject a mock aiohttp.ClientSession."""
        self._session = session
        log.debug("Injected mock HTTP session")

    def _inject_base_url(self, base_url: str) -> None:
        """For testing: override the base URL."""
        self._base_url = base_url
        log.debug("Injected base URL: %s", base_url)

    def _inject_wifi_manager(self, wifi_manager: Any) -> None:
        """For testing: inject a mock WiFiManager."""
        self._wifi_manager = wifi_manager
        log.debug("Injected mock WiFi manager")
