"""
Tests for WiFi AP transport layer — mocked at the HTTP boundary.

No real hardware or WiFi connection required.  These tests verify:
  - Discovery via HTTP probe to GoPro at 10.5.5.9:8080
  - State machine transitions (DISCONNECTED → DISCOVERING → CONNECTED → STREAMING)
  - HTTP API calls to GoPro at 10.5.5.9:8080
  - Stream start/stop lifecycle with proper API endpoints
  - Keep-alive behavior and failure detection
  - Health check validation (including streaming status verification)
  - WiFi disconnection handling (timeouts, OSError, connection drops)
  - Full lifecycle: discover → connect → stream → stop → disconnect
  - Mid-stream WiFi loss and recovery
  - Error handling and timeout paths
  - Configuration validation (resolution / FOV guards)
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gomaxwebcam.transport.base import Transport, TransportState, StreamInfo
from gomaxwebcam.transport.wifi_ap import (
    WiFiAPTransport,
    GOPRO_AP_IP,
    GOPRO_HTTP_PORT,
    GOPRO_UDP_PORT,
    RES_480,
    RES_1080,
    RES_720,
    FOV_WIDE,
    FOV_LINEAR,
    WEBCAM_STATUS_HIGH_POWER_PREVIEW,
    WEBCAM_STATUS_LOW_POWER_PREVIEW,
    WEBCAM_STATUS_IDLE,
    WEBCAM_STATUS_OFF,
    WEBCAM_ERROR_SUCCESS,
    ALLOWED_RESOLUTIONS,
    ALLOWED_FOVS,
    EP_WEBCAM_START,
    EP_WEBCAM_STOP,
    EP_WEBCAM_EXIT,
    EP_WEBCAM_STATUS,
    EP_KEEP_ALIVE,
)

# All tests in this module are unit tests — no GoPro hardware needed
pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Mock HTTP response helper
# ---------------------------------------------------------------------------


class MockResponse:
    """Simulates an aiohttp response with async context manager support."""

    def __init__(self, status: int = 200, json_data: Any = None):
        self.status = status
        self._json_data = json_data if json_data is not None else {}

    async def json(self, content_type=None):
        return self._json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _RaisingResponse:
    """A fake response context manager that raises an exception on enter."""

    def __init__(self, exc: Exception):
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, *args):
        pass


class MockSession:
    """Simulates aiohttp.ClientSession with GET tracking.

    Responses can be MockResponse instances or Exception instances.
    If an Exception is mapped to an endpoint, the GET will raise it.
    """

    def __init__(self, responses: dict[str, MockResponse | Exception] | None = None):
        self._responses = responses or {}
        self._default_response = MockResponse(200, {})
        self.get_calls: list[tuple[str, dict]] = []
        self.closed = False

    def _find_response(self, url: str) -> MockResponse | Exception:
        """Match URL to a response (checks endpoint suffix)."""
        for pattern, resp in self._responses.items():
            if pattern in url:
                return resp
        return self._default_response

    def get(self, url: str, **kwargs) -> MockResponse | _RaisingResponse:
        self.get_calls.append((url, kwargs))
        resp = self._find_response(url)
        if isinstance(resp, Exception):
            return _RaisingResponse(resp)
        return resp

    async def close(self):
        self.closed = True


def _make_status_response(
    status: int = WEBCAM_STATUS_HIGH_POWER_PREVIEW,
    error: int = WEBCAM_ERROR_SUCCESS,
) -> MockResponse:
    """Create a webcam_status response."""
    return MockResponse(200, {"status": status, "error": error})


def _make_start_response(
    error: int = WEBCAM_ERROR_SUCCESS,
) -> MockResponse:
    """Create a webcam_start response."""
    return MockResponse(200, {"error": error})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def transport():
    """Create a WiFiAPTransport for testing."""
    return WiFiAPTransport(
        camera_ip=GOPRO_AP_IP,
        http_port=GOPRO_HTTP_PORT,
        udp_port=GOPRO_UDP_PORT,
    )


@pytest.fixture
def mock_session():
    """Create a mock session with default GoPro responses."""
    return MockSession(responses={
        EP_WEBCAM_STATUS: _make_status_response(),
        EP_WEBCAM_START: _make_start_response(),
        EP_WEBCAM_STOP: MockResponse(200, {}),
        EP_WEBCAM_EXIT: MockResponse(200, {}),
        EP_KEEP_ALIVE: MockResponse(200, {}),
    })


@pytest.fixture
def connected_transport(transport, mock_session):
    """Create a transport that's already in CONNECTED state."""
    transport._inject_session(mock_session)
    transport._set_state(TransportState.CONNECTED)
    transport._stats.camera_model = f"GoPro@{GOPRO_AP_IP} (WiFi AP)"
    transport._stats.camera_serial = "wifi-ap"
    return transport


# ---------------------------------------------------------------------------
# Tests: Init
# ---------------------------------------------------------------------------


class TestWiFiAPTransportInit:
    """Construction and default values."""

    def test_initial_state_is_disconnected(self, transport):
        assert transport.state == TransportState.DISCONNECTED

    def test_default_name(self, transport):
        assert transport.name == "WiFi AP"

    def test_not_connected_initially(self, transport):
        assert not transport.is_connected
        assert not transport.is_streaming

    def test_stream_info_none_initially(self, transport):
        assert transport.stream_info is None

    def test_default_camera_ip(self, transport):
        assert transport.camera_ip == "10.5.5.9"

    def test_default_http_port(self, transport):
        assert transport.http_port == 8080

    def test_default_udp_port(self, transport):
        assert transport.udp_port == 8554

    def test_custom_camera_ip(self):
        t = WiFiAPTransport(camera_ip="192.168.1.100")
        assert t.camera_ip == "192.168.1.100"

    def test_is_subclass_of_transport_abc(self, transport):
        assert isinstance(transport, Transport)

    def test_stats_transport_type(self, transport):
        assert transport.stats.transport_type == "WiFi AP"


# ---------------------------------------------------------------------------
# Tests: Discover
# ---------------------------------------------------------------------------


class TestWiFiAPTransportDiscover:
    """Discovery phase tests (pinging GoPro at 10.5.5.9)."""

    @pytest.mark.anyio
    async def test_discover_success(self, transport, mock_session):
        transport._inject_session(mock_session)
        result = await transport.discover(timeout=2.0)
        assert result is True

    @pytest.mark.anyio
    async def test_discover_sets_discovering_state(self, transport, mock_session):
        states: list = []
        transport.add_state_listener(lambda old, new: states.append((old, new)))

        transport._inject_session(mock_session)
        await transport.discover()

        assert (TransportState.DISCONNECTED, TransportState.DISCOVERING) in states

    @pytest.mark.anyio
    async def test_discover_http_error(self, transport):
        session = MockSession(responses={
            EP_WEBCAM_STATUS: MockResponse(500, {"error": "internal"}),
        })
        transport._inject_session(session)
        result = await transport.discover()
        assert result is False
        assert transport.state == TransportState.ERROR

    @pytest.mark.anyio
    async def test_discover_network_error(self, transport):
        """If the camera is unreachable, discover returns False."""
        import aiohttp

        session = MagicMock()
        session.get = MagicMock(side_effect=aiohttp.ClientError("Connection refused"))
        session.close = AsyncMock()
        transport._inject_session(session)

        result = await transport.discover()
        assert result is False
        assert transport.state == TransportState.ERROR

    @pytest.mark.anyio
    async def test_discover_timeout_gopro_out_of_range(self, transport):
        """Discovery fails on timeout (GoPro out of WiFi range)."""
        session = MockSession(responses={
            EP_WEBCAM_STATUS: asyncio.TimeoutError(),
        })
        transport._inject_session(session)

        result = await transport.discover(timeout=2.0)
        assert result is False
        assert transport.state == TransportState.ERROR

    @pytest.mark.anyio
    async def test_discover_os_error_wifi_adapter_down(self, transport):
        """Discovery fails on OSError (WiFi adapter disabled/down)."""
        session = MockSession(responses={
            EP_WEBCAM_STATUS: OSError("Network is unreachable"),
        })
        transport._inject_session(session)

        result = await transport.discover(timeout=2.0)
        assert result is False
        assert transport.state == TransportState.ERROR

    @pytest.mark.anyio
    async def test_discover_connection_refused(self, transport):
        """Discovery fails when connection refused (wrong network)."""
        session = MockSession(responses={
            EP_WEBCAM_STATUS: ConnectionRefusedError("Connection refused"),
        })
        transport._inject_session(session)

        result = await transport.discover(timeout=2.0)
        assert result is False
        assert transport.state == TransportState.ERROR


# ---------------------------------------------------------------------------
# Tests: Connect
# ---------------------------------------------------------------------------


class TestWiFiAPTransportConnect:
    """Connection phase tests."""

    @pytest.mark.anyio
    async def test_connect_success(self, transport):
        """Connect should create a session and verify the API."""
        # Patch aiohttp.ClientSession to return our mock
        mock_resp = _make_status_response()
        mock_session = MockSession(responses={
            EP_WEBCAM_STATUS: mock_resp,
        })

        with patch("gomaxwebcam.transport.wifi_ap.aiohttp.ClientSession", return_value=mock_session):
            result = await transport.connect()

        assert result is True
        assert transport.state == TransportState.CONNECTED
        assert transport._session is mock_session

    @pytest.mark.anyio
    async def test_connect_records_camera_info(self, transport):
        mock_session = MockSession(responses={
            EP_WEBCAM_STATUS: _make_status_response(),
        })

        with patch("gomaxwebcam.transport.wifi_ap.aiohttp.ClientSession", return_value=mock_session):
            await transport.connect()

        assert transport.stats.camera_serial == "wifi-ap"
        assert "10.5.5.9" in transport.stats.camera_model
        assert "WiFi AP" in transport.stats.camera_model

    @pytest.mark.anyio
    async def test_connect_api_error(self, transport):
        """If the API returns non-200, connect should fail."""
        mock_session = MockSession(responses={
            EP_WEBCAM_STATUS: MockResponse(503, {}),
        })

        with patch("gomaxwebcam.transport.wifi_ap.aiohttp.ClientSession", return_value=mock_session):
            result = await transport.connect()

        assert result is False
        assert transport.state == TransportState.ERROR

    @pytest.mark.anyio
    async def test_connect_timeout(self, transport):
        """Connect fails on timeout (GoPro slow to respond over WiFi)."""
        session = MockSession(responses={
            EP_WEBCAM_STATUS: asyncio.TimeoutError(),
        })

        with patch("gomaxwebcam.transport.wifi_ap.aiohttp.ClientSession", return_value=session):
            result = await transport.connect()

        assert result is False
        assert transport.state == TransportState.ERROR
        assert "timeout" in (transport.stats.last_error or "").lower()

    @pytest.mark.anyio
    async def test_connect_network_error_wifi_drop(self, transport):
        """Connect fails on OSError (WiFi dropped mid-handshake)."""
        session = MockSession(responses={
            EP_WEBCAM_STATUS: OSError("Network is unreachable"),
        })

        with patch("gomaxwebcam.transport.wifi_ap.aiohttp.ClientSession", return_value=session):
            result = await transport.connect()

        assert result is False
        assert transport.state == TransportState.ERROR

    @pytest.mark.anyio
    async def test_connect_state_transitions(self, transport):
        states: list = []
        transport.add_state_listener(lambda old, new: states.append(new))

        mock_session = MockSession(responses={
            EP_WEBCAM_STATUS: _make_status_response(),
        })

        with patch("gomaxwebcam.transport.wifi_ap.aiohttp.ClientSession", return_value=mock_session):
            await transport.connect()

        assert TransportState.CONNECTING in states
        assert TransportState.CONNECTED in states


# ---------------------------------------------------------------------------
# Tests: Start Stream
# ---------------------------------------------------------------------------


class TestWiFiAPTransportStream:
    """Stream start/stop tests."""

    @pytest.mark.anyio
    async def test_start_stream_not_connected(self, transport):
        result = await transport.start_stream()
        assert result is None

    @pytest.mark.anyio
    async def test_start_stream_success(self, connected_transport, mock_session):
        info = await connected_transport.start_stream()

        assert info is not None
        assert info.port == GOPRO_UDP_PORT
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 30
        assert info.protocol == "udp"
        assert info.codec == "h264"
        assert connected_transport.state == TransportState.STREAMING
        assert connected_transport.is_streaming

    @pytest.mark.anyio
    async def test_start_stream_calls_webcam_start_endpoint(self, connected_transport, mock_session):
        await connected_transport.start_stream()

        # Verify webcam_start was called
        start_calls = [
            (url, kw) for url, kw in mock_session.get_calls
            if EP_WEBCAM_START in url
        ]
        assert len(start_calls) >= 1

        # Verify params include resolution, fov, port, protocol
        url, kwargs = start_calls[0]
        params = kwargs.get("params", {})
        assert params.get("res") == str(RES_1080)
        assert params.get("fov") == str(FOV_WIDE)
        assert params.get("port") == str(GOPRO_UDP_PORT)
        assert params.get("protocol") == "TS"

    @pytest.mark.anyio
    async def test_start_stream_sets_stream_info(self, connected_transport, mock_session):
        info = await connected_transport.start_stream()

        assert connected_transport.stream_info is info
        assert connected_transport.stream_info is not None

    @pytest.mark.anyio
    async def test_start_stream_webcam_error(self, connected_transport):
        """If webcam_start returns an error, start_stream should return None."""
        session = MockSession(responses={
            EP_WEBCAM_START: MockResponse(200, {"error": 7}),  # UNAVAILABLE
            EP_WEBCAM_STATUS: _make_status_response(WEBCAM_STATUS_OFF),
            EP_WEBCAM_STOP: MockResponse(200, {}),
        })
        connected_transport._inject_session(session)

        info = await connected_transport.start_stream()
        assert info is None
        assert not connected_transport.is_streaming

    @pytest.mark.anyio
    async def test_start_stream_http_failure(self, connected_transport):
        """If webcam_start returns non-200, start_stream should return None."""
        session = MockSession(responses={
            EP_WEBCAM_START: MockResponse(500, {}),
            EP_WEBCAM_STATUS: _make_status_response(),
        })
        connected_transport._inject_session(session)

        info = await connected_transport.start_stream()
        assert info is None

    @pytest.mark.anyio
    async def test_start_stream_status_timeout(self, connected_transport):
        """If webcam never reaches STREAMING, start_stream should return None."""
        session = MockSession(responses={
            EP_WEBCAM_START: _make_start_response(),
            EP_WEBCAM_STATUS: _make_status_response(WEBCAM_STATUS_OFF),
            EP_WEBCAM_STOP: MockResponse(200, {}),
        })
        connected_transport._inject_session(session)

        # Patch the poll timeout to be very short
        with patch("gomaxwebcam.transport.wifi_ap.WEBCAM_STATUS_POLL_MAX_S", 0.5), \
             patch("gomaxwebcam.transport.wifi_ap.WEBCAM_STATUS_POLL_INTERVAL_S", 0.1):
            info = await connected_transport.start_stream()

        assert info is None

    @pytest.mark.anyio
    async def test_stop_stream(self, connected_transport, mock_session):
        await connected_transport.start_stream()
        await connected_transport.stop_stream()

        assert connected_transport.state == TransportState.CONNECTED
        assert connected_transport.stream_info is None

        # Verify stop + exit were called
        stop_calls = [url for url, _ in mock_session.get_calls if EP_WEBCAM_STOP in url]
        exit_calls = [url for url, _ in mock_session.get_calls if EP_WEBCAM_EXIT in url]
        assert len(stop_calls) >= 1
        assert len(exit_calls) >= 1

    @pytest.mark.anyio
    async def test_stop_stream_when_not_streaming_is_noop(self, connected_transport, mock_session):
        # Not streaming — stop should be a no-op
        await connected_transport.stop_stream()

        stop_calls = [url for url, _ in mock_session.get_calls if EP_WEBCAM_STOP in url]
        assert len(stop_calls) == 0

    @pytest.mark.anyio
    async def test_start_stream_connection_drop(self, connected_transport):
        """start_stream handles WiFi drop during webcam_start call."""
        session = MockSession(responses={
            EP_WEBCAM_START: OSError("Network unreachable"),
            EP_WEBCAM_STATUS: _make_status_response(),
        })
        connected_transport._inject_session(session)

        info = await connected_transport.start_stream()
        assert info is None

    @pytest.mark.anyio
    async def test_stream_720p(self, connected_transport, mock_session):
        connected_transport.resolution = RES_720
        info = await connected_transport.start_stream()

        assert info is not None
        assert info.width == 1280
        assert info.height == 720

    @pytest.mark.anyio
    async def test_stream_480p(self, connected_transport, mock_session):
        connected_transport.resolution = RES_480
        info = await connected_transport.start_stream()

        assert info is not None
        assert info.width == 854
        assert info.height == 480

    @pytest.mark.anyio
    async def test_start_stream_returns_correct_stream_info_fields(self, connected_transport, mock_session):
        """StreamInfo from WiFi AP matches expected frozen dataclass."""
        info = await connected_transport.start_stream()
        assert info == StreamInfo(
            protocol="udp", host="0.0.0.0", port=8554,
            width=1920, height=1080, fps=30, codec="h264",
        )

    @pytest.mark.anyio
    async def test_stream_info_is_frozen(self, connected_transport, mock_session):
        """StreamInfo is immutable."""
        info = await connected_transport.start_stream()
        with pytest.raises(AttributeError):
            info.port = 9999  # type: ignore


# ---------------------------------------------------------------------------
# Tests: Disconnect
# ---------------------------------------------------------------------------


class TestWiFiAPTransportDisconnect:
    """Disconnect and cleanup tests."""

    @pytest.mark.anyio
    async def test_disconnect_from_connected(self, connected_transport, mock_session):
        await connected_transport.disconnect()

        assert connected_transport.state == TransportState.DISCONNECTED
        assert mock_session.closed

    @pytest.mark.anyio
    async def test_disconnect_from_streaming(self, connected_transport, mock_session):
        await connected_transport.start_stream()
        await connected_transport.disconnect()

        assert connected_transport.state == TransportState.DISCONNECTED
        assert mock_session.closed

        # Should have stopped stream first
        stop_calls = [url for url, _ in mock_session.get_calls if EP_WEBCAM_STOP in url]
        assert len(stop_calls) >= 1

    @pytest.mark.anyio
    async def test_disconnect_clears_session(self, connected_transport, mock_session):
        await connected_transport.disconnect()
        assert connected_transport._session is None

    @pytest.mark.anyio
    async def test_disconnect_updates_uptime(self, connected_transport, mock_session):
        import time
        connected_transport._connected_at = time.monotonic()
        await asyncio.sleep(0.05)
        await connected_transport.disconnect()

        assert connected_transport.stats.connection_uptime_s > 0

    @pytest.mark.anyio
    async def test_disconnect_clears_stream_info(self, connected_transport, mock_session):
        """disconnect clears stream_info."""
        await connected_transport.start_stream()
        assert connected_transport.stream_info is not None
        await connected_transport.disconnect()
        assert connected_transport.stream_info is None

    @pytest.mark.anyio
    async def test_disconnect_idempotent(self, transport):
        """disconnect from DISCONNECTED state is safe (no-op)."""
        await transport.disconnect()
        assert transport.state == TransportState.DISCONNECTED


# ---------------------------------------------------------------------------
# Tests: WiFi disconnection handling
# ---------------------------------------------------------------------------


class TestWiFiAPDisconnectionHandling:
    """Tests verifying WiFi AP transport handles network disruptions.

    These simulate real WiFi AP scenarios:
    - GoPro hotspot going out of range
    - Camera power off / sleep
    - WiFi interference causing connection drops
    - Mid-stream network loss
    """

    @pytest.mark.anyio
    async def test_keep_alive_detects_wifi_loss(self, connected_transport):
        """keep_alive returns False when WiFi connection drops (OSError)."""
        session = MockSession(responses={
            EP_KEEP_ALIVE: OSError("Network is unreachable"),
            EP_WEBCAM_STATUS: OSError("Network is unreachable"),
        })
        connected_transport._inject_session(session)

        result = await connected_transport.keep_alive()

        assert result is False
        assert connected_transport.stats.keepalives_failed >= 1

    @pytest.mark.anyio
    async def test_keep_alive_detects_timeout(self, connected_transport):
        """keep_alive returns False when GoPro stops responding (timeout)."""
        session = MockSession(responses={
            EP_KEEP_ALIVE: asyncio.TimeoutError(),
            EP_WEBCAM_STATUS: asyncio.TimeoutError(),
        })
        connected_transport._inject_session(session)

        result = await connected_transport.keep_alive()

        assert result is False
        assert connected_transport.stats.keepalives_failed >= 1

    @pytest.mark.anyio
    async def test_keep_alive_detects_connection_reset(self, connected_transport):
        """keep_alive returns False on ConnectionResetError (camera power off)."""
        session = MockSession(responses={
            EP_KEEP_ALIVE: ConnectionResetError("Connection reset by peer"),
            EP_WEBCAM_STATUS: ConnectionResetError("Connection reset by peer"),
        })
        connected_transport._inject_session(session)

        result = await connected_transport.keep_alive()

        assert result is False
        assert connected_transport.stats.keepalives_failed >= 1

    @pytest.mark.anyio
    async def test_health_check_fails_on_wifi_loss(self, connected_transport):
        """health_check returns False when WiFi AP is lost."""
        session = MockSession(responses={
            EP_WEBCAM_STATUS: OSError("No route to host"),
        })
        connected_transport._inject_session(session)

        result = await connected_transport.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_health_check_not_connected_returns_false(self, transport):
        """health_check returns False when not connected."""
        result = await transport.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_health_check_streaming_status_low_power_ok(self, connected_transport, mock_session):
        """health_check accepts LOW_POWER_PREVIEW as valid streaming state."""
        await connected_transport.start_stream()

        lp_session = MockSession(responses={
            EP_WEBCAM_STATUS: _make_status_response(WEBCAM_STATUS_LOW_POWER_PREVIEW),
        })
        connected_transport._inject_session(lp_session)

        result = await connected_transport.health_check()
        assert result is True

    @pytest.mark.anyio
    async def test_stream_start_wifi_drop_during_status_poll(self, connected_transport):
        """start_stream handles WiFi drop during status polling gracefully."""
        session = MockSession(responses={
            EP_WEBCAM_START: _make_start_response(),
            # Status always returns OFF — camera never reaches streaming
            EP_WEBCAM_STATUS: _make_status_response(WEBCAM_STATUS_OFF),
            EP_WEBCAM_STOP: MockResponse(200, {}),
        })
        connected_transport._inject_session(session)

        with patch("gomaxwebcam.transport.wifi_ap.WEBCAM_STATUS_POLL_MAX_S", 0.3), \
             patch("gomaxwebcam.transport.wifi_ap.WEBCAM_STATUS_POLL_INTERVAL_S", 0.05):
            info = await connected_transport.start_stream()

        assert info is None
        assert "timeout" in (connected_transport.stats.last_error or "").lower()

    @pytest.mark.anyio
    async def test_stop_stream_survives_network_error(self, connected_transport, mock_session):
        """stop_stream doesn't raise even if WiFi AP is gone."""
        await connected_transport.start_stream()

        # Replace with a broken session for the stop phase
        broken_session = MockSession(responses={
            EP_WEBCAM_STOP: OSError("Network unreachable"),
            EP_WEBCAM_EXIT: OSError("Network unreachable"),
        })
        connected_transport._inject_session(broken_session)

        # Should not raise — must handle gracefully
        await connected_transport.stop_stream()
        assert connected_transport.state == TransportState.CONNECTED

    @pytest.mark.anyio
    async def test_disconnect_survives_network_error(self, connected_transport, mock_session):
        """disconnect doesn't raise even if WiFi network is gone during cleanup."""
        await connected_transport.start_stream()

        broken_session = MockSession(responses={
            EP_WEBCAM_STOP: OSError("Connection refused"),
            EP_WEBCAM_EXIT: OSError("Connection refused"),
        })
        connected_transport._inject_session(broken_session)

        await connected_transport.disconnect()
        assert connected_transport.state == TransportState.DISCONNECTED

    @pytest.mark.anyio
    async def test_consecutive_keepalive_failures_tracked(self, connected_transport):
        """Multiple consecutive keep-alive failures increment the failure counter."""
        session = MockSession(responses={
            EP_KEEP_ALIVE: OSError("WiFi lost"),
            EP_WEBCAM_STATUS: OSError("WiFi lost"),
        })
        connected_transport._inject_session(session)

        for _ in range(5):
            await connected_transport.keep_alive()

        assert connected_transport.stats.keepalives_failed >= 5


# ---------------------------------------------------------------------------
# Tests: Keep-alive
# ---------------------------------------------------------------------------


class TestWiFiAPTransportKeepAlive:
    """Keep-alive and health check tests."""

    @pytest.mark.anyio
    async def test_keep_alive_success(self, connected_transport, mock_session):
        result = await connected_transport.keep_alive()
        assert result is True
        assert connected_transport.stats.keepalives_sent == 1

    @pytest.mark.anyio
    async def test_keep_alive_not_connected(self, transport):
        result = await transport.keep_alive()
        assert result is False

    @pytest.mark.anyio
    async def test_keep_alive_failure(self, connected_transport):
        session = MockSession(responses={
            EP_KEEP_ALIVE: MockResponse(500, {}),
            EP_WEBCAM_STATUS: MockResponse(500, {}),
        })
        connected_transport._inject_session(session)

        result = await connected_transport.keep_alive()
        assert result is False
        assert connected_transport.stats.keepalives_failed == 1

    @pytest.mark.anyio
    async def test_health_check_streaming(self, connected_transport, mock_session):
        await connected_transport.start_stream()
        result = await connected_transport.health_check()
        assert result is True

    @pytest.mark.anyio
    async def test_health_check_wrong_status(self, connected_transport, mock_session):
        """If we're streaming but camera says OFF, health check should fail."""
        await connected_transport.start_stream()

        # Swap to a session that reports OFF status
        bad_session = MockSession(responses={
            EP_WEBCAM_STATUS: _make_status_response(WEBCAM_STATUS_OFF),
        })
        connected_transport._inject_session(bad_session)

        result = await connected_transport.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_keep_alive_no_session(self, transport):
        """keep_alive returns False when no session exists."""
        transport._set_state(TransportState.CONNECTED)
        result = await transport.keep_alive()
        assert result is False

    @pytest.mark.anyio
    async def test_keep_alive_increments_sent_counter(self, connected_transport, mock_session):
        """Multiple successful keep-alives increment the sent counter."""
        await connected_transport.keep_alive()
        await connected_transport.keep_alive()
        await connected_transport.keep_alive()

        assert connected_transport.stats.keepalives_sent == 3

    @pytest.mark.anyio
    async def test_keep_alive_fallback_to_webcam_status(self, connected_transport):
        """keep_alive falls back to webcam_status when keep_alive endpoint returns non-200."""
        session = MockSession(responses={
            EP_KEEP_ALIVE: MockResponse(500, {}),
            EP_WEBCAM_STATUS: _make_status_response(),  # Returns 200
        })
        connected_transport._inject_session(session)

        result = await connected_transport.keep_alive()
        # Should succeed via the fallback path
        assert result is True


# ---------------------------------------------------------------------------
# Tests: Configuration
# ---------------------------------------------------------------------------


class TestWiFiAPTransportConfig:
    """Configuration validation tests."""

    def test_set_valid_resolution(self, transport):
        transport.resolution = RES_720
        assert transport.resolution == RES_720

    def test_set_invalid_resolution_raises(self, transport):
        with pytest.raises(ValueError, match="not allowed"):
            transport.resolution = 99

    def test_set_valid_fov(self, transport):
        transport.fov = FOV_LINEAR
        assert transport.fov == FOV_LINEAR

    def test_set_invalid_fov_raises(self, transport):
        with pytest.raises(ValueError, match="not allowed"):
            transport.fov = 99

    def test_all_valid_resolutions(self, transport):
        for res in ALLOWED_RESOLUTIONS:
            transport.resolution = res
            assert transport.resolution == res

    def test_all_valid_fovs(self, transport):
        for fov in ALLOWED_FOVS:
            transport.fov = fov
            assert transport.fov == fov

    def test_camera_ip_property(self, transport):
        assert transport.camera_ip == "10.5.5.9"

    def test_http_port_property(self, transport):
        assert transport.http_port == GOPRO_HTTP_PORT

    def test_udp_port_property(self, transport):
        assert transport.udp_port == GOPRO_UDP_PORT


# ---------------------------------------------------------------------------
# Tests: State Listeners
# ---------------------------------------------------------------------------


class TestWiFiAPTransportStateListeners:
    """State listener callback tests."""

    @pytest.mark.anyio
    async def test_listener_fires_on_connect(self, transport):
        transitions: list = []
        transport.add_state_listener(lambda old, new: transitions.append((old, new)))

        mock_session = MockSession(responses={
            EP_WEBCAM_STATUS: _make_status_response(),
        })

        with patch("gomaxwebcam.transport.wifi_ap.aiohttp.ClientSession", return_value=mock_session):
            await transport.connect()

        assert (TransportState.DISCONNECTED, TransportState.CONNECTING) in transitions
        assert (TransportState.CONNECTING, TransportState.CONNECTED) in transitions

    @pytest.mark.anyio
    async def test_listener_fires_on_disconnect(self, connected_transport, mock_session):
        transitions: list = []
        connected_transport.add_state_listener(lambda old, new: transitions.append((old, new)))
        await connected_transport.disconnect()

        assert (TransportState.CONNECTED, TransportState.DISCONNECTED) in transitions

    @pytest.mark.anyio
    async def test_listener_fires_on_streaming(self, connected_transport, mock_session):
        """Listener receives transition to STREAMING."""
        transitions: list = []
        connected_transport.add_state_listener(
            lambda old, new: transitions.append((old, new))
        )

        await connected_transport.start_stream()

        assert (TransportState.CONNECTED, TransportState.STREAMING) in transitions

    @pytest.mark.anyio
    async def test_listener_exception_doesnt_crash(self, transport):
        """A broken listener should not prevent state transitions."""
        def bad_listener(old, new):
            raise RuntimeError("listener boom")

        transport.add_state_listener(bad_listener)

        mock_session = MockSession(responses={
            EP_WEBCAM_STATUS: _make_status_response(),
        })

        with patch("gomaxwebcam.transport.wifi_ap.aiohttp.ClientSession", return_value=mock_session):
            # Should not raise
            await transport.connect()


# ---------------------------------------------------------------------------
# Tests: Full lifecycle integration
# ---------------------------------------------------------------------------


class TestWiFiAPFullLifecycle:
    """End-to-end lifecycle: discover → connect → stream → stop → disconnect.

    These integration tests verify the complete WiFi AP transport workflow
    against the GoPro hotspot at 10.5.5.9.
    """

    @pytest.mark.anyio
    async def test_full_lifecycle(self, transport):
        """Complete happy path lifecycle over WiFi AP at 10.5.5.9."""
        states: list = []
        transport.add_state_listener(lambda old, new: states.append(new))

        # 1. Discover — GoPro responds at 10.5.5.9:8080
        discover_session = MockSession(responses={
            EP_WEBCAM_STATUS: _make_status_response(WEBCAM_STATUS_IDLE),
        })
        transport._inject_session(discover_session)
        discovered = await transport.discover(timeout=2.0)
        assert discovered is True

        # 2. Connect — HTTP session established
        connect_session = MockSession(responses={
            EP_WEBCAM_STATUS: _make_status_response(WEBCAM_STATUS_IDLE),
        })
        with patch("gomaxwebcam.transport.wifi_ap.aiohttp.ClientSession", return_value=connect_session):
            connected = await transport.connect()
        assert connected is True
        assert transport.state == TransportState.CONNECTED

        # 3. Start stream — webcam_start, poll until HIGH_POWER_PREVIEW
        stream_session = MockSession(responses={
            EP_WEBCAM_START: _make_start_response(),
            EP_WEBCAM_STATUS: _make_status_response(WEBCAM_STATUS_HIGH_POWER_PREVIEW),
            EP_KEEP_ALIVE: MockResponse(200, {}),
            EP_WEBCAM_STOP: MockResponse(200, {}),
            EP_WEBCAM_EXIT: MockResponse(200, {}),
        })
        transport._inject_session(stream_session)
        info = await transport.start_stream()
        assert info is not None
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 30
        assert transport.state == TransportState.STREAMING

        # 4. Keep-alive succeeds
        alive = await transport.keep_alive()
        assert alive is True

        # 5. Health check passes
        healthy = await transport.health_check()
        assert healthy is True

        # 6. Stop stream
        await transport.stop_stream()
        assert transport.state == TransportState.CONNECTED
        assert transport.stream_info is None

        # 7. Disconnect
        await transport.disconnect()
        assert transport.state == TransportState.DISCONNECTED

        # Verify complete state transition history
        assert TransportState.DISCOVERING in states
        assert TransportState.CONNECTING in states
        assert TransportState.CONNECTED in states
        assert TransportState.STREAMING in states
        assert TransportState.DISCONNECTED in states

    @pytest.mark.anyio
    async def test_lifecycle_with_mid_stream_wifi_loss(self, transport):
        """WiFi drops during streaming — verifies graceful degradation."""
        # Connect successfully
        connect_session = MockSession(responses={
            EP_WEBCAM_STATUS: _make_status_response(WEBCAM_STATUS_IDLE),
        })
        with patch("gomaxwebcam.transport.wifi_ap.aiohttp.ClientSession", return_value=connect_session):
            await transport.connect()

        # Start streaming
        stream_session = MockSession(responses={
            EP_WEBCAM_START: _make_start_response(),
            EP_WEBCAM_STATUS: _make_status_response(WEBCAM_STATUS_HIGH_POWER_PREVIEW),
            EP_KEEP_ALIVE: MockResponse(200, {}),
            EP_WEBCAM_STOP: MockResponse(200, {}),
            EP_WEBCAM_EXIT: MockResponse(200, {}),
        })
        transport._inject_session(stream_session)
        info = await transport.start_stream()
        assert info is not None
        assert transport.is_streaming

        # WiFi drops — all HTTP calls fail
        broken_session = MockSession(responses={
            EP_KEEP_ALIVE: OSError("Network unreachable"),
            EP_WEBCAM_STATUS: OSError("Network unreachable"),
            EP_WEBCAM_STOP: OSError("Network unreachable"),
            EP_WEBCAM_EXIT: OSError("Network unreachable"),
        })
        transport._inject_session(broken_session)

        # Keep-alive detects the loss
        ka_result = await transport.keep_alive()
        assert ka_result is False

        # Health check detects the loss
        hc_result = await transport.health_check()
        assert hc_result is False

        # Disconnect still works cleanly despite network being gone
        await transport.disconnect()
        assert transport.state == TransportState.DISCONNECTED
        assert transport.stream_info is None

    @pytest.mark.anyio
    async def test_reconnect_after_disconnect(self, transport):
        """Transport can reconnect after a clean disconnect cycle."""
        # First connection
        session1 = MockSession(responses={
            EP_WEBCAM_STATUS: _make_status_response(),
        })
        with patch("gomaxwebcam.transport.wifi_ap.aiohttp.ClientSession", return_value=session1):
            assert await transport.connect() is True

        await transport.disconnect()
        assert transport.state == TransportState.DISCONNECTED

        # Second connection — should work fine
        session2 = MockSession(responses={
            EP_WEBCAM_STATUS: _make_status_response(),
        })
        with patch("gomaxwebcam.transport.wifi_ap.aiohttp.ClientSession", return_value=session2):
            assert await transport.connect() is True

        assert transport.state == TransportState.CONNECTED

    @pytest.mark.anyio
    async def test_reconnect_after_error(self, transport):
        """Transport can reconnect after a failed connection attempt."""
        # First attempt fails
        fail_session = MockSession(responses={
            EP_WEBCAM_STATUS: OSError("Network unreachable"),
        })
        with patch("gomaxwebcam.transport.wifi_ap.aiohttp.ClientSession", return_value=fail_session):
            result = await transport.connect()
        assert result is False
        assert transport.state == TransportState.ERROR

        # Clean up from error state
        await transport.disconnect()

        # Second attempt succeeds
        ok_session = MockSession(responses={
            EP_WEBCAM_STATUS: _make_status_response(),
        })
        with patch("gomaxwebcam.transport.wifi_ap.aiohttp.ClientSession", return_value=ok_session):
            result = await transport.connect()
        assert result is True
        assert transport.state == TransportState.CONNECTED


# ---------------------------------------------------------------------------
# Tests: Stats
# ---------------------------------------------------------------------------


class TestWiFiAPTransportStats:
    """TransportStats tracking tests."""

    def test_initial_stats(self, transport):
        assert transport.stats.keepalives_sent == 0
        assert transport.stats.keepalives_failed == 0
        assert transport.stats.transport_type == "WiFi AP"
        assert transport.stats.last_error is None

    @pytest.mark.anyio
    async def test_keepalive_stats_increment(self, connected_transport, mock_session):
        await connected_transport.keep_alive()
        await connected_transport.keep_alive()
        assert connected_transport.stats.keepalives_sent == 2

    @pytest.mark.anyio
    async def test_stats_record_keepalive_failures(self, connected_transport):
        """Stats track failed keep-alive attempts from network errors."""
        session = MockSession(responses={
            EP_KEEP_ALIVE: OSError("WiFi lost"),
            EP_WEBCAM_STATUS: OSError("WiFi lost"),
        })
        connected_transport._inject_session(session)

        await connected_transport.keep_alive()

        assert connected_transport.stats.keepalives_failed >= 1
        assert connected_transport.stats.keepalives_sent == 0
