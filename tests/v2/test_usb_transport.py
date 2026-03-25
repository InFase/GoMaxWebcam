"""
Tests for USB transport layer — mocked at the open-gopro SDK boundary.

No real hardware required.  These tests verify:
  - State machine transitions (DISCONNECTED → DISCOVERING → CONNECTED → STREAMING)
  - Stream start/stop lifecycle with proper SDK calls
  - Keep-alive behavior and failure detection
  - Health check validation
  - Error handling and timeout paths
  - Configuration validation (resolution / FOV guards)
  - Edge cases: double connect, reconnect after error, concurrent operations
"""

from __future__ import annotations

import asyncio
import types
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mark all async tests to run with anyio (asyncio backend)
# Individual tests use @pytest.mark.anyio

# ---------------------------------------------------------------------------
# Mock SDK enums (so tests don't need open-gopro installed)
# ---------------------------------------------------------------------------


@dataclass
class MockWebcamResponse:
    status: Any = None
    error: Any = None


class MockWebcamStatus:
    OFF = 0
    IDLE = 1
    HIGH_POWER_PREVIEW = 2
    LOW_POWER_PREVIEW = 3


class MockWebcamError:
    SUCCESS = 0
    SHUTTER = 4
    UNAVAILABLE = 7


class MockWebcamResolution:
    NOT_APPLICABLE = 0
    RES_480 = 4
    RES_720 = 7
    RES_1080 = 12

    def __init__(self, val):
        self._val = val


class MockWebcamFOV:
    WIDE = 0
    NARROW = 2
    SUPERVIEW = 3
    LINEAR = 4

    def __init__(self, val):
        self._val = val


class MockWebcamProtocol:
    TS = "TS"
    RTSP = "RTSP"


class MockGoProResp:
    """Simulates open_gopro.GoProResp."""

    def __init__(self, ok: bool = True, data: Any = None):
        self.ok = ok
        self.data = data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_gopro() -> MagicMock:
    """Create a mock WiredGoPro with async command methods."""
    gopro = MagicMock()
    gopro.ip_address = "172.20.100.51"

    # open / close
    gopro.open = AsyncMock()
    gopro.close = AsyncMock()

    # http_command namespace
    cmd = MagicMock()
    cmd.webcam_start = AsyncMock(return_value=MockGoProResp(
        ok=True,
        data=MockWebcamResponse(
            status=MockWebcamStatus.IDLE,
            error=MockWebcamError.SUCCESS,
        ),
    ))
    cmd.webcam_stop = AsyncMock(return_value=MockGoProResp(ok=True))
    cmd.webcam_exit = AsyncMock(return_value=MockGoProResp(ok=True))
    cmd.webcam_status = AsyncMock(return_value=MockGoProResp(
        ok=True,
        data=MockWebcamResponse(
            status=MockWebcamStatus.HIGH_POWER_PREVIEW,
            error=MockWebcamError.SUCCESS,
        ),
    ))
    gopro.http_command = cmd

    return gopro


@pytest.fixture
def mock_gopro():
    return _make_mock_gopro()


@pytest.fixture
def transport():
    """Create a USBTransport without triggering imports."""
    from gomaxwebcam.transport.usb import USBTransport
    return USBTransport(serial="123", udp_port=8554)


# ---------------------------------------------------------------------------
# Patch the SDK imports inside usb.py
# ---------------------------------------------------------------------------


def _patch_sdk_enums():
    """Patch open_gopro.models.streaming enums for testing."""
    mock_streaming = types.ModuleType("open_gopro.models.streaming")
    mock_streaming.WebcamResolution = MockWebcamResolution
    mock_streaming.WebcamFOV = MockWebcamFOV
    mock_streaming.WebcamProtocol = MockWebcamProtocol
    mock_streaming.WebcamError = MockWebcamError
    mock_streaming.WebcamStatus = MockWebcamStatus
    return patch.dict("sys.modules", {
        "open_gopro.models.streaming": mock_streaming,
    })


# ---------------------------------------------------------------------------
# Tests — Initialization
# ---------------------------------------------------------------------------


class TestUSBTransportInit:
    """Construction and default values."""

    def test_initial_state_is_disconnected(self, transport):
        from gomaxwebcam.transport.base import TransportState
        assert transport.state == TransportState.DISCONNECTED

    def test_default_name(self, transport):
        assert transport.name == "USB"

    def test_not_connected_initially(self, transport):
        assert not transport.is_connected
        assert not transport.is_streaming

    def test_stream_info_none_initially(self, transport):
        assert transport.stream_info is None

    def test_gopro_none_initially(self, transport):
        assert transport._gopro is None

    def test_default_resolution_is_1080(self, transport):
        from gomaxwebcam.transport.usb import RES_1080
        assert transport.resolution == RES_1080

    def test_default_fov_is_wide(self, transport):
        from gomaxwebcam.transport.usb import FOV_WIDE
        assert transport.fov == FOV_WIDE

    def test_custom_serial(self):
        from gomaxwebcam.transport.usb import USBTransport
        t = USBTransport(serial="456")
        assert t.serial == "456"

    def test_none_serial(self):
        from gomaxwebcam.transport.usb import USBTransport
        t = USBTransport(serial=None)
        assert t.serial is None


# ---------------------------------------------------------------------------
# Tests — Discovery
# ---------------------------------------------------------------------------


class TestUSBTransportDiscover:
    """Discovery phase tests."""

    @pytest.mark.anyio
    async def test_discover_creates_sdk_handle(self, transport):
        mock_wired = MagicMock()
        with patch.dict("sys.modules", {"open_gopro": MagicMock(WiredGoPro=mock_wired)}):
            result = await transport.discover(timeout=5.0)
            assert result is True

    @pytest.mark.anyio
    async def test_discover_sets_discovering_state(self, transport):
        from gomaxwebcam.transport.base import TransportState

        states: list = []
        transport.add_state_listener(lambda old, new: states.append((old, new)))

        with patch.dict("sys.modules", {"open_gopro": MagicMock(WiredGoPro=MagicMock())}):
            await transport.discover()

        assert (TransportState.DISCONNECTED, TransportState.DISCOVERING) in states

    @pytest.mark.anyio
    async def test_discover_import_error(self, transport):
        """If open-gopro is not installed, discover should return False."""
        from gomaxwebcam.transport.base import TransportState

        import sys
        saved = sys.modules.get("open_gopro")
        sys.modules["open_gopro"] = None  # type: ignore
        try:
            result = await transport.discover()
            assert result is False or transport.state == TransportState.ERROR
        finally:
            if saved is not None:
                sys.modules["open_gopro"] = saved
            else:
                sys.modules.pop("open_gopro", None)


# ---------------------------------------------------------------------------
# Tests — Connect
# ---------------------------------------------------------------------------


class TestUSBTransportConnect:
    """Connection phase tests."""

    @pytest.mark.anyio
    async def test_connect_without_discover_fails(self, transport):
        result = await transport.connect()
        assert result is False

    @pytest.mark.anyio
    async def test_connect_success(self, transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        transport._inject_gopro(mock_gopro)
        result = await transport.connect()

        assert result is True
        assert transport.state == TransportState.CONNECTED
        mock_gopro.open.assert_awaited_once()

    @pytest.mark.anyio
    async def test_connect_records_camera_info(self, transport, mock_gopro):
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        assert transport.stats.camera_serial is not None
        assert "172.20.100.51" in transport.stats.camera_model

    @pytest.mark.anyio
    async def test_connect_timeout(self, transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        mock_gopro.open = AsyncMock(side_effect=asyncio.TimeoutError)
        transport._inject_gopro(mock_gopro)

        result = await transport.connect()
        assert result is False
        assert transport.state == TransportState.ERROR

    @pytest.mark.anyio
    async def test_connect_exception(self, transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        mock_gopro.open = AsyncMock(side_effect=ConnectionError("USB disconnected"))
        transport._inject_gopro(mock_gopro)

        result = await transport.connect()
        assert result is False
        assert transport.state == TransportState.ERROR
        assert "USB disconnected" in transport.stats.last_error

    @pytest.mark.anyio
    async def test_state_transitions_on_connect(self, transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        states: list = []
        transport.add_state_listener(lambda old, new: states.append(new))

        transport._inject_gopro(mock_gopro)
        await transport.connect()

        assert TransportState.CONNECTING in states
        assert TransportState.CONNECTED in states

    @pytest.mark.anyio
    async def test_connect_os_error(self, transport, mock_gopro):
        """OSError (e.g., network adapter down) should fail gracefully."""
        from gomaxwebcam.transport.base import TransportState

        mock_gopro.open = AsyncMock(side_effect=OSError("Network adapter not found"))
        transport._inject_gopro(mock_gopro)

        result = await transport.connect()
        assert result is False
        assert transport.state == TransportState.ERROR
        assert "Network adapter" in transport.stats.last_error

    @pytest.mark.anyio
    async def test_connect_runtime_error(self, transport, mock_gopro):
        """RuntimeError from SDK should fail gracefully."""
        from gomaxwebcam.transport.base import TransportState

        mock_gopro.open = AsyncMock(side_effect=RuntimeError("SDK internal error"))
        transport._inject_gopro(mock_gopro)

        result = await transport.connect()
        assert result is False
        assert transport.state == TransportState.ERROR
        assert "SDK internal" in transport.stats.last_error

    @pytest.mark.anyio
    async def test_connect_sets_connected_at_timestamp(self, transport, mock_gopro):
        """Connected timestamp should be set for uptime tracking."""
        transport._inject_gopro(mock_gopro)
        await transport.connect()
        assert transport._connected_at > 0

    @pytest.mark.anyio
    async def test_connect_extracts_camera_ip(self, transport, mock_gopro):
        """Camera IP should be extracted from the SDK handle."""
        mock_gopro.ip_address = "172.20.100.99"
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        assert transport._camera_ip == "172.20.100.99"

    @pytest.mark.anyio
    async def test_connect_without_ip_address_attr(self, transport, mock_gopro):
        """If SDK handle has no ip_address, use 'unknown'."""
        del mock_gopro.ip_address
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        assert transport._camera_ip == "unknown"

    @pytest.mark.anyio
    async def test_reconnect_after_error(self, transport, mock_gopro):
        """Should be able to reconnect after an error state."""
        from gomaxwebcam.transport.base import TransportState

        # First connect fails
        mock_gopro.open = AsyncMock(side_effect=ConnectionError("fail"))
        transport._inject_gopro(mock_gopro)
        result = await transport.connect()
        assert result is False
        assert transport.state == TransportState.ERROR

        # Second connect succeeds
        mock_gopro.open = AsyncMock()
        result = await transport.connect()
        assert result is True
        assert transport.state == TransportState.CONNECTED

    @pytest.mark.anyio
    async def test_connect_cancelled(self, transport, mock_gopro):
        """CancelledError should propagate (not be swallowed)."""
        mock_gopro.open = AsyncMock(side_effect=asyncio.CancelledError)
        transport._inject_gopro(mock_gopro)

        # CancelledError may be caught as Exception or propagate
        # Either is acceptable as long as it doesn't leave transport in bad state
        try:
            result = await transport.connect()
            # If caught, should indicate failure
            assert result is False
        except asyncio.CancelledError:
            pass  # Propagation is also acceptable


# ---------------------------------------------------------------------------
# Tests — Stream Start/Stop
# ---------------------------------------------------------------------------


class TestUSBTransportStream:
    """Stream start/stop tests."""

    @pytest.mark.anyio
    async def test_start_stream_not_connected(self, transport):
        result = await transport.start_stream()
        assert result is None

    @pytest.mark.anyio
    async def test_start_stream_success(self, transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            info = await transport.start_stream()

        assert info is not None
        assert info.port == 8554
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 30
        assert info.protocol == "udp"
        assert transport.state == TransportState.STREAMING
        assert transport.is_streaming

    @pytest.mark.anyio
    async def test_start_stream_calls_sdk_webcam_start(self, transport, mock_gopro):
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            await transport.start_stream()

        mock_gopro.http_command.webcam_start.assert_awaited_once()

    @pytest.mark.anyio
    async def test_start_stream_sets_stream_info(self, transport, mock_gopro):
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            info = await transport.start_stream()

        assert transport.stream_info is info
        assert transport.stream_info is not None

    @pytest.mark.anyio
    async def test_start_stream_webcam_error(self, transport, mock_gopro):
        """If webcam_start returns an error, start_stream should return None."""
        mock_gopro.http_command.webcam_start = AsyncMock(return_value=MockGoProResp(
            ok=True,
            data=MockWebcamResponse(
                status=MockWebcamStatus.OFF,
                error=MockWebcamError.UNAVAILABLE,
            ),
        ))
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            info = await transport.start_stream()

        assert info is None
        assert not transport.is_streaming

    @pytest.mark.anyio
    async def test_start_stream_sdk_failure(self, transport, mock_gopro):
        """If webcam_start returns ok=False, start_stream should return None."""
        mock_gopro.http_command.webcam_start = AsyncMock(
            return_value=MockGoProResp(ok=False, data=None)
        )
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            info = await transport.start_stream()

        assert info is None

    @pytest.mark.anyio
    async def test_stop_stream(self, transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            await transport.start_stream()

        await transport.stop_stream()

        assert transport.state == TransportState.CONNECTED
        assert transport.stream_info is None
        mock_gopro.http_command.webcam_stop.assert_awaited_once()
        mock_gopro.http_command.webcam_exit.assert_awaited_once()

    @pytest.mark.anyio
    async def test_stop_stream_when_not_streaming_is_noop(self, transport, mock_gopro):
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        # Not streaming — stop should be a no-op
        await transport.stop_stream()
        mock_gopro.http_command.webcam_stop.assert_not_awaited()

    @pytest.mark.anyio
    async def test_stream_720p(self, transport, mock_gopro):
        from gomaxwebcam.transport.usb import RES_720
        transport._inject_gopro(mock_gopro)
        transport.resolution = RES_720
        await transport.connect()

        with _patch_sdk_enums():
            info = await transport.start_stream()

        assert info is not None
        assert info.width == 1280
        assert info.height == 720

    @pytest.mark.anyio
    async def test_stream_480p(self, transport, mock_gopro):
        """480p stream should return correct dimensions."""
        from gomaxwebcam.transport.usb import RES_480
        transport._inject_gopro(mock_gopro)
        transport.resolution = RES_480
        await transport.connect()

        with _patch_sdk_enums():
            info = await transport.start_stream()

        assert info is not None
        assert info.width == 854
        assert info.height == 480

    @pytest.mark.anyio
    async def test_start_stream_general_exception(self, transport, mock_gopro):
        """Generic exception during stream start should return None."""
        mock_gopro.http_command.webcam_start = AsyncMock(
            side_effect=RuntimeError("Camera firmware error")
        )
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            info = await transport.start_stream()

        assert info is None
        assert "Camera firmware error" in transport.stats.last_error

    @pytest.mark.anyio
    async def test_start_stream_connection_error(self, transport, mock_gopro):
        """ConnectionError during stream start should return None."""
        mock_gopro.http_command.webcam_start = AsyncMock(
            side_effect=ConnectionError("USB cable unplugged")
        )
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            info = await transport.start_stream()

        assert info is None
        assert transport.stats.last_error is not None

    @pytest.mark.anyio
    async def test_start_stream_timeout_exception(self, transport, mock_gopro):
        """TimeoutError during webcam_start should return None."""
        mock_gopro.http_command.webcam_start = AsyncMock(
            side_effect=asyncio.TimeoutError
        )
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            info = await transport.start_stream()

        assert info is None

    @pytest.mark.anyio
    async def test_start_stream_no_gopro_handle(self, transport):
        """Start stream with no gopro handle should return None."""
        from gomaxwebcam.transport.base import TransportState
        # Manually set connected state without a gopro handle (edge case)
        transport._set_state(TransportState.CONNECTED)

        result = await transport.start_stream()
        assert result is None

    @pytest.mark.anyio
    async def test_start_stream_from_error_state(self, transport, mock_gopro):
        """Cannot start stream from ERROR state."""
        from gomaxwebcam.transport.base import TransportState

        transport._inject_gopro(mock_gopro)
        transport._set_state(TransportState.ERROR)

        with _patch_sdk_enums():
            info = await transport.start_stream()

        assert info is None

    @pytest.mark.anyio
    async def test_start_stream_from_disconnected_state(self, transport, mock_gopro):
        """Cannot start stream from DISCONNECTED state."""
        transport._inject_gopro(mock_gopro)

        with _patch_sdk_enums():
            info = await transport.start_stream()

        assert info is None

    @pytest.mark.anyio
    async def test_start_stream_shutter_error(self, transport, mock_gopro):
        """Shutter error code from webcam_start should fail."""
        mock_gopro.http_command.webcam_start = AsyncMock(return_value=MockGoProResp(
            ok=True,
            data=MockWebcamResponse(
                status=MockWebcamStatus.IDLE,
                error=MockWebcamError.SHUTTER,
            ),
        ))
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            info = await transport.start_stream()

        assert info is None

    @pytest.mark.anyio
    async def test_stop_stream_webcam_stop_error(self, transport, mock_gopro):
        """webcam_stop error should not prevent state transition."""
        from gomaxwebcam.transport.base import TransportState

        mock_gopro.http_command.webcam_stop = AsyncMock(
            side_effect=ConnectionError("USB lost")
        )
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            await transport.start_stream()

        # Should not raise even though webcam_stop fails
        await transport.stop_stream()
        assert transport.state == TransportState.CONNECTED

    @pytest.mark.anyio
    async def test_stop_stream_webcam_exit_error(self, transport, mock_gopro):
        """webcam_exit error should not prevent state transition."""
        from gomaxwebcam.transport.base import TransportState

        mock_gopro.http_command.webcam_exit = AsyncMock(
            side_effect=RuntimeError("exit failed")
        )
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            await transport.start_stream()

        await transport.stop_stream()
        assert transport.state == TransportState.CONNECTED
        assert transport.stream_info is None

    @pytest.mark.anyio
    async def test_stop_stream_clears_stream_info(self, transport, mock_gopro):
        """Stopping the stream should clear the stream_info."""
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            info = await transport.start_stream()
            assert info is not None

        await transport.stop_stream()
        assert transport.stream_info is None

    @pytest.mark.anyio
    async def test_stream_state_transitions(self, transport, mock_gopro):
        """Full stream lifecycle state transitions."""
        from gomaxwebcam.transport.base import TransportState

        states: list = []
        transport.add_state_listener(lambda old, new: states.append((old, new)))

        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            await transport.start_stream()

        await transport.stop_stream()

        assert (TransportState.CONNECTED, TransportState.STREAMING) in states
        assert (TransportState.STREAMING, TransportState.CONNECTED) in states

    @pytest.mark.anyio
    async def test_start_stream_codec_is_h264(self, transport, mock_gopro):
        """Stream info should report h264 codec."""
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            info = await transport.start_stream()

        assert info is not None
        assert info.codec == "h264"

    @pytest.mark.anyio
    async def test_start_stream_host_is_any(self, transport, mock_gopro):
        """Stream info host should be 0.0.0.0 (listen on all interfaces)."""
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            info = await transport.start_stream()

        assert info is not None
        assert info.host == "0.0.0.0"

    @pytest.mark.anyio
    async def test_start_stream_status_poll_timeout(self, transport, mock_gopro):
        """If status never reaches HIGH_POWER_PREVIEW, should return None."""
        import gomaxwebcam.transport.usb as usb_mod

        # webcam_start succeeds but status always returns OFF
        mock_gopro.http_command.webcam_status = AsyncMock(return_value=MockGoProResp(
            ok=True,
            data=MockWebcamResponse(
                status=MockWebcamStatus.OFF,
                error=MockWebcamError.SUCCESS,
            ),
        ))

        transport._inject_gopro(mock_gopro)
        await transport.connect()

        # Shorten timeouts for test speed
        orig_poll_max = usb_mod.WEBCAM_STATUS_POLL_MAX_S
        orig_poll_interval = usb_mod.WEBCAM_STATUS_POLL_INTERVAL_S
        usb_mod.WEBCAM_STATUS_POLL_MAX_S = 0.2
        usb_mod.WEBCAM_STATUS_POLL_INTERVAL_S = 0.05
        try:
            with _patch_sdk_enums():
                info = await transport.start_stream()
        finally:
            usb_mod.WEBCAM_STATUS_POLL_MAX_S = orig_poll_max
            usb_mod.WEBCAM_STATUS_POLL_INTERVAL_S = orig_poll_interval

        assert info is None
        assert "timeout" in (transport.stats.last_error or "").lower()

    @pytest.mark.anyio
    async def test_start_stream_status_poll_errors_recover(self, transport, mock_gopro):
        """Status poll errors should be tolerated if status eventually succeeds."""
        import gomaxwebcam.transport.usb as usb_mod

        call_count = 0

        async def flaky_status():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient error")
            return MockGoProResp(
                ok=True,
                data=MockWebcamResponse(
                    status=MockWebcamStatus.HIGH_POWER_PREVIEW,
                    error=MockWebcamError.SUCCESS,
                ),
            )

        mock_gopro.http_command.webcam_status = flaky_status

        transport._inject_gopro(mock_gopro)
        await transport.connect()

        orig_interval = usb_mod.WEBCAM_STATUS_POLL_INTERVAL_S
        usb_mod.WEBCAM_STATUS_POLL_INTERVAL_S = 0.01
        try:
            with _patch_sdk_enums():
                info = await transport.start_stream()
        finally:
            usb_mod.WEBCAM_STATUS_POLL_INTERVAL_S = orig_interval

        assert info is not None
        assert transport.is_streaming

    @pytest.mark.anyio
    async def test_start_stream_low_power_preview_accepted(self, transport, mock_gopro):
        """LOW_POWER_PREVIEW should also be accepted as streaming."""
        mock_gopro.http_command.webcam_status = AsyncMock(return_value=MockGoProResp(
            ok=True,
            data=MockWebcamResponse(
                status=MockWebcamStatus.LOW_POWER_PREVIEW,
                error=MockWebcamError.SUCCESS,
            ),
        ))

        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            info = await transport.start_stream()

        assert info is not None
        assert transport.is_streaming

    @pytest.mark.anyio
    async def test_start_stream_custom_port(self):
        """Stream info should reflect the configured UDP port."""
        from gomaxwebcam.transport.usb import USBTransport

        t = USBTransport(serial="123", udp_port=9000)
        gopro = _make_mock_gopro()
        t._inject_gopro(gopro)
        await t.connect()

        with _patch_sdk_enums():
            info = await t.start_stream()

        assert info is not None
        assert info.port == 9000


# ---------------------------------------------------------------------------
# Tests — Disconnect
# ---------------------------------------------------------------------------


class TestUSBTransportDisconnect:
    """Disconnect and cleanup tests."""

    @pytest.mark.anyio
    async def test_disconnect_from_connected(self, transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        transport._inject_gopro(mock_gopro)
        await transport.connect()
        await transport.disconnect()

        assert transport.state == TransportState.DISCONNECTED
        mock_gopro.close.assert_awaited_once()

    @pytest.mark.anyio
    async def test_disconnect_from_streaming(self, transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            await transport.start_stream()

        await transport.disconnect()

        assert transport.state == TransportState.DISCONNECTED
        # Should have stopped stream first
        mock_gopro.http_command.webcam_stop.assert_awaited()
        mock_gopro.close.assert_awaited()

    @pytest.mark.anyio
    async def test_disconnect_clears_gopro(self, transport, mock_gopro):
        transport._inject_gopro(mock_gopro)
        await transport.connect()
        await transport.disconnect()

        assert transport._gopro is None

    @pytest.mark.anyio
    async def test_disconnect_updates_uptime(self, transport, mock_gopro):
        transport._inject_gopro(mock_gopro)
        await transport.connect()
        await asyncio.sleep(0.05)
        await transport.disconnect()

        assert transport.stats.connection_uptime_s > 0

    @pytest.mark.anyio
    async def test_disconnect_clears_camera_ip(self, transport, mock_gopro):
        """Camera IP should be cleared on disconnect."""
        transport._inject_gopro(mock_gopro)
        await transport.connect()
        assert transport._camera_ip is not None

        await transport.disconnect()
        assert transport._camera_ip is None

    @pytest.mark.anyio
    async def test_disconnect_clears_stream_info(self, transport, mock_gopro):
        """Stream info should be cleared on disconnect."""
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            await transport.start_stream()

        await transport.disconnect()
        assert transport.stream_info is None

    @pytest.mark.anyio
    async def test_disconnect_close_error_handled(self, transport, mock_gopro):
        """Error in gopro.close() should not prevent disconnect."""
        from gomaxwebcam.transport.base import TransportState

        mock_gopro.close = AsyncMock(side_effect=RuntimeError("close failed"))
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        # Should not raise
        await transport.disconnect()
        assert transport.state == TransportState.DISCONNECTED
        assert transport._gopro is None

    @pytest.mark.anyio
    async def test_disconnect_when_already_disconnected(self, transport):
        """Disconnect from disconnected state should be a no-op."""
        from gomaxwebcam.transport.base import TransportState

        await transport.disconnect()
        assert transport.state == TransportState.DISCONNECTED

    @pytest.mark.anyio
    async def test_disconnect_stops_keepalive_task(self, transport, mock_gopro):
        """Disconnect should cancel the keepalive background task."""
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            await transport.start_stream()

        # Keepalive task should be running
        assert transport._keepalive_task is not None

        await transport.disconnect()
        # After disconnect, keepalive task should be cleared
        assert transport._keepalive_task is None or transport._keepalive_task.done()


# ---------------------------------------------------------------------------
# Tests — Keep-Alive & Health Check
# ---------------------------------------------------------------------------


class TestUSBTransportKeepAlive:
    """Keep-alive and health check tests."""

    @pytest.mark.anyio
    async def test_keep_alive_success(self, transport, mock_gopro):
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        result = await transport.keep_alive()
        assert result is True
        assert transport.stats.keepalives_sent == 1

    @pytest.mark.anyio
    async def test_keep_alive_not_connected(self, transport):
        result = await transport.keep_alive()
        assert result is False

    @pytest.mark.anyio
    async def test_keep_alive_failure(self, transport, mock_gopro):
        mock_gopro.http_command.webcam_status = AsyncMock(
            side_effect=ConnectionError("Network unreachable")
        )
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        result = await transport.keep_alive()
        assert result is False
        assert transport.stats.keepalives_failed == 1

    @pytest.mark.anyio
    async def test_keep_alive_response_not_ok(self, transport, mock_gopro):
        """If webcam_status returns ok=False, keepalive should fail."""
        mock_gopro.http_command.webcam_status = AsyncMock(
            return_value=MockGoProResp(ok=False, data=None)
        )
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        result = await transport.keep_alive()
        assert result is False
        assert transport.stats.keepalives_failed == 1

    @pytest.mark.anyio
    async def test_keep_alive_timeout_error(self, transport, mock_gopro):
        """Timeout during keepalive should count as failure."""
        mock_gopro.http_command.webcam_status = AsyncMock(
            side_effect=asyncio.TimeoutError
        )
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        result = await transport.keep_alive()
        assert result is False
        assert transport.stats.keepalives_failed == 1

    @pytest.mark.anyio
    async def test_keep_alive_multiple_failures_tracked(self, transport, mock_gopro):
        """Multiple failures should increment the counter."""
        mock_gopro.http_command.webcam_status = AsyncMock(
            side_effect=ConnectionError("fail")
        )
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        await transport.keep_alive()
        await transport.keep_alive()
        await transport.keep_alive()

        assert transport.stats.keepalives_failed == 3

    @pytest.mark.anyio
    async def test_keep_alive_records_last_error(self, transport, mock_gopro):
        """Keepalive failure should record the error message."""
        mock_gopro.http_command.webcam_status = AsyncMock(
            side_effect=ConnectionError("Camera lost")
        )
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        await transport.keep_alive()
        assert "Camera lost" in transport.stats.last_error

    @pytest.mark.anyio
    async def test_health_check_streaming(self, transport, mock_gopro):
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            await transport.start_stream()
            result = await transport.health_check()

        assert result is True

    @pytest.mark.anyio
    async def test_health_check_wrong_status(self, transport, mock_gopro):
        """If we're streaming but camera says OFF, health check should fail."""
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            await transport.start_stream()

            # Now change the status response to OFF
            mock_gopro.http_command.webcam_status = AsyncMock(return_value=MockGoProResp(
                ok=True,
                data=MockWebcamResponse(
                    status=MockWebcamStatus.OFF,
                    error=MockWebcamError.SUCCESS,
                ),
            ))

            result = await transport.health_check()

        assert result is False

    @pytest.mark.anyio
    async def test_health_check_not_connected(self, transport):
        """Health check when not connected should return False."""
        result = await transport.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_health_check_no_gopro_handle(self, transport):
        """Health check without gopro handle should return False."""
        from gomaxwebcam.transport.base import TransportState
        transport._set_state(TransportState.CONNECTED)

        result = await transport.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_health_check_exception(self, transport, mock_gopro):
        """Exception during health check should return False."""
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        mock_gopro.http_command.webcam_status = AsyncMock(
            side_effect=OSError("adapter down")
        )

        result = await transport.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_health_check_connected_not_streaming(self, transport, mock_gopro):
        """Health check when connected (not streaming) should check basic connectivity."""
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        result = await transport.health_check()
        assert result is True

    @pytest.mark.anyio
    async def test_health_check_status_resp_not_ok(self, transport, mock_gopro):
        """If webcam_status returns ok=False, health check should fail."""
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        mock_gopro.http_command.webcam_status = AsyncMock(
            return_value=MockGoProResp(ok=False, data=None)
        )

        result = await transport.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_health_check_low_power_preview_accepted(self, transport, mock_gopro):
        """LOW_POWER_PREVIEW should pass health check when streaming."""
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            await transport.start_stream()

            mock_gopro.http_command.webcam_status = AsyncMock(return_value=MockGoProResp(
                ok=True,
                data=MockWebcamResponse(
                    status=MockWebcamStatus.LOW_POWER_PREVIEW,
                    error=MockWebcamError.SUCCESS,
                ),
            ))

            result = await transport.health_check()

        assert result is True


# ---------------------------------------------------------------------------
# Tests — Configuration
# ---------------------------------------------------------------------------


class TestUSBTransportConfig:
    """Configuration validation tests."""

    def test_set_valid_resolution(self, transport):
        from gomaxwebcam.transport.usb import RES_720
        transport.resolution = RES_720
        assert transport.resolution == RES_720

    def test_set_invalid_resolution_raises(self, transport):
        with pytest.raises(ValueError, match="not allowed"):
            transport.resolution = 99

    def test_set_valid_fov(self, transport):
        from gomaxwebcam.transport.usb import FOV_LINEAR
        transport.fov = FOV_LINEAR
        assert transport.fov == FOV_LINEAR

    def test_set_invalid_fov_raises(self, transport):
        with pytest.raises(ValueError, match="not allowed"):
            transport.fov = 99

    def test_serial_property(self, transport):
        assert transport.serial == "123"

    def test_udp_port_property(self, transport):
        assert transport.udp_port == 8554

    def test_set_all_valid_resolutions(self, transport):
        from gomaxwebcam.transport.usb import RES_480, RES_720, RES_1080
        for res in (RES_480, RES_720, RES_1080):
            transport.resolution = res
            assert transport.resolution == res

    def test_set_all_valid_fovs(self, transport):
        from gomaxwebcam.transport.usb import FOV_WIDE, FOV_NARROW, FOV_SUPERVIEW, FOV_LINEAR
        for fov in (FOV_WIDE, FOV_NARROW, FOV_SUPERVIEW, FOV_LINEAR):
            transport.fov = fov
            assert transport.fov == fov


# ---------------------------------------------------------------------------
# Tests — State Listeners
# ---------------------------------------------------------------------------


class TestUSBTransportStateListeners:
    """State listener callback tests."""

    @pytest.mark.anyio
    async def test_listener_fires_on_connect(self, transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        transitions: list = []
        transport.add_state_listener(lambda old, new: transitions.append((old, new)))

        transport._inject_gopro(mock_gopro)
        await transport.connect()

        assert len(transitions) >= 2
        assert (TransportState.DISCONNECTED, TransportState.CONNECTING) in transitions
        assert (TransportState.CONNECTING, TransportState.CONNECTED) in transitions

    @pytest.mark.anyio
    async def test_listener_fires_on_disconnect(self, transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        transport._inject_gopro(mock_gopro)
        await transport.connect()

        transitions: list = []
        transport.add_state_listener(lambda old, new: transitions.append((old, new)))
        await transport.disconnect()

        assert (TransportState.CONNECTED, TransportState.DISCONNECTED) in transitions

    @pytest.mark.anyio
    async def test_listener_exception_doesnt_crash(self, transport, mock_gopro):
        """A broken listener should not prevent state transitions."""
        def bad_listener(old, new):
            raise RuntimeError("listener boom")

        transport.add_state_listener(bad_listener)
        transport._inject_gopro(mock_gopro)

        # Should not raise
        await transport.connect()

    @pytest.mark.anyio
    async def test_multiple_listeners(self, transport, mock_gopro):
        """Multiple listeners should all be called."""
        results_a: list = []
        results_b: list = []
        transport.add_state_listener(lambda old, new: results_a.append(new))
        transport.add_state_listener(lambda old, new: results_b.append(new))

        transport._inject_gopro(mock_gopro)
        await transport.connect()

        assert results_a == results_b
        assert len(results_a) >= 2

    @pytest.mark.anyio
    async def test_listener_fires_on_error(self, transport, mock_gopro):
        """Listener should see transitions to ERROR state."""
        from gomaxwebcam.transport.base import TransportState

        transitions: list = []
        transport.add_state_listener(lambda old, new: transitions.append((old, new)))

        mock_gopro.open = AsyncMock(side_effect=ConnectionError("fail"))
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        assert (TransportState.CONNECTING, TransportState.ERROR) in transitions

    @pytest.mark.anyio
    async def test_listener_full_lifecycle(self, transport, mock_gopro):
        """Listener should see all state transitions across full lifecycle."""
        from gomaxwebcam.transport.base import TransportState

        transitions: list = []
        transport.add_state_listener(lambda old, new: transitions.append((old, new)))

        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            await transport.start_stream()

        await transport.stop_stream()
        await transport.disconnect()

        expected = [
            (TransportState.DISCONNECTED, TransportState.CONNECTING),
            (TransportState.CONNECTING, TransportState.CONNECTED),
            (TransportState.CONNECTED, TransportState.STREAMING),
            (TransportState.STREAMING, TransportState.CONNECTED),
            (TransportState.CONNECTED, TransportState.DISCONNECTED),
        ]
        assert transitions == expected


# ---------------------------------------------------------------------------
# Tests — StreamInfo & TransportStats
# ---------------------------------------------------------------------------


class TestStreamInfo:
    """StreamInfo dataclass tests."""

    def test_defaults(self):
        from gomaxwebcam.transport.base import StreamInfo
        info = StreamInfo()
        assert info.host == "0.0.0.0"
        assert info.port == 8554
        assert info.protocol == "udp"

    def test_frozen(self):
        from gomaxwebcam.transport.base import StreamInfo
        info = StreamInfo()
        with pytest.raises(AttributeError):
            info.port = 9999  # type: ignore

    def test_custom_values(self):
        from gomaxwebcam.transport.base import StreamInfo
        info = StreamInfo(protocol="tcp", host="127.0.0.1", port=9000, width=1280, height=720, fps=60)
        assert info.protocol == "tcp"
        assert info.host == "127.0.0.1"
        assert info.port == 9000
        assert info.width == 1280
        assert info.height == 720
        assert info.fps == 60


class TestTransportStats:
    """TransportStats tracking tests."""

    def test_initial_stats(self, transport):
        assert transport.stats.keepalives_sent == 0
        assert transport.stats.keepalives_failed == 0
        assert transport.stats.transport_type == "USB"
        assert transport.stats.last_error is None

    @pytest.mark.anyio
    async def test_keepalive_stats_increment(self, transport, mock_gopro):
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        await transport.keep_alive()
        await transport.keep_alive()

        assert transport.stats.keepalives_sent == 2

    @pytest.mark.anyio
    async def test_error_recorded_on_connect_failure(self, transport, mock_gopro):
        """Stats should record error message on connect failure."""
        mock_gopro.open = AsyncMock(side_effect=OSError("USB power failure"))
        transport._inject_gopro(mock_gopro)

        await transport.connect()
        assert transport.stats.last_error == "USB power failure"

    @pytest.mark.anyio
    async def test_uptime_tracking(self, transport, mock_gopro):
        """Uptime should be tracked from connect to disconnect."""
        transport._inject_gopro(mock_gopro)
        await transport.connect()
        await asyncio.sleep(0.05)
        await transport.disconnect()

        assert transport.stats.connection_uptime_s >= 0.04

    @pytest.mark.anyio
    async def test_camera_serial_recorded(self, transport, mock_gopro):
        """Camera serial should be recorded in stats on connect."""
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        assert transport.stats.camera_serial == "123"

    @pytest.mark.anyio
    async def test_camera_model_recorded(self, transport, mock_gopro):
        """Camera model (IP-based) should be recorded in stats on connect."""
        transport._inject_gopro(mock_gopro)
        await transport.connect()

        assert "GoPro@" in transport.stats.camera_model


# ---------------------------------------------------------------------------
# Tests — Inject Helpers
# ---------------------------------------------------------------------------


class TestInjectHelpers:
    """Test helper injection methods."""

    def test_inject_gopro(self, transport):
        gopro = MagicMock()
        gopro.ip_address = "10.0.0.1"
        transport._inject_gopro(gopro)

        assert transport._gopro is gopro
        assert transport._camera_ip == "10.0.0.1"
        assert "10.0.0.1" in transport.stats.camera_model

    def test_inject_gopro_without_ip(self, transport):
        """Inject a mock without ip_address attr should use default."""
        gopro = MagicMock(spec=[])
        transport._inject_gopro(gopro)

        assert transport._gopro is gopro
        assert transport._camera_ip == "172.20.100.51"

    def test_inject_camera_ip(self, transport):
        transport._inject_camera_ip("192.168.1.100")

        assert transport._camera_ip == "192.168.1.100"
        assert "192.168.1.100" in transport.stats.camera_model


# ---------------------------------------------------------------------------
# Tests — Full Lifecycle Integration
# ---------------------------------------------------------------------------


class TestUSBTransportLifecycle:
    """Integration tests for full connect → stream → disconnect lifecycle."""

    @pytest.mark.anyio
    async def test_full_lifecycle(self, transport, mock_gopro):
        """Complete lifecycle: connect → start_stream → stop_stream → disconnect."""
        from gomaxwebcam.transport.base import TransportState

        transport._inject_gopro(mock_gopro)

        # Connect
        assert await transport.connect() is True
        assert transport.state == TransportState.CONNECTED

        # Start stream
        with _patch_sdk_enums():
            info = await transport.start_stream()
        assert info is not None
        assert transport.state == TransportState.STREAMING

        # Keep alive while streaming
        result = await transport.keep_alive()
        assert result is True

        # Health check while streaming
        result = await transport.health_check()
        assert result is True

        # Stop stream
        await transport.stop_stream()
        assert transport.state == TransportState.CONNECTED

        # Disconnect
        await transport.disconnect()
        assert transport.state == TransportState.DISCONNECTED
        assert transport._gopro is None

    @pytest.mark.anyio
    async def test_connect_stream_disconnect_stream_disconnect(self, transport, mock_gopro):
        """Connect → stream → disconnect → reconnect → stream → disconnect."""
        from gomaxwebcam.transport.base import TransportState

        transport._inject_gopro(mock_gopro)
        await transport.connect()

        with _patch_sdk_enums():
            info = await transport.start_stream()
        assert info is not None

        await transport.disconnect()
        assert transport.state == TransportState.DISCONNECTED

        # Reconnect with new mock (since disconnect clears _gopro)
        new_gopro = _make_mock_gopro()
        transport._inject_gopro(new_gopro)
        assert await transport.connect() is True

        with _patch_sdk_enums():
            info2 = await transport.start_stream()
        assert info2 is not None

        await transport.disconnect()
        assert transport.state == TransportState.DISCONNECTED

    @pytest.mark.anyio
    async def test_error_recovery_lifecycle(self, transport, mock_gopro):
        """Connect failure → reconnect → stream successfully."""
        from gomaxwebcam.transport.base import TransportState

        # First attempt fails
        mock_gopro.open = AsyncMock(side_effect=ConnectionError("USB error"))
        transport._inject_gopro(mock_gopro)
        assert await transport.connect() is False
        assert transport.state == TransportState.ERROR

        # Second attempt succeeds
        mock_gopro.open = AsyncMock()
        assert await transport.connect() is True
        assert transport.state == TransportState.CONNECTED

        # Stream works after recovery
        with _patch_sdk_enums():
            info = await transport.start_stream()
        assert info is not None
        assert transport.is_streaming
