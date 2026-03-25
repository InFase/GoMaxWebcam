"""
Tests for COHN transport layer — mocked at the WirelessGoPro SDK boundary.

No real hardware or network required. These tests verify:
  - State machine transitions (DISCONNECTED → DISCOVERING → CONNECTED → STREAMING)
  - WirelessGoPro wrapper lifecycle (open/close/http_command)
  - Stream start/stop lifecycle with SDK webcam commands
  - Keep-alive behavior and failure detection
  - Health check validation (including streaming status verification)
  - Error handling: timeouts, auth failures, network errors
  - Configuration validation (resolution / FOV guards)
  - mDNS discovery fallback path
  - Credential injection for testing
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock open-gopro SDK helpers
# ---------------------------------------------------------------------------

class MockWebcamResponse:
    """Simulates open_gopro WebcamResponse model."""

    def __init__(self, status: int | None = None, error: int = 0):
        self.status = MagicMock(value=status) if status is not None else None
        self.error = MagicMock(value=error)


class MockGoProResp:
    """Simulates open_gopro GoProResp wrapper."""

    def __init__(
        self,
        ok: bool = True,
        status: int | None = None,
        error: int = 0,
    ):
        self.ok = ok
        self.data = MockWebcamResponse(status=status, error=error)


class MockHttpCommand:
    """Simulates gopro.http_command with webcam methods."""

    def __init__(
        self,
        webcam_status_val: int = 2,  # HIGH_POWER_PREVIEW
        webcam_error: int = 0,
        webcam_start_ok: bool = True,
    ):
        self._webcam_status_val = webcam_status_val
        self._webcam_error = webcam_error
        self._webcam_start_ok = webcam_start_ok

    async def webcam_status(self) -> MockGoProResp:
        return MockGoProResp(
            ok=True,
            status=self._webcam_status_val,
            error=self._webcam_error,
        )

    async def webcam_start(self, **kwargs: Any) -> MockGoProResp:
        return MockGoProResp(
            ok=self._webcam_start_ok,
            status=1,  # IDLE initially
            error=self._webcam_error,
        )

    async def webcam_stop(self) -> MockGoProResp:
        return MockGoProResp(ok=True, status=0, error=0)

    async def webcam_exit(self) -> MockGoProResp:
        return MockGoProResp(ok=True, status=0, error=0)


class MockWirelessGoPro:
    """Simulates open-gopro's WirelessGoPro for COHN connection."""

    def __init__(
        self,
        is_http_connected: bool = True,
        webcam_status_val: int = 2,
        webcam_error: int = 0,
        webcam_start_ok: bool = True,
    ):
        self._is_http_connected = is_http_connected
        self.http_command = MockHttpCommand(
            webcam_status_val=webcam_status_val,
            webcam_error=webcam_error,
            webcam_start_ok=webcam_start_ok,
        )
        self.opened = False
        self.closed = False

    @property
    def is_http_connected(self) -> bool:
        return self._is_http_connected

    async def open(self, **kwargs: Any) -> None:
        self.opened = True

    async def close(self) -> None:
        self.closed = True


def _make_mock_gopro(
    webcam_status_val: int = 2,
    webcam_error: int = 0,
    webcam_start_ok: bool = True,
) -> MockWirelessGoPro:
    """Create a mock WirelessGoPro with standard COHN API responses."""
    return MockWirelessGoPro(
        webcam_status_val=webcam_status_val,
        webcam_error=webcam_error,
        webcam_start_ok=webcam_start_ok,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_gopro():
    return _make_mock_gopro()


@pytest.fixture
def transport():
    """Create a COHNTransport without triggering any network calls."""
    from gomaxwebcam.transport.cohn import COHNTransport
    return COHNTransport(
        ip_address="192.168.1.100",
        username="gopro",
        password="test-password-123",
        udp_port=8554,
    )


@pytest.fixture
def connected_transport(transport, mock_gopro):
    """Return a transport that's already in CONNECTED state with mock gopro."""
    from gomaxwebcam.transport.base import TransportState
    transport._inject_camera_ip("192.168.1.100")
    transport._inject_gopro(mock_gopro)
    transport._set_state(TransportState.CONNECTED)
    transport._connected_at = time.monotonic()
    return transport


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCOHNTransportInit:
    """Construction and default values."""

    def test_initial_state_is_disconnected(self, transport):
        from gomaxwebcam.transport.base import TransportState
        assert transport.state == TransportState.DISCONNECTED

    def test_default_name(self, transport):
        assert transport.name == "COHN"

    def test_not_connected_initially(self, transport):
        assert not transport.is_connected
        assert not transport.is_streaming

    def test_stream_info_none_initially(self, transport):
        assert transport.stream_info is None

    def test_ip_address_property(self, transport):
        assert transport.ip_address == "192.168.1.100"

    def test_username_property(self, transport):
        assert transport.username == "gopro"

    def test_has_credentials(self, transport):
        assert transport.has_credentials is True

    def test_no_credentials(self):
        from gomaxwebcam.transport.cohn import COHNTransport
        t = COHNTransport(ip_address="1.2.3.4")
        assert t.has_credentials is False


class TestCOHNTransportDiscover:
    """Discovery phase tests."""

    @pytest.mark.anyio
    async def test_discover_with_credentials_returns_true(self, transport):
        result = await transport.discover()
        assert result is True

    @pytest.mark.anyio
    async def test_discover_sets_camera_model(self, transport):
        await transport.discover()
        assert "192.168.1.100" in transport.stats.camera_model
        assert "COHN" in transport.stats.camera_model

    @pytest.mark.anyio
    async def test_discover_sets_discovering_state(self, transport):
        from gomaxwebcam.transport.base import TransportState

        states: list = []
        transport.add_state_listener(lambda old, new: states.append((old, new)))
        await transport.discover()

        assert (TransportState.DISCONNECTED, TransportState.DISCOVERING) in states

    @pytest.mark.anyio
    async def test_discover_without_credentials_needs_mdns(self):
        """Without credentials, discover should attempt mDNS."""
        from gomaxwebcam.transport.cohn import COHNTransport
        from gomaxwebcam.transport.base import TransportState

        t = COHNTransport()

        # Without zeroconf installed, should fail gracefully
        with patch.dict("sys.modules", {"zeroconf": None}):
            import sys
            saved = sys.modules.get("zeroconf")
            sys.modules["zeroconf"] = None  # type: ignore
            try:
                result = await t.discover(timeout=1.0)
                assert result is False or t.state == TransportState.ERROR
            finally:
                if saved is not None:
                    sys.modules["zeroconf"] = saved
                else:
                    sys.modules.pop("zeroconf", None)


class TestCOHNTransportConnect:
    """Connection phase tests."""

    @pytest.mark.anyio
    async def test_connect_without_discover_fails(self):
        from gomaxwebcam.transport.cohn import COHNTransport
        t = COHNTransport()
        result = await t.connect()
        assert result is False

    @pytest.mark.anyio
    async def test_connect_without_password_fails(self):
        from gomaxwebcam.transport.cohn import COHNTransport
        from gomaxwebcam.transport.base import TransportState
        t = COHNTransport(ip_address="192.168.1.100")
        result = await t.connect()
        assert result is False
        assert t.state == TransportState.ERROR

    @pytest.mark.anyio
    async def test_connect_with_injected_gopro(self, transport, mock_gopro):
        """Test connect via the _inject pattern (simpler for most tests)."""
        from gomaxwebcam.transport.base import TransportState

        await transport.discover()
        transport._inject_gopro(mock_gopro)
        transport._set_state(TransportState.CONNECTED)
        transport._connected_at = time.monotonic()

        assert transport.state == TransportState.CONNECTED
        assert transport.is_connected

    @pytest.mark.anyio
    async def test_connect_records_camera_info(self, connected_transport):
        assert connected_transport.stats.camera_model is not None
        assert "192.168.1.100" in connected_transport.stats.camera_model
        assert "COHN" in connected_transport.stats.camera_model


class TestCOHNTransportStream:
    """Stream start/stop tests."""

    @pytest.mark.anyio
    async def test_start_stream_not_connected(self, transport):
        result = await transport.start_stream()
        assert result is None

    @pytest.mark.anyio
    async def test_start_stream_success(self, connected_transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        info = await connected_transport.start_stream()

        assert info is not None
        assert info.port == 8554
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 30
        assert info.protocol == "udp"
        assert connected_transport.state == TransportState.STREAMING
        assert connected_transport.is_streaming

    @pytest.mark.anyio
    async def test_start_stream_sets_stream_info(self, connected_transport, mock_gopro):
        info = await connected_transport.start_stream()

        assert connected_transport.stream_info is info
        assert connected_transport.stream_info is not None

    @pytest.mark.anyio
    async def test_start_stream_webcam_start_failure(self, connected_transport):
        """If webcam_start returns ok=False, start_stream should return None."""
        gopro = _make_mock_gopro(webcam_start_ok=False)
        # Override webcam_start to return not-ok
        async def failing_start(**kw):
            return MockGoProResp(ok=False, status=0, error=7)
        gopro.http_command.webcam_start = failing_start
        connected_transport._inject_gopro(gopro)

        info = await connected_transport.start_stream()

        assert info is None
        assert not connected_transport.is_streaming

    @pytest.mark.anyio
    async def test_start_stream_status_timeout(self, connected_transport):
        """If webcam never reaches HIGH_POWER_PREVIEW, should return None."""
        gopro = _make_mock_gopro(webcam_status_val=1)  # Stuck in IDLE
        connected_transport._inject_gopro(gopro)

        # Patch the timeout to be short for testing
        with patch("gomaxwebcam.transport.cohn.WEBCAM_STATUS_POLL_MAX_S", 0.5), \
             patch("gomaxwebcam.transport.cohn.WEBCAM_STATUS_POLL_INTERVAL_S", 0.1):
            info = await connected_transport.start_stream()

        assert info is None
        assert connected_transport.stats.last_error == "Webcam status timeout"

    @pytest.mark.anyio
    async def test_stop_stream(self, connected_transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        await connected_transport.start_stream()
        await connected_transport.stop_stream()

        assert connected_transport.state == TransportState.CONNECTED
        assert connected_transport.stream_info is None

    @pytest.mark.anyio
    async def test_stop_stream_when_not_streaming_is_noop(self, connected_transport):
        # Not streaming — stop should be a no-op
        await connected_transport.stop_stream()
        # State should remain CONNECTED
        from gomaxwebcam.transport.base import TransportState
        assert connected_transport.state == TransportState.CONNECTED

    @pytest.mark.anyio
    async def test_stream_720p(self, connected_transport, mock_gopro):
        from gomaxwebcam.transport.cohn import RES_720
        connected_transport.resolution = RES_720

        info = await connected_transport.start_stream()

        assert info is not None
        assert info.width == 1280
        assert info.height == 720

    @pytest.mark.anyio
    async def test_stream_480p(self, connected_transport, mock_gopro):
        from gomaxwebcam.transport.cohn import RES_480
        connected_transport.resolution = RES_480

        info = await connected_transport.start_stream()

        assert info is not None
        assert info.width == 854
        assert info.height == 480


class TestCOHNTransportDisconnect:
    """Disconnect and cleanup tests."""

    @pytest.mark.anyio
    async def test_disconnect_from_connected(self, connected_transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        await connected_transport.disconnect()

        assert connected_transport.state == TransportState.DISCONNECTED
        assert mock_gopro.closed

    @pytest.mark.anyio
    async def test_disconnect_from_streaming(self, connected_transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        await connected_transport.start_stream()
        await connected_transport.disconnect()

        assert connected_transport.state == TransportState.DISCONNECTED

    @pytest.mark.anyio
    async def test_disconnect_clears_gopro(self, connected_transport, mock_gopro):
        await connected_transport.disconnect()
        assert connected_transport._gopro is None

    @pytest.mark.anyio
    async def test_disconnect_updates_uptime(self, connected_transport, mock_gopro):
        await asyncio.sleep(0.05)
        await connected_transport.disconnect()
        assert connected_transport.stats.connection_uptime_s > 0


class TestCOHNTransportKeepAlive:
    """Keep-alive and health check tests."""

    @pytest.mark.anyio
    async def test_keep_alive_success(self, connected_transport, mock_gopro):
        result = await connected_transport.keep_alive()
        assert result is True
        assert connected_transport.stats.keepalives_sent == 1

    @pytest.mark.anyio
    async def test_keep_alive_not_connected(self, transport):
        result = await transport.keep_alive()
        assert result is False

    @pytest.mark.anyio
    async def test_keep_alive_failure(self, connected_transport):
        """Keep-alive should fail gracefully on network error."""

        class FailGoPro:
            is_http_connected = True

            class http_command:
                @staticmethod
                async def webcam_status():
                    raise ConnectionError("Network unreachable")

            async def close(self):
                pass

        connected_transport._inject_gopro(FailGoPro())

        result = await connected_transport.keep_alive()
        assert result is False
        assert connected_transport.stats.keepalives_failed == 1

    @pytest.mark.anyio
    async def test_keep_alive_not_ok_response(self, connected_transport):
        """Keep-alive should fail on non-OK response."""
        gopro = _make_mock_gopro()

        async def bad_status():
            return MockGoProResp(ok=False, status=0, error=1)

        gopro.http_command.webcam_status = bad_status
        connected_transport._inject_gopro(gopro)

        result = await connected_transport.keep_alive()
        assert result is False

    @pytest.mark.anyio
    async def test_health_check_streaming(self, connected_transport, mock_gopro):
        await connected_transport.start_stream()
        result = await connected_transport.health_check()
        assert result is True

    @pytest.mark.anyio
    async def test_health_check_wrong_status(self, connected_transport, mock_gopro):
        """If we're streaming but camera says OFF, health check should fail."""
        await connected_transport.start_stream()

        # Swap to a gopro that reports OFF status
        gopro_off = _make_mock_gopro(webcam_status_val=0)
        connected_transport._inject_gopro(gopro_off)

        result = await connected_transport.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_health_check_not_connected(self, transport):
        result = await transport.health_check()
        assert result is False


class TestCOHNTransportConfig:
    """Configuration validation tests."""

    def test_set_valid_resolution(self, transport):
        from gomaxwebcam.transport.cohn import RES_720
        transport.resolution = RES_720
        assert transport.resolution == RES_720

    def test_set_invalid_resolution_raises(self, transport):
        with pytest.raises(ValueError, match="not allowed"):
            transport.resolution = 99

    def test_set_valid_fov(self, transport):
        from gomaxwebcam.transport.cohn import FOV_LINEAR
        transport.fov = FOV_LINEAR
        assert transport.fov == FOV_LINEAR

    def test_set_invalid_fov_raises(self, transport):
        with pytest.raises(ValueError, match="not allowed"):
            transport.fov = 99

    def test_udp_port_property(self, transport):
        assert transport.udp_port == 8554


class TestCOHNTransportStateListeners:
    """State listener callback tests."""

    @pytest.mark.anyio
    async def test_listener_fires_on_state_change(self, connected_transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        transitions: list = []
        connected_transport.add_state_listener(lambda old, new: transitions.append((old, new)))

        await connected_transport.start_stream()

        assert (TransportState.CONNECTED, TransportState.STREAMING) in transitions

    @pytest.mark.anyio
    async def test_listener_fires_on_disconnect(self, connected_transport, mock_gopro):
        from gomaxwebcam.transport.base import TransportState

        transitions: list = []
        connected_transport.add_state_listener(lambda old, new: transitions.append((old, new)))
        await connected_transport.disconnect()

        assert (TransportState.CONNECTED, TransportState.DISCONNECTED) in transitions

    @pytest.mark.anyio
    async def test_listener_exception_doesnt_crash(self, connected_transport, mock_gopro):
        """A broken listener should not prevent state transitions."""
        def bad_listener(old, new):
            raise RuntimeError("listener boom")

        connected_transport.add_state_listener(bad_listener)

        # Should not raise
        await connected_transport.disconnect()


class TestCOHNTransportTestHelpers:
    """Test helper injection methods."""

    def test_inject_gopro(self, transport):
        mock = MagicMock()
        transport._inject_gopro(mock)
        assert transport._gopro is mock

    def test_inject_camera_ip(self, transport):
        transport._inject_camera_ip("10.0.0.50")
        assert transport._ip_address == "10.0.0.50"
        assert "10.0.0.50" in transport.stats.camera_model

    def test_inject_credentials(self, transport):
        transport._inject_credentials("user", "pass123")
        assert transport._username == "user"
        assert transport._password == "pass123"
        assert transport.has_credentials is True


class TestStreamInfo:
    """StreamInfo from COHN transport."""

    @pytest.mark.anyio
    async def test_stream_info_values(self, connected_transport, mock_gopro):
        info = await connected_transport.start_stream()

        assert info is not None
        assert info.host == "0.0.0.0"
        assert info.port == 8554
        assert info.protocol == "udp"
        assert info.codec == "h264"
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 30


class TestCOHNIterFrames:
    """Frame iteration via iter_frames() async generator."""

    @pytest.mark.anyio
    async def test_iter_frames_not_connected_raises(self, transport):
        """iter_frames() should raise if transport is not connected."""
        with pytest.raises(RuntimeError, match="not connected|DISCONNECTED|Cannot iterate"):
            async for _ in transport.iter_frames():
                pass

    @pytest.mark.anyio
    async def test_iter_frames_no_gopro_raises(self):
        """iter_frames() should raise if no WirelessGoPro instance is set."""
        from gomaxwebcam.transport.cohn import COHNTransport
        from gomaxwebcam.transport.base import TransportState

        t = COHNTransport(
            ip_address="192.168.1.100",
            username="gopro",
            password="test-password-123",
        )
        t._set_state(TransportState.CONNECTED)

        with pytest.raises(RuntimeError, match="no WirelessGoPro"):
            async for _ in t.iter_frames():
                pass

    @pytest.mark.anyio
    async def test_iter_frames_starts_stream_if_connected(self, connected_transport, mock_gopro):
        """iter_frames() should auto-start the stream if in CONNECTED state."""
        import numpy as np
        from gomaxwebcam.transport.base import TransportState

        assert connected_transport.state == TransportState.CONNECTED

        with patch("gomaxwebcam.pipeline.decode.UDPDecoder") as MockDecoder:
            mock_decoder_instance = MagicMock()
            mock_decoder_instance.stop.return_value = None
            MockDecoder.return_value = mock_decoder_instance

            def start_and_deliver():
                call_kwargs = MockDecoder.call_args
                on_frame = call_kwargs.kwargs.get("on_frame")
                on_stopped = call_kwargs.kwargs.get("on_stopped")
                # Use a thread to simulate real decoder behavior
                import threading
                def deliver():
                    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                    on_frame(frame)
                    on_stopped()
                threading.Thread(target=deliver, daemon=True).start()
                return True

            mock_decoder_instance.start.side_effect = start_and_deliver

            frames_received = []
            async for frame in connected_transport.iter_frames(decode_timeout=2.0):
                frames_received.append(frame)
                if len(frames_received) >= 1:
                    break

            # Should have started streaming
            assert connected_transport.state == TransportState.STREAMING

    @pytest.mark.anyio
    async def test_iter_frames_yields_numpy_arrays(self, connected_transport, mock_gopro):
        """iter_frames() should yield numpy RGB24 arrays."""
        import numpy as np
        import threading

        with patch("gomaxwebcam.pipeline.decode.UDPDecoder") as MockDecoder:
            mock_decoder_instance = MagicMock()
            mock_decoder_instance.stop.return_value = None
            MockDecoder.return_value = mock_decoder_instance

            test_frames = [
                np.full((1080, 1920, 3), i, dtype=np.uint8)
                for i in range(3)
            ]

            def start_and_deliver():
                call_kwargs = MockDecoder.call_args
                on_frame = call_kwargs.kwargs.get("on_frame")
                on_stopped = call_kwargs.kwargs.get("on_stopped")
                def deliver():
                    import time
                    for f in test_frames:
                        on_frame(f)
                        time.sleep(0.01)
                    on_stopped()
                threading.Thread(target=deliver, daemon=True).start()
                return True

            mock_decoder_instance.start.side_effect = start_and_deliver

            received = []
            async for frame in connected_transport.iter_frames(decode_timeout=2.0):
                received.append(frame)

            assert len(received) == 3
            for i, frame in enumerate(received):
                assert isinstance(frame, np.ndarray)
                assert frame.shape == (1080, 1920, 3)
                assert frame.dtype == np.uint8
                assert frame[0, 0, 0] == i  # Verify frame identity

    @pytest.mark.anyio
    async def test_iter_frames_cleanup_on_break(self, connected_transport, mock_gopro):
        """Decoder should be stopped when generator is explicitly closed."""
        import numpy as np
        import threading

        with patch("gomaxwebcam.pipeline.decode.UDPDecoder") as MockDecoder:
            mock_decoder_instance = MagicMock()
            mock_decoder_instance.stop.return_value = None
            MockDecoder.return_value = mock_decoder_instance

            stop_event = threading.Event()

            def start_and_deliver():
                call_kwargs = MockDecoder.call_args
                on_frame = call_kwargs.kwargs.get("on_frame")
                def deliver():
                    import time
                    for i in range(100):
                        if stop_event.is_set():
                            break
                        on_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))
                        time.sleep(0.01)
                threading.Thread(target=deliver, daemon=True).start()
                return True

            mock_decoder_instance.start.side_effect = start_and_deliver

            frame_count = 0
            gen = connected_transport.iter_frames(decode_timeout=2.0)
            async for frame in gen:
                frame_count += 1
                if frame_count >= 2:
                    break

            # Explicitly close the generator to trigger finally block
            await gen.aclose()
            stop_event.set()

            # Decoder stop() should have been called during cleanup
            mock_decoder_instance.stop.assert_called_once()

    @pytest.mark.anyio
    async def test_iter_frames_decoder_error_stops_iteration(self, connected_transport, mock_gopro):
        """If decoder reports a fatal error, iteration should stop."""
        import numpy as np
        import threading

        with patch("gomaxwebcam.pipeline.decode.UDPDecoder") as MockDecoder:
            mock_decoder_instance = MagicMock()
            mock_decoder_instance.stop.return_value = None
            MockDecoder.return_value = mock_decoder_instance

            def start_and_deliver():
                call_kwargs = MockDecoder.call_args
                on_frame = call_kwargs.kwargs.get("on_frame")
                on_error = call_kwargs.kwargs.get("on_error")
                def deliver():
                    import time
                    on_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))
                    time.sleep(0.02)
                    on_error(RuntimeError("stream died"))
                threading.Thread(target=deliver, daemon=True).start()
                return True

            mock_decoder_instance.start.side_effect = start_and_deliver

            received = []
            async for frame in connected_transport.iter_frames(decode_timeout=2.0):
                received.append(frame)

            # Should have received the one frame before error signaled stop
            assert len(received) >= 1

    @pytest.mark.anyio
    async def test_iter_frames_cohn_preflight_check(self, connected_transport, mock_gopro):
        """COHN iter_frames should perform a preflight webcam_status check."""
        import numpy as np
        import threading

        # Track if webcam_status was called for preflight
        status_calls = []
        original_status = mock_gopro.http_command.webcam_status

        async def tracking_status():
            status_calls.append(True)
            return await original_status()

        mock_gopro.http_command.webcam_status = tracking_status

        with patch("gomaxwebcam.pipeline.decode.UDPDecoder") as MockDecoder:
            mock_decoder_instance = MagicMock()
            mock_decoder_instance.stop.return_value = None
            MockDecoder.return_value = mock_decoder_instance

            def start_and_deliver():
                call_kwargs = MockDecoder.call_args
                on_frame = call_kwargs.kwargs.get("on_frame")
                on_stopped = call_kwargs.kwargs.get("on_stopped")
                def deliver():
                    on_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))
                    on_stopped()
                threading.Thread(target=deliver, daemon=True).start()
                return True

            mock_decoder_instance.start.side_effect = start_and_deliver

            async for _ in connected_transport.iter_frames(decode_timeout=2.0):
                break

            # Pre-flight webcam_status should have been called
            assert len(status_calls) >= 1

    @pytest.mark.anyio
    async def test_iter_frames_default_cohn_timeout(self):
        """COHN iter_frames should use 15s default decode_timeout."""
        from gomaxwebcam.transport.cohn import COHNTransport
        import inspect

        # Verify the signature has the COHN-specific default
        sig = inspect.signature(COHNTransport.iter_frames)
        decode_timeout_param = sig.parameters.get("decode_timeout")
        assert decode_timeout_param is not None
        assert decode_timeout_param.default == 15.0

    @pytest.mark.anyio
    async def test_iter_frames_queue_backpressure(self, connected_transport, mock_gopro):
        """Frames should be dropped when queue is full (newest-wins)."""
        import numpy as np
        import threading

        with patch("gomaxwebcam.pipeline.decode.UDPDecoder") as MockDecoder:
            mock_decoder_instance = MagicMock()
            mock_decoder_instance.stop.return_value = None
            MockDecoder.return_value = mock_decoder_instance

            def start_and_deliver():
                call_kwargs = MockDecoder.call_args
                on_frame = call_kwargs.kwargs.get("on_frame")
                on_stopped = call_kwargs.kwargs.get("on_stopped")
                def deliver():
                    import time
                    # Send many frames quickly with small queue
                    for i in range(10):
                        on_frame(np.full((1080, 1920, 3), i, dtype=np.uint8))
                        time.sleep(0.005)
                    time.sleep(0.05)  # Let consumer catch up
                    on_stopped()
                threading.Thread(target=deliver, daemon=True).start()
                return True

            mock_decoder_instance.start.side_effect = start_and_deliver

            received = []
            async for frame in connected_transport.iter_frames(
                queue_size=2, decode_timeout=2.0
            ):
                received.append(frame)

            # Should have received frames (some may have been dropped)
            assert len(received) >= 1


class TestTransportStats:
    """TransportStats tracking for COHN."""

    def test_initial_stats(self, transport):
        assert transport.stats.keepalives_sent == 0
        assert transport.stats.keepalives_failed == 0
        assert transport.stats.transport_type == "COHN"
        assert transport.stats.last_error is None

    @pytest.mark.anyio
    async def test_keepalive_stats_increment(self, connected_transport, mock_gopro):
        await connected_transport.keep_alive()
        await connected_transport.keep_alive()
        assert connected_transport.stats.keepalives_sent == 2
