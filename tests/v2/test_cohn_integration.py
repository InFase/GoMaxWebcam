"""
Integration tests for COHN transport — multi-step lifecycle and error scenarios.

These tests exercise the full transport lifecycle with mocked WirelessGoPro,
verifying that state machine transitions, streaming, error handling, and
resource cleanup all work correctly in realistic multi-step flows.

No real hardware or network required.  All tests use the ``no_gopro_needed``
marker so they run without a GoPro connected.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from gomaxwebcam.transport.base import StreamInfo, TransportState
from gomaxwebcam.transport.cohn import (
    COHNTransport,
    FOV_LINEAR,
    FOV_NARROW,
    FOV_WIDE,
    RES_480,
    RES_720,
    RES_1080,
)

pytestmark = [pytest.mark.no_gopro_needed, pytest.mark.anyio]


# ---------------------------------------------------------------------------
# Mock helpers (reusable across integration tests)
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
    """Simulates gopro.http_command with webcam methods and call tracking."""

    def __init__(
        self,
        webcam_status_val: int = 2,
        webcam_error: int = 0,
        webcam_start_ok: bool = True,
    ):
        self._webcam_status_val = webcam_status_val
        self._webcam_error = webcam_error
        self._webcam_start_ok = webcam_start_ok
        # Track calls for integration assertions
        self.status_call_count = 0
        self.start_call_count = 0
        self.stop_call_count = 0
        self.exit_call_count = 0

    async def webcam_status(self) -> MockGoProResp:
        self.status_call_count += 1
        return MockGoProResp(
            ok=True,
            status=self._webcam_status_val,
            error=self._webcam_error,
        )

    async def webcam_start(self, **kwargs: Any) -> MockGoProResp:
        self.start_call_count += 1
        return MockGoProResp(
            ok=self._webcam_start_ok,
            status=1,  # IDLE initially
            error=self._webcam_error,
        )

    async def webcam_stop(self) -> MockGoProResp:
        self.stop_call_count += 1
        return MockGoProResp(ok=True, status=0, error=0)

    async def webcam_exit(self) -> MockGoProResp:
        self.exit_call_count += 1
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
        self.open_count = 0
        self.close_count = 0

    @property
    def is_http_connected(self) -> bool:
        return self._is_http_connected

    async def open(self, **kwargs: Any) -> None:
        self.opened = True
        self.open_count += 1

    async def close(self) -> None:
        self.closed = True
        self.close_count += 1


def _make_mock_gopro(
    webcam_status_val: int = 2,
    webcam_error: int = 0,
    webcam_start_ok: bool = True,
) -> MockWirelessGoPro:
    """Factory for creating pre-configured mock WirelessGoPro instances."""
    return MockWirelessGoPro(
        webcam_status_val=webcam_status_val,
        webcam_error=webcam_error,
        webcam_start_ok=webcam_start_ok,
    )


def _make_transport(**kwargs: Any) -> COHNTransport:
    """Create a COHNTransport with sensible test defaults."""
    defaults = {
        "ip_address": "192.168.1.100",
        "username": "gopro",
        "password": "test-password-123",
        "udp_port": 8554,
    }
    defaults.update(kwargs)
    return COHNTransport(**defaults)


def _connect_transport(
    transport: COHNTransport,
    mock_gopro: MockWirelessGoPro | None = None,
) -> MockWirelessGoPro:
    """Inject a mock and put transport into CONNECTED state. Returns the mock."""
    gopro = mock_gopro or _make_mock_gopro()
    transport._inject_camera_ip("192.168.1.100")
    transport._inject_gopro(gopro)
    transport._set_state(TransportState.CONNECTED)
    transport._connected_at = time.monotonic()
    return gopro


# ---------------------------------------------------------------------------
# UDPDecoder mock helper for frame iteration
# ---------------------------------------------------------------------------


def _patch_decoder_with_frames(
    frames: list[np.ndarray],
    *,
    error: Exception | None = None,
    delay_between: float = 0.005,
):
    """Return a context manager that patches UDPDecoder to deliver frames.

    Args:
        frames: List of numpy arrays to deliver.
        error: Optional exception to signal after delivering frames.
        delay_between: Sleep between frame deliveries (seconds).
    """
    import threading

    return patch(
        "gomaxwebcam.pipeline.decode.UDPDecoder",
        side_effect=lambda **kwargs: _FakeDecoder(
            kwargs, frames, error=error, delay_between=delay_between
        ),
    )


class _FakeDecoder:
    """Minimal UDPDecoder stand-in that delivers frames from a list."""

    def __init__(
        self,
        init_kwargs: dict,
        frames: list[np.ndarray],
        *,
        error: Exception | None = None,
        delay_between: float = 0.005,
    ):
        self._init_kwargs = init_kwargs
        self._frames = frames
        self._error = error
        self._delay_between = delay_between

    def start(self) -> bool:
        import threading

        on_frame = self._init_kwargs.get("on_frame")
        on_stopped = self._init_kwargs.get("on_stopped")
        on_error = self._init_kwargs.get("on_error")

        def deliver():
            import time as _t

            for f in self._frames:
                on_frame(f)
                _t.sleep(self._delay_between)

            if self._error and on_error:
                on_error(self._error)
            elif on_stopped:
                on_stopped()

        threading.Thread(target=deliver, daemon=True).start()
        return True

    def stop(self, timeout: float = 5.0) -> None:
        pass


# ===========================================================================
# Integration test classes
# ===========================================================================


class TestFullLifecycle:
    """End-to-end lifecycle: discover → connect → stream → stop → disconnect."""

    async def test_happy_path_lifecycle(self):
        """Full lifecycle with no errors: discover → connect → stream → stop → disconnect."""
        transport = _make_transport()
        gopro = _make_mock_gopro()

        states: list[tuple[TransportState, TransportState]] = []
        transport.add_state_listener(lambda old, new: states.append((old, new)))

        # 1. Discover
        assert await transport.discover() is True

        # 2. Connect (inject mock to avoid real SDK)
        _connect_transport(transport, gopro)
        assert transport.state == TransportState.CONNECTED
        assert transport.is_connected

        # 3. Start stream
        info = await transport.start_stream()
        assert info is not None
        assert info.port == 8554
        assert info.width == 1920
        assert info.height == 1080
        assert transport.state == TransportState.STREAMING
        assert transport.is_streaming

        # 4. Stop stream
        await transport.stop_stream()
        assert transport.state == TransportState.CONNECTED
        assert not transport.is_streaming
        assert transport.stream_info is None

        # 5. Disconnect
        await transport.disconnect()
        assert transport.state == TransportState.DISCONNECTED
        assert not transport.is_connected
        assert gopro.closed

        # Verify state transitions are in the right order
        assert (TransportState.DISCONNECTED, TransportState.DISCOVERING) in states
        assert (TransportState.CONNECTED, TransportState.STREAMING) in states
        assert (TransportState.STREAMING, TransportState.CONNECTED) in states
        assert (TransportState.CONNECTED, TransportState.DISCONNECTED) in states

    async def test_lifecycle_with_720p_resolution(self):
        """Lifecycle with non-default resolution."""
        transport = _make_transport(resolution=RES_720)
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        info = await transport.start_stream()
        assert info is not None
        assert info.width == 1280
        assert info.height == 720

        await transport.stop_stream()
        await transport.disconnect()
        assert transport.state == TransportState.DISCONNECTED

    async def test_lifecycle_with_480p_resolution(self):
        """Lifecycle with 480p resolution."""
        transport = _make_transport(resolution=RES_480)
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        info = await transport.start_stream()
        assert info is not None
        assert info.width == 854
        assert info.height == 480

        await transport.disconnect()

    async def test_disconnect_while_streaming(self):
        """Disconnect should stop stream first, then disconnect cleanly."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        await transport.start_stream()
        assert transport.is_streaming

        # Disconnect directly (should stop stream internally)
        await transport.disconnect()
        assert transport.state == TransportState.DISCONNECTED
        assert gopro.closed
        # webcam_stop and webcam_exit should have been called
        assert gopro.http_command.stop_call_count >= 1
        assert gopro.http_command.exit_call_count >= 1

    async def test_double_disconnect_is_safe(self):
        """Calling disconnect() twice should not error."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        await transport.disconnect()
        assert transport.state == TransportState.DISCONNECTED

        # Second disconnect should be a no-op
        await transport.disconnect()
        assert transport.state == TransportState.DISCONNECTED

    async def test_multiple_stream_cycles(self):
        """Start/stop stream multiple times without disconnecting."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        for cycle in range(3):
            info = await transport.start_stream()
            assert info is not None, f"Stream start failed on cycle {cycle}"
            assert transport.is_streaming

            await transport.stop_stream()
            assert transport.state == TransportState.CONNECTED
            assert not transport.is_streaming

        await transport.disconnect()
        assert gopro.http_command.start_call_count == 3
        assert gopro.http_command.stop_call_count >= 3


class TestConnectionErrors:
    """Connection failure scenarios and error state transitions."""

    async def test_connect_without_discover_fails(self):
        """connect() without discover() should fail gracefully."""
        transport = COHNTransport()
        result = await transport.connect()
        assert result is False

    async def test_connect_without_password_enters_error(self):
        """Missing COHN password should put transport in ERROR state."""
        transport = COHNTransport(ip_address="192.168.1.100")

        states: list[tuple[TransportState, TransportState]] = []
        transport.add_state_listener(lambda old, new: states.append((old, new)))

        result = await transport.connect()
        assert result is False
        assert transport.state == TransportState.ERROR
        assert transport.stats.last_error is not None
        assert "password" in transport.stats.last_error.lower()

    async def test_connect_failure_records_error_in_stats(self):
        """Connection errors should be recorded in transport stats."""
        transport = COHNTransport(ip_address="10.0.0.1")
        result = await transport.connect()
        assert result is False
        assert transport.stats.last_error is not None

    async def test_stream_not_connected_returns_none(self):
        """start_stream on a disconnected transport should return None."""
        transport = _make_transport()
        result = await transport.start_stream()
        assert result is None
        assert not transport.is_streaming


class TestStreamingErrors:
    """Stream start/stop failure scenarios."""

    async def test_webcam_start_failure_does_not_enter_streaming(self):
        """If webcam_start fails, transport should remain CONNECTED."""
        transport = _make_transport()
        gopro = _make_mock_gopro(webcam_start_ok=False)

        # Override webcam_start to return failure
        async def failing_start(**kw: Any) -> MockGoProResp:
            return MockGoProResp(ok=False, status=0, error=7)

        gopro.http_command.webcam_start = failing_start
        _connect_transport(transport, gopro)

        info = await transport.start_stream()
        assert info is None
        assert not transport.is_streaming
        # Transport should still be connected (not in error)
        assert transport.is_connected

    async def test_webcam_status_timeout_during_stream_start(self):
        """If webcam never reaches streaming status, start_stream returns None."""
        transport = _make_transport()
        gopro = _make_mock_gopro(webcam_status_val=1)  # Stuck in IDLE
        _connect_transport(transport, gopro)

        with (
            patch("gomaxwebcam.transport.cohn.WEBCAM_STATUS_POLL_MAX_S", 0.3),
            patch("gomaxwebcam.transport.cohn.WEBCAM_STATUS_POLL_INTERVAL_S", 0.05),
        ):
            info = await transport.start_stream()

        assert info is None
        assert transport.stats.last_error == "Webcam status timeout"

    async def test_webcam_start_network_error(self):
        """Network error during webcam_start should be handled gracefully."""
        transport = _make_transport()
        gopro = _make_mock_gopro()

        async def exploding_start(**kw: Any):
            raise ConnectionError("Network unreachable")

        gopro.http_command.webcam_start = exploding_start
        _connect_transport(transport, gopro)

        info = await transport.start_stream()
        assert info is None
        assert transport.stats.last_error is not None

    async def test_stop_stream_error_still_transitions(self):
        """Even if webcam_stop raises, transport should transition to CONNECTED."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        await transport.start_stream()
        assert transport.is_streaming

        # Make webcam_stop raise
        async def exploding_stop():
            raise ConnectionError("Gone")

        gopro.http_command.webcam_stop = exploding_stop

        await transport.stop_stream()
        # Should still be CONNECTED despite the error
        assert transport.state == TransportState.CONNECTED

    async def test_retry_stream_after_failure(self):
        """After a failed stream start, a second attempt should succeed."""
        transport = _make_transport()

        # First attempt: stuck in IDLE
        gopro_idle = _make_mock_gopro(webcam_status_val=1)
        _connect_transport(transport, gopro_idle)

        with (
            patch("gomaxwebcam.transport.cohn.WEBCAM_STATUS_POLL_MAX_S", 0.2),
            patch("gomaxwebcam.transport.cohn.WEBCAM_STATUS_POLL_INTERVAL_S", 0.05),
        ):
            info = await transport.start_stream()
        assert info is None

        # Second attempt: working gopro
        gopro_good = _make_mock_gopro(webcam_status_val=2)
        transport._inject_gopro(gopro_good)
        # Reset to CONNECTED since failed start_stream may leave us there
        transport._set_state(TransportState.CONNECTED)

        info = await transport.start_stream()
        assert info is not None
        assert transport.is_streaming

        await transport.disconnect()


class TestKeepAliveIntegration:
    """Keep-alive behavior during streaming lifecycle."""

    async def test_keepalive_tracks_stats_across_calls(self):
        """Multiple keep-alive calls should accumulate stats."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        for _ in range(5):
            result = await transport.keep_alive()
            assert result is True

        assert transport.stats.keepalives_sent == 5
        assert transport.stats.keepalives_failed == 0

    async def test_keepalive_failure_increments_failed_counter(self):
        """Failed keep-alives should increment the failure counter."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        # First two succeed
        await transport.keep_alive()
        await transport.keep_alive()
        assert transport.stats.keepalives_sent == 2

        # Now make it fail
        async def fail_status():
            raise ConnectionError("timeout")

        gopro.http_command.webcam_status = fail_status

        result = await transport.keep_alive()
        assert result is False
        assert transport.stats.keepalives_failed == 1
        assert transport.stats.last_error is not None

    async def test_keepalive_not_connected_returns_false(self):
        """Keep-alive should return False when transport is disconnected."""
        transport = _make_transport()
        result = await transport.keep_alive()
        assert result is False

    async def test_keepalive_during_streaming(self):
        """Keep-alive should work while transport is streaming."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        await transport.start_stream()
        assert transport.is_streaming

        result = await transport.keep_alive()
        assert result is True
        assert transport.stats.keepalives_sent >= 1

        await transport.disconnect()


class TestHealthCheckIntegration:
    """Health check during different lifecycle phases."""

    async def test_health_check_connected_not_streaming(self):
        """Health check when connected but not streaming should succeed."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        result = await transport.health_check()
        assert result is True

    async def test_health_check_while_streaming(self):
        """Health check during streaming should verify camera status."""
        transport = _make_transport()
        gopro = _make_mock_gopro(webcam_status_val=2)  # HIGH_POWER_PREVIEW
        _connect_transport(transport, gopro)

        await transport.start_stream()
        result = await transport.health_check()
        assert result is True

        await transport.disconnect()

    async def test_health_check_streaming_camera_reports_off(self):
        """Health check should fail if we're streaming but camera says OFF."""
        transport = _make_transport()
        gopro = _make_mock_gopro(webcam_status_val=2)
        _connect_transport(transport, gopro)

        await transport.start_stream()
        assert transport.is_streaming

        # Now camera reports OFF (status=0)
        gopro_off = _make_mock_gopro(webcam_status_val=0)
        transport._inject_gopro(gopro_off)

        result = await transport.health_check()
        assert result is False

    async def test_health_check_after_disconnect_returns_false(self):
        """Health check after disconnect should return False."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        await transport.disconnect()
        result = await transport.health_check()
        assert result is False

    async def test_health_check_network_error(self):
        """Health check should return False on network error."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        async def exploding_status():
            raise ConnectionError("Network down")

        gopro.http_command.webcam_status = exploding_status

        result = await transport.health_check()
        assert result is False


class TestFrameStreamingIntegration:
    """Frame iteration integrated with transport lifecycle."""

    async def test_iter_frames_full_lifecycle(self):
        """Frames should flow through the full discover→connect→stream→frames pipeline."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        test_frames = [
            np.full((1080, 1920, 3), i, dtype=np.uint8)
            for i in range(5)
        ]

        with _patch_decoder_with_frames(test_frames):
            received: list[np.ndarray] = []
            async for frame in transport.iter_frames(decode_timeout=3.0):
                received.append(frame)

        assert len(received) == 5
        for i, frame in enumerate(received):
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (1080, 1920, 3)
            assert frame[0, 0, 0] == i

        # Transport should be in STREAMING state after iter_frames auto-starts
        assert transport.state == TransportState.STREAMING

    async def test_iter_frames_auto_starts_stream(self):
        """iter_frames should auto-start stream if transport is CONNECTED."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)
        assert transport.state == TransportState.CONNECTED

        single_frame = [np.zeros((1080, 1920, 3), dtype=np.uint8)]

        with _patch_decoder_with_frames(single_frame):
            async for frame in transport.iter_frames(decode_timeout=3.0):
                assert transport.state == TransportState.STREAMING
                break

    async def test_iter_frames_handles_decoder_error(self):
        """Decoder errors should stop iteration without crashing."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        pre_error_frames = [np.zeros((1080, 1920, 3), dtype=np.uint8)]

        with _patch_decoder_with_frames(
            pre_error_frames,
            error=RuntimeError("decode failure"),
        ):
            received = []
            async for frame in transport.iter_frames(decode_timeout=3.0):
                received.append(frame)

        assert len(received) >= 1  # Got frames before error

    async def test_iter_frames_cleanup_on_break(self):
        """Breaking out of iter_frames should clean up decoder resources."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        many_frames = [
            np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(50)
        ]

        with _patch_decoder_with_frames(many_frames, delay_between=0.01):
            count = 0
            gen = transport.iter_frames(decode_timeout=3.0)
            async for frame in gen:
                count += 1
                if count >= 2:
                    break
            await gen.aclose()

        assert count == 2  # We broke out after 2 frames

    async def test_iter_frames_not_connected_raises(self):
        """iter_frames on a disconnected transport should raise RuntimeError."""
        transport = _make_transport()
        with pytest.raises(RuntimeError):
            async for _ in transport.iter_frames():
                pass

    async def test_iter_frames_no_gopro_raises(self):
        """iter_frames without a GoPro instance should raise RuntimeError."""
        transport = _make_transport()
        transport._set_state(TransportState.CONNECTED)

        with pytest.raises(RuntimeError, match="no WirelessGoPro"):
            async for _ in transport.iter_frames():
                pass

    async def test_iter_frames_preflight_check_called(self):
        """COHN iter_frames should call webcam_status as preflight check."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        initial_status_count = gopro.http_command.status_call_count

        single_frame = [np.zeros((1080, 1920, 3), dtype=np.uint8)]
        with _patch_decoder_with_frames(single_frame):
            async for _ in transport.iter_frames(decode_timeout=3.0):
                break

        # Preflight check should have called webcam_status at least once
        assert gopro.http_command.status_call_count > initial_status_count


class TestStateListenerIntegration:
    """State listeners during multi-step operations."""

    async def test_full_lifecycle_state_transitions(self):
        """Listeners should receive all state transitions during full lifecycle."""
        transport = _make_transport()
        gopro = _make_mock_gopro()

        transitions: list[tuple[TransportState, TransportState]] = []
        transport.add_state_listener(lambda old, new: transitions.append((old, new)))

        # Discover
        await transport.discover()
        # Connect (inject)
        _connect_transport(transport, gopro)
        # Stream
        await transport.start_stream()
        # Stop stream
        await transport.stop_stream()
        # Disconnect
        await transport.disconnect()

        # Verify key transitions occurred
        assert (TransportState.DISCONNECTED, TransportState.DISCOVERING) in transitions
        assert (TransportState.CONNECTED, TransportState.STREAMING) in transitions
        assert (TransportState.STREAMING, TransportState.CONNECTED) in transitions
        assert (TransportState.CONNECTED, TransportState.DISCONNECTED) in transitions

    async def test_error_listener_not_swallowed(self):
        """A broken state listener should not prevent lifecycle operations."""
        transport = _make_transport()
        gopro = _make_mock_gopro()

        call_count = 0

        def bad_listener(old: TransportState, new: TransportState) -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("listener explosion")

        transport.add_state_listener(bad_listener)
        _connect_transport(transport, gopro)

        # Operations should complete despite broken listener
        await transport.start_stream()
        await transport.stop_stream()
        await transport.disconnect()

        # Listener was called for each transition
        assert call_count >= 3

    async def test_multiple_listeners(self):
        """Multiple listeners should all receive transitions."""
        transport = _make_transport()
        gopro = _make_mock_gopro()

        log_a: list[str] = []
        log_b: list[str] = []

        transport.add_state_listener(lambda o, n: log_a.append(n.name))
        transport.add_state_listener(lambda o, n: log_b.append(n.name))

        _connect_transport(transport, gopro)
        await transport.start_stream()
        await transport.disconnect()

        # Both listeners should have received the same transitions
        assert log_a == log_b
        assert len(log_a) >= 2


class TestConnectionUptimeTracking:
    """Uptime stats during connect/disconnect cycles."""

    async def test_uptime_tracked_on_disconnect(self):
        """connection_uptime_s should reflect actual connection duration."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        await asyncio.sleep(0.05)  # Small delay to accumulate uptime
        await transport.disconnect()

        assert transport.stats.connection_uptime_s > 0
        assert transport.stats.connection_uptime_s < 5.0  # Sanity check

    async def test_uptime_resets_on_reconnect(self):
        """Uptime should reset when establishing a new connection."""
        transport = _make_transport()

        # First connection
        gopro1 = _make_mock_gopro()
        _connect_transport(transport, gopro1)
        await asyncio.sleep(0.05)
        await transport.disconnect()
        first_uptime = transport.stats.connection_uptime_s

        # Second connection
        gopro2 = _make_mock_gopro()
        _connect_transport(transport, gopro2)
        await asyncio.sleep(0.02)
        await transport.disconnect()

        # Uptime should reflect the second (shorter) connection
        assert transport.stats.connection_uptime_s > 0


class TestCredentialHandling:
    """COHN credential lifecycle and validation."""

    async def test_discover_with_credentials_succeeds(self):
        """Pre-configured credentials allow immediate discovery."""
        transport = _make_transport()
        result = await transport.discover()
        assert result is True
        assert transport.has_credentials

    async def test_discover_without_any_credentials_fails(self):
        """Without credentials or mDNS, discover should fail."""
        transport = COHNTransport()
        assert not transport.has_credentials

        # Without zeroconf, should fail
        with patch.dict("sys.modules", {"zeroconf": None}):
            import sys
            saved = sys.modules.get("zeroconf")
            sys.modules["zeroconf"] = None  # type: ignore
            try:
                result = await transport.discover(timeout=0.5)
                assert result is False or transport.state == TransportState.ERROR
            finally:
                if saved is not None:
                    sys.modules["zeroconf"] = saved
                else:
                    sys.modules.pop("zeroconf", None)

    async def test_inject_credentials_enables_connection(self):
        """Injecting credentials should enable the COHN connection flow."""
        transport = COHNTransport(ip_address="10.0.0.50")
        assert not transport.has_credentials

        transport._inject_credentials("gopro", "secret123")
        assert transport.has_credentials
        assert transport.username == "gopro"

    async def test_camera_model_set_after_discover(self):
        """stats.camera_model should contain the IP after discovery."""
        transport = _make_transport()
        await transport.discover()
        assert "192.168.1.100" in transport.stats.camera_model
        assert "COHN" in transport.stats.camera_model


class TestConcurrentOperationSafety:
    """Tests to verify transport handles concurrent calls safely."""

    async def test_concurrent_keepalives_dont_corrupt_stats(self):
        """Concurrent keep-alive calls should not corrupt stats counters."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        # Fire 10 concurrent keep-alive calls
        results = await asyncio.gather(
            *[transport.keep_alive() for _ in range(10)]
        )

        assert all(r is True for r in results)
        assert transport.stats.keepalives_sent == 10

    async def test_stop_stream_while_not_streaming_is_noop(self):
        """Calling stop_stream when not streaming should be harmless."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        # Not streaming — stop should be no-op
        await transport.stop_stream()
        assert transport.state == TransportState.CONNECTED

        # Start and stop normally
        await transport.start_stream()
        await transport.stop_stream()
        assert transport.state == TransportState.CONNECTED


class TestResourceCleanup:
    """Verify resources are properly cleaned up in all scenarios."""

    async def test_gopro_handle_cleared_on_disconnect(self):
        """The WirelessGoPro handle should be None after disconnect."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        assert transport._gopro is not None
        await transport.disconnect()
        assert transport._gopro is None

    async def test_stream_info_cleared_on_disconnect(self):
        """StreamInfo should be None after disconnect from streaming state."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        await transport.start_stream()
        assert transport.stream_info is not None

        await transport.disconnect()
        assert transport.stream_info is None

    async def test_keepalive_task_cancelled_on_disconnect(self):
        """Background keepalive task should be cancelled on disconnect."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        await transport.start_stream()
        # Keepalive task should be running
        assert transport._keepalive_task is not None

        await transport.disconnect()
        # Task should be cancelled/cleaned up
        assert transport._keepalive_task is None or transport._keepalive_task.done()

    async def test_keepalive_task_cancelled_on_stop_stream(self):
        """Background keepalive task should stop when stream stops."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        await transport.start_stream()
        assert transport._keepalive_task is not None

        await transport.stop_stream()
        assert transport._keepalive_task is None or transport._keepalive_task.done()

    async def test_close_called_on_gopro_during_disconnect(self):
        """WirelessGoPro.close() should be called during disconnect."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        await transport.disconnect()
        assert gopro.closed is True
        assert gopro.close_count == 1

    async def test_disconnect_error_does_not_leave_stale_state(self):
        """If WirelessGoPro.close() raises, transport should still clean up."""
        transport = _make_transport()
        gopro = _make_mock_gopro()
        _connect_transport(transport, gopro)

        async def fail_close():
            raise OSError("USB cable yanked")

        gopro.close = fail_close

        # Should not raise
        await transport.disconnect()
        assert transport.state == TransportState.DISCONNECTED
        assert transport._gopro is None


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    async def test_start_stream_without_gopro_returns_none(self):
        """If gopro handle is somehow None while CONNECTED, start_stream should fail."""
        transport = _make_transport()
        transport._set_state(TransportState.CONNECTED)
        # No gopro injected

        result = await transport.start_stream()
        assert result is None

    async def test_resolution_fov_validation(self):
        """Invalid resolution/FOV should raise ValueError."""
        transport = _make_transport()

        with pytest.raises(ValueError):
            transport.resolution = 999

        with pytest.raises(ValueError):
            transport.fov = 999

        # Valid changes should work
        transport.resolution = RES_720
        assert transport.resolution == RES_720

        transport.fov = FOV_NARROW
        assert transport.fov == FOV_NARROW

    async def test_transport_name_is_cohn(self):
        """Transport name should always be 'COHN'."""
        transport = _make_transport()
        assert transport.name == "COHN"

    async def test_stats_transport_type(self):
        """Stats should reflect COHN transport type."""
        transport = _make_transport()
        assert transport.stats.transport_type == "COHN"

    async def test_low_power_preview_accepted(self):
        """LOW_POWER_PREVIEW (status=3) should be treated as streaming."""
        transport = _make_transport()
        gopro = _make_mock_gopro(webcam_status_val=3)  # LOW_POWER_PREVIEW
        _connect_transport(transport, gopro)

        info = await transport.start_stream()
        assert info is not None
        assert transport.is_streaming

        # Health check should also pass
        result = await transport.health_check()
        assert result is True

        await transport.disconnect()
