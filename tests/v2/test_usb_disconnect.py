"""
Tests for USB disconnect detection — health-check monitoring and event emission.

No real hardware required.  All tests use the ``no_gopro_needed`` marker
and mock the open-gopro SDK boundary.

Verifies:
  - Consecutive keep-alive failures trigger confirming health check
  - Confirmed failure emits disconnect event and sets ERROR state
  - Successful health check after failures resets the counter
  - Disconnect listener callbacks are invoked with correct args
  - Threshold is configurable via max_consecutive_failures
  - Counter resets on successful keep-alive, connect, and disconnect
"""

from __future__ import annotations

import asyncio
import types
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# All tests are mocked — no GoPro hardware required
pytestmark = pytest.mark.no_gopro_needed

# ---------------------------------------------------------------------------
# Mock SDK enums (same pattern as test_usb_transport.py)
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


class MockWebcamResolution:
    RES_1080 = 12
    def __init__(self, val):
        self._val = val


class MockWebcamFOV:
    WIDE = 0
    def __init__(self, val):
        self._val = val


class MockWebcamProtocol:
    TS = "TS"


class MockGoProResp:
    def __init__(self, ok: bool = True, data: Any = None):
        self.ok = ok
        self.data = data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_gopro() -> MagicMock:
    """Create a mock WiredGoPro with async command methods."""
    gopro = MagicMock()
    gopro.ip_address = "172.20.100.51"
    gopro.open = AsyncMock()
    gopro.close = AsyncMock()

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


async def _connect_and_stream(transport, gopro):
    """Helper: inject mock, connect, and start streaming."""
    transport._inject_gopro(gopro)
    from gomaxwebcam.transport.base import TransportState
    transport._set_state(TransportState.CONNECTED)
    with _patch_sdk_enums():
        info = await transport.start_stream()
    assert info is not None
    return info


# ---------------------------------------------------------------------------
# Tests — Disconnect Detection
# ---------------------------------------------------------------------------


class TestUSBDisconnectDetection:
    """Health-check monitoring detects USB disconnection."""

    @pytest.mark.anyio
    async def test_consecutive_failures_trigger_disconnect(self):
        """After N consecutive keep-alive failures + failed health check,
        transport emits disconnect and enters ERROR state."""
        from gomaxwebcam.transport.usb import USBTransport
        from gomaxwebcam.transport.base import TransportState

        # Use low failure threshold and fast interval for test speed
        transport = USBTransport(
            serial="123",
            keepalive_interval=0.05,
            max_consecutive_failures=2,
        )
        gopro = _make_mock_gopro()

        # Start streaming
        with _patch_sdk_enums():
            await _connect_and_stream(transport, gopro)

        # Now make keep-alive fail
        gopro.http_command.webcam_status = AsyncMock(
            return_value=MockGoProResp(ok=False)
        )

        # Track disconnect events
        disconnect_events = []
        transport.add_disconnect_listener(
            lambda t, reason: disconnect_events.append((t.name, reason))
        )

        # Wait for the keepalive loop to detect failures
        # 2 failures * 0.05s interval + health check time + margin
        await asyncio.sleep(0.5)

        assert transport.state == TransportState.ERROR
        assert len(disconnect_events) == 1
        assert disconnect_events[0] == ("USB", "health_check_failed")
        assert "disconnect" in transport.stats.last_error.lower()

    @pytest.mark.anyio
    async def test_recovery_resets_failure_counter(self):
        """A successful keep-alive after failures resets the counter."""
        from gomaxwebcam.transport.usb import USBTransport
        from gomaxwebcam.transport.base import TransportState

        transport = USBTransport(
            serial="123",
            keepalive_interval=0.05,
            max_consecutive_failures=3,
        )
        gopro = _make_mock_gopro()

        with _patch_sdk_enums():
            await _connect_and_stream(transport, gopro)

        # Fail twice (below threshold of 3)
        call_count = 0
        original_ok = MockGoProResp(
            ok=True,
            data=MockWebcamResponse(
                status=MockWebcamStatus.HIGH_POWER_PREVIEW,
                error=MockWebcamError.SUCCESS,
            ),
        )

        async def _alternating_status(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return MockGoProResp(ok=False)
            return original_ok

        gopro.http_command.webcam_status = AsyncMock(side_effect=_alternating_status)

        disconnect_events = []
        transport.add_disconnect_listener(
            lambda t, reason: disconnect_events.append(reason)
        )

        # Wait for a few cycles
        await asyncio.sleep(0.4)

        # Should still be streaming — failures recovered before threshold
        assert transport.state == TransportState.STREAMING
        assert len(disconnect_events) == 0
        assert transport._consecutive_failures == 0

        # Cleanup
        transport._stop_keepalive_task()

    @pytest.mark.anyio
    async def test_health_check_recovery_after_threshold(self):
        """If health check passes after reaching threshold, no disconnect."""
        from gomaxwebcam.transport.usb import USBTransport
        from gomaxwebcam.transport.base import TransportState

        transport = USBTransport(
            serial="123",
            keepalive_interval=0.05,
            max_consecutive_failures=2,
        )
        gopro = _make_mock_gopro()

        with _patch_sdk_enums():
            await _connect_and_stream(transport, gopro)

        # Keep-alive fails, but health_check passes
        call_count = 0

        async def _failing_then_ok(**kwargs):
            nonlocal call_count
            call_count += 1
            # First 2 calls = keep-alive failures, 3rd = health check success
            if call_count <= 2:
                return MockGoProResp(ok=False)
            return MockGoProResp(
                ok=True,
                data=MockWebcamResponse(
                    status=MockWebcamStatus.HIGH_POWER_PREVIEW,
                ),
            )

        gopro.http_command.webcam_status = AsyncMock(side_effect=_failing_then_ok)

        disconnect_events = []
        transport.add_disconnect_listener(
            lambda t, reason: disconnect_events.append(reason)
        )

        await asyncio.sleep(0.4)

        # Health check saved us — should still be streaming
        assert transport.state == TransportState.STREAMING
        assert len(disconnect_events) == 0

        # Cleanup
        transport._stop_keepalive_task()

    @pytest.mark.anyio
    async def test_exception_in_keepalive_counts_as_failure(self):
        """Exceptions from webcam_status count as failures."""
        from gomaxwebcam.transport.usb import USBTransport
        from gomaxwebcam.transport.base import TransportState

        transport = USBTransport(
            serial="123",
            keepalive_interval=0.05,
            max_consecutive_failures=2,
        )
        gopro = _make_mock_gopro()

        with _patch_sdk_enums():
            await _connect_and_stream(transport, gopro)

        # Make keep-alive raise exceptions
        gopro.http_command.webcam_status = AsyncMock(
            side_effect=ConnectionError("USB cable unplugged")
        )

        disconnect_events = []
        transport.add_disconnect_listener(
            lambda t, reason: disconnect_events.append(reason)
        )

        await asyncio.sleep(0.5)

        assert transport.state == TransportState.ERROR
        assert len(disconnect_events) == 1

    @pytest.mark.anyio
    async def test_connect_resets_failure_counter(self):
        """connect() resets the consecutive failure counter."""
        from gomaxwebcam.transport.usb import USBTransport

        transport = USBTransport(serial="123")
        gopro = _make_mock_gopro()
        transport._inject_gopro(gopro)

        # Simulate some prior failures
        transport._consecutive_failures = 5

        await transport.connect()

        assert transport._consecutive_failures == 0

    @pytest.mark.anyio
    async def test_disconnect_resets_failure_counter(self):
        """disconnect() resets the consecutive failure counter."""
        from gomaxwebcam.transport.usb import USBTransport

        transport = USBTransport(serial="123")
        gopro = _make_mock_gopro()
        transport._inject_gopro(gopro)

        await transport.connect()
        transport._consecutive_failures = 5
        await transport.disconnect()

        assert transport._consecutive_failures == 0

    @pytest.mark.anyio
    async def test_multiple_disconnect_listeners(self):
        """Multiple listeners all get called on disconnect."""
        from gomaxwebcam.transport.usb import USBTransport
        from gomaxwebcam.transport.base import TransportState

        transport = USBTransport(
            serial="123",
            keepalive_interval=0.05,
            max_consecutive_failures=1,
        )
        gopro = _make_mock_gopro()

        with _patch_sdk_enums():
            await _connect_and_stream(transport, gopro)

        gopro.http_command.webcam_status = AsyncMock(
            return_value=MockGoProResp(ok=False)
        )

        results_a = []
        results_b = []
        transport.add_disconnect_listener(
            lambda t, reason: results_a.append(reason)
        )
        transport.add_disconnect_listener(
            lambda t, reason: results_b.append(reason)
        )

        await asyncio.sleep(0.4)

        assert len(results_a) == 1
        assert len(results_b) == 1
        assert results_a[0] == "health_check_failed"

    @pytest.mark.anyio
    async def test_configurable_threshold(self):
        """max_consecutive_failures is respected by the keepalive loop."""
        from gomaxwebcam.transport.usb import USBTransport
        from gomaxwebcam.transport.base import TransportState

        # High threshold — should not trigger disconnect with only 2 failures
        transport = USBTransport(
            serial="123",
            keepalive_interval=0.05,
            max_consecutive_failures=10,
        )
        gopro = _make_mock_gopro()

        with _patch_sdk_enums():
            await _connect_and_stream(transport, gopro)

        # Fail 4 times then recover
        call_count = 0

        async def _limited_failures(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                return MockGoProResp(ok=False)
            return MockGoProResp(
                ok=True,
                data=MockWebcamResponse(
                    status=MockWebcamStatus.HIGH_POWER_PREVIEW,
                ),
            )

        gopro.http_command.webcam_status = AsyncMock(side_effect=_limited_failures)

        disconnect_events = []
        transport.add_disconnect_listener(
            lambda t, reason: disconnect_events.append(reason)
        )

        await asyncio.sleep(0.6)

        # Threshold is 10, only 4 failures — should not disconnect
        assert transport.state == TransportState.STREAMING
        assert len(disconnect_events) == 0

        transport._stop_keepalive_task()


class TestDisconnectListenerBase:
    """Tests for the base Transport disconnect listener mechanism."""

    def test_add_disconnect_listener(self):
        """Can register disconnect listeners on base transport."""
        from gomaxwebcam.transport.usb import USBTransport

        transport = USBTransport(serial="123")
        calls = []
        transport.add_disconnect_listener(lambda t, r: calls.append(r))

        # Manually emit
        transport._emit_disconnect("test_reason")

        assert calls == ["test_reason"]

    def test_emit_disconnect_with_no_listeners(self):
        """Emitting disconnect with no listeners doesn't raise."""
        from gomaxwebcam.transport.usb import USBTransport

        transport = USBTransport(serial="123")
        # Should not raise
        transport._emit_disconnect("test_reason")

    def test_listener_exception_doesnt_break_others(self):
        """An exception in one listener doesn't prevent others from firing."""
        from gomaxwebcam.transport.usb import USBTransport

        transport = USBTransport(serial="123")
        calls = []

        transport.add_disconnect_listener(lambda t, r: (_ for _ in ()).throw(ValueError("boom")))
        transport.add_disconnect_listener(lambda t, r: calls.append(r))

        # The bad listener raises but second one should still fire
        transport._emit_disconnect("test")
        assert calls == ["test"]
