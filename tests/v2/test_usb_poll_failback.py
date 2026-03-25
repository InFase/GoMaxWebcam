"""
test_usb_poll_failback.py — Tests for USB device presence polling and fail-back.

Verifies Sub-AC 4.1:
  - USB polling loop starts when COHN becomes the active transport
  - USB polling loop stops when USB becomes the active transport
  - Polling detects USB device and emits USB_REDETECTED event
  - Polling triggers force_failover("USB") when device is found
  - Polling is cancelled on manager stop
  - Polling handles discovery errors gracefully
  - Polling does not start if USB transport is not registered

All tests are pure unit tests with mocks — no GoPro or hardware needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gomaxwebcam.events import EventBus, EventType
from gomaxwebcam.transport.base import Transport, TransportState, StreamInfo
from gomaxwebcam.transport_manager import (
    TransportManager,
    TransportManagerConfig,
    ManagerState,
)

# All tests in this module are pure unit tests with mocks — no GoPro needed.
pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_transport(name: str, *, discover_ok: bool = True,
                         connect_ok: bool = True,
                         stream_ok: bool = True) -> MagicMock:
    """Create a mock Transport with controllable discover/connect/start_stream."""
    t = MagicMock(spec=Transport)
    t.name = name
    t._name = name
    t.state = TransportState.DISCONNECTED
    t.is_streaming = False
    t.is_connected = False
    t.stream_info = StreamInfo(port=8554)

    t.discover = AsyncMock(return_value=discover_ok)
    t.connect = AsyncMock(return_value=connect_ok)
    t.start_stream = AsyncMock(
        return_value=StreamInfo(port=8554) if stream_ok else None,
    )
    t.stop_stream = AsyncMock()
    t.disconnect = AsyncMock()
    t.health_check = AsyncMock(return_value=True)
    t.keep_alive = AsyncMock(return_value=True)

    t._disconnect_listeners = []
    t.add_disconnect_listener = MagicMock(
        side_effect=lambda cb: t._disconnect_listeners.append(cb),
    )

    return t


class FakeDeviceInfo:
    """Minimal fake GoProDeviceInfo for test assertions."""
    def __init__(self, serial: str = "C3531350067212", ip: str = "172.22.112.51"):
        self.serial_number = serial
        self.camera_ip = ip


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_usb_poll_starts_when_cohn_active():
    """USB polling should start when COHN becomes the active transport."""
    bus = EventBus()
    cfg = TransportManagerConfig(usb_poll_interval_s=0.1)
    mgr = TransportManager(bus, config=cfg)

    usb = _make_mock_transport("USB", discover_ok=False)  # USB not available
    cohn = _make_mock_transport("COHN")

    mgr.register_transport("USB", usb)
    mgr.register_transport("COHN", cohn)

    # Patch _check_usb_presence to never find USB (we just want to verify polling starts)
    with patch.object(mgr, "_check_usb_presence", new_callable=AsyncMock, return_value=None):
        # Start with COHN (USB discover fails)
        mgr._running = True
        await mgr._notify_transport_switch("COHN")

        # Polling task should have been created
        assert mgr._usb_poll_task is not None
        assert not mgr._usb_poll_task.done()

        # Clean up
        mgr._stop_usb_poll()
        mgr._running = False


@pytest.mark.asyncio
async def test_usb_poll_stops_when_usb_active():
    """USB polling should stop when USB becomes the active transport."""
    bus = EventBus()
    mgr = TransportManager(bus)

    usb = _make_mock_transport("USB")
    cohn = _make_mock_transport("COHN")

    mgr.register_transport("USB", usb)
    mgr.register_transport("COHN", cohn)

    with patch.object(mgr, "_check_usb_presence", new_callable=AsyncMock, return_value=None):
        # Start polling (COHN active)
        mgr._running = True
        await mgr._notify_transport_switch("COHN")
        assert mgr._usb_poll_task is not None

        # Switch to USB — poll should stop
        await mgr._notify_transport_switch("USB")
        assert mgr._usb_poll_task is None

        mgr._running = False


@pytest.mark.asyncio
async def test_usb_poll_emits_event_on_detection():
    """When USB device is found during polling, USB_REDETECTED event should be published."""
    bus = EventBus()
    cfg = TransportManagerConfig(usb_poll_interval_s=0.05)
    mgr = TransportManager(bus, config=cfg)

    usb = _make_mock_transport("USB")
    cohn = _make_mock_transport("COHN")

    mgr.register_transport("USB", usb)
    mgr.register_transport("COHN", cohn)

    fake_device = FakeDeviceInfo()
    events_received = []

    # Subscribe to USB_REDETECTED events
    sub = bus.subscribe(event_types={EventType.USB_REDETECTED})

    mgr._running = True
    mgr._active_name = "COHN"
    mgr._state = ManagerState.ACTIVE

    # Patch _check_usb_presence to return a device after a short delay
    call_count = 0

    async def _fake_check():
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            return fake_device
        return None

    with patch.object(mgr, "_check_usb_presence", side_effect=_fake_check):
        # Also patch force_failover to succeed without actually doing anything
        with patch.object(mgr, "force_failover", new_callable=AsyncMock, return_value=True) as mock_failover:
            # Start the poll loop directly (bypass initial delay)
            with patch("gomaxwebcam.transport_manager.USB_POLL_INITIAL_DELAY_S", 0.01):
                mgr._usb_poll_task = asyncio.create_task(mgr._usb_poll_loop())

                # Wait for the poll to detect USB and trigger failback
                await asyncio.sleep(0.5)

                # force_failover should have been called with "USB"
                mock_failover.assert_called_once_with("USB")

    # Check that USB_REDETECTED event was published
    found = [e for e in bus.recent_events if e.type == EventType.USB_REDETECTED]
    assert len(found) >= 1
    assert found[0].data["camera_serial"] == "C3531350067212"
    assert found[0].data["camera_ip"] == "172.22.112.51"

    mgr._stop_usb_poll()
    mgr._running = False


@pytest.mark.asyncio
async def test_usb_poll_retries_on_failback_failure():
    """If force_failover fails, the poll loop should keep trying."""
    bus = EventBus()
    cfg = TransportManagerConfig(usb_poll_interval_s=0.05)
    mgr = TransportManager(bus, config=cfg)

    usb = _make_mock_transport("USB")
    cohn = _make_mock_transport("COHN")

    mgr.register_transport("USB", usb)
    mgr.register_transport("COHN", cohn)
    mgr._running = True
    mgr._active_name = "COHN"
    mgr._state = ManagerState.ACTIVE

    failover_calls = 0

    async def _failover_side_effect(target):
        nonlocal failover_calls
        failover_calls += 1
        if failover_calls < 2:
            return False  # First attempt fails
        mgr._active_name = "USB"  # Second attempt succeeds
        return True

    with patch.object(mgr, "_check_usb_presence", new_callable=AsyncMock, return_value=FakeDeviceInfo()):
        with patch.object(mgr, "force_failover", side_effect=_failover_side_effect):
            with patch("gomaxwebcam.transport_manager.USB_POLL_INITIAL_DELAY_S", 0.01):
                mgr._usb_poll_task = asyncio.create_task(mgr._usb_poll_loop())
                await asyncio.sleep(0.5)

    assert failover_calls >= 2, f"Expected at least 2 failover attempts, got {failover_calls}"

    mgr._stop_usb_poll()
    mgr._running = False


@pytest.mark.asyncio
async def test_usb_poll_handles_discovery_errors():
    """Polling should handle errors from _check_usb_presence gracefully."""
    bus = EventBus()
    cfg = TransportManagerConfig(usb_poll_interval_s=0.05)
    mgr = TransportManager(bus, config=cfg)

    usb = _make_mock_transport("USB")
    cohn = _make_mock_transport("COHN")

    mgr.register_transport("USB", usb)
    mgr.register_transport("COHN", cohn)
    mgr._running = True
    mgr._active_name = "COHN"

    call_count = 0

    async def _error_then_find():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise OSError("WMI query failed")
        # After errors, stop the loop by setting active to USB
        mgr._active_name = "USB"
        return None

    with patch.object(mgr, "_check_usb_presence", side_effect=_error_then_find):
        with patch("gomaxwebcam.transport_manager.USB_POLL_INITIAL_DELAY_S", 0.01):
            mgr._usb_poll_task = asyncio.create_task(mgr._usb_poll_loop())
            await asyncio.sleep(0.5)

    # Should have survived the errors and continued
    assert call_count >= 2

    mgr._stop_usb_poll()
    mgr._running = False


@pytest.mark.asyncio
async def test_usb_poll_no_start_without_usb_transport():
    """USB polling should not start if USB transport is not registered."""
    bus = EventBus()
    mgr = TransportManager(bus)

    cohn = _make_mock_transport("COHN")
    mgr.register_transport("COHN", cohn)
    # USB is NOT registered

    mgr._running = True
    await mgr._notify_transport_switch("COHN")

    # No poll task should be created
    assert mgr._usb_poll_task is None

    mgr._running = False


@pytest.mark.asyncio
async def test_usb_poll_cancelled_on_stop():
    """USB polling should be cancelled when the manager is stopped."""
    bus = EventBus()
    cfg = TransportManagerConfig(usb_poll_interval_s=0.1)
    mgr = TransportManager(bus, config=cfg)

    usb = _make_mock_transport("USB")
    cohn = _make_mock_transport("COHN")

    mgr.register_transport("USB", usb)
    mgr.register_transport("COHN", cohn)

    with patch.object(mgr, "_check_usb_presence", new_callable=AsyncMock, return_value=None):
        mgr._running = True
        mgr._active_name = "COHN"
        mgr._state = ManagerState.ACTIVE
        await mgr._notify_transport_switch("COHN")

        assert mgr._usb_poll_task is not None
        task = mgr._usb_poll_task

        # Stop the manager
        await mgr.stop()

        # Allow cancellation to propagate
        await asyncio.sleep(0.05)

        # Task should be cancelled or done
        assert mgr._usb_poll_task is None
        assert task.done() or task.cancelled()


@pytest.mark.asyncio
async def test_usb_poll_status_reported():
    """get_status() should report whether USB polling is active."""
    bus = EventBus()
    mgr = TransportManager(bus)

    usb = _make_mock_transport("USB")
    cohn = _make_mock_transport("COHN")

    mgr.register_transport("USB", usb)
    mgr.register_transport("COHN", cohn)

    # Initially no polling
    status = mgr.get_status()
    assert status["usb_poll_active"] is False

    with patch.object(mgr, "_check_usb_presence", new_callable=AsyncMock, return_value=None):
        mgr._running = True
        await mgr._notify_transport_switch("COHN")
        await asyncio.sleep(0.05)  # Let task start

        status = mgr.get_status()
        assert status["usb_poll_active"] is True

        mgr._stop_usb_poll()
        mgr._running = False
