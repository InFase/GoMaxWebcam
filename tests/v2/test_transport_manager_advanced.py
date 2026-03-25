"""
test_transport_manager_advanced.py — Advanced TransportManager unit tests.

Covers edge cases NOT tested in test_transport_manager_failover.py:
  - stop() during active failover aborts cleanly
  - stop() cancels recovery loop
  - unregister_transport removes from failover chain
  - start() with preferred transport override
  - start() when already running returns current state
  - start() with empty priority list fails gracefully
  - get_status() during FAILING_OVER state
  - active_transport property returns correct transport object
  - is_running property tracks lifecycle
  - force_failover to unregistered target fails gracefully
  - force_failover when not running fails gracefully
  - Backoff attempt resets after successful failover

All tests are pure unit tests with mocks — no GoPro or hardware needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gomaxwebcam.events import EventBus
from gomaxwebcam.transport.base import Transport, TransportState, StreamInfo
from gomaxwebcam.transport_manager import (
    TransportManager,
    TransportManagerConfig,
    ManagerState,
)

pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_transport(
    name: str,
    *,
    discover_ok: bool = True,
    connect_ok: bool = True,
    stream_ok: bool = True,
) -> MagicMock:
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


def _fast_config(**overrides) -> TransportManagerConfig:
    defaults = dict(
        priority=["USB", "COHN"],
        discover_timeout_s=2.0,
        connect_timeout_s=2.0,
        stream_timeout_s=2.0,
        backoff_base_s=0.05,
        backoff_cap_s=0.1,
        auto_recovery=False,
    )
    defaults.update(overrides)
    return TransportManagerConfig(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def usb_transport():
    return _make_mock_transport("USB")


@pytest.fixture
def cohn_transport():
    return _make_mock_transport("COHN")


@pytest.fixture
def manager(event_bus, usb_transport, cohn_transport):
    mgr = TransportManager(event_bus, config=_fast_config())
    mgr.register_transport("USB", usb_transport)
    mgr.register_transport("COHN", cohn_transport)
    return mgr


# ---------------------------------------------------------------------------
# Tests: stop() edge cases
# ---------------------------------------------------------------------------

class TestStopEdgeCases:
    """stop() edge cases during failover and recovery."""

    @pytest.mark.asyncio
    async def test_stop_during_recovery_cancels_recovery_task(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """stop() cancels the recovery loop task."""
        mgr = TransportManager(
            event_bus,
            config=_fast_config(auto_recovery=True, backoff_base_s=0.5),
        )
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        await mgr.start()

        # Make both fail during failover
        cohn_transport.discover = AsyncMock(return_value=False)
        usb_transport.discover = AsyncMock(return_value=False)

        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "unplugged")
        await asyncio.sleep(0.2)

        assert mgr.state == ManagerState.RECOVERING

        # Stop during recovery — should not raise
        await mgr.stop()
        assert mgr.state == ManagerState.STOPPED
        assert not mgr.is_running

    @pytest.mark.asyncio
    async def test_stop_when_idle_is_noop(self, event_bus):
        """stop() on an idle (never started) manager is a safe no-op."""
        mgr = TransportManager(event_bus)
        await mgr.stop()
        assert mgr.state == ManagerState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_disconnects_active_transport(
        self, manager, usb_transport,
    ):
        """stop() calls disconnect on the active transport."""
        await manager.start()
        assert manager.active_transport_name == "USB"

        await manager.stop()

        usb_transport.disconnect.assert_awaited()
        assert manager.active_transport_name == ""

    @pytest.mark.asyncio
    async def test_stop_handles_disconnect_error_gracefully(
        self, manager, usb_transport,
    ):
        """stop() handles transport.disconnect() raising an exception."""
        usb_transport.disconnect = AsyncMock(side_effect=RuntimeError("USB hung"))

        await manager.start()
        await manager.stop()  # Should not raise

        assert manager.state == ManagerState.STOPPED


# ---------------------------------------------------------------------------
# Tests: start() edge cases
# ---------------------------------------------------------------------------

class TestStartEdgeCases:
    """start() edge cases."""

    @pytest.mark.asyncio
    async def test_start_with_preferred_transport(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """start(preferred='COHN') starts COHN instead of USB."""
        mgr = TransportManager(event_bus, config=_fast_config())
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        ok = await mgr.start(preferred="COHN")

        assert ok is True
        assert mgr.active_transport_name == "COHN"
        cohn_transport.discover.assert_awaited_once()
        usb_transport.discover.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_start_when_already_running_returns_current(
        self, manager,
    ):
        """start() when already running returns True if active."""
        ok1 = await manager.start()
        assert ok1 is True

        ok2 = await manager.start()
        assert ok2 is True
        assert manager.state == ManagerState.ACTIVE

    @pytest.mark.asyncio
    async def test_start_with_unregistered_preferred_fails(
        self, event_bus, usb_transport,
    ):
        """start(preferred='NONEXISTENT') returns False."""
        mgr = TransportManager(event_bus, config=_fast_config())
        mgr.register_transport("USB", usb_transport)

        ok = await mgr.start(preferred="NONEXISTENT")

        assert ok is False
        assert mgr.state == ManagerState.IDLE

    @pytest.mark.asyncio
    async def test_start_with_empty_priority_and_no_preferred(
        self, event_bus, usb_transport,
    ):
        """start() with empty priority list and no preferred fails."""
        cfg = _fast_config(priority=[])
        mgr = TransportManager(event_bus, config=cfg)
        mgr.register_transport("USB", usb_transport)

        ok = await mgr.start()
        assert ok is False

    @pytest.mark.asyncio
    async def test_start_connect_failure_tries_next(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """When USB connect fails, start() tries COHN."""
        usb_transport.connect = AsyncMock(return_value=False)

        mgr = TransportManager(event_bus, config=_fast_config())
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        ok = await mgr.start()

        assert ok is True
        assert mgr.active_transport_name == "COHN"

    @pytest.mark.asyncio
    async def test_start_stream_failure_tries_next(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """When USB start_stream fails, start() tries COHN."""
        usb_transport.start_stream = AsyncMock(return_value=None)

        mgr = TransportManager(event_bus, config=_fast_config())
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        ok = await mgr.start()

        assert ok is True
        assert mgr.active_transport_name == "COHN"


# ---------------------------------------------------------------------------
# Tests: unregister_transport
# ---------------------------------------------------------------------------

class TestUnregister:
    """unregister_transport edge cases."""

    def test_unregister_removes_transport(self, event_bus, usb_transport, cohn_transport):
        """Unregistered transport is removed from registered_transports."""
        mgr = TransportManager(event_bus)
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        assert "COHN" in mgr.registered_transports
        mgr.unregister_transport("COHN")
        assert "COHN" not in mgr.registered_transports

    def test_unregister_nonexistent_is_noop(self, event_bus):
        """Unregistering a non-existent transport is a safe no-op."""
        mgr = TransportManager(event_bus)
        mgr.unregister_transport("NONEXISTENT")  # Should not raise


# ---------------------------------------------------------------------------
# Tests: Properties
# ---------------------------------------------------------------------------

class TestProperties:
    """Property tests."""

    @pytest.mark.asyncio
    async def test_active_transport_returns_object(
        self, manager, usb_transport,
    ):
        """active_transport returns the Transport instance."""
        await manager.start()
        assert manager.active_transport is usb_transport

    @pytest.mark.asyncio
    async def test_active_transport_none_when_idle(self, event_bus):
        """active_transport is None when no transport is active."""
        mgr = TransportManager(event_bus)
        assert mgr.active_transport is None

    @pytest.mark.asyncio
    async def test_is_running_tracks_lifecycle(self, manager):
        """is_running reflects the manager lifecycle."""
        assert not manager.is_running

        await manager.start()
        assert manager.is_running

        await manager.stop()
        assert not manager.is_running


# ---------------------------------------------------------------------------
# Tests: get_status() states
# ---------------------------------------------------------------------------

class TestGetStatus:
    """get_status() during different states."""

    @pytest.mark.asyncio
    async def test_status_idle(self, event_bus):
        """get_status() returns IDLE before start."""
        mgr = TransportManager(event_bus)
        status = mgr.get_status()
        assert status["state"] == "IDLE"
        assert status["active_transport"] == ""

    @pytest.mark.asyncio
    async def test_status_stopped(self, manager):
        """get_status() returns STOPPED after stop."""
        await manager.start()
        await manager.stop()
        status = manager.get_status()
        assert status["state"] == "STOPPED"

    @pytest.mark.asyncio
    async def test_status_recovering(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """get_status() returns RECOVERING when all transports exhausted."""
        mgr = TransportManager(event_bus, config=_fast_config())
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        await mgr.start()

        cohn_transport.discover = AsyncMock(return_value=False)
        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "unplugged")
        await asyncio.sleep(0.2)

        status = mgr.get_status()
        assert status["state"] == "RECOVERING"

        await mgr.stop()


# ---------------------------------------------------------------------------
# Tests: force_failover edge cases
# ---------------------------------------------------------------------------

class TestForceFailoverEdgeCases:
    """force_failover edge cases."""

    @pytest.mark.asyncio
    async def test_force_failover_to_unregistered_target(self, manager):
        """force_failover to an unregistered name returns False."""
        await manager.start()
        ok = await manager.force_failover("WIFI_AP")
        assert ok is False
        assert manager.active_transport_name == "USB"
        await manager.stop()

    @pytest.mark.asyncio
    async def test_force_failover_when_target_fails(
        self, manager, cohn_transport,
    ):
        """force_failover returns False when the target transport fails."""
        await manager.start()

        cohn_transport.discover = AsyncMock(return_value=False)

        ok = await manager.force_failover("COHN")
        assert ok is False
        # Should still be on USB (unchanged)
        assert manager.active_transport_name == "USB"

        await manager.stop()


# ---------------------------------------------------------------------------
# Tests: Backoff state management
# ---------------------------------------------------------------------------

class TestBackoffState:
    """Backoff attempt tracking."""

    @pytest.mark.asyncio
    async def test_backoff_resets_after_successful_failover(
        self, manager, usb_transport, cohn_transport,
    ):
        """_backoff_attempt resets to 0 after successful failover."""
        await manager.start()

        # Simulate disconnect → successful failover
        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "disconnect")
        await asyncio.sleep(0.2)

        assert manager.state == ManagerState.ACTIVE
        assert manager._backoff_attempt == 0

        await manager.stop()


# ---------------------------------------------------------------------------
# Tests: Disconnect from unknown/unregistered transport
# ---------------------------------------------------------------------------

class TestDisconnectEdgeCases:
    """Disconnect listener edge cases."""

    @pytest.mark.asyncio
    async def test_disconnect_after_stop_ignored(
        self, manager, usb_transport,
    ):
        """Disconnect events after stop() are ignored."""
        await manager.start()
        await manager.stop()

        # Fire disconnect after stop
        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "late_disconnect")
        await asyncio.sleep(0.1)

        # Should still be stopped
        assert manager.state == ManagerState.STOPPED
        assert manager.failover_count == 0
