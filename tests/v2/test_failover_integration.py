"""
test_failover_integration.py — End-to-end integration test for USB->COHN reactive failover.

Proves the full failover path through the real component stack:
  AppOrchestrator -> TransportManager -> [USB, COHN] with EventBus wiring

Scenario:
  1. USB transport starts successfully and becomes active
  2. USB transport emits a disconnect event (cable pull)
  3. TransportManager triggers freeze-frame callback
  4. TransportManager fails over to COHN transport
  5. On COHN success: unfreeze callback fires
  6. on_transport_switch propagates to CameraManager
  7. EventBus receives the full event sequence
  8. Recovery loop reconnects USB after COHN also fails

All transports are mocked -- no GoPro hardware or network required.
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
from gomaxwebcam.orchestrator import AppOrchestrator
from gomaxwebcam.camera_manager import CameraManager

# All tests are mocked -- no GoPro needed.
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
    t.is_streaming = stream_ok
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

    # Track disconnect listeners registered by TransportManager
    t._disconnect_listeners = []
    t.add_disconnect_listener = MagicMock(
        side_effect=lambda cb: t._disconnect_listeners.append(cb),
    )

    return t


def _fast_config() -> TransportManagerConfig:
    """Config with short timeouts for fast test execution."""
    return TransportManagerConfig(
        priority=["USB", "COHN"],
        discover_timeout_s=2.0,
        connect_timeout_s=2.0,
        stream_timeout_s=2.0,
        backoff_base_s=0.05,
        backoff_cap_s=0.1,
        auto_recovery=True,
    )


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
def transport_manager(event_bus, usb_transport, cohn_transport):
    """TransportManager with USB + COHN registered, fast backoff."""
    mgr = TransportManager(event_bus, config=_fast_config())
    mgr.register_transport("USB", usb_transport)
    mgr.register_transport("COHN", cohn_transport)
    return mgr


# ---------------------------------------------------------------------------
# Integration tests: Full failover path through real component stack
# ---------------------------------------------------------------------------

class TestUSBToCOHNFailoverIntegration:
    """End-to-end: USB connected -> disconnect event -> COHN failover."""

    @pytest.mark.asyncio
    async def test_full_usb_to_cohn_failover_path(
        self, event_bus, transport_manager, usb_transport, cohn_transport,
    ):
        """Complete reactive failover: USB disconnect -> freeze -> COHN connect -> unfreeze.

        Verifies the entire event chain from disconnect through recovery with
        real TransportManager + EventBus (only transports are mocked).
        """
        # Track callbacks
        freeze_log: list[str] = []
        unfreeze_log: list[str] = []
        switch_log: list[tuple[str, object]] = []

        transport_manager.on_freeze = lambda: freeze_log.append("frozen")
        transport_manager.on_unfreeze = lambda: unfreeze_log.append("unfrozen")
        transport_manager.on_transport_switch = lambda name, t: switch_log.append((name, t))

        # Phase 1: Start on USB
        ok = await transport_manager.start()
        assert ok is True
        assert transport_manager.active_transport_name == "USB"
        assert transport_manager.state == ManagerState.ACTIVE

        # Phase 2: USB cable pull (simulate disconnect)
        assert len(usb_transport._disconnect_listeners) == 1
        listener = usb_transport._disconnect_listeners[0]
        usb_transport.is_streaming = True  # was streaming before pull
        listener(usb_transport, "usb_cable_unplugged")

        # Let async failover complete
        await asyncio.sleep(0.2)

        # Phase 3: Verify freeze-frame was triggered immediately
        assert freeze_log == ["frozen"], "Freeze must fire before failover attempt"

        # Phase 4: Verify USB was cleaned up
        usb_transport.stop_stream.assert_awaited()
        usb_transport.disconnect.assert_awaited()

        # Phase 5: Verify COHN was discovered, connected, and started
        cohn_transport.discover.assert_awaited()
        cohn_transport.connect.assert_awaited()
        cohn_transport.start_stream.assert_awaited()

        # Phase 6: Verify active transport switched to COHN
        assert transport_manager.active_transport_name == "COHN"
        assert transport_manager.state == ManagerState.ACTIVE

        # Phase 7: Verify unfreeze was called after COHN success
        assert unfreeze_log == ["unfrozen"]

        # Phase 8: Verify transport_switch callback was invoked
        # switch_log includes startup (USB) + failover (COHN)
        assert len(switch_log) == 2
        assert switch_log[0][0] == "USB"
        assert switch_log[0][1] is usb_transport
        assert switch_log[1][0] == "COHN"
        assert switch_log[1][1] is cohn_transport

        # Phase 9: Verify failover counter
        assert transport_manager.failover_count == 1

        # Phase 10: Verify EventBus received the event sequence
        recent = event_bus.recent_events
        error_events = [e for e in recent if e.type == EventType.ERROR]
        transport_events = [e for e in recent if e.type == EventType.TRANSPORT]

        # Must have disconnect error event
        assert any(
            "disconnected" in e.data.get("message", "").lower()
            for e in error_events
        ), f"Expected disconnect error event, got: {[e.data for e in error_events]}"

        # Must have failover transport event (USB -> COHN)
        assert any(
            e.data.get("new_transport") == "COHN"
            and e.data.get("reason") == "failover"
            for e in transport_events
        ), f"Expected failover transport event, got: {[e.data for e in transport_events]}"

        await transport_manager.stop()

    @pytest.mark.asyncio
    async def test_failover_event_ordering(
        self, event_bus, transport_manager, usb_transport, cohn_transport,
    ):
        """Freeze must happen BEFORE COHN discover; unfreeze AFTER COHN stream starts."""
        call_order: list[str] = []

        transport_manager.on_freeze = lambda: call_order.append("freeze")
        transport_manager.on_unfreeze = lambda: call_order.append("unfreeze")

        # Instrument COHN mocks to track ordering
        original_discover = cohn_transport.discover

        async def _discover_tracking():
            call_order.append("cohn_discover")
            return await original_discover()

        cohn_transport.discover = AsyncMock(side_effect=_discover_tracking)

        original_start_stream = cohn_transport.start_stream

        async def _stream_tracking():
            call_order.append("cohn_start_stream")
            return await original_start_stream()

        cohn_transport.start_stream = AsyncMock(side_effect=_stream_tracking)

        await transport_manager.start()

        # Trigger USB disconnect
        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "cable_pull")
        await asyncio.sleep(0.2)

        # Verify ordering: freeze -> discover -> stream -> unfreeze
        assert call_order.index("freeze") < call_order.index("cohn_discover"), \
            f"freeze must precede discover, got: {call_order}"
        assert call_order.index("cohn_start_stream") < call_order.index("unfreeze"), \
            f"stream must precede unfreeze, got: {call_order}"

        await transport_manager.stop()

    @pytest.mark.asyncio
    async def test_failover_with_cohn_also_failing_enters_recovery(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """When both USB and COHN fail, manager enters RECOVERING state."""
        # Use a config with auto_recovery disabled so recovery loop
        # doesn't reconnect USB before we can assert RECOVERING state.
        cfg = TransportManagerConfig(
            priority=["USB", "COHN"],
            discover_timeout_s=2.0,
            connect_timeout_s=2.0,
            stream_timeout_s=2.0,
            backoff_base_s=0.05,
            backoff_cap_s=0.1,
            auto_recovery=False,  # Disable so we can inspect RECOVERING
        )
        mgr = TransportManager(event_bus, config=cfg)
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        await mgr.start()
        assert mgr.active_transport_name == "USB"

        # Make COHN fail during failover
        cohn_transport.discover = AsyncMock(return_value=False)

        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "usb_unplugged")
        await asyncio.sleep(0.2)

        assert mgr.state == ManagerState.RECOVERING

        # Verify exhaustion error event
        recent = event_bus.recent_events
        exhaustion_events = [
            e for e in recent
            if e.type == EventType.ERROR
            and "exhausted" in e.data.get("message", "").lower()
        ]
        assert len(exhaustion_events) >= 1

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_recovery_loop_reconnects_usb_after_both_fail(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """Recovery loop retries and reconnects USB after both transports fail."""
        mgr = TransportManager(event_bus, config=_fast_config())
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        await mgr.start()
        assert mgr.active_transport_name == "USB"

        # Make both fail during failover
        cohn_transport.discover = AsyncMock(return_value=False)
        usb_transport.discover = AsyncMock(return_value=False)

        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "cable_pull")
        await asyncio.sleep(0.2)

        assert mgr.state == ManagerState.RECOVERING

        # Now USB comes back online
        usb_transport.discover = AsyncMock(return_value=True)
        usb_transport.connect = AsyncMock(return_value=True)
        usb_transport.start_stream = AsyncMock(return_value=StreamInfo(port=8554))

        # Wait for recovery loop to reconnect (fast backoff)
        await asyncio.sleep(0.5)

        assert mgr.state == ManagerState.ACTIVE
        assert mgr.active_transport_name == "USB"

        # Verify recovery transport event
        recent = event_bus.recent_events
        recovery_events = [
            e for e in recent
            if e.type == EventType.TRANSPORT
            and e.data.get("reason") == "recovery"
        ]
        assert len(recovery_events) >= 1

        await mgr.stop()


class TestOrchestratorIntegration:
    """End-to-end: AppOrchestrator wires callbacks that bridge failover to pipeline."""

    @pytest.mark.asyncio
    async def test_orchestrator_wires_freeze_unfreeze_through_failover(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """AppOrchestrator freeze/unfreeze callbacks reach the pipeline during failover."""
        # Build real TransportManager
        tm = TransportManager(event_bus, config=_fast_config())
        tm.register_transport("USB", usb_transport)
        tm.register_transport("COHN", cohn_transport)

        # Mock CameraManager
        cm = MagicMock(spec=CameraManager)
        cm.start = AsyncMock()
        cm.stop = AsyncMock()
        cm.set_transport = MagicMock()
        cm.set_pipeline = MagicMock()

        # Build orchestrator with mocked internals
        with patch("gomaxwebcam.orchestrator.CohnCredentialStore") as mock_store_cls:
            mock_store_cls.return_value.list_cameras.return_value = []

            orch = AppOrchestrator(
                event_bus=event_bus,
                camera_manager=cm,
                transport_manager=tm,
            )

        # Set up a mock pipeline so freeze/unfreeze reach it
        mock_pipeline = MagicMock()
        mock_pipeline.freeze = MagicMock()
        mock_pipeline.unfreeze = MagicMock()
        mock_pipeline.stop = AsyncMock()
        orch._pipeline = mock_pipeline

        # Wire callbacks manually (normally done in start(), but we skip
        # full start to avoid eager COHN reconnect complications)
        orch._wire_callbacks()

        # Start TransportManager on USB
        ok = await tm.start()
        assert ok is True
        assert tm.active_transport_name == "USB"

        # Simulate USB disconnect
        usb_transport.is_streaming = True
        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "hardware_disconnect")
        await asyncio.sleep(0.3)

        # Verify pipeline freeze was called through orchestrator wiring
        mock_pipeline.freeze.assert_called_once()

        # Verify pipeline unfreeze was called after COHN success
        mock_pipeline.unfreeze.assert_called_once()

        # Verify CameraManager was notified of transport switch
        cm.set_transport.assert_called_with(cohn_transport)

        # Verify final state
        assert tm.active_transport_name == "COHN"
        assert tm.state == ManagerState.ACTIVE

        await tm.stop()

    @pytest.mark.asyncio
    async def test_orchestrator_transport_switch_updates_camera_manager(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """on_transport_switch callback updates CameraManager via disconnect failover."""
        tm = TransportManager(event_bus, config=_fast_config())
        tm.register_transport("USB", usb_transport)
        tm.register_transport("COHN", cohn_transport)

        cm = MagicMock(spec=CameraManager)
        cm.start = AsyncMock()
        cm.stop = AsyncMock()
        cm.set_transport = MagicMock()
        cm.set_pipeline = MagicMock()

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore") as mock_store_cls:
            mock_store_cls.return_value.list_cameras.return_value = []
            orch = AppOrchestrator(
                event_bus=event_bus,
                camera_manager=cm,
                transport_manager=tm,
            )

        orch._wire_callbacks()
        await tm.start()

        # Simulate USB disconnect to trigger reactive failover
        # (on_transport_switch is called in _handle_disconnect path)
        usb_transport.is_streaming = True
        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "cable_pull")
        await asyncio.sleep(0.3)

        # CameraManager.set_transport should have been called with COHN transport
        cm.set_transport.assert_called_with(cohn_transport)
        assert tm.active_transport_name == "COHN"

        await tm.stop()


class TestConcurrentFailoverIntegration:
    """Verify failover serialization under rapid disconnect events."""

    @pytest.mark.asyncio
    async def test_rapid_disconnects_produce_single_failover(
        self, event_bus, transport_manager, usb_transport, cohn_transport,
    ):
        """Multiple rapid USB disconnect events result in exactly one failover."""
        freeze_count: list[int] = []
        transport_manager.on_freeze = lambda: freeze_count.append(1)
        transport_manager.on_unfreeze = lambda: None

        await transport_manager.start()

        listener = usb_transport._disconnect_listeners[0]

        # Fire 5 rapid disconnects
        for i in range(5):
            listener(usb_transport, f"reason_{i}")

        await asyncio.sleep(0.3)

        # Only one failover should execute
        assert transport_manager.failover_count == 1
        assert len(freeze_count) == 1
        assert transport_manager.active_transport_name == "COHN"
        assert transport_manager.state == ManagerState.ACTIVE

        await transport_manager.stop()


class TestStatusDuringFailover:
    """Verify get_status() reflects state transitions during failover."""

    @pytest.mark.asyncio
    async def test_status_reflects_active_transport_after_failover(
        self, transport_manager, usb_transport, cohn_transport,
    ):
        """get_status() shows COHN as active transport after USB->COHN failover."""
        transport_manager.on_freeze = lambda: None
        transport_manager.on_unfreeze = lambda: None

        await transport_manager.start()

        status_before = transport_manager.get_status()
        assert status_before["active_transport"] == "USB"
        assert status_before["failover_count"] == 0

        # Trigger failover
        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "disconnect")
        await asyncio.sleep(0.2)

        status_after = transport_manager.get_status()
        assert status_after["active_transport"] == "COHN"
        assert status_after["state"] == "ACTIVE"
        assert status_after["failover_count"] == 1

        await transport_manager.stop()


class TestEventBusIntegration:
    """Verify the full event sequence published during failover."""

    @pytest.mark.asyncio
    async def test_event_sequence_usb_startup_then_failover_to_cohn(
        self, event_bus, transport_manager, usb_transport, cohn_transport,
    ):
        """EventBus receives startup event, error event, and failover event in order."""
        transport_manager.on_freeze = lambda: None
        transport_manager.on_unfreeze = lambda: None

        await transport_manager.start()

        # Clear event snapshot
        initial_events = list(event_bus.recent_events)

        # Verify startup transport event
        startup_events = [
            e for e in initial_events
            if e.type == EventType.TRANSPORT
            and e.data.get("reason") == "startup"
        ]
        assert len(startup_events) == 1
        assert startup_events[0].data["new_transport"] == "USB"

        # Trigger failover
        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "health_check_failed")
        await asyncio.sleep(0.2)

        # Get all events after startup
        all_events = list(event_bus.recent_events)
        post_startup = [e for e in all_events if e not in initial_events]

        # Must have: error (disconnect) + transport (failover)
        error_events = [e for e in post_startup if e.type == EventType.ERROR]
        transport_events = [e for e in post_startup if e.type == EventType.TRANSPORT]

        assert len(error_events) >= 1, "Must publish disconnect error event"
        assert len(transport_events) >= 1, "Must publish failover transport event"

        failover_event = transport_events[0]
        assert failover_event.data["old_transport"] == "USB"
        assert failover_event.data["new_transport"] == "COHN"
        assert failover_event.data["reason"] == "failover"

        await transport_manager.stop()
