"""
test_transport_manager_failover.py — Tests for TransportManager failover orchestration.

Verifies Sub-AC 3b:
  - On disconnect event, freeze-frame is triggered
  - Failover switches active transport from USB to COHN
  - Unfreeze is called on successful failover
  - on_transport_switch callback is invoked with new transport
  - EventBus receives transport change and error events
  - Concurrent disconnect events are serialized by failover lock
  - Recovery loop activates when all transports are exhausted
  - Orchestrator _on_freeze / _on_unfreeze call pipeline.freeze()/unfreeze()

All tests are pure unit tests with mocks — no GoPro or hardware needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

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

    # Track disconnect listeners registered by TransportManager
    t._disconnect_listeners = []
    t.add_disconnect_listener = MagicMock(
        side_effect=lambda cb: t._disconnect_listeners.append(cb),
    )

    return t


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def event_loop_ref(request):
    """Get the current asyncio event loop for the test (pytest-asyncio managed)."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        # Not inside an async test — create a loop for sync fixtures
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()
        return


@pytest.fixture
def event_bus():
    bus = EventBus()
    # Loop will be set inside async tests via set_loop; no eager call here
    return bus


@pytest.fixture
def usb_transport():
    return _make_mock_transport("USB")


@pytest.fixture
def cohn_transport():
    return _make_mock_transport("COHN")


@pytest.fixture
def manager(event_bus, usb_transport, cohn_transport):
    """Pre-wired TransportManager with USB + COHN registered."""
    cfg = TransportManagerConfig(
        priority=["USB", "COHN"],
        discover_timeout_s=2.0,
        connect_timeout_s=2.0,
        stream_timeout_s=2.0,
        backoff_base_s=0.1,
        backoff_cap_s=0.5,
    )
    mgr = TransportManager(event_bus, config=cfg)
    mgr.register_transport("USB", usb_transport)
    mgr.register_transport("COHN", cohn_transport)
    return mgr


# ---------------------------------------------------------------------------
# Tests: Registration & disconnect listener wiring
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_wires_disconnect_listener(self, usb_transport):
        """register_transport adds a disconnect listener to the transport."""
        bus = EventBus()
        mgr = TransportManager(bus)
        mgr.register_transport("USB", usb_transport)

        usb_transport.add_disconnect_listener.assert_called_once()
        assert len(usb_transport._disconnect_listeners) == 1

    def test_registered_transports_returns_copy(self, manager, usb_transport, cohn_transport):
        registered = manager.registered_transports
        assert "USB" in registered
        assert "COHN" in registered
        assert registered["USB"] is usb_transport


# ---------------------------------------------------------------------------
# Tests: Startup
# ---------------------------------------------------------------------------

class TestStartup:
    @pytest.mark.asyncio
    async def test_start_connects_first_priority(self, manager, usb_transport):
        """start() discovers, connects, and starts stream on the first transport."""
        ok = await manager.start()

        assert ok is True
        assert manager.state == ManagerState.ACTIVE
        assert manager.active_transport_name == "USB"
        usb_transport.discover.assert_awaited_once()
        usb_transport.connect.assert_awaited_once()
        usb_transport.start_stream.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_falls_back_to_cohn_on_usb_failure(
        self, manager, usb_transport, cohn_transport,
    ):
        """If USB fails at startup, falls back to COHN."""
        usb_transport.discover = AsyncMock(return_value=False)

        ok = await manager.start()

        assert ok is True
        assert manager.active_transport_name == "COHN"
        cohn_transport.discover.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_returns_false_when_all_fail(
        self, manager, usb_transport, cohn_transport,
    ):
        """start() returns False when all transports fail."""
        usb_transport.discover = AsyncMock(return_value=False)
        cohn_transport.discover = AsyncMock(return_value=False)

        ok = await manager.start()

        assert ok is False
        assert manager.state == ManagerState.IDLE


# ---------------------------------------------------------------------------
# Tests: Failover on disconnect
# ---------------------------------------------------------------------------

class TestFailover:
    @pytest.mark.asyncio
    async def test_disconnect_triggers_freeze_then_failover_then_unfreeze(
        self, manager, usb_transport, cohn_transport,
    ):
        """Full failover flow: disconnect → freeze → switch to COHN → unfreeze."""
        freeze_calls = []
        unfreeze_calls = []
        switch_calls = []

        manager.on_freeze = lambda: freeze_calls.append("frozen")
        manager.on_unfreeze = lambda: unfreeze_calls.append("unfrozen")
        manager.on_transport_switch = lambda name, t: switch_calls.append(name)

        # Start on USB
        await manager.start()
        assert manager.active_transport_name == "USB"

        # Simulate USB disconnect
        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "usb_unplugged")

        # Let the async failover task run
        await asyncio.sleep(0.1)

        # Verify freeze was called
        assert freeze_calls == ["frozen"]

        # Verify switched to COHN
        assert manager.active_transport_name == "COHN"
        assert manager.state == ManagerState.ACTIVE
        cohn_transport.discover.assert_awaited()
        cohn_transport.connect.assert_awaited()
        cohn_transport.start_stream.assert_awaited()

        # Verify unfreeze was called after successful failover
        assert unfreeze_calls == ["unfrozen"]

        # Verify transport switch callback was called (USB at startup, COHN at failover)
        assert switch_calls == ["USB", "COHN"]

        # Verify failover count incremented
        assert manager.failover_count == 1

        await manager.stop()

    @pytest.mark.asyncio
    async def test_disconnect_publishes_events_to_bus(
        self, manager, event_bus, usb_transport,
    ):
        """Failover publishes error + transport events to EventBus."""
        await manager.start()

        # Capture events
        initial_count = event_bus.event_count

        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "health_check_failed")
        await asyncio.sleep(0.1)

        # Check events were published
        recent = event_bus.recent_events
        error_events = [e for e in recent if e.type == EventType.ERROR]
        transport_events = [e for e in recent if e.type == EventType.TRANSPORT]

        assert len(error_events) >= 1
        assert any("disconnected" in e.data.get("message", "").lower()
                    for e in error_events)
        assert len(transport_events) >= 2  # startup + failover

        await manager.stop()

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_failed_transport(
        self, manager, usb_transport,
    ):
        """Failover calls disconnect on the failed transport."""
        await manager.start()

        usb_transport.is_streaming = True

        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "cable_unplugged")
        await asyncio.sleep(0.1)

        # USB transport should have been cleaned up
        usb_transport.stop_stream.assert_awaited()
        usb_transport.disconnect.assert_awaited()

        await manager.stop()

    @pytest.mark.asyncio
    async def test_non_active_disconnect_is_ignored(
        self, manager, usb_transport, cohn_transport,
    ):
        """Disconnect from a non-active transport is ignored."""
        await manager.start()
        assert manager.active_transport_name == "USB"

        # COHN disconnects — should be ignored since USB is active
        cohn_listener = cohn_transport._disconnect_listeners[0]
        cohn_listener(cohn_transport, "wifi_lost")
        await asyncio.sleep(0.1)

        # Still on USB, no failover
        assert manager.active_transport_name == "USB"
        assert manager.failover_count == 0

        await manager.stop()

    @pytest.mark.asyncio
    async def test_concurrent_disconnects_are_serialized(
        self, manager, usb_transport, cohn_transport,
    ):
        """Multiple rapid disconnect events don't cause concurrent failovers."""
        freeze_count = []
        manager.on_freeze = lambda: freeze_count.append(1)

        await manager.start()

        listener = usb_transport._disconnect_listeners[0]
        # Fire 3 rapid disconnects
        listener(usb_transport, "reason1")
        listener(usb_transport, "reason2")
        listener(usb_transport, "reason3")

        await asyncio.sleep(0.2)

        # Only one failover should have occurred
        assert manager.failover_count == 1
        assert len(freeze_count) == 1

        await manager.stop()


# ---------------------------------------------------------------------------
# Tests: All transports exhausted → recovery
# ---------------------------------------------------------------------------

class TestRecovery:
    @pytest.mark.asyncio
    async def test_all_exhausted_enters_recovery(
        self, manager, usb_transport, cohn_transport,
    ):
        """When all failover candidates fail, enters RECOVERING state."""
        await manager.start()

        # Make COHN fail during failover
        cohn_transport.discover = AsyncMock(return_value=False)

        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "usb_unplugged")
        await asyncio.sleep(0.1)

        assert manager.state == ManagerState.RECOVERING

        await manager.stop()

    @pytest.mark.asyncio
    async def test_recovery_loop_retries_and_connects(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """Recovery loop retries transports with backoff and reconnects."""
        cfg = TransportManagerConfig(
            priority=["USB", "COHN"],
            discover_timeout_s=2.0,
            connect_timeout_s=2.0,
            stream_timeout_s=2.0,
            backoff_base_s=0.05,  # Fast for testing
            backoff_cap_s=0.1,
            auto_recovery=True,
        )
        mgr = TransportManager(event_bus, config=cfg)
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        await mgr.start()

        # Make both fail during failover
        cohn_transport.discover = AsyncMock(return_value=False)
        usb_transport.discover = AsyncMock(return_value=False)

        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "unplugged")
        await asyncio.sleep(0.1)

        assert mgr.state == ManagerState.RECOVERING

        # Now make USB available again
        usb_transport.discover = AsyncMock(return_value=True)
        usb_transport.connect = AsyncMock(return_value=True)
        usb_transport.start_stream = AsyncMock(return_value=StreamInfo(port=8554))

        # Wait for recovery to kick in
        await asyncio.sleep(0.5)

        assert mgr.state == ManagerState.ACTIVE
        assert mgr.active_transport_name == "USB"

        await mgr.stop()


# ---------------------------------------------------------------------------
# Tests: Manual failover
# ---------------------------------------------------------------------------

class TestManualFailover:
    @pytest.mark.asyncio
    async def test_force_failover_to_named_transport(
        self, manager, usb_transport, cohn_transport,
    ):
        """force_failover(target) switches to the named transport."""
        manager.on_freeze = MagicMock()
        manager.on_unfreeze = MagicMock()

        await manager.start()
        assert manager.active_transport_name == "USB"

        ok = await manager.force_failover("COHN")

        assert ok is True
        assert manager.active_transport_name == "COHN"
        manager.on_freeze.assert_called_once()
        manager.on_unfreeze.assert_called_once()

        await manager.stop()

    @pytest.mark.asyncio
    async def test_force_failover_to_next_in_chain(
        self, manager, usb_transport, cohn_transport,
    ):
        """force_failover() with no target picks next in priority."""
        await manager.start()
        assert manager.active_transport_name == "USB"

        ok = await manager.force_failover()

        assert ok is True
        assert manager.active_transport_name == "COHN"

        await manager.stop()


# ---------------------------------------------------------------------------
# Tests: Orchestrator freeze/unfreeze wiring to pipeline
# ---------------------------------------------------------------------------

class TestOrchestratorFreezeWiring:
    @pytest.mark.asyncio
    async def test_on_freeze_calls_pipeline_freeze(self):
        """Orchestrator._on_freeze calls pipeline.freeze() when pipeline exists."""
        from gomaxwebcam.orchestrator import AppOrchestrator

        bus = MagicMock()
        bus.set_loop = MagicMock()
        bus.shutdown = AsyncMock()

        cm = MagicMock()
        cm.start = AsyncMock()
        cm.stop = AsyncMock()

        tm = MagicMock()
        tm.start = AsyncMock(return_value=True)
        tm.stop = AsyncMock()
        tm.on_transport_switch = None
        tm.on_freeze = None
        tm.on_unfreeze = None

        orch = AppOrchestrator(
            event_bus=bus,
            camera_manager=cm,
            transport_manager=tm,
        )

        # Set a mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.freeze = MagicMock()
        mock_pipeline.unfreeze = MagicMock()
        orch._pipeline = mock_pipeline

        # Call _on_freeze
        orch._on_freeze()
        mock_pipeline.freeze.assert_called_once()

        # Call _on_unfreeze
        orch._on_unfreeze()
        mock_pipeline.unfreeze.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_freeze_noop_without_pipeline(self):
        """Orchestrator._on_freeze is a no-op when no pipeline is set."""
        from gomaxwebcam.orchestrator import AppOrchestrator

        bus = MagicMock()
        bus.set_loop = MagicMock()
        bus.shutdown = AsyncMock()

        cm = MagicMock()
        cm.start = AsyncMock()
        cm.stop = AsyncMock()

        tm = MagicMock()
        tm.start = AsyncMock(return_value=True)
        tm.stop = AsyncMock()
        tm.on_transport_switch = None
        tm.on_freeze = None
        tm.on_unfreeze = None

        orch = AppOrchestrator(
            event_bus=bus,
            camera_manager=cm,
            transport_manager=tm,
        )

        # No pipeline set — should not raise
        orch._on_freeze()
        orch._on_unfreeze()


# ---------------------------------------------------------------------------
# Tests: Pipeline freeze/unfreeze methods
# ---------------------------------------------------------------------------

class TestPipelineFreezeUnfreeze:
    def test_freeze_enters_freeze_frame_state(self):
        """Pipeline.freeze() transitions from STREAMING to FREEZE_FRAME."""
        from gomaxwebcam.pipeline.frame_pipeline import FramePipeline, PipelineState

        pipeline = FramePipeline()
        # Manually set state to STREAMING to simulate running pipeline
        pipeline._state = PipelineState.STREAMING

        freeze_cb = MagicMock()
        pipeline.on_freeze = freeze_cb

        pipeline.freeze()

        assert pipeline.state == PipelineState.FREEZE_FRAME
        assert pipeline._was_frozen is True
        freeze_cb.assert_called_once()

    def test_freeze_noop_when_not_streaming(self):
        """Pipeline.freeze() is a no-op when not in STREAMING state."""
        from gomaxwebcam.pipeline.frame_pipeline import FramePipeline, PipelineState

        pipeline = FramePipeline()
        assert pipeline.state == PipelineState.STOPPED

        pipeline.freeze()

        assert pipeline.state == PipelineState.STOPPED

    def test_unfreeze_noop_when_not_frozen(self):
        """Pipeline.unfreeze() is a no-op when not in FREEZE_FRAME state."""
        from gomaxwebcam.pipeline.frame_pipeline import FramePipeline, PipelineState

        pipeline = FramePipeline()
        pipeline._state = PipelineState.STREAMING

        pipeline.unfreeze()

        # State should not change — unfreeze only prepares for resume
        assert pipeline.state == PipelineState.STREAMING

    def test_unfreeze_when_frozen_keeps_freeze_state(self):
        """Pipeline.unfreeze() does not immediately change state — waits for first frame."""
        from gomaxwebcam.pipeline.frame_pipeline import FramePipeline, PipelineState

        pipeline = FramePipeline()
        pipeline._state = PipelineState.FREEZE_FRAME
        pipeline._was_frozen = True

        pipeline.unfreeze()

        # Still frozen — state transitions on first decoded frame
        assert pipeline.state == PipelineState.FREEZE_FRAME


# ---------------------------------------------------------------------------
# Tests: Status reporting
# ---------------------------------------------------------------------------

class TestStatus:
    @pytest.mark.asyncio
    async def test_get_status_returns_correct_fields(self, manager):
        """get_status() returns all expected fields."""
        await manager.start()

        status = manager.get_status()

        assert status["state"] == "ACTIVE"
        assert status["active_transport"] == "USB"
        assert "USB" in status["registered_transports"]
        assert "COHN" in status["registered_transports"]
        assert status["failover_count"] == 0
        assert status["priority"] == ["USB", "COHN"]
        assert "uptime_s" in status

        await manager.stop()


# ---------------------------------------------------------------------------
# Tests: USB fail-back (COHN -> USB when USB is re-detected)
# ---------------------------------------------------------------------------

class TestUSBFailBack:
    """Tests for fail-back from COHN to USB when USB device is re-detected.

    The TransportManager starts a background USB polling loop whenever COHN
    is the active transport.  When the poll finds a GoPro on USB, it:
      1. Publishes a USB_REDETECTED event on the EventBus
      2. Calls force_failover("USB") to gracefully switch back
      3. Uses freeze-frame bridging during the transition
    """

    @pytest.mark.asyncio
    async def test_usb_poll_starts_when_cohn_becomes_active(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """USB polling starts automatically when COHN becomes active transport."""
        cfg = TransportManagerConfig(
            priority=["USB", "COHN"],
            discover_timeout_s=2.0,
            connect_timeout_s=2.0,
            stream_timeout_s=2.0,
            usb_poll_interval_s=60.0,  # Long interval - we just check task creation
        )
        mgr = TransportManager(event_bus, config=cfg)
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        # Make USB fail at startup so COHN becomes active
        usb_transport.discover = AsyncMock(return_value=False)

        await mgr.start()
        assert mgr.active_transport_name == "COHN"

        # USB poll task should have been started
        assert mgr._usb_poll_task is not None
        assert not mgr._usb_poll_task.done()

        status = mgr.get_status()
        assert status["usb_poll_active"] is True

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_usb_poll_not_started_when_usb_is_active(
        self, manager, usb_transport,
    ):
        """USB polling is NOT started when USB is the active transport."""
        await manager.start()
        assert manager.active_transport_name == "USB"

        # No USB poll when USB is already active
        assert manager._usb_poll_task is None

        status = manager.get_status()
        assert status["usb_poll_active"] is False

        await manager.stop()

    @pytest.mark.asyncio
    async def test_usb_poll_stops_on_manager_stop(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """USB poll task is cancelled when manager stops."""
        cfg = TransportManagerConfig(
            priority=["USB", "COHN"],
            discover_timeout_s=2.0,
            connect_timeout_s=2.0,
            stream_timeout_s=2.0,
            usb_poll_interval_s=60.0,
        )
        mgr = TransportManager(event_bus, config=cfg)
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        usb_transport.discover = AsyncMock(return_value=False)
        await mgr.start()
        assert mgr._usb_poll_task is not None

        await mgr.stop()

        # Poll task should be cancelled / None after stop
        assert mgr._usb_poll_task is None

    @pytest.mark.asyncio
    async def test_usb_redetected_triggers_failback_to_usb(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """When USB is re-detected during COHN, fail-back switches to USB with freeze/unfreeze."""
        cfg = TransportManagerConfig(
            priority=["USB", "COHN"],
            discover_timeout_s=2.0,
            connect_timeout_s=2.0,
            stream_timeout_s=2.0,
            usb_poll_interval_s=0.05,  # Fast polling for test
        )
        mgr = TransportManager(event_bus, config=cfg)
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        freeze_calls = []
        unfreeze_calls = []
        switch_calls = []
        mgr.on_freeze = lambda: freeze_calls.append("frozen")
        mgr.on_unfreeze = lambda: unfreeze_calls.append("unfrozen")
        mgr.on_transport_switch = lambda name, t: switch_calls.append(name)

        # Start on COHN (USB fails at startup)
        usb_transport.discover = AsyncMock(return_value=False)
        await mgr.start()
        assert mgr.active_transport_name == "COHN"

        # Now make USB available again for failback
        usb_transport.discover = AsyncMock(return_value=True)
        usb_transport.connect = AsyncMock(return_value=True)
        usb_transport.start_stream = AsyncMock(return_value=StreamInfo(port=8554))

        # Mock _check_usb_presence to return a device
        mock_device = MagicMock()
        mock_device.serial_number = "C123456"
        mock_device.camera_ip = "172.20.100.51"

        # Patch the initial delay constant so polling starts immediately
        with patch("gomaxwebcam.transport_manager.USB_POLL_INITIAL_DELAY_S", 0.01), \
             patch.object(mgr, "_check_usb_presence", new_callable=AsyncMock, return_value=mock_device):
            # Cancel existing poll task (started with real delay) and restart
            mgr._stop_usb_poll()
            mgr._start_usb_poll()
            # Wait for poll to detect USB and trigger failback
            await asyncio.sleep(0.5)

        # Should have failed back to USB
        assert mgr.active_transport_name == "USB"
        assert mgr.state == ManagerState.ACTIVE

        # Freeze/unfreeze should have been called during failback
        assert len(freeze_calls) >= 1
        assert len(unfreeze_calls) >= 1

        # Transport switch callback should show COHN->USB
        assert "USB" in switch_calls

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_usb_redetected_publishes_event(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """USB re-detection publishes a USB_REDETECTED event on the EventBus."""
        cfg = TransportManagerConfig(
            priority=["USB", "COHN"],
            discover_timeout_s=2.0,
            connect_timeout_s=2.0,
            stream_timeout_s=2.0,
            usb_poll_interval_s=0.05,
        )
        mgr = TransportManager(event_bus, config=cfg)
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        # Start on COHN
        usb_transport.discover = AsyncMock(return_value=False)
        await mgr.start()
        assert mgr.active_transport_name == "COHN"

        # Now make USB available for failback
        usb_transport.discover = AsyncMock(return_value=True)
        usb_transport.connect = AsyncMock(return_value=True)
        usb_transport.start_stream = AsyncMock(return_value=StreamInfo(port=8554))

        mock_device = MagicMock()
        mock_device.serial_number = "C999888"
        mock_device.camera_ip = "172.20.100.51"

        with patch("gomaxwebcam.transport_manager.USB_POLL_INITIAL_DELAY_S", 0.01), \
             patch.object(mgr, "_check_usb_presence", new_callable=AsyncMock, return_value=mock_device):
            mgr._stop_usb_poll()
            mgr._start_usb_poll()
            await asyncio.sleep(0.5)

        # Verify USB_REDETECTED event was published
        usb_events = [
            e for e in event_bus.recent_events
            if e.type == EventType.USB_REDETECTED
        ]
        assert len(usb_events) >= 1
        assert usb_events[0].data["camera_serial"] == "C999888"

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_usb_failback_retries_on_failure(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """If USB failback fails, the poll loop keeps retrying."""
        cfg = TransportManagerConfig(
            priority=["USB", "COHN"],
            discover_timeout_s=2.0,
            connect_timeout_s=2.0,
            stream_timeout_s=2.0,
            usb_poll_interval_s=0.05,
        )
        mgr = TransportManager(event_bus, config=cfg)
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        # Start on COHN
        usb_transport.discover = AsyncMock(return_value=False)
        await mgr.start()
        assert mgr.active_transport_name == "COHN"

        # USB is detected but connect fails first time, succeeds second
        call_count = 0

        async def failing_then_succeeding_connect():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return False
            return True

        usb_transport.discover = AsyncMock(return_value=True)
        usb_transport.connect = AsyncMock(side_effect=failing_then_succeeding_connect)
        usb_transport.start_stream = AsyncMock(return_value=StreamInfo(port=8554))

        mock_device = MagicMock()
        mock_device.serial_number = "C111222"
        mock_device.camera_ip = "172.20.100.51"

        with patch("gomaxwebcam.transport_manager.USB_POLL_INITIAL_DELAY_S", 0.01), \
             patch.object(mgr, "_check_usb_presence", new_callable=AsyncMock, return_value=mock_device):
            mgr._stop_usb_poll()
            mgr._start_usb_poll()
            await asyncio.sleep(1.0)

        # Should eventually succeed
        assert mgr.active_transport_name == "USB"
        assert mgr.state == ManagerState.ACTIVE

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_usb_poll_stops_after_successful_failback(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """USB poll task stops after a successful fail-back to USB."""
        cfg = TransportManagerConfig(
            priority=["USB", "COHN"],
            discover_timeout_s=2.0,
            connect_timeout_s=2.0,
            stream_timeout_s=2.0,
            usb_poll_interval_s=0.05,
        )
        mgr = TransportManager(event_bus, config=cfg)
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        usb_transport.discover = AsyncMock(return_value=False)
        await mgr.start()
        assert mgr.active_transport_name == "COHN"
        assert mgr._usb_poll_task is not None

        # Make USB available
        usb_transport.discover = AsyncMock(return_value=True)
        usb_transport.connect = AsyncMock(return_value=True)
        usb_transport.start_stream = AsyncMock(return_value=StreamInfo(port=8554))

        mock_device = MagicMock()
        mock_device.serial_number = "C555666"
        mock_device.camera_ip = "172.20.100.51"

        with patch("gomaxwebcam.transport_manager.USB_POLL_INITIAL_DELAY_S", 0.01), \
             patch.object(mgr, "_check_usb_presence", new_callable=AsyncMock, return_value=mock_device):
            mgr._stop_usb_poll()
            mgr._start_usb_poll()
            await asyncio.sleep(0.5)

        assert mgr.active_transport_name == "USB"

        # After failback, USB poll should be stopped
        status = mgr.get_status()
        assert status["usb_poll_active"] is False

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_usb_poll_no_device_continues_polling(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """When no USB device is detected, polling continues without failback."""
        cfg = TransportManagerConfig(
            priority=["USB", "COHN"],
            discover_timeout_s=2.0,
            connect_timeout_s=2.0,
            stream_timeout_s=2.0,
            usb_poll_interval_s=0.05,
        )
        mgr = TransportManager(event_bus, config=cfg)
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        usb_transport.discover = AsyncMock(return_value=False)
        await mgr.start()
        assert mgr.active_transport_name == "COHN"

        # No USB device found - restart poll with short initial delay
        with patch("gomaxwebcam.transport_manager.USB_POLL_INITIAL_DELAY_S", 0.01), \
             patch.object(mgr, "_check_usb_presence", new_callable=AsyncMock, return_value=None):
            mgr._stop_usb_poll()
            mgr._start_usb_poll()
            await asyncio.sleep(0.3)

        # Still on COHN, poll still running
        assert mgr.active_transport_name == "COHN"
        assert mgr._usb_poll_task is not None
        assert not mgr._usb_poll_task.done()

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_failback_uses_freeze_frame_bridge(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """Fail-back from COHN to USB uses freeze-frame bridging via force_failover."""
        cfg = TransportManagerConfig(
            priority=["USB", "COHN"],
            discover_timeout_s=2.0,
            connect_timeout_s=2.0,
            stream_timeout_s=2.0,
            usb_poll_interval_s=0.05,
        )
        mgr = TransportManager(event_bus, config=cfg)
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)

        # Track freeze/unfreeze sequence
        call_order = []
        mgr.on_freeze = lambda: call_order.append("freeze")
        mgr.on_unfreeze = lambda: call_order.append("unfreeze")
        mgr.on_transport_switch = lambda name, t: call_order.append(f"switch:{name}")

        usb_transport.discover = AsyncMock(return_value=False)
        await mgr.start()
        assert mgr.active_transport_name == "COHN"
        call_order.clear()  # Reset after startup

        # Make USB available
        usb_transport.discover = AsyncMock(return_value=True)
        usb_transport.connect = AsyncMock(return_value=True)
        usb_transport.start_stream = AsyncMock(return_value=StreamInfo(port=8554))

        mock_device = MagicMock()
        mock_device.serial_number = "C777888"
        mock_device.camera_ip = "172.20.100.51"

        with patch("gomaxwebcam.transport_manager.USB_POLL_INITIAL_DELAY_S", 0.01), \
             patch.object(mgr, "_check_usb_presence", new_callable=AsyncMock, return_value=mock_device):
            mgr._stop_usb_poll()
            mgr._start_usb_poll()
            await asyncio.sleep(0.5)

        # Verify the fail-back sequence: freeze -> switch -> unfreeze
        assert "freeze" in call_order
        assert "unfreeze" in call_order
        assert "switch:USB" in call_order

        # Verify order: freeze before unfreeze
        freeze_idx = call_order.index("freeze")
        unfreeze_idx = call_order.index("unfreeze")
        assert freeze_idx < unfreeze_idx

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_full_failover_and_failback_round_trip(
        self, event_bus, usb_transport, cohn_transport,
    ):
        """Full round-trip: USB -> COHN (on disconnect) -> USB (on re-detect)."""
        cfg = TransportManagerConfig(
            priority=["USB", "COHN"],
            discover_timeout_s=2.0,
            connect_timeout_s=2.0,
            stream_timeout_s=2.0,
            usb_poll_interval_s=0.05,
        )
        mgr = TransportManager(event_bus, config=cfg)
        mgr.register_transport("USB", usb_transport)
        mgr.register_transport("COHN", cohn_transport)
        mgr.on_transport_switch = lambda name, t: None

        # Step 1: Start on USB
        await mgr.start()
        assert mgr.active_transport_name == "USB"
        assert mgr._usb_poll_task is None  # No poll when USB active

        # Step 2: USB disconnects -> failover to COHN
        listener = usb_transport._disconnect_listeners[0]
        listener(usb_transport, "cable_unplugged")
        await asyncio.sleep(0.2)

        assert mgr.active_transport_name == "COHN"
        assert mgr.state == ManagerState.ACTIVE
        assert mgr.failover_count == 1
        # USB poll should now be running
        assert mgr._usb_poll_task is not None

        # Step 3: USB re-detected -> fail-back to USB
        usb_transport.discover = AsyncMock(return_value=True)
        usb_transport.connect = AsyncMock(return_value=True)
        usb_transport.start_stream = AsyncMock(return_value=StreamInfo(port=8554))

        mock_device = MagicMock()
        mock_device.serial_number = "C333444"
        mock_device.camera_ip = "172.20.100.51"

        with patch("gomaxwebcam.transport_manager.USB_POLL_INITIAL_DELAY_S", 0.01), \
             patch.object(mgr, "_check_usb_presence", new_callable=AsyncMock, return_value=mock_device):
            # Restart poll with short delay
            mgr._stop_usb_poll()
            mgr._start_usb_poll()
            await asyncio.sleep(0.5)

        assert mgr.active_transport_name == "USB"
        assert mgr.state == ManagerState.ACTIVE

        # Verify events include the full round trip
        transport_events = [
            e for e in event_bus.recent_events
            if e.type == EventType.TRANSPORT
        ]
        # startup(USB) + failover(COHN) + failback(USB)
        assert len(transport_events) >= 3

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_usb_poll_not_started_without_usb_registered(
        self, event_bus, cohn_transport,
    ):
        """USB polling is not started if no USB transport is registered."""
        cfg = TransportManagerConfig(
            priority=["COHN"],
            discover_timeout_s=2.0,
            connect_timeout_s=2.0,
            stream_timeout_s=2.0,
        )
        mgr = TransportManager(event_bus, config=cfg)
        mgr.register_transport("COHN", cohn_transport)

        await mgr.start()
        assert mgr.active_transport_name == "COHN"

        # No USB transport registered - poll should not start
        assert mgr._usb_poll_task is None

        await mgr.stop()
