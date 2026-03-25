"""
tests/v2/test_camera_manager_sse.py — Tests for CameraManager → SSE EventBus wiring.

Verifies that the camera manager publishes status updates (battery polling,
transport changes, connection events) to the SSE event bus.

All tests mock at the Transport ABC boundary — no real hardware or network.

Test categories:
  1. EventBus: publish/subscribe, backpressure, type filtering, shutdown
  2. Connection events: transport state changes → SSE events
  3. Transport events: transport type changes → SSE events
  4. Battery events: battery polling → SSE events
  5. Pipeline events: pipeline state changes → SSE events
  6. Status snapshots: periodic full status → SSE events
  7. Error events: transport/pipeline errors → SSE events
  8. Integration: full wiring from transport → manager → bus → subscriber
"""

import asyncio
import time
from typing import Optional
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest

from gomaxwebcam.events import (
    EventBus,
    Event,
    EventType,
    EventSubscription,
    connection_event,
    transport_event,
    battery_event,
    pipeline_event,
    error_event,
    status_event,
)
from gomaxwebcam.camera_manager import CameraManager, CameraManagerConfig
from gomaxwebcam.transport.base import Transport, TransportState, TransportStats
from gomaxwebcam.pipeline.frame_pipeline import PipelineState, PipelineStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockTransport(Transport):
    """Minimal mock transport for testing CameraManager wiring."""

    def __init__(self, name: str = "USB"):
        super().__init__(name=name)
        self._mock_battery: Optional[int] = None

    async def discover(self, timeout: float = 10.0) -> bool:
        self._set_state(TransportState.DISCOVERING)
        self._set_state(TransportState.DISCONNECTED)
        return True

    async def connect(self) -> bool:
        self._set_state(TransportState.CONNECTING)
        self._set_state(TransportState.CONNECTED)
        return True

    async def start_stream(self):
        self._set_state(TransportState.STREAMING)
        return None

    async def stop_stream(self) -> None:
        self._set_state(TransportState.CONNECTED)

    async def disconnect(self) -> None:
        self._set_state(TransportState.DISCONNECTED)

    async def keep_alive(self) -> bool:
        return True

    def simulate_state(self, state: TransportState) -> None:
        """Test helper: force a state transition."""
        self._set_state(state)

    def simulate_error(self, message: str) -> None:
        """Test helper: simulate an error."""
        self._stats.last_error = message
        self._set_state(TransportState.ERROR)


class MockPipeline:
    """Minimal mock pipeline for testing CameraManager wiring."""

    def __init__(self):
        self.state = PipelineState.STOPPED
        self.is_frozen = False
        self.on_state_change = None
        self.on_freeze = None
        self.on_unfreeze = None
        self._stats = PipelineStats()

    def get_stats(self) -> PipelineStats:
        return self._stats

    def simulate_state(self, new_state: PipelineState) -> None:
        """Test helper: trigger state change callback."""
        self.state = new_state
        if self.on_state_change:
            self.on_state_change(new_state)

    def simulate_freeze(self) -> None:
        """Test helper: trigger freeze callback."""
        self.state = PipelineState.FREEZE_FRAME
        self.is_frozen = True
        if self.on_freeze:
            self.on_freeze()

    def simulate_unfreeze(self) -> None:
        """Test helper: trigger unfreeze callback."""
        self.state = PipelineState.STREAMING
        self.is_frozen = False
        if self.on_unfreeze:
            self.on_unfreeze()


async def collect_events(
    bus: EventBus,
    timeout: float = 0.1,
    max_events: int = 100,
    event_types: Optional[set[EventType]] = None,
) -> list[Event]:
    """Collect events from the bus within a timeout."""
    events = []
    sub = bus.subscribe(event_types=event_types)
    try:
        while len(events) < max_events:
            try:
                event = await asyncio.wait_for(sub.__anext__(), timeout=timeout)
                events.append(event)
            except (asyncio.TimeoutError, StopAsyncIteration):
                break
    finally:
        bus.unsubscribe(sub)
    return events


# ---------------------------------------------------------------------------
# 1. EventBus tests
# ---------------------------------------------------------------------------

class TestEventBus:
    """Tests for the core EventBus pub/sub mechanism."""

    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self):
        """Events published to the bus reach subscribers."""
        bus = EventBus()
        sub = bus.subscribe()

        event = battery_event(level=85)
        bus.publish(event)

        received = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert received.type == EventType.BATTERY
        assert received.data["level"] == 85

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        """Multiple subscribers all receive the same event."""
        bus = EventBus()
        sub1 = bus.subscribe()
        sub2 = bus.subscribe()

        bus.publish(battery_event(level=50))

        r1 = await asyncio.wait_for(sub1.__anext__(), timeout=1.0)
        r2 = await asyncio.wait_for(sub2.__anext__(), timeout=1.0)

        assert r1.data["level"] == 50
        assert r2.data["level"] == 50
        assert bus.subscriber_count == 2

        bus.unsubscribe(sub1)
        bus.unsubscribe(sub2)

    @pytest.mark.asyncio
    async def test_type_filtering(self):
        """Subscribers with type filter only receive matching events."""
        bus = EventBus()
        sub = bus.subscribe(event_types={EventType.BATTERY})

        bus.publish(connection_event("A", "B"))
        bus.publish(battery_event(level=75))
        bus.publish(pipeline_event("A", "B"))

        received = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert received.type == EventType.BATTERY
        assert received.data["level"] == 75

        # No more events should be waiting (the others were filtered)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sub.__anext__(), timeout=0.05)

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_backpressure_drops_oldest(self):
        """When subscriber queue is full, oldest events are dropped."""
        bus = EventBus(max_queue_size=2)
        sub = bus.subscribe()

        # Publish 5 events — only latest 2 should survive
        for i in range(5):
            bus.publish(battery_event(level=i * 10))

        events = []
        for _ in range(2):
            e = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
            events.append(e)

        # Should have the last 2 events (level 30, 40)
        levels = [e.data["level"] for e in events]
        assert levels == [30, 40]

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_shutdown_signals_subscribers(self):
        """Bus shutdown causes subscribers to receive StopAsyncIteration."""
        bus = EventBus()
        sub = bus.subscribe()

        await bus.shutdown()

        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(sub.__anext__(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_event_count(self):
        """Bus tracks total events published."""
        bus = EventBus()
        assert bus.event_count == 0

        bus.publish(battery_event(50))
        bus.publish(battery_event(60))
        assert bus.event_count == 2

    @pytest.mark.asyncio
    async def test_event_history(self):
        """Bus maintains bounded event history."""
        bus = EventBus()
        for i in range(5):
            bus.publish(battery_event(level=i))

        history = bus.recent_events
        assert len(history) == 5
        assert history[0].data["level"] == 0
        assert history[-1].data["level"] == 4

    @pytest.mark.asyncio
    async def test_subscribe_with_history_replay(self):
        """Subscriber with include_history=True gets past events."""
        bus = EventBus()
        bus.publish(battery_event(level=10))
        bus.publish(battery_event(level=20))

        sub = bus.subscribe(include_history=True)

        # Should get replayed history events
        e1 = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        e2 = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert e1.data["level"] == 10
        assert e2.data["level"] == 20

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_subscriber(self):
        """Unsubscribing removes the subscriber from the bus."""
        bus = EventBus()
        sub = bus.subscribe()
        assert bus.subscriber_count == 1

        bus.unsubscribe(sub)
        assert bus.subscriber_count == 0


# ---------------------------------------------------------------------------
# 2. Event serialization tests
# ---------------------------------------------------------------------------

class TestEventSerialization:
    """Tests for Event data model and SSE formatting."""

    def test_event_to_sse(self):
        """Event.to_sse() produces valid SSE format."""
        event = battery_event(level=85)
        sse = event.to_sse()

        assert sse.startswith("event: battery\n")
        assert '"level": 85' in sse
        assert sse.endswith("\n\n")

    def test_event_to_dict(self):
        """Event.to_dict() produces a serializable dict."""
        event = connection_event("DISCONNECTED", "CONNECTING", "USB")
        d = event.to_dict()

        assert d["type"] == "connection"
        assert d["data"]["old_state"] == "DISCONNECTED"
        assert d["data"]["new_state"] == "CONNECTING"
        assert "timestamp" in d

    def test_connection_event_factory(self):
        """connection_event() creates correct event type and data."""
        event = connection_event("A", "B", "USB", "test detail")
        assert event.type == EventType.CONNECTION
        assert event.data["old_state"] == "A"
        assert event.data["new_state"] == "B"
        assert event.data["transport"] == "USB"
        assert event.data["detail"] == "test detail"

    def test_transport_event_factory(self):
        """transport_event() creates correct event."""
        event = transport_event("USB", "WiFi AP", "failover")
        assert event.type == EventType.TRANSPORT
        assert event.data["old_transport"] == "USB"
        assert event.data["new_transport"] == "WiFi AP"

    def test_battery_event_factory(self):
        """battery_event() creates correct event."""
        event = battery_event(level=42, charging=True)
        assert event.type == EventType.BATTERY
        assert event.data["level"] == 42
        assert event.data["charging"] is True

    def test_pipeline_event_factory(self):
        """pipeline_event() creates correct event."""
        event = pipeline_event("STOPPED", "STREAMING", "started")
        assert event.type == EventType.PIPELINE
        assert event.data["old_state"] == "STOPPED"
        assert event.data["new_state"] == "STREAMING"

    def test_error_event_factory(self):
        """error_event() creates correct event."""
        event = error_event("transport.USB", "Connection lost", recoverable=True)
        assert event.type == EventType.ERROR
        assert event.data["source"] == "transport.USB"
        assert event.data["recoverable"] is True

    def test_status_event_factory(self):
        """status_event() creates correct event."""
        data = {"battery": 85, "fps": 30.0}
        event = status_event(data)
        assert event.type == EventType.STATUS
        assert event.data["battery"] == 85


# ---------------------------------------------------------------------------
# 3. CameraManager connection event tests
# ---------------------------------------------------------------------------

class TestCameraManagerConnectionEvents:
    """Tests that transport state changes publish connection events."""

    @pytest.mark.asyncio
    async def test_transport_state_change_publishes_connection_event(self):
        """Transport state change publishes a CONNECTION event."""
        bus = EventBus()
        manager = CameraManager(bus)
        transport = MockTransport("USB")

        manager.set_transport(transport)
        sub = bus.subscribe(event_types={EventType.CONNECTION})

        # Simulate transport state change
        transport.simulate_state(TransportState.CONNECTING)

        event = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert event.type == EventType.CONNECTION
        assert event.data["new_state"] == "CONNECTING"
        assert event.data["transport"] == "USB"

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_transport_error_publishes_connection_and_error_events(self):
        """Transport ERROR state publishes both connection and error events."""
        bus = EventBus()
        manager = CameraManager(bus)
        transport = MockTransport("USB")

        # Subscribe BEFORE setting transport so we catch all events
        sub = bus.subscribe()

        manager.set_transport(transport)

        # Drain the initial transport event
        initial = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert initial.type == EventType.TRANSPORT

        # Simulate error
        transport.simulate_error("Connection timeout")

        events = []
        for _ in range(3):
            try:
                e = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
                events.append(e)
            except asyncio.TimeoutError:
                break

        types = [e.type for e in events]
        assert EventType.CONNECTION in types
        assert EventType.ERROR in types

        error_evt = next(e for e in events if e.type == EventType.ERROR)
        assert "Connection timeout" in error_evt.data["message"]

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_full_lifecycle_connection_events(self):
        """Full connect → stream → disconnect lifecycle publishes events."""
        bus = EventBus()
        manager = CameraManager(bus)
        transport = MockTransport("USB")

        manager.set_transport(transport)
        sub = bus.subscribe(event_types={EventType.CONNECTION})

        # Walk through lifecycle
        await transport.connect()       # DISCONNECTED → CONNECTING → CONNECTED
        await transport.start_stream()  # CONNECTED → STREAMING
        await transport.stop_stream()   # STREAMING → CONNECTED
        await transport.disconnect()    # CONNECTED → DISCONNECTED

        events = []
        for _ in range(10):
            try:
                e = await asyncio.wait_for(sub.__anext__(), timeout=0.1)
                events.append(e)
            except asyncio.TimeoutError:
                break

        states = [e.data["new_state"] for e in events]
        assert "CONNECTING" in states
        assert "CONNECTED" in states
        assert "STREAMING" in states
        assert "DISCONNECTED" in states

        bus.unsubscribe(sub)


# ---------------------------------------------------------------------------
# 4. CameraManager transport change events
# ---------------------------------------------------------------------------

class TestCameraManagerTransportEvents:
    """Tests that transport type changes publish transport events."""

    @pytest.mark.asyncio
    async def test_set_transport_publishes_transport_event(self):
        """Setting a transport publishes a TRANSPORT change event."""
        bus = EventBus()
        manager = CameraManager(bus)
        sub = bus.subscribe(event_types={EventType.TRANSPORT})

        transport = MockTransport("USB")
        manager.set_transport(transport)

        event = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert event.type == EventType.TRANSPORT
        assert event.data["old_transport"] == "none"
        assert event.data["new_transport"] == "USB"

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_transport_failover_publishes_event(self):
        """Switching from USB to WiFi AP publishes a failover event."""
        bus = EventBus()
        manager = CameraManager(bus)

        usb = MockTransport("USB")
        wifi = MockTransport("WiFi AP")

        manager.set_transport(usb)

        sub = bus.subscribe(event_types={EventType.TRANSPORT})

        manager.set_transport(wifi)

        event = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert event.data["old_transport"] == "USB"
        assert event.data["new_transport"] == "WiFi AP"
        assert event.data["reason"] == "failover"

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_clear_transport_publishes_event(self):
        """Setting transport to None publishes disconnect event."""
        bus = EventBus()
        manager = CameraManager(bus)

        transport = MockTransport("USB")
        manager.set_transport(transport)

        sub = bus.subscribe(event_types={EventType.TRANSPORT})

        manager.set_transport(None)

        event = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert event.data["old_transport"] == "USB"
        assert event.data["new_transport"] == "none"

        bus.unsubscribe(sub)


# ---------------------------------------------------------------------------
# 5. CameraManager battery events
# ---------------------------------------------------------------------------

class TestCameraManagerBatteryEvents:
    """Tests that battery updates publish battery events."""

    @pytest.mark.asyncio
    async def test_set_battery_publishes_event(self):
        """Manually setting battery level publishes a BATTERY event."""
        bus = EventBus()
        manager = CameraManager(bus)
        sub = bus.subscribe(event_types={EventType.BATTERY})

        manager.set_battery_level(85)

        event = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert event.type == EventType.BATTERY
        assert event.data["level"] == 85
        assert event.data["charging"] is False

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_battery_change_with_charging(self):
        """Battery event includes charging state."""
        bus = EventBus()
        manager = CameraManager(bus)
        sub = bus.subscribe(event_types={EventType.BATTERY})

        manager.set_battery_level(42, charging=True)

        event = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert event.data["level"] == 42
        assert event.data["charging"] is True

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_same_battery_level_no_event(self):
        """Setting same battery level does NOT publish a duplicate event."""
        bus = EventBus()
        manager = CameraManager(bus)

        manager.set_battery_level(50)
        sub = bus.subscribe(event_types={EventType.BATTERY})

        # Set same level again
        manager.set_battery_level(50)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sub.__anext__(), timeout=0.1)

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_battery_poll_loop_publishes_on_change(self):
        """Battery poll loop publishes events when level changes."""
        bus = EventBus()
        config = CameraManagerConfig(
            battery_poll_interval=0.05,  # Fast polling for test
            status_poll_interval=100.0,  # Slow status to avoid noise
        )
        manager = CameraManager(bus, config=config)

        transport = MockTransport("USB")
        transport._set_state(TransportState.CONNECTED)
        manager.set_transport(transport)

        # Mock _query_battery to return a sequence of values
        battery_values = iter([80, 80, 75, 70])

        async def mock_query():
            try:
                return next(battery_values)
            except StopIteration:
                return None

        manager._query_battery = mock_query  # type: ignore[assignment]

        sub = bus.subscribe(event_types={EventType.BATTERY})

        await manager.start()
        try:
            # Collect battery events
            events = []
            for _ in range(3):
                try:
                    e = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
                    events.append(e)
                except asyncio.TimeoutError:
                    break

            # Should see changes: 80 (initial), 75, 70
            levels = [e.data["level"] for e in events]
            assert 80 in levels
            assert 75 in levels
        finally:
            await manager.stop()
            bus.unsubscribe(sub)


# ---------------------------------------------------------------------------
# 6. CameraManager pipeline events
# ---------------------------------------------------------------------------

class TestCameraManagerPipelineEvents:
    """Tests that pipeline state changes publish pipeline events."""

    @pytest.mark.asyncio
    async def test_pipeline_state_change_publishes_event(self):
        """Pipeline state change triggers a PIPELINE event."""
        bus = EventBus()
        manager = CameraManager(bus)
        pipeline = MockPipeline()

        manager.set_pipeline(pipeline)
        sub = bus.subscribe(event_types={EventType.PIPELINE})

        pipeline.simulate_state(PipelineState.STREAMING)

        event = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert event.type == EventType.PIPELINE
        assert event.data["new_state"] == "STREAMING"

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_pipeline_freeze_publishes_event(self):
        """Pipeline freeze callback publishes a freeze PIPELINE event."""
        bus = EventBus()
        manager = CameraManager(bus)
        pipeline = MockPipeline()

        manager.set_pipeline(pipeline)
        sub = bus.subscribe(event_types={EventType.PIPELINE})

        pipeline.simulate_freeze()

        event = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert event.data["new_state"] == "FREEZE_FRAME"
        assert "last good frame" in event.data["detail"]

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_pipeline_unfreeze_publishes_event(self):
        """Pipeline unfreeze callback publishes a resume PIPELINE event."""
        bus = EventBus()
        manager = CameraManager(bus)
        pipeline = MockPipeline()

        manager.set_pipeline(pipeline)
        sub = bus.subscribe(event_types={EventType.PIPELINE})

        pipeline.simulate_unfreeze()

        event = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert event.data["new_state"] == "STREAMING"
        assert "resumed" in event.data["detail"]

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_set_pipeline_wires_all_callbacks(self):
        """set_pipeline wires on_state_change, on_freeze, on_unfreeze."""
        bus = EventBus()
        manager = CameraManager(bus)
        pipeline = MockPipeline()

        manager.set_pipeline(pipeline)

        assert pipeline.on_state_change is not None
        assert pipeline.on_freeze is not None
        assert pipeline.on_unfreeze is not None

    @pytest.mark.asyncio
    async def test_replace_pipeline_clears_old_callbacks(self):
        """Replacing pipeline clears callbacks on old pipeline."""
        bus = EventBus()
        manager = CameraManager(bus)

        old_pipeline = MockPipeline()
        new_pipeline = MockPipeline()

        manager.set_pipeline(old_pipeline)
        manager.set_pipeline(new_pipeline)

        assert old_pipeline.on_state_change is None
        assert old_pipeline.on_freeze is None
        assert new_pipeline.on_state_change is not None


# ---------------------------------------------------------------------------
# 7. CameraManager status snapshot events
# ---------------------------------------------------------------------------

class TestCameraManagerStatusEvents:
    """Tests that periodic status snapshots are published."""

    @pytest.mark.asyncio
    async def test_status_poll_publishes_events(self):
        """Status poll loop publishes STATUS events periodically."""
        bus = EventBus()
        config = CameraManagerConfig(
            battery_poll_enabled=False,
            status_poll_interval=0.05,
        )
        manager = CameraManager(bus, config=config)

        transport = MockTransport("USB")
        transport._set_state(TransportState.CONNECTED)
        manager.set_transport(transport)

        sub = bus.subscribe(event_types={EventType.STATUS})

        await manager.start()
        try:
            event = await asyncio.wait_for(sub.__anext__(), timeout=2.0)
            assert event.type == EventType.STATUS
            assert event.data["transport_type"] == "USB"
            assert event.data["connection_state"] == "CONNECTED"
            assert "battery_level" in event.data
            assert "subscribers" in event.data
        finally:
            await manager.stop()
            bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_status_snapshot_includes_pipeline_stats(self):
        """Status snapshot includes pipeline stats when pipeline is set."""
        bus = EventBus()
        manager = CameraManager(bus)

        pipeline = MockPipeline()
        pipeline.state = PipelineState.STREAMING
        pipeline._stats = PipelineStats(
            decoder_fps=29.5,
            vcam_fps=30.0,
            decoder_frames=1000,
            vcam_frames=1000,
        )
        manager.set_pipeline(pipeline)

        snapshot = manager.get_status_snapshot()
        assert snapshot["pipeline_state"] == "STREAMING"
        assert snapshot["decoder_fps"] == 29.5
        assert snapshot["vcam_fps"] == 30.0

    @pytest.mark.asyncio
    async def test_status_snapshot_without_transport(self):
        """Status snapshot works when no transport is set."""
        bus = EventBus()
        manager = CameraManager(bus)

        snapshot = manager.get_status_snapshot()
        assert snapshot["connection_state"] == "DISCONNECTED"
        assert snapshot["is_connected"] is False
        assert snapshot["transport_type"] == "none"


# ---------------------------------------------------------------------------
# 8. Integration tests
# ---------------------------------------------------------------------------

class TestCameraManagerIntegration:
    """End-to-end wiring tests: transport → manager → bus → subscriber."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_events(self):
        """Full lifecycle publishes all expected event types."""
        bus = EventBus()
        config = CameraManagerConfig(
            battery_poll_enabled=False,
            status_poll_interval=100.0,
        )
        manager = CameraManager(bus, config=config)
        sub = bus.subscribe()

        # Set up
        transport = MockTransport("USB")
        pipeline = MockPipeline()

        # 1. Set transport → TRANSPORT event
        manager.set_transport(transport)

        # 2. Connect → CONNECTION events
        await transport.connect()

        # 3. Set pipeline
        manager.set_pipeline(pipeline)

        # 4. Pipeline starts → PIPELINE event
        pipeline.simulate_state(PipelineState.STREAMING)

        # 5. Battery update → BATTERY event
        manager.set_battery_level(85)

        # 6. Pipeline freeze → PIPELINE event
        pipeline.simulate_freeze()

        # Collect all events
        events = []
        for _ in range(20):
            try:
                e = await asyncio.wait_for(sub.__anext__(), timeout=0.1)
                events.append(e)
            except asyncio.TimeoutError:
                break

        types = {e.type for e in events}
        assert EventType.TRANSPORT in types
        assert EventType.CONNECTION in types
        assert EventType.PIPELINE in types
        assert EventType.BATTERY in types

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_manager_start_stop(self):
        """Manager start/stop manages background tasks correctly."""
        bus = EventBus()
        config = CameraManagerConfig(
            battery_poll_interval=0.05,
            status_poll_interval=0.05,
        )
        manager = CameraManager(bus, config=config)

        assert not manager.is_running

        await manager.start()
        assert manager.is_running

        # Give tasks a moment to run
        await asyncio.sleep(0.15)

        await manager.stop()
        assert not manager.is_running

    @pytest.mark.asyncio
    async def test_events_published_counter(self):
        """Manager tracks total events published."""
        bus = EventBus()
        manager = CameraManager(bus)

        assert manager.events_published == 0

        transport = MockTransport("USB")
        manager.set_transport(transport)

        # At least 1 event (transport change)
        assert manager.events_published >= 1

        manager.set_battery_level(50)
        assert manager.events_published >= 2

    @pytest.mark.asyncio
    async def test_transport_state_listener_removed_on_replacement(self):
        """Old transport's state listener is removed when transport is replaced."""
        bus = EventBus()
        manager = CameraManager(bus)

        t1 = MockTransport("USB")
        t2 = MockTransport("WiFi AP")

        manager.set_transport(t1)
        initial_listeners = len(t1._state_listeners)

        manager.set_transport(t2)

        # t1 should have one fewer listener
        assert len(t1._state_listeners) < initial_listeners

        # t2 should have the manager's listener
        assert len(t2._state_listeners) >= 1

    @pytest.mark.asyncio
    async def test_sse_format_output(self):
        """Events produce valid SSE-formatted strings."""
        bus = EventBus()
        manager = CameraManager(bus)

        sub = bus.subscribe()
        manager.set_battery_level(65)

        event = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        sse_str = event.to_sse()

        # Validate SSE format
        lines = sse_str.split("\n")
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        assert lines[2] == ""  # Empty line delimiter
        assert lines[3] == ""  # Trailing empty line

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_concurrent_publishers(self):
        """Multiple publishers (transport + pipeline) can publish concurrently."""
        bus = EventBus()
        manager = CameraManager(bus)

        transport = MockTransport("USB")
        pipeline = MockPipeline()

        # Subscribe BEFORE setting transport
        sub = bus.subscribe()

        manager.set_transport(transport)
        manager.set_pipeline(pipeline)

        # Drain the initial transport event
        initial = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert initial.type == EventType.TRANSPORT

        # Simultaneous state changes
        transport.simulate_state(TransportState.CONNECTING)
        pipeline.simulate_state(PipelineState.STARTING)
        manager.set_battery_level(90)

        events = []
        for _ in range(5):
            try:
                e = await asyncio.wait_for(sub.__anext__(), timeout=0.5)
                events.append(e)
            except asyncio.TimeoutError:
                break

        types = {e.type for e in events}
        assert EventType.CONNECTION in types
        assert EventType.PIPELINE in types
        assert EventType.BATTERY in types

        bus.unsubscribe(sub)
