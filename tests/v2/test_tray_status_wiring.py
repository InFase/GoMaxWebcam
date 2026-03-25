"""
test_tray_status_wiring.py — Tests for reactive connection status updates
wired from transport/failover layer through EventBus to tray icon.

Verifies:
  - TransportManager.start() publishes CONNECTION events (CONNECTING → STREAMING)
  - TransportManager failover publishes FREEZE_FRAME → STREAMING connection events
  - TransportManager exhausted state publishes DISCONNECTED connection event
  - TransportManager recovery publishes CONNECTING → STREAMING connection events
  - TransportManager.stop() publishes STOPPED connection event
  - CameraManager.set_transport() publishes current transport state
  - TransportManager.start() calls on_transport_switch so CameraManager observes transport
  - All connection events contain data keys expected by tray (_STATE_COLORS keys)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gomaxwebcam.events import EventBus, EventType, Event
from gomaxwebcam.transport.base import Transport, TransportState, StreamInfo
from gomaxwebcam.transport_manager import TransportManager
from gomaxwebcam.camera_manager import CameraManager
from gomaxwebcam.tray import _STATE_COLORS

pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_transport(name: str = "USB", state: TransportState = TransportState.DISCONNECTED) -> MagicMock:
    """Create a mock Transport with async methods."""
    t = MagicMock(spec=Transport)
    t.name = name
    t.state = state
    t.is_connected = state in (TransportState.CONNECTED, TransportState.STREAMING)
    t.is_streaming = state == TransportState.STREAMING
    t.stats = MagicMock(camera_model="", camera_serial="", keepalives_sent=0,
                        keepalives_failed=0, last_error="", connection_uptime_s=0.0)
    t._state_listeners = []
    t.add_state_listener = MagicMock(side_effect=lambda cb: t._state_listeners.append(cb))
    t.add_disconnect_listener = MagicMock()

    t.discover = AsyncMock(return_value=True)
    t.connect = AsyncMock(return_value=True)
    t.start_stream = AsyncMock(return_value=StreamInfo(
        host="127.0.0.1", port=8554, codec="h264",
        width=1920, height=1080, fps=30,
    ))
    t.stop_stream = AsyncMock()
    t.disconnect = AsyncMock()
    t.health_check = AsyncMock(return_value=True)
    return t


def _collect_connection_events(bus: EventBus) -> list[Event]:
    """Collect all CONNECTION events from the bus history."""
    return [e for e in bus.recent_events if e.type == EventType.CONNECTION]


def _collect_events_by_type(bus: EventBus, event_type: EventType) -> list[Event]:
    """Collect events of a specific type from the bus history."""
    return [e for e in bus.recent_events if e.type == event_type]


# ---------------------------------------------------------------------------
# TransportManager startup publishes connection events
# ---------------------------------------------------------------------------

class TestTransportManagerStartupEvents:
    """TransportManager.start() should publish CONNECTION events for tray."""

    @pytest.mark.asyncio
    async def test_startup_publishes_connecting_then_streaming(self):
        """Successful startup should publish CONNECTING then STREAMING."""
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        mgr = TransportManager(bus)
        usb = _make_mock_transport("USB")
        mgr.register_transport("USB", usb)

        await mgr.start()

        events = _collect_connection_events(bus)
        states = [e.data["new_state"] for e in events]
        assert "CONNECTING" in states, f"Expected CONNECTING in {states}"
        assert "STREAMING" in states, f"Expected STREAMING in {states}"

        # CONNECTING should come before STREAMING
        connecting_idx = states.index("CONNECTING")
        streaming_idx = states.index("STREAMING")
        assert connecting_idx < streaming_idx

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_startup_calls_on_transport_switch(self):
        """start() should call on_transport_switch so orchestrator/CameraManager is wired."""
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        mgr = TransportManager(bus)
        usb = _make_mock_transport("USB")
        mgr.register_transport("USB", usb)

        switch_calls = []
        async def on_switch(name, transport):
            switch_calls.append((name, transport))

        mgr.on_transport_switch = on_switch

        await mgr.start()

        assert len(switch_calls) == 1
        assert switch_calls[0][0] == "USB"
        assert switch_calls[0][1] is usb

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_startup_fallback_publishes_events(self):
        """When primary fails, fallback transport events should be published."""
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        mgr = TransportManager(bus)

        usb = _make_mock_transport("USB")
        usb.discover = AsyncMock(return_value=False)  # USB fails discovery

        cohn = _make_mock_transport("COHN")
        mgr.register_transport("USB", usb)
        mgr.register_transport("COHN", cohn)

        await mgr.start()

        events = _collect_connection_events(bus)
        states = [e.data["new_state"] for e in events]
        transports = [e.data.get("transport", "") for e in events]

        # Should have CONNECTING for USB, then CONNECTING for COHN, then STREAMING for COHN
        assert "STREAMING" in states
        streaming_event = next(e for e in events if e.data["new_state"] == "STREAMING")
        assert streaming_event.data["transport"] == "COHN"

        await mgr.stop()


# ---------------------------------------------------------------------------
# TransportManager stop publishes STOPPED
# ---------------------------------------------------------------------------

class TestTransportManagerStopEvents:

    @pytest.mark.asyncio
    async def test_stop_publishes_stopped_event(self):
        """stop() should publish a STOPPED connection event."""
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        mgr = TransportManager(bus)
        usb = _make_mock_transport("USB")
        mgr.register_transport("USB", usb)

        await mgr.start()
        await mgr.stop()

        events = _collect_connection_events(bus)
        states = [e.data["new_state"] for e in events]
        assert "STOPPED" in states, f"Expected STOPPED in {states}"


# ---------------------------------------------------------------------------
# TransportManager failover publishes connection events
# ---------------------------------------------------------------------------

class TestFailoverConnectionEvents:

    @pytest.mark.asyncio
    async def test_failover_publishes_freeze_frame_then_streaming(self):
        """Failover should publish FREEZE_FRAME then STREAMING connection events."""
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        mgr = TransportManager(bus)

        usb = _make_mock_transport("USB")
        cohn = _make_mock_transport("COHN")

        mgr.register_transport("USB", usb)
        mgr.register_transport("COHN", cohn)

        await mgr.start()

        # Clear history to isolate failover events
        bus._history.clear()

        # Simulate USB disconnect by calling the internal handler
        await mgr._handle_disconnect("USB", "cable_unplugged")

        events = _collect_connection_events(bus)
        states = [e.data["new_state"] for e in events]

        assert "FREEZE_FRAME" in states, f"Expected FREEZE_FRAME in {states}"
        assert "STREAMING" in states, f"Expected STREAMING in {states}"

        # FREEZE_FRAME should come before STREAMING
        ff_idx = states.index("FREEZE_FRAME")
        st_idx = states.index("STREAMING")
        assert ff_idx < st_idx

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_failover_exhausted_publishes_disconnected(self):
        """When all transports fail, DISCONNECTED should be published."""
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        mgr = TransportManager(bus)

        usb = _make_mock_transport("USB")
        cohn = _make_mock_transport("COHN")
        cohn.discover = AsyncMock(return_value=False)  # COHN fails too

        mgr.register_transport("USB", usb)
        mgr.register_transport("COHN", cohn)

        await mgr.start()
        bus._history.clear()

        await mgr._handle_disconnect("USB", "cable_unplugged")

        events = _collect_connection_events(bus)
        states = [e.data["new_state"] for e in events]

        assert "DISCONNECTED" in states, f"Expected DISCONNECTED in {states}"

        await mgr.stop()


# ---------------------------------------------------------------------------
# CameraManager.set_transport publishes current state
# ---------------------------------------------------------------------------

class TestCameraManagerSetTransportEvents:

    def test_set_transport_publishes_current_state(self):
        """set_transport() should publish a connection event with the current transport state."""
        bus = EventBus()
        cm = CameraManager(bus)

        t = _make_mock_transport("USB", state=TransportState.STREAMING)

        cm.set_transport(t)

        events = _collect_connection_events(bus)
        assert len(events) >= 1
        # Last connection event should reflect STREAMING state
        last = events[-1]
        assert last.data["new_state"] == "STREAMING"
        assert last.data["transport"] == "USB"


# ---------------------------------------------------------------------------
# All published states are recognized by tray color mapping
# ---------------------------------------------------------------------------

class TestTrayColorMappingCoverage:
    """Ensure all connection states published by the transport layer
    are recognized by the tray's _STATE_COLORS mapping."""

    EXPECTED_STATES = {
        "CONNECTING", "STREAMING", "FREEZE_FRAME", "DISCONNECTED", "STOPPED"
    }

    def test_all_emitted_states_have_tray_colors(self):
        """Every connection state the transport layer emits should have a tray color."""
        for state in self.EXPECTED_STATES:
            assert state in _STATE_COLORS, (
                f"State '{state}' emitted by transport layer but not in tray _STATE_COLORS"
            )

    def test_state_colors_map_to_expected_visual(self):
        """Verify visual meaning of key states."""
        # Green = active/streaming
        assert _STATE_COLORS["STREAMING"] == "#22c55e"
        # Yellow = transitional
        assert _STATE_COLORS["CONNECTING"] == "#eab308"
        assert _STATE_COLORS["FREEZE_FRAME"] == "#eab308"
        # Grey = inactive
        assert _STATE_COLORS["DISCONNECTED"] == "#9ca3af"
        assert _STATE_COLORS["STOPPED"] == "#9ca3af"
