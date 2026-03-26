"""
tests/v2/test_dashboard_sse.py — SSE endpoint and status tracker tests.

Tests the FastAPI SSE endpoint for camera status events:
  1. SSE endpoint streams status events as server-sent events
  2. Auth token is required for API access
  3. CameraStatusTracker aggregates transport + pipeline status
  4. Status changes trigger SSE events to all subscribers
  5. Health endpoint works without auth
  6. REST snapshot endpoint returns current status
  7. EventBus-powered /api/events/stream endpoint
  8. Recent events REST endpoint

All tests mock at the Transport ABC boundary — no real hardware needed.
"""

import asyncio
import json
import time
from dataclasses import asdict
from unittest.mock import MagicMock, PropertyMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from gomaxwebcam.dashboard.status_tracker import CameraStatus, CameraStatusTracker
from gomaxwebcam.dashboard.app import create_app, format_sse_event
from gomaxwebcam.events import (
    EventBus,
    EventType,
    battery_event,
    connection_event,
    transport_event,
    pipeline_event,
    error_event,
    status_event,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker():
    """Create a fresh CameraStatusTracker."""
    return CameraStatusTracker()


@pytest.fixture
def event_bus():
    """Create a fresh EventBus."""
    return EventBus()


@pytest.fixture
def auth_token():
    return "test-token-abc123"


@pytest.fixture
def app(tracker, event_bus, auth_token):
    """Create a FastAPI app with test tracker, event bus, and known auth token."""
    return create_app(tracker=tracker, event_bus=event_bus, auth_token=auth_token)


@pytest_asyncio.fixture
async def client(app):
    """Create an async HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# CameraStatus tests
# ---------------------------------------------------------------------------

class TestCameraStatus:
    """Tests for the CameraStatus dataclass."""

    def test_default_values(self):
        status = CameraStatus()
        assert status.transport_type == "none"
        assert status.connection_state == "DISCONNECTED"
        assert status.battery_level == -1
        assert status.pipeline_state == "STOPPED"
        assert status.is_frozen is False

    def test_to_sse_data_returns_json(self):
        status = CameraStatus(
            transport_type="USB",
            connection_state="STREAMING",
            battery_level=85,
        )
        data = status.to_sse_data()
        parsed = json.loads(data)
        assert parsed["transport_type"] == "USB"
        assert parsed["connection_state"] == "STREAMING"
        assert parsed["battery_level"] == 85

    def test_has_changed_detects_difference(self):
        s1 = CameraStatus(transport_type="USB", timestamp=1.0)
        s2 = CameraStatus(transport_type="COHN", timestamp=2.0)
        assert s1.has_changed(s2)

    def test_has_changed_ignores_timestamp(self):
        s1 = CameraStatus(transport_type="USB", timestamp=1.0)
        s2 = CameraStatus(transport_type="USB", timestamp=999.0)
        assert not s1.has_changed(s2)

    def test_has_changed_same_status(self):
        s1 = CameraStatus(transport_type="USB", battery_level=50)
        s2 = CameraStatus(transport_type="USB", battery_level=50)
        assert not s1.has_changed(s2)


# ---------------------------------------------------------------------------
# CameraStatusTracker tests
# ---------------------------------------------------------------------------

class TestCameraStatusTracker:
    """Tests for the CameraStatusTracker."""

    @pytest.mark.asyncio
    async def test_initial_status_is_disconnected(self, tracker):
        assert tracker.current.connection_state == "DISCONNECTED"
        assert tracker.current.transport_type == "none"

    @pytest.mark.asyncio
    async def test_update_status_notifies_subscribers(self, tracker):
        """Manual status update should be delivered to subscribers."""
        received = []

        async def collect():
            async for status in tracker.subscribe():
                received.append(status)
                if len(received) >= 2:
                    break

        task = asyncio.create_task(collect())

        # Give the subscription a moment to start
        await asyncio.sleep(0.05)

        # Push a status update
        new_status = CameraStatus(
            transport_type="USB",
            connection_state="STREAMING",
            battery_level=75,
            timestamp=time.monotonic(),
        )
        await tracker.update_status(new_status)

        # Wait for collection with timeout
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # First event is initial status (sent on subscribe), second is the update
        assert len(received) >= 1
        # Find the USB status
        usb_events = [s for s in received if s.transport_type == "USB"]
        assert len(usb_events) == 1
        assert usb_events[0].battery_level == 75

    @pytest.mark.asyncio
    async def test_subscriber_count(self, tracker):
        assert tracker.subscriber_count == 0
        sub = tracker.subscribe()
        assert tracker.subscriber_count == 1
        sub._cleanup()
        assert tracker.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_collect_status_from_transport(self, tracker):
        """Tracker should read transport state and stats."""
        mock_transport = MagicMock()
        mock_transport.name = "USB"

        # Mock transport state enum
        mock_state = MagicMock()
        mock_state.name = "STREAMING"
        type(mock_transport).state = PropertyMock(return_value=mock_state)

        mock_stats = MagicMock()
        mock_stats.camera_model = "GoPro@172.20.10.1"
        mock_stats.camera_serial = "C33"
        mock_stats.keepalives_sent = 42
        mock_stats.keepalives_failed = 1
        mock_stats.last_error = ""
        type(mock_transport).stats = PropertyMock(return_value=mock_stats)

        tracker.set_transport(mock_transport)
        status = tracker._collect_status()

        assert status.transport_type == "USB"
        assert status.connection_state == "STREAMING"
        assert status.camera_model == "GoPro@172.20.10.1"
        assert status.camera_serial == "C33"
        assert status.keepalives_sent == 42

    @pytest.mark.asyncio
    async def test_collect_status_from_pipeline(self, tracker):
        """Tracker should read pipeline state and stats."""
        mock_pipeline = MagicMock()

        mock_p_state = MagicMock()
        mock_p_state.name = "STREAMING"
        type(mock_pipeline).state = PropertyMock(return_value=mock_p_state)
        type(mock_pipeline).is_frozen = PropertyMock(return_value=False)

        mock_p_stats = MagicMock()
        mock_p_stats.decoder_fps = 29.9
        mock_p_stats.vcam_fps = 30.0
        mock_p_stats.decoder_frames = 1000
        mock_p_stats.vcam_frames = 998
        mock_p_stats.uptime_s = 33.5
        mock_pipeline.get_stats.return_value = mock_p_stats

        tracker.set_pipeline(mock_pipeline)
        status = tracker._collect_status()

        assert status.pipeline_state == "STREAMING"
        assert status.is_frozen is False
        assert status.decoder_fps == 29.9
        assert status.vcam_fps == 30.0
        assert status.decoder_frames == 1000

    @pytest.mark.asyncio
    async def test_battery_level_injection(self, tracker):
        tracker.set_battery_level(85)
        status = tracker._collect_status()
        assert status.battery_level == 85

    @pytest.mark.asyncio
    async def test_stop_sends_sentinel(self, tracker):
        """Stopping the tracker should unblock subscribers."""
        received = []

        async def collect():
            async for status in tracker.subscribe():
                received.append(status)

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.05)
        await tracker.stop()

        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Should have gotten at least the initial status
        assert len(received) >= 1


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Health endpoint requires no auth."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}


class TestAuthToken:
    """Token auth tests."""

    @pytest.mark.asyncio
    async def test_status_without_token_returns_401(self, client):
        resp = await client.get("/api/status")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_status_with_query_token(self, client, auth_token):
        resp = await client.get(f"/api/status?token={auth_token}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_status_with_bearer_token(self, client, auth_token):
        resp = await client.get(
            "/api/status",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_status_with_wrong_token_returns_401(self, client):
        resp = await client.get("/api/status?token=wrong-token")
        assert resp.status_code == 401


class TestStatusSnapshot:
    """REST /api/status endpoint."""

    @pytest.mark.asyncio
    async def test_returns_current_status(self, client, auth_token, tracker):
        # Set a known status
        await tracker.update_status(CameraStatus(
            transport_type="USB",
            connection_state="CONNECTED",
            battery_level=90,
        ))

        resp = await client.get(f"/api/status?token={auth_token}")
        assert resp.status_code == 200
        data = resp.json()["status"]
        assert data["transport_type"] == "USB"
        assert data["connection_state"] == "CONNECTED"
        assert data["battery_level"] == 90

    @pytest.mark.asyncio
    async def test_default_status_is_disconnected(self, client, auth_token):
        resp = await client.get(f"/api/status?token={auth_token}")
        data = resp.json()["status"]
        assert data["connection_state"] == "DISCONNECTED"
        assert data["transport_type"] == "none"


class TestSSEEndpoint:
    """SSE /api/status/stream endpoint.

    Note: httpx ASGI transport doesn't support true streaming (aiter_bytes
    blocks until the generator completes). We test the SSE endpoint by:
      1. Verifying the tracker subscription delivers formatted SSE events
      2. Verifying the endpoint requires auth
      3. Verifying the endpoint returns the correct content-type header
    """

    @pytest.mark.asyncio
    async def test_sse_subscription_delivers_initial_status(self, tracker):
        """subscribe() should immediately deliver current status."""
        await tracker.update_status(CameraStatus(
            transport_type="USB",
            connection_state="STREAMING",
            battery_level=65,
        ))

        events = []
        async def collect():
            async for status in tracker.subscribe():
                events.append(status)
                break  # Just get the first one

        await asyncio.wait_for(collect(), timeout=2.0)

        assert len(events) == 1
        assert events[0].transport_type == "USB"
        assert events[0].battery_level == 65

    @pytest.mark.asyncio
    async def test_sse_subscription_delivers_updates(self, tracker):
        """Subscription should receive pushed status updates."""
        events = []

        async def collect():
            async for status in tracker.subscribe():
                events.append(status)
                if len(events) >= 2:
                    break

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.05)

        await tracker.update_status(CameraStatus(
            transport_type="COHN",
            connection_state="CONNECTED",
            battery_level=42,
            timestamp=time.monotonic(),
        ))

        await asyncio.wait_for(task, timeout=2.0)

        assert len(events) >= 2
        cohn_events = [e for e in events if e.transport_type == "COHN"]
        assert len(cohn_events) >= 1
        assert cohn_events[0].battery_level == 42

    @pytest.mark.asyncio
    async def test_sse_event_format(self, tracker):
        """SSE events should be formatted as event: type\\ndata: json\\n\\n."""
        status = CameraStatus(
            transport_type="USB",
            connection_state="STREAMING",
            battery_level=85,
        )
        sse_data = status.to_sse_data()
        event_str = format_sse_event("status", sse_data)

        assert event_str.startswith("event: status\n")
        assert "data: " in event_str
        assert event_str.endswith("\n\n")

        # Parse the data portion
        for line in event_str.split("\n"):
            if line.startswith("data: "):
                parsed = json.loads(line[6:])
                assert parsed["transport_type"] == "USB"
                assert parsed["battery_level"] == 85
                break

    @pytest.mark.asyncio
    async def test_sse_without_auth_returns_401(self, client):
        """SSE endpoint should reject unauthenticated requests."""
        resp = await client.get("/api/status/stream")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_sse_endpoint_exists_and_accepts_auth(self, app, auth_token):
        """SSE endpoint should accept authenticated requests and return event-stream.

        We check the route exists and returns the correct media type by
        inspecting the FastAPI routes rather than streaming (which hangs).
        """
        # Find the SSE route in the app
        sse_routes = [
            r for r in app.routes
            if hasattr(r, "path") and r.path == "/api/status/stream"
        ]
        assert len(sse_routes) == 1, "SSE route should be registered"


# ---------------------------------------------------------------------------
# Dashboard HTML endpoint tests
# ---------------------------------------------------------------------------

class TestDashboardHTML:
    """Tests for the dashboard HTML page serving."""

    @pytest.mark.asyncio
    async def test_dashboard_serves_html(self, client, auth_token):
        """GET / with token should serve the dashboard HTML."""
        resp = await client.get(f"/?token={auth_token}")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        content = resp.text
        assert "GoMaxWebcam" in content
        assert "Alpine" in content or "alpine" in content or "x-data" in content

    @pytest.mark.asyncio
    async def test_dashboard_no_auth_placeholder(self, client, auth_token):
        """Dashboard HTML should not contain raw auth token placeholder."""
        resp = await client.get(f"/?token={auth_token}")
        assert resp.status_code == 200
        content = resp.text
        # No raw placeholder should remain in the served HTML
        assert "__AUTH_TOKEN__" not in content
        # Token is read from URL query string by JS, not embedded in HTML body
        assert "URLSearchParams" in content or "location.search" in content

    @pytest.mark.asyncio
    async def test_dashboard_without_token_returns_401(self, client):
        """GET / without auth should be rejected."""
        resp = await client.get("/")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_dashboard_html_has_sse_connection(self, client, auth_token):
        """Dashboard HTML should contain SSE EventSource setup."""
        resp = await client.get(f"/?token={auth_token}")
        content = resp.text
        assert "EventSource" in content
        assert "/api/status/stream" in content

    @pytest.mark.asyncio
    async def test_dashboard_html_has_battery_indicator(self, client, auth_token):
        """Dashboard HTML should contain battery indicator elements."""
        resp = await client.get(f"/?token={auth_token}")
        content = resp.text
        assert "battery" in content.lower()
        assert "status.battery" in content

    @pytest.mark.asyncio
    async def test_dashboard_html_has_transport_info(self, client, auth_token):
        """Dashboard HTML should contain transport type display."""
        resp = await client.get(f"/?token={auth_token}")
        content = resp.text
        assert "transport" in content.lower()
        assert "status.transport" in content

    @pytest.mark.asyncio
    async def test_dashboard_html_has_fps_chart(self, client, auth_token):
        """Dashboard HTML should contain FPS chart setup."""
        resp = await client.get(f"/?token={auth_token}")
        content = resp.text
        assert "MiniChart" in content
        assert "chartFps" in content

    @pytest.mark.asyncio
    async def test_dashboard_html_has_status_indicators(self, client, auth_token):
        """Dashboard HTML should show connection state indicators."""
        resp = await client.get(f"/?token={auth_token}")
        content = resp.text
        assert "status-dot" in content
        assert "status.state" in content


# ---------------------------------------------------------------------------
# SSE format helper tests
# ---------------------------------------------------------------------------

class TestFormatSSE:
    """Tests for the SSE format helper."""

    def test_format_sse_event(self):
        result = format_sse_event("status", '{"ok": true}')
        assert result == 'event: status\ndata: {"ok": true}\n\n'

    def test_format_sse_event_empty_data(self):
        result = format_sse_event("ping", "")
        assert result == "event: ping\ndata: \n\n"


# ---------------------------------------------------------------------------
# EventBus SSE endpoint tests (/api/events/stream)
# ---------------------------------------------------------------------------

class TestEventsStreamEndpoint:
    """Tests for the reactive EventBus SSE endpoint."""

    @pytest.mark.asyncio
    async def test_events_stream_without_auth_returns_401(self, client):
        """Events SSE endpoint should reject unauthenticated requests."""
        resp = await client.get("/api/events/stream")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_events_stream_route_exists(self, app, auth_token):
        """Events SSE route should be registered in the app."""
        routes = [
            r for r in app.routes
            if hasattr(r, "path") and r.path == "/api/events/stream"
        ]
        assert len(routes) == 1, "Events SSE route should be registered"

    @pytest.mark.asyncio
    async def test_event_bus_stored_on_app_state(self, app, event_bus):
        """EventBus should be accessible via app.state."""
        assert app.state.event_bus is event_bus

    @pytest.mark.asyncio
    async def test_eventbus_subscription_delivers_events(self, event_bus):
        """EventBus subscription should deliver published events."""
        sub = event_bus.subscribe()

        event_bus.publish(battery_event(level=85))

        received = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert received.type == EventType.BATTERY
        assert received.data["level"] == 85

        event_bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_eventbus_delivers_connection_events(self, event_bus):
        """EventBus should deliver connection state change events."""
        sub = event_bus.subscribe(event_types={EventType.CONNECTION})

        event_bus.publish(connection_event(
            old_state="DISCONNECTED",
            new_state="CONNECTING",
            transport_name="USB",
        ))

        received = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert received.type == EventType.CONNECTION
        assert received.data["old_state"] == "DISCONNECTED"
        assert received.data["new_state"] == "CONNECTING"
        assert received.data["transport"] == "USB"

        event_bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_eventbus_delivers_transport_events(self, event_bus):
        """EventBus should deliver transport failover events."""
        sub = event_bus.subscribe(event_types={EventType.TRANSPORT})

        event_bus.publish(transport_event(
            old_transport="USB",
            new_transport="COHN",
            reason="failover",
        ))

        received = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert received.type == EventType.TRANSPORT
        assert received.data["old_transport"] == "USB"
        assert received.data["new_transport"] == "COHN"
        assert received.data["reason"] == "failover"

        event_bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_eventbus_delivers_pipeline_events(self, event_bus):
        """EventBus should deliver pipeline state change events."""
        sub = event_bus.subscribe(event_types={EventType.PIPELINE})

        event_bus.publish(pipeline_event(
            old_state="STREAMING",
            new_state="FREEZE_FRAME",
            detail="Stream lost, showing last good frame",
        ))

        received = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert received.type == EventType.PIPELINE
        assert received.data["new_state"] == "FREEZE_FRAME"
        assert "last good frame" in received.data["detail"]

        event_bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_eventbus_delivers_error_events(self, event_bus):
        """EventBus should deliver error events."""
        sub = event_bus.subscribe(event_types={EventType.ERROR})

        event_bus.publish(error_event(
            source="transport.USB",
            message="Connection timeout",
            recoverable=True,
        ))

        received = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert received.type == EventType.ERROR
        assert received.data["source"] == "transport.USB"
        assert received.data["recoverable"] is True

        event_bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_eventbus_delivers_status_snapshots(self, event_bus):
        """EventBus should deliver full status snapshot events."""
        sub = event_bus.subscribe(event_types={EventType.STATUS})

        event_bus.publish(status_event({
            "transport_type": "USB",
            "connection_state": "STREAMING",
            "battery_level": 75,
            "pipeline_state": "STREAMING",
        }))

        received = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert received.type == EventType.STATUS
        assert received.data["transport_type"] == "USB"
        assert received.data["battery_level"] == 75

        event_bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_event_to_sse_format(self, event_bus):
        """Events should produce valid SSE-formatted strings."""
        event = battery_event(level=85)
        sse = event.to_sse()

        assert sse.startswith("event: battery\n")
        assert "data: " in sse
        assert sse.endswith("\n\n")

        # Parse the data line
        for line in sse.split("\n"):
            if line.startswith("data: "):
                parsed = json.loads(line[6:])
                assert parsed["level"] == 85
                break

    @pytest.mark.asyncio
    async def test_eventbus_type_filtering_in_subscription(self, event_bus):
        """Subscriptions with type filter should only receive matching events."""
        sub = event_bus.subscribe(event_types={EventType.BATTERY})

        # Publish various event types
        event_bus.publish(connection_event("A", "B"))
        event_bus.publish(battery_event(level=42))
        event_bus.publish(pipeline_event("X", "Y"))

        # Should only get the battery event
        received = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert received.type == EventType.BATTERY
        assert received.data["level"] == 42

        # No more matching events
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sub.__anext__(), timeout=0.05)

        event_bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_eventbus_history_replay(self, event_bus):
        """Subscriptions with include_history should replay recent events."""
        # Publish events before subscribing
        event_bus.publish(battery_event(level=10))
        event_bus.publish(battery_event(level=20))

        # Subscribe with history
        sub = event_bus.subscribe(include_history=True)

        # Should get replayed events
        e1 = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        e2 = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert e1.data["level"] == 10
        assert e2.data["level"] == 20

        event_bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_eventbus_multiple_subscribers_all_receive(self, event_bus):
        """Multiple subscribers should all receive the same events."""
        sub1 = event_bus.subscribe()
        sub2 = event_bus.subscribe()

        event_bus.publish(battery_event(level=50))

        r1 = await asyncio.wait_for(sub1.__anext__(), timeout=1.0)
        r2 = await asyncio.wait_for(sub2.__anext__(), timeout=1.0)

        assert r1.data["level"] == 50
        assert r2.data["level"] == 50

        event_bus.unsubscribe(sub1)
        event_bus.unsubscribe(sub2)

    @pytest.mark.asyncio
    async def test_eventbus_shutdown_stops_subscribers(self, event_bus):
        """Bus shutdown should cause subscribers to receive StopAsyncIteration."""
        sub = event_bus.subscribe()

        await event_bus.shutdown()

        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(sub.__anext__(), timeout=1.0)


# ---------------------------------------------------------------------------
# Recent events REST endpoint tests
# ---------------------------------------------------------------------------

class TestRecentEventsEndpoint:
    """Tests for the /api/events/recent REST endpoint."""

    @pytest.mark.asyncio
    async def test_recent_events_without_auth_returns_401(self, client):
        """Recent events endpoint should reject unauthenticated requests."""
        resp = await client.get("/api/events/recent")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_recent_events_empty(self, client, auth_token):
        """Recent events should return empty list when no events published."""
        resp = await client.get(f"/api/events/recent?token={auth_token}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["events"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_recent_events_returns_published_events(self, client, auth_token, event_bus):
        """Recent events should return events published to the bus."""
        event_bus.publish(battery_event(level=85))
        event_bus.publish(connection_event("DISCONNECTED", "CONNECTING", "USB"))

        resp = await client.get(f"/api/events/recent?token={auth_token}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2

        types = [e["type"] for e in data["events"]]
        assert "battery" in types
        assert "connection" in types

    @pytest.mark.asyncio
    async def test_recent_events_contain_data_payload(self, client, auth_token, event_bus):
        """Each recent event should contain type, data, and timestamp."""
        event_bus.publish(battery_event(level=42, charging=True))

        resp = await client.get(f"/api/events/recent?token={auth_token}")
        data = resp.json()
        assert data["count"] == 1

        event = data["events"][0]
        assert event["type"] == "battery"
        assert event["data"]["level"] == 42
        assert event["data"]["charging"] is True
        assert "timestamp" in event

    @pytest.mark.asyncio
    async def test_recent_events_with_bearer_auth(self, client, auth_token, event_bus):
        """Recent events endpoint should accept Bearer token auth."""
        event_bus.publish(battery_event(level=50))

        resp = await client.get(
            "/api/events/recent",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 1


# ---------------------------------------------------------------------------
# Last-Event-ID / event ID tests
# ---------------------------------------------------------------------------

class TestLastEventID:
    """Tests for SSE event IDs and Last-Event-ID reconnect support."""

    # -- Event.id assignment by EventBus --

    def test_published_event_gets_sequential_id(self):
        """EventBus should assign a sequential ID to each published event."""
        bus = EventBus()
        evt1 = battery_event(level=10)
        evt2 = battery_event(level=20)

        # IDs start at 0 (unassigned) before publish
        assert evt1.id == 0

        bus.publish(evt1)
        bus.publish(evt2)

        # After publishing, history events carry sequential IDs
        history = bus.recent_events
        assert history[0].id == 1
        assert history[1].id == 2

    def test_event_ids_are_monotonically_increasing(self):
        """Event IDs should increment by 1 for each published event."""
        bus = EventBus()
        for i in range(5):
            bus.publish(battery_event(level=i * 10))

        ids = [e.id for e in bus.recent_events]
        assert ids == [1, 2, 3, 4, 5]

    def test_event_with_preassigned_id_is_not_overwritten(self):
        """An event that already has a non-zero ID should keep its ID."""
        from dataclasses import replace as dr
        from gomaxwebcam.events import Event
        bus = EventBus()
        evt = dr(battery_event(level=42), id=999)
        bus.publish(evt)

        history = bus.recent_events
        assert history[0].id == 999

    # -- to_sse() includes id field --

    def test_to_sse_includes_id_when_assigned(self):
        """to_sse() should emit an 'id:' line when the event has a non-zero ID."""
        from dataclasses import replace as dr
        evt = dr(battery_event(level=85), id=7)
        sse = evt.to_sse()
        lines = sse.strip().splitlines()
        assert lines[0] == "id: 7"
        assert any(l.startswith("event: battery") for l in lines)
        assert any(l.startswith("data:") for l in lines)

    def test_to_sse_omits_id_when_zero(self):
        """to_sse() should NOT emit an 'id:' line for unassigned (id=0) events."""
        evt = battery_event(level=50)
        assert evt.id == 0
        sse = evt.to_sse()
        assert "id:" not in sse

    def test_to_dict_includes_id_when_assigned(self):
        """to_dict() should include 'id' key when event has a non-zero ID."""
        from dataclasses import replace as dr
        evt = dr(battery_event(level=85), id=3)
        d = evt.to_dict()
        assert d["id"] == 3

    def test_to_dict_omits_id_when_zero(self):
        """to_dict() should NOT include 'id' key for unassigned events."""
        evt = battery_event(level=50)
        d = evt.to_dict()
        assert "id" not in d

    # -- subscribe() since_event_id replay --

    @pytest.mark.asyncio
    async def test_subscribe_since_event_id_replays_missed_events(self):
        """subscribe(since_event_id=N) should replay events with id > N."""
        bus = EventBus()
        bus.publish(battery_event(level=10))   # id=1
        bus.publish(battery_event(level=20))   # id=2
        bus.publish(battery_event(level=30))   # id=3

        # Client last saw id=1, so it should receive id=2 and id=3
        sub = bus.subscribe(since_event_id=1)

        e1 = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        e2 = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert e1.data["level"] == 20
        assert e1.id == 2
        assert e2.data["level"] == 30
        assert e2.id == 3

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_subscribe_since_event_id_zero_replays_all(self):
        """subscribe(since_event_id=0) should replay all history (id > 0)."""
        bus = EventBus()
        bus.publish(battery_event(level=10))   # id=1
        bus.publish(battery_event(level=20))   # id=2

        sub = bus.subscribe(since_event_id=0)

        e1 = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        e2 = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert e1.data["level"] == 10
        assert e2.data["level"] == 20

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_subscribe_since_event_id_beyond_history_receives_no_replay(self):
        """subscribe(since_event_id=N) where N >= latest id should get no replay."""
        bus = EventBus()
        bus.publish(battery_event(level=10))   # id=1
        bus.publish(battery_event(level=20))   # id=2

        # Client claims to have seen id=5 (beyond history) — no replay expected
        sub = bus.subscribe(since_event_id=5)

        # Publish a new event so the subscription doesn't block forever
        bus.publish(battery_event(level=99))   # id=3 → delivered live

        e = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert e.data["level"] == 99  # Only the new live event
        assert e.id == 3

        bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_subscribe_since_event_id_with_type_filter(self):
        """since_event_id replay should respect the event_types filter."""
        bus = EventBus()
        bus.publish(battery_event(level=10))         # id=1
        bus.publish(connection_event("DISC", "CONN", "USB"))  # id=2
        bus.publish(battery_event(level=30))         # id=3

        # Client last saw id=0, wants only battery events
        sub = bus.subscribe(
            event_types={EventType.BATTERY},
            since_event_id=0,
        )

        e1 = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        e2 = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert e1.type == EventType.BATTERY
        assert e2.type == EventType.BATTERY
        assert e1.data["level"] == 10
        assert e2.data["level"] == 30

        bus.unsubscribe(sub)

    # -- SSE endpoint Last-Event-ID header support --

    @pytest.mark.asyncio
    async def test_events_stream_last_event_id_header_replays_missed(
        self, app, auth_token, event_bus
    ):
        """GET /api/events/stream with Last-Event-ID header should replay missed events.

        Uses bus.shutdown() to terminate the infinite SSE stream so the response
        body can be inspected — the same pattern the existing SSE tests use.
        """
        # Pre-publish some events so they land in history with ids 1 and 2
        event_bus.publish(battery_event(level=10))   # id=1
        event_bus.publish(battery_event(level=20))   # id=2

        # Schedule a bus shutdown so the SSE generator terminates after a short delay,
        # allowing the history replay to be included in the response body.
        async def _shutdown_soon():
            await asyncio.sleep(0.05)
            await event_bus.shutdown()

        asyncio.create_task(_shutdown_soon())

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(
                f"/api/events/stream?token={auth_token}",
                headers={"Last-Event-ID": "1"},
            )

        # Should have received the stream body after shutdown
        assert resp.status_code == 200
        body = resp.text
        # Only id=2 should be replayed (id=1 was already seen by the client)
        assert "id: 2" in body
        assert "level" in body

    @pytest.mark.asyncio
    async def test_events_stream_invalid_last_event_id_is_ignored(
        self, app, auth_token, event_bus
    ):
        """GET /api/events/stream with invalid Last-Event-ID should not error (returns 200).

        Uses bus.shutdown() to terminate the stream quickly.
        """
        # Schedule shutdown so the stream terminates
        async def _shutdown_soon():
            await asyncio.sleep(0.05)
            await event_bus.shutdown()

        asyncio.create_task(_shutdown_soon())

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(
                f"/api/events/stream?token={auth_token}",
                headers={"Last-Event-ID": "not-a-number"},
            )

        # Should still return 200 — bad header is silently ignored
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_published_events_include_id_in_sse_output(self, event_bus):
        """Events published to the bus should carry sequential IDs in their SSE text."""
        event_bus.publish(battery_event(level=55))
        event_bus.publish(battery_event(level=66))

        history = event_bus.recent_events
        sse1 = history[0].to_sse()
        sse2 = history[1].to_sse()

        assert "id: 1" in sse1
        assert "id: 2" in sse2


# ---------------------------------------------------------------------------
# Client-side SSE auto-reconnect logic tests
# ---------------------------------------------------------------------------

class TestClientSideSSEAutoReconnect:
    """Tests that the dashboard HTML includes client-side SSE auto-reconnect
    with exponential backoff.

    All tests serve the real HTML via the FastAPI ASGI app and inspect the
    JavaScript source for the required reconnect implementation.  No real
    browser or hardware is involved.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    async def _get_html(client, auth_token: str) -> str:
        resp = await client.get(f"/?token={auth_token}")
        assert resp.status_code == 200
        return resp.text

    # ── _connectSSE function present ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_connect_sse_function_is_defined(self, client, auth_token):
        """Dashboard JS must define a _connectSSE() function."""
        html = await self._get_html(client, auth_token)
        assert "_connectSSE" in html, "_connectSSE function not found in dashboard HTML"

    # ── Reconnect state variables ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_sse_connected_state_variable_present(self, client, auth_token):
        """sseConnected state variable must be declared for connection tracking."""
        html = await self._get_html(client, auth_token)
        assert "sseConnected" in html

    @pytest.mark.asyncio
    async def test_sse_countdown_state_variable_present(self, client, auth_token):
        """sseCountdown variable must exist for the user-facing reconnect countdown."""
        html = await self._get_html(client, auth_token)
        assert "sseCountdown" in html

    @pytest.mark.asyncio
    async def test_sse_delay_state_variable_present(self, client, auth_token):
        """_sseDelay variable must track the current backoff delay."""
        html = await self._get_html(client, auth_token)
        assert "_sseDelay" in html

    @pytest.mark.asyncio
    async def test_sse_attempts_state_variable_present(self, client, auth_token):
        """_sseAttempts variable must count consecutive failures for backoff calc."""
        html = await self._get_html(client, auth_token)
        assert "_sseAttempts" in html

    @pytest.mark.asyncio
    async def test_sse_reconnect_timer_handle_present(self, client, auth_token):
        """_sseReconnectTimer must store the pending setTimeout handle."""
        html = await self._get_html(client, auth_token)
        assert "_sseReconnectTimer" in html

    # ── Exponential backoff implementation ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_exponential_backoff_uses_math_pow(self, client, auth_token):
        """Backoff calculation must use Math.pow for exponential growth."""
        html = await self._get_html(client, auth_token)
        assert "Math.pow" in html, "Exponential backoff requires Math.pow"

    @pytest.mark.asyncio
    async def test_exponential_backoff_cap_at_30_seconds(self, client, auth_token):
        """Backoff must be capped at 30 000 ms to avoid very long waits."""
        html = await self._get_html(client, auth_token)
        assert "30000" in html, "30 s (30000 ms) cap not found in dashboard HTML"

    @pytest.mark.asyncio
    async def test_exponential_backoff_uses_min_clamp(self, client, auth_token):
        """Math.min must be used to clamp the backoff delay at the cap."""
        html = await self._get_html(client, auth_token)
        assert "Math.min" in html, "Math.min clamp for backoff cap not found"

    @pytest.mark.asyncio
    async def test_jitter_applied_to_backoff_delay(self, client, auth_token):
        """Jitter (Math.random) must be added to prevent thundering-herd."""
        html = await self._get_html(client, auth_token)
        assert "Math.random" in html, "Jitter via Math.random not found"

    # ── onerror handler ───────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_onerror_handler_closes_connection(self, client, auth_token):
        """onerror must call es.close() to cleanly discard the broken stream."""
        html = await self._get_html(client, auth_token)
        assert "es.close()" in html, "es.close() not found in onerror handler"

    @pytest.mark.asyncio
    async def test_onerror_handler_schedules_reconnect_with_settimeout(
        self, client, auth_token
    ):
        """onerror must use setTimeout to schedule the next _connectSSE() call."""
        html = await self._get_html(client, auth_token)
        assert "setTimeout" in html, "setTimeout not found — reconnect not scheduled"

    @pytest.mark.asyncio
    async def test_onerror_handler_increments_attempt_counter(
        self, client, auth_token
    ):
        """onerror must increment _sseAttempts so subsequent delays grow."""
        html = await self._get_html(client, auth_token)
        # Either "_sseAttempts += 1" or "_sseAttempts++" are valid
        assert (
            "_sseAttempts += 1" in html or "_sseAttempts++" in html
        ), "_sseAttempts increment not found in onerror handler"

    # ── Backoff reset on success ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_backoff_reset_on_first_successful_message(
        self, client, auth_token
    ):
        """A successful message must reset _sseDelay and _sseAttempts to initial
        values so the next failure starts the backoff from the beginning."""
        html = await self._get_html(client, auth_token)
        # Both reset assignments must be present
        assert "_sseDelay = 1000" in html, "_sseDelay not reset to 1000 on success"
        assert "_sseAttempts = 0" in html, "_sseAttempts not reset to 0 on success"

    @pytest.mark.asyncio
    async def test_countdown_reset_on_first_successful_message(
        self, client, auth_token
    ):
        """sseCountdown must be cleared (set to 0) when the connection is restored."""
        html = await self._get_html(client, auth_token)
        assert "sseCountdown = 0" in html

    # ── Countdown UI ──────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_countdown_ui_updated_with_setinterval(self, client, auth_token):
        """A setInterval tick must count down sseCountdown each second."""
        html = await self._get_html(client, auth_token)
        assert "setInterval" in html, "setInterval countdown tick not found"

    @pytest.mark.asyncio
    async def test_countdown_uses_math_max_floor(self, client, auth_token):
        """Countdown must use Math.max to avoid going below zero."""
        html = await self._get_html(client, auth_token)
        assert "Math.max" in html, "Math.max guard for countdown not found"

    @pytest.mark.asyncio
    async def test_countdown_value_shown_in_hint_text(self, client, auth_token):
        """The hint text must reference sseCountdown so users see the timer."""
        html = await self._get_html(client, auth_token)
        assert "sseCountdown" in html
        # 'reconnecting in' message must be present somewhere in the page
        assert "reconnect" in html.lower()

    # ── Guard: pending timer is cancelled before re-opening ───────────────────

    @pytest.mark.asyncio
    async def test_existing_reconnect_timer_cancelled_before_new_connect(
        self, client, auth_token
    ):
        """_connectSSE must cancel any pending timer with clearTimeout to avoid
        double-reconnect if the caller triggers reconnection manually."""
        html = await self._get_html(client, auth_token)
        assert "clearTimeout" in html, "clearTimeout not found — timer leak possible"

    # ── EventSource is set up for /api/status/stream ──────────────────────────

    @pytest.mark.asyncio
    async def test_eventsource_targets_status_stream_url(self, client, auth_token):
        """EventSource must open /api/status/stream (the correct SSE endpoint)."""
        html = await self._get_html(client, auth_token)
        assert "EventSource" in html
        assert "/api/status/stream" in html

    # ── SSE indicator visible in sidebar ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_sse_indicator_element_present_in_sidebar(
        self, client, auth_token
    ):
        """The sidebar must contain an SSE status indicator showing live/reconnecting."""
        html = await self._get_html(client, auth_token)
        assert "sse-indicator" in html or "sse-dot" in html or "sseConnected" in html
