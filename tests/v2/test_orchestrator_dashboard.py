"""
test_orchestrator_dashboard.py — Tests for AppOrchestrator FastAPI lifecycle integration.

Verifies:
  - create_dashboard() produces a working FastAPI app
  - Lifespan startup calls orch.start()
  - Lifespan shutdown calls orch.stop()
  - Orchestrator reference is accessible on app.state
  - create_dashboard() can only be called once
  - dashboard_url property returns the expected URL
  - Auth token is properly forwarded to the app
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gomaxwebcam.orchestrator import AppOrchestrator

# All tests in this module are pure unit tests with mocks — no GoPro needed.
pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_event_bus():
    bus = MagicMock()
    bus.set_loop = MagicMock()
    bus.shutdown = AsyncMock()
    bus.subscribe = MagicMock()
    bus.recent_events = []
    return bus


@pytest.fixture
def mock_camera_manager():
    cm = MagicMock()
    cm.start = AsyncMock()
    cm.stop = AsyncMock()
    cm.set_transport = MagicMock()
    cm.transport = None
    return cm


@pytest.fixture
def mock_transport_manager():
    tm = MagicMock()
    tm.start = AsyncMock()
    tm.stop = AsyncMock()
    tm.on_transport_switch = None
    tm.on_freeze = None
    tm.on_unfreeze = None
    return tm


@pytest.fixture
def mock_status_tracker():
    st = MagicMock()
    st.start = AsyncMock()
    st.stop = AsyncMock()
    st.set_transport = MagicMock()
    return st


@pytest.fixture
def orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager, mock_status_tracker):
    return AppOrchestrator(
        event_bus=mock_event_bus,
        camera_manager=mock_camera_manager,
        transport_manager=mock_transport_manager,
        status_tracker=mock_status_tracker,
    )


# ---------------------------------------------------------------------------
# Tests: create_dashboard
# ---------------------------------------------------------------------------

class TestCreateDashboard:
    """Tests for AppOrchestrator.create_dashboard()."""

    def test_creates_fastapi_app(self, orchestrator):
        """create_dashboard() returns a FastAPI app instance."""
        from fastapi import FastAPI

        app = orchestrator.create_dashboard(auth_token="test-token", host="127.0.0.1", port=9999)
        assert isinstance(app, FastAPI)

    def test_stores_app_on_orchestrator(self, orchestrator):
        """The created app is accessible via orchestrator.app."""
        app = orchestrator.create_dashboard(auth_token="test-token", host="127.0.0.1", port=9999)
        assert orchestrator.app is app

    def test_stores_auth_token(self, orchestrator):
        """Auth token is stored on the orchestrator."""
        orchestrator.create_dashboard(auth_token="my-secret-token", host="127.0.0.1", port=9999)
        assert orchestrator.auth_token == "my-secret-token"

    def test_generates_auth_token_when_none(self, orchestrator):
        """A random token is generated when none is provided."""
        orchestrator.create_dashboard(host="127.0.0.1", port=9999)
        assert orchestrator.auth_token is not None
        assert len(orchestrator.auth_token) > 10

    def test_dashboard_url_property(self, orchestrator):
        """dashboard_url returns full URL with auth token."""
        orchestrator.create_dashboard(auth_token="abc123", host="127.0.0.1", port=8080)
        assert orchestrator.dashboard_url == "http://127.0.0.1:8080/?token=abc123"

    def test_dashboard_url_none_before_create(self, orchestrator):
        """dashboard_url is None before create_dashboard() is called."""
        assert orchestrator.dashboard_url is None

    def test_cannot_create_twice(self, orchestrator):
        """create_dashboard() raises RuntimeError on second call."""
        orchestrator.create_dashboard(auth_token="token1", host="127.0.0.1", port=9999)
        with pytest.raises(RuntimeError, match="already created"):
            orchestrator.create_dashboard(auth_token="token2", host="127.0.0.1", port=9998)

    def test_orchestrator_on_app_state(self, orchestrator):
        """The orchestrator is accessible on app.state.orchestrator."""
        app = orchestrator.create_dashboard(auth_token="test", host="127.0.0.1", port=9999)
        assert app.state.orchestrator is orchestrator

    def test_event_bus_on_app_state(self, orchestrator, mock_event_bus):
        """The EventBus is accessible on app.state.event_bus."""
        app = orchestrator.create_dashboard(auth_token="test", host="127.0.0.1", port=9999)
        assert app.state.event_bus is mock_event_bus

    def test_tracker_on_app_state(self, orchestrator, mock_status_tracker):
        """The status tracker is accessible on app.state.tracker."""
        app = orchestrator.create_dashboard(auth_token="test", host="127.0.0.1", port=9999)
        assert app.state.tracker is mock_status_tracker

    def test_auth_token_on_app_state(self, orchestrator):
        """The auth token is accessible on app.state.auth_token."""
        app = orchestrator.create_dashboard(auth_token="my-token", host="127.0.0.1", port=9999)
        assert app.state.auth_token == "my-token"


# ---------------------------------------------------------------------------
# Tests: Lifespan integration
# ---------------------------------------------------------------------------

class TestLifespanIntegration:
    """Tests for the FastAPI lifespan → orchestrator start/stop wiring."""

    @pytest.mark.asyncio
    async def test_lifespan_calls_start_and_stop(self, orchestrator):
        """The lifespan context manager calls start() on entry and stop() on exit."""
        app = orchestrator.create_dashboard(auth_token="test", host="127.0.0.1", port=9999)

        # Patch start/stop to track calls without side effects
        orchestrator.start = AsyncMock()
        orchestrator.stop = AsyncMock()

        # Exercise the lifespan
        lifespan = app.router.lifespan_context
        async with lifespan(app):
            orchestrator.start.assert_awaited_once()
            orchestrator.stop.assert_not_awaited()

        # After exiting, stop should have been called
        orchestrator.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_stop_called_on_exception(self, orchestrator):
        """stop() is called even if an exception occurs during the lifespan body."""
        app = orchestrator.create_dashboard(auth_token="test", host="127.0.0.1", port=9999)

        orchestrator.start = AsyncMock()
        orchestrator.stop = AsyncMock()

        lifespan = app.router.lifespan_context

        with pytest.raises(ValueError):
            async with lifespan(app):
                raise ValueError("simulated crash")

        # stop() must be called for cleanup even after exception
        orchestrator.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_health_endpoint_available(self, orchestrator):
        """The /health endpoint is accessible on the created app."""
        from httpx import AsyncClient, ASGITransport

        app = orchestrator.create_dashboard(auth_token="test", host="127.0.0.1", port=9999)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"ok": True}
