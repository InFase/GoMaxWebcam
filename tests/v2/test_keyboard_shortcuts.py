"""
tests/v2/test_keyboard_shortcuts.py — Tests for keyboard shortcut backend endpoints.

Tests the shortcut-triggered API endpoints:
  1. POST /api/camera/pause        — P key: pause/resume stream
  2. POST /api/camera/reconnect    — R key: force reconnect
  3. POST /api/camera/visibility   — V key: toggle virtual camera visibility

Also tests FramePipeline.hide() / show() visibility methods.

All tests mock at the orchestrator/pipeline boundary — no real hardware needed.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from gomaxwebcam.dashboard.app import create_app
from gomaxwebcam.dashboard.status_tracker import CameraStatusTracker
from gomaxwebcam.events import EventBus

pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def auth_token():
    return "test-token-shortcuts-xyz"


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def tracker():
    return CameraStatusTracker()


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.freeze = MagicMock()
    pipeline.unfreeze = MagicMock()
    pipeline.hide = MagicMock()
    pipeline.show = MagicMock()
    return pipeline


@pytest.fixture
def mock_orchestrator(mock_pipeline):
    orch = MagicMock()
    orch.pipeline = mock_pipeline
    tm = MagicMock()
    tm.force_failover = AsyncMock(return_value=True)
    orch.transport_manager = tm
    return orch


@pytest.fixture
def app(tracker, event_bus, auth_token, mock_orchestrator):
    application = create_app(
        tracker=tracker,
        event_bus=event_bus,
        auth_token=auth_token,
    )
    application.state.orchestrator = mock_orchestrator
    return application


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def auth(token):
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# POST /api/camera/pause  (P key)
# ---------------------------------------------------------------------------

class TestPauseShortcut:
    """Tests for the P-key pause/resume endpoint."""

    @pytest.mark.asyncio
    async def test_pause_stream(self, client, auth_token, mock_pipeline):
        resp = await client.post(
            "/api/camera/pause",
            json={"paused": True},
            headers=auth(auth_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["paused"] is True
        mock_pipeline.freeze.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_stream(self, client, auth_token, mock_pipeline):
        resp = await client.post(
            "/api/camera/pause",
            json={"paused": False},
            headers=auth(auth_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["paused"] is False
        mock_pipeline.unfreeze.assert_called_once()

    @pytest.mark.asyncio
    async def test_pause_no_pipeline(self, client, auth_token, app, mock_orchestrator):
        mock_orchestrator.pipeline = None
        resp = await client.post(
            "/api/camera/pause",
            json={"paused": True},
            headers=auth(auth_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "detail" in data  # friendly message when camera not connected

    @pytest.mark.asyncio
    async def test_pause_no_orchestrator(self, client, auth_token, app):
        app.state.orchestrator = None
        resp = await client.post(
            "/api/camera/pause",
            json={"paused": True},
            headers=auth(auth_token),
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_pause_requires_auth(self, client):
        resp = await client.post("/api/camera/pause", json={"paused": True})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /api/camera/reconnect  (R key)
# ---------------------------------------------------------------------------

class TestReconnectShortcut:
    """Tests for the R-key reconnect endpoint."""

    @pytest.mark.asyncio
    async def test_reconnect_triggers_failover(
        self, client, auth_token, mock_orchestrator
    ):
        resp = await client.post(
            "/api/camera/reconnect",
            json={},
            headers=auth(auth_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True

    @pytest.mark.asyncio
    async def test_reconnect_no_orchestrator(self, client, auth_token, app):
        app.state.orchestrator = None
        resp = await client.post(
            "/api/camera/reconnect",
            json={},
            headers=auth(auth_token),
        )
        assert resp.status_code == 503
        assert "error" in resp.json()

    @pytest.mark.asyncio
    async def test_reconnect_no_transport_manager(
        self, client, auth_token, mock_orchestrator
    ):
        mock_orchestrator.transport_manager = None
        resp = await client.post(
            "/api/camera/reconnect",
            json={},
            headers=auth(auth_token),
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_reconnect_requires_auth(self, client):
        resp = await client.post("/api/camera/reconnect", json={})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /api/camera/visibility  (V key)
# ---------------------------------------------------------------------------

class TestVisibilityShortcut:
    """Tests for the V-key virtual camera visibility toggle endpoint."""

    @pytest.mark.asyncio
    async def test_hide_virtual_camera(self, client, auth_token, mock_pipeline):
        resp = await client.post(
            "/api/camera/visibility",
            json={"hidden": True},
            headers=auth(auth_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["hidden"] is True
        mock_pipeline.hide.assert_called_once()

    @pytest.mark.asyncio
    async def test_show_virtual_camera(self, client, auth_token, mock_pipeline):
        resp = await client.post(
            "/api/camera/visibility",
            json={"hidden": False},
            headers=auth(auth_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["hidden"] is False
        mock_pipeline.show.assert_called_once()

    @pytest.mark.asyncio
    async def test_visibility_no_pipeline(self, client, auth_token, mock_orchestrator):
        mock_orchestrator.pipeline = None
        resp = await client.post(
            "/api/camera/visibility",
            json={"hidden": True},
            headers=auth(auth_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "detail" in data  # friendly message when no pipeline

    @pytest.mark.asyncio
    async def test_visibility_no_orchestrator(self, client, auth_token, app):
        app.state.orchestrator = None
        resp = await client.post(
            "/api/camera/visibility",
            json={"hidden": True},
            headers=auth(auth_token),
        )
        assert resp.status_code == 200
        # No orchestrator → no pipeline → graceful fallback
        assert resp.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_visibility_requires_auth(self, client):
        resp = await client.post("/api/camera/visibility", json={"hidden": True})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_visibility_default_hidden_true(self, client, auth_token, mock_pipeline):
        """Body with no 'hidden' key should default to hiding (True)."""
        resp = await client.post(
            "/api/camera/visibility",
            json={},
            headers=auth(auth_token),
        )
        assert resp.status_code == 200
        assert resp.json()["hidden"] is True
        mock_pipeline.hide.assert_called_once()


# ---------------------------------------------------------------------------
# FramePipeline.hide() / show()  unit tests
# ---------------------------------------------------------------------------

class TestPipelineVisibility:
    """Unit tests for FramePipeline hide/show methods and blank frame logic."""

    def _make_pipeline(self):
        from gomaxwebcam.pipeline.frame_pipeline import FramePipeline, PipelineConfig
        pipeline = FramePipeline(config=PipelineConfig())
        return pipeline

    def test_hidden_flag_starts_false(self):
        pipeline = self._make_pipeline()
        assert pipeline._hidden is False

    def test_hide_sets_flag(self):
        pipeline = self._make_pipeline()
        pipeline.hide()
        assert pipeline._hidden is True

    def test_show_clears_flag(self):
        pipeline = self._make_pipeline()
        pipeline.hide()
        pipeline.show()
        assert pipeline._hidden is False

    def test_hide_show_idempotent(self):
        pipeline = self._make_pipeline()
        pipeline.hide()
        pipeline.hide()  # double-hide is fine
        assert pipeline._hidden is True
        pipeline.show()
        pipeline.show()  # double-show is fine
        assert pipeline._hidden is False

    def test_decoded_frame_sends_blank_when_hidden(self):
        """When hidden, _on_decoded_frame sends zeros, not the real frame."""
        pipeline = self._make_pipeline()
        pipeline._hidden = True

        submitted_frames = []
        sink = MagicMock()
        sink.submit_frame = MagicMock(side_effect=submitted_frames.append)
        pipeline._vcam_sink = sink

        # Simulate a coloured frame arriving from the decoder
        real_frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        pipeline._on_decoded_frame(real_frame)

        assert len(submitted_frames) == 1
        blank = submitted_frames[0]
        assert blank.shape == real_frame.shape
        assert blank.dtype == real_frame.dtype
        assert np.all(blank == 0), "Hidden mode must produce all-black frames"

    def test_decoded_frame_sends_real_when_visible(self):
        """When visible, _on_decoded_frame sends the original frame."""
        pipeline = self._make_pipeline()
        pipeline._hidden = False

        submitted_frames = []
        sink = MagicMock()
        sink.submit_frame = MagicMock(side_effect=submitted_frames.append)
        pipeline._vcam_sink = sink

        real_frame = np.full((1080, 1920, 3), 200, dtype=np.uint8)
        pipeline._on_decoded_frame(real_frame)

        assert len(submitted_frames) == 1
        assert submitted_frames[0] is real_frame, "Visible mode must pass frame through"

    def test_toggle_visibility_live(self):
        """Toggling hide/show mid-stream changes what frames are submitted."""
        pipeline = self._make_pipeline()

        submitted_frames = []
        sink = MagicMock()
        sink.submit_frame = MagicMock(side_effect=submitted_frames.append)
        pipeline._vcam_sink = sink

        frame = np.full((1080, 1920, 3), 99, dtype=np.uint8)

        # Visible: real frame passes through
        pipeline._on_decoded_frame(frame)
        assert np.all(submitted_frames[-1] == 99)

        # Hide: blank frame
        pipeline.hide()
        pipeline._on_decoded_frame(frame)
        assert np.all(submitted_frames[-1] == 0)

        # Show: real frame again
        pipeline.show()
        pipeline._on_decoded_frame(frame)
        assert np.all(submitted_frames[-1] == 99)
