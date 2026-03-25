"""
tests/v2/test_dashboard_controls.py — Tests for new dashboard POST endpoints.

Tests the camera control, transport, and settings endpoints:
  1. POST /api/camera/resolution — set resolution with validation
  2. POST /api/camera/fov — set FOV with validation
  3. POST /api/transport/priority — reorder transport priority
  4. POST /api/transport/switch — manual transport switch
  5. POST /api/settings/auto-start — toggle auto-start
  6. POST /api/settings — update general settings
  7. GET /api/settings — retrieve current settings
  8. Auth required for all endpoints

All tests mock at the orchestrator/config boundary — no real hardware needed.
Memory footprint is minimal (no pipeline, no frames).
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from gomaxwebcam.dashboard.status_tracker import CameraStatusTracker
from gomaxwebcam.dashboard.app import create_app
from gomaxwebcam.events import EventBus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker():
    return CameraStatusTracker()


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def auth_token():
    return "test-token-controls-123"


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock Config object that writes to a temp directory."""
    from gomaxwebcam.config import Config
    cfg = Config(config_dir=tmp_path)
    cfg.video.resolution = "1080p"
    cfg.video.fov = "wide"
    cfg.transport.priority = ["usb", "cohn", "wifi_ap"]
    cfg.transport.ble_wake_mode = "always_on"
    cfg.logging.debug = False
    return cfg


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator with a mock transport manager."""
    orch = MagicMock()
    orch.set_resolution = AsyncMock()
    orch.set_fov = AsyncMock()

    tm = MagicMock()
    tm.force_failover = AsyncMock(return_value=True)
    tm._config = MagicMock()
    tm._config.priority = ["USB", "COHN", "WiFi AP"]
    tm.get_status = MagicMock(return_value={
        "state": "ACTIVE",
        "active_transport": "USB",
        "priority": ["USB", "COHN", "WiFi AP"],
        "failover_count": 0,
    })

    orch.transport_manager = tm
    return orch


@pytest.fixture
def app(tracker, event_bus, auth_token, mock_orchestrator, mock_config):
    """Create FastAPI app with mocked orchestrator and config."""
    application = create_app(
        tracker=tracker, event_bus=event_bus, auth_token=auth_token,
    )
    application.state.orchestrator = mock_orchestrator
    application.state.config = mock_config
    return application


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Resolution endpoint tests
# ---------------------------------------------------------------------------

class TestSetResolution:
    """POST /api/camera/resolution tests."""

    @pytest.mark.asyncio
    async def test_set_resolution_1080p(self, client, auth_token, mock_config):
        resp = await client.post(
            "/api/camera/resolution",
            json={"resolution": "1080p"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["resolution"] == "1080p"
        assert mock_config.video.resolution == "1080p"

    @pytest.mark.asyncio
    async def test_set_resolution_720p(self, client, auth_token, mock_config):
        resp = await client.post(
            "/api/camera/resolution",
            json={"resolution": "720p"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        assert resp.json()["resolution"] == "720p"
        assert mock_config.video.resolution == "720p"

    @pytest.mark.asyncio
    async def test_set_resolution_480p(self, client, auth_token, mock_config):
        resp = await client.post(
            "/api/camera/resolution",
            json={"resolution": "480p"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        assert resp.json()["resolution"] == "480p"

    @pytest.mark.asyncio
    async def test_set_resolution_invalid(self, client, auth_token):
        resp = await client.post(
            "/api/camera/resolution",
            json={"resolution": "4k"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 400
        assert "error" in resp.json()

    @pytest.mark.asyncio
    async def test_set_resolution_no_auth(self, client):
        resp = await client.post(
            "/api/camera/resolution",
            json={"resolution": "1080p"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_set_resolution_calls_orchestrator(
        self, client, auth_token, mock_orchestrator,
    ):
        resp = await client.post(
            "/api/camera/resolution",
            json={"resolution": "720p"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        mock_orchestrator.set_resolution.assert_awaited_once_with("720p")


# ---------------------------------------------------------------------------
# FOV endpoint tests
# ---------------------------------------------------------------------------

class TestSetFov:
    """POST /api/camera/fov tests."""

    @pytest.mark.asyncio
    async def test_set_fov_wide(self, client, auth_token, mock_config):
        resp = await client.post(
            "/api/camera/fov",
            json={"fov": "wide"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        assert resp.json()["fov"] == "wide"
        assert mock_config.video.fov == "wide"

    @pytest.mark.asyncio
    async def test_set_fov_linear(self, client, auth_token):
        resp = await client.post(
            "/api/camera/fov",
            json={"fov": "linear"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        assert resp.json()["fov"] == "linear"

    @pytest.mark.asyncio
    async def test_set_fov_narrow(self, client, auth_token):
        resp = await client.post(
            "/api/camera/fov",
            json={"fov": "narrow"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_set_fov_superview(self, client, auth_token):
        resp = await client.post(
            "/api/camera/fov",
            json={"fov": "superview"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        assert resp.json()["fov"] == "superview"

    @pytest.mark.asyncio
    async def test_set_fov_invalid(self, client, auth_token):
        resp = await client.post(
            "/api/camera/fov",
            json={"fov": "fisheye"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_set_fov_no_auth(self, client):
        resp = await client.post("/api/camera/fov", json={"fov": "wide"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Transport priority tests
# ---------------------------------------------------------------------------

class TestTransportPriority:
    """POST /api/transport/priority tests."""

    @pytest.mark.asyncio
    async def test_set_priority(self, client, auth_token, mock_config):
        resp = await client.post(
            "/api/transport/priority",
            json={"priority": ["COHN", "USB", "WiFi AP"]},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["priority"] == ["cohn", "usb", "wifi_ap"]
        assert mock_config.transport.priority == ["cohn", "usb", "wifi_ap"]

    @pytest.mark.asyncio
    async def test_set_priority_empty(self, client, auth_token):
        resp = await client.post(
            "/api/transport/priority",
            json={"priority": []},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_set_priority_invalid_type(self, client, auth_token):
        resp = await client.post(
            "/api/transport/priority",
            json={"priority": "usb"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_set_priority_no_auth(self, client):
        resp = await client.post(
            "/api/transport/priority",
            json={"priority": ["USB"]},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Transport switch tests
# ---------------------------------------------------------------------------

class TestTransportSwitch:
    """POST /api/transport/switch tests."""

    @pytest.mark.asyncio
    async def test_switch_to_usb(self, client, auth_token, mock_orchestrator):
        resp = await client.post(
            "/api/transport/switch",
            json={"transport": "usb"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["transport"] == "USB"
        mock_orchestrator.transport_manager.force_failover.assert_awaited_once_with("USB")

    @pytest.mark.asyncio
    async def test_switch_to_cohn(self, client, auth_token, mock_orchestrator):
        resp = await client.post(
            "/api/transport/switch",
            json={"transport": "cohn"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        assert resp.json()["transport"] == "COHN"

    @pytest.mark.asyncio
    async def test_switch_to_wifi_ap(self, client, auth_token, mock_orchestrator):
        resp = await client.post(
            "/api/transport/switch",
            json={"transport": "wifi_ap"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        assert resp.json()["transport"] == "WiFi AP"

    @pytest.mark.asyncio
    async def test_switch_empty_name(self, client, auth_token):
        resp = await client.post(
            "/api/transport/switch",
            json={"transport": ""},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_switch_failed(self, client, auth_token, mock_orchestrator):
        mock_orchestrator.transport_manager.force_failover = AsyncMock(return_value=False)
        resp = await client.post(
            "/api/transport/switch",
            json={"transport": "usb"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is False

    @pytest.mark.asyncio
    async def test_switch_no_orchestrator(self, client, auth_token, app):
        app.state.orchestrator = None
        resp = await client.post(
            "/api/transport/switch",
            json={"transport": "usb"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_switch_no_auth(self, client):
        resp = await client.post(
            "/api/transport/switch",
            json={"transport": "usb"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Auto-start endpoint tests
# ---------------------------------------------------------------------------

class TestAutoStart:
    """POST /api/settings/auto-start tests."""

    @pytest.mark.asyncio
    async def test_enable_auto_start(self, client, auth_token):
        resp = await client.post(
            "/api/settings/auto-start",
            json={"enabled": True},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert resp.json()["auto_start"] is True

    @pytest.mark.asyncio
    async def test_disable_auto_start(self, client, auth_token):
        resp = await client.post(
            "/api/settings/auto-start",
            json={"enabled": False},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        assert resp.json()["auto_start"] is False


# ---------------------------------------------------------------------------
# General settings endpoint tests
# ---------------------------------------------------------------------------

class TestSettings:
    """POST /api/settings and GET /api/settings tests."""

    @pytest.mark.asyncio
    async def test_update_ble_wake_mode(self, client, auth_token, mock_config):
        resp = await client.post(
            "/api/settings",
            json={"ble_wake_mode": "battery_saver"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        assert "ble_wake_mode" in resp.json()["updated"]
        assert mock_config.transport.ble_wake_mode == "battery_saver"

    @pytest.mark.asyncio
    async def test_update_debug_logging(self, client, auth_token, mock_config):
        resp = await client.post(
            "/api/settings",
            json={"debug_logging": True},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        assert mock_config.logging.debug is True

    @pytest.mark.asyncio
    async def test_update_multiple_settings(self, client, auth_token, mock_config):
        resp = await client.post(
            "/api/settings",
            json={"ble_wake_mode": "always_on", "debug_logging": False},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "ble_wake_mode" in data["updated"]
        assert "debug_logging" in data["updated"]

    @pytest.mark.asyncio
    async def test_get_settings(self, client, auth_token, mock_config):
        resp = await client.get(
            "/api/settings",
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "settings" in data
        s = data["settings"]
        assert s["resolution"] == "1080p"
        assert s["fov"] == "wide"
        assert s["transport_priority"] == ["usb", "cohn", "wifi_ap"]

    @pytest.mark.asyncio
    async def test_get_settings_no_auth(self, client):
        resp = await client.get("/api/settings")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_settings_no_config_falls_back_to_disk(self, client, auth_token, app):
        """When config is not cached on app.state, _load_config loads from disk."""
        app.state.config = None
        resp = await client.post(
            "/api/settings",
            json={"debug_logging": True},
            headers=auth_headers(auth_token),
        )
        # Should succeed by loading config from disk (creates default if missing)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Config persistence tests
# ---------------------------------------------------------------------------

class TestConfigPersistence:
    """Verify that settings endpoints persist to config.toml."""

    @pytest.mark.asyncio
    async def test_resolution_persisted(self, client, auth_token, mock_config, tmp_path):
        """Resolution change is written to disk."""
        mock_config._config_dir = tmp_path
        mock_config._config_path = tmp_path / "config.toml"

        resp = await client.post(
            "/api/camera/resolution",
            json={"resolution": "720p"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200

        # Config was saved — the file should exist
        assert mock_config._config_path.exists()

    @pytest.mark.asyncio
    async def test_fov_persisted(self, client, auth_token, mock_config, tmp_path):
        """FOV change is written to disk."""
        mock_config._config_dir = tmp_path
        mock_config._config_path = tmp_path / "config.toml"

        resp = await client.post(
            "/api/camera/fov",
            json={"fov": "linear"},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        assert mock_config._config_path.exists()

    @pytest.mark.asyncio
    async def test_priority_persisted(self, client, auth_token, mock_config, tmp_path):
        """Transport priority change is written to disk."""
        mock_config._config_dir = tmp_path
        mock_config._config_path = tmp_path / "config.toml"

        resp = await client.post(
            "/api/transport/priority",
            json={"priority": ["COHN", "USB"]},
            headers=auth_headers(auth_token),
        )
        assert resp.status_code == 200
        assert mock_config._config_path.exists()
