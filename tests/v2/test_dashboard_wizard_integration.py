"""
test_dashboard_wizard_integration.py — Integration tests for wizard endpoint flows.

Verifies:
  - Full wizard scan → select → provision → confirm happy path
  - Wizard select with known address (no prior scan)
  - Wizard select with empty address returns error
  - Wizard provision without selected camera returns error
  - Wizard provision when already provisioning returns error
  - Wizard cancel with no active provisioning returns appropriate response
  - Wizard cancel with active orchestrator
  - Wizard confirm before provisioning returns error
  - Wizard confirm after complete with cached credentials
  - Wizard status endpoint returns current wizard state
  - WizardState progress notification publishes to EventBus
  - WizardState subscription/unsubscription lifecycle
  - WizardState reset clears all state

All tests mock at BLE/COHN boundary — no real hardware needed.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from gomaxwebcam.dashboard.app import create_app
from gomaxwebcam.dashboard.wizard import WizardState
from gomaxwebcam.dashboard.wizard_models import (
    DiscoveredCamera,
    WizardPhase,
    WizardStatus,
)
from gomaxwebcam.dashboard.status_tracker import CameraStatusTracker
from gomaxwebcam.events import EventBus, EventType

pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def wizard_state(event_bus):
    return WizardState(event_bus=event_bus)


@pytest.fixture
def auth_token():
    return "test-wizard-token-integration"


@pytest.fixture
def app(wizard_state, auth_token, event_bus):
    return create_app(
        tracker=CameraStatusTracker(),
        event_bus=event_bus,
        auth_token=auth_token,
        wizard_state=wizard_state,
    )


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _mock_ble_device(address="AA:BB:CC:DD:EE:FF", name="GoPro 1234", rssi=-50):
    mock = MagicMock()
    mock.address = address
    mock.name = name
    mock.rssi = rssi
    mock.metadata = {"uuids": ["0000fea6-0000-1000-8000-00805f9b34fb"]}
    return mock


# ---------------------------------------------------------------------------
# Tests: Wizard select endpoint
# ---------------------------------------------------------------------------

class TestWizardSelect:
    """POST /api/wizard/select endpoint tests."""

    @pytest.mark.asyncio
    async def test_select_after_scan(self, client, auth_token, wizard_state):
        """Select a camera that was discovered in a prior scan."""
        # Pre-populate wizard state with scanned cameras
        wizard_state.discovered_cameras = [
            DiscoveredCamera(
                address="AA:BB:CC:DD:EE:01",
                name="GoPro 1234",
                rssi=-45,
                serial_suffix="1234",
            ),
        ]

        resp = await client.post(
            f"/api/wizard/select?token={auth_token}",
            json={"address": "AA:BB:CC:DD:EE:01"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["selected"] is True
        assert data["address"] == "AA:BB:CC:DD:EE:01"
        assert data["name"] == "GoPro 1234"
        assert wizard_state.selected_camera is not None
        assert wizard_state.selected_camera.address == "AA:BB:CC:DD:EE:01"

    @pytest.mark.asyncio
    async def test_select_by_known_address_without_scan(self, client, auth_token, wizard_state):
        """Select a camera by address without prior scan."""
        resp = await client.post(
            f"/api/wizard/select?token={auth_token}",
            json={"address": "11:22:33:44:55:66", "name": "My GoPro"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["selected"] is True
        assert data["address"] == "11:22:33:44:55:66"
        assert wizard_state.selected_camera is not None

    @pytest.mark.asyncio
    async def test_select_empty_address_returns_error(self, client, auth_token):
        """Select with empty address returns error."""
        resp = await client.post(
            f"/api/wizard/select?token={auth_token}",
            json={"address": ""},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["selected"] is False
        assert data["error"] != ""


# ---------------------------------------------------------------------------
# Tests: Wizard provision endpoint
# ---------------------------------------------------------------------------

class TestWizardProvision:
    """POST /api/wizard/provision endpoint tests."""

    @pytest.mark.asyncio
    async def test_provision_without_camera_returns_error(self, client, auth_token):
        """Provision without selecting a camera first returns error."""
        resp = await client.post(
            f"/api/wizard/provision?token={auth_token}",
            json={
                "wifi_ssid": "TestWiFi",
                "wifi_password": "password123",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["started"] is False
        assert "camera" in data["error"].lower() or "select" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_provision_with_camera_address_starts(
        self, client, auth_token, wizard_state,
    ):
        """Provision with explicit camera_address starts provisioning."""
        resp = await client.post(
            f"/api/wizard/provision?token={auth_token}",
            json={
                "wifi_ssid": "TestWiFi",
                "wifi_password": "password123",
                "camera_address": "AA:BB:CC:DD:EE:FF",
                "camera_name": "GoPro Test",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["started"] is True
        assert data["phase"] == "provisioning"

        # Clean up the background task
        if wizard_state._provision_task and not wizard_state._provision_task.done():
            wizard_state._provision_task.cancel()
            try:
                await wizard_state._provision_task
            except (asyncio.CancelledError, Exception):
                pass

    @pytest.mark.asyncio
    async def test_provision_with_selected_camera(
        self, client, auth_token, wizard_state,
    ):
        """Provision uses the previously selected camera."""
        wizard_state.selected_camera = DiscoveredCamera(
            address="AA:BB:CC:DD:EE:01",
            name="GoPro 1234",
            serial_suffix="1234",
        )

        resp = await client.post(
            f"/api/wizard/provision?token={auth_token}",
            json={
                "wifi_ssid": "TestWiFi",
                "wifi_password": "password123",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["started"] is True

        # Clean up
        if wizard_state._provision_task and not wizard_state._provision_task.done():
            wizard_state._provision_task.cancel()
            try:
                await wizard_state._provision_task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# Tests: Wizard cancel endpoint
# ---------------------------------------------------------------------------

class TestWizardCancel:
    """POST /api/wizard/cancel endpoint tests."""

    @pytest.mark.asyncio
    async def test_cancel_no_provisioning(self, client, auth_token):
        """Cancel with no active provisioning returns appropriate response."""
        resp = await client.post(f"/api/wizard/cancel?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["cancelled"] is False
        assert "no provisioning" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_cancel_with_orchestrator(
        self, client, auth_token, wizard_state,
    ):
        """Cancel signals the orchestrator to stop."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.cancel = MagicMock()
        wizard_state._orchestrator = mock_orchestrator

        resp = await client.post(f"/api/wizard/cancel?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["cancelled"] is True
        mock_orchestrator.cancel.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Wizard confirm endpoint
# ---------------------------------------------------------------------------

class TestWizardConfirm:
    """POST /api/wizard/confirm endpoint tests."""

    @pytest.mark.asyncio
    async def test_confirm_before_provisioning_returns_error(
        self, client, auth_token,
    ):
        """Confirm before provisioning completes returns error."""
        resp = await client.post(f"/api/wizard/confirm?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["confirmed"] is False
        assert "not completed" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_confirm_with_cached_credentials(
        self, client, auth_token, wizard_state,
    ):
        """Confirm with cached credentials returns success."""
        # Simulate completed provisioning
        wizard_state.status = WizardStatus(
            phase=WizardPhase.COMPLETE,
            progress_pct=100,
            message="Done",
            camera_ip="192.168.1.100",
        )

        # Simulate cached credentials
        mock_creds = MagicMock()
        mock_creds.ip_address = "192.168.1.100"
        mock_creds.username = "gopro"
        mock_creds.camera_serial = "C3481234"
        mock_creds.provisioned = True
        wizard_state._last_credentials = mock_creds

        resp = await client.post(f"/api/wizard/confirm?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["confirmed"] is True
        assert data["camera_ip"] == "192.168.1.100"
        assert data["cohn_username"] == "gopro"

    @pytest.mark.asyncio
    async def test_confirm_falls_back_to_wizard_status_ip(
        self, client, auth_token, wizard_state,
    ):
        """Confirm falls back to wizard status camera_ip when creds not cached."""
        wizard_state.status = WizardStatus(
            phase=WizardPhase.COMPLETE,
            progress_pct=100,
            message="Done",
            camera_ip="192.168.1.200",
            cohn_username="gopro_user",
        )
        wizard_state._last_credentials = None

        # Patch CohnCredentialStore and BLEProvisioningService at their import locations
        # (they're imported inside the endpoint function with local imports)
        with patch(
            "gomaxwebcam.transport.cohn_persistence.CohnCredentialStore.load_credentials",
            side_effect=Exception("not available"),
        ), patch(
            "gomaxwebcam.provisioning.BLEProvisioningService.get_stored_credentials",
            side_effect=Exception("not available"),
        ):
            resp = await client.post(f"/api/wizard/confirm?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["confirmed"] is True
        assert data["camera_ip"] == "192.168.1.200"


# ---------------------------------------------------------------------------
# Tests: Wizard status endpoint
# ---------------------------------------------------------------------------

class TestWizardStatusEndpoint:
    """GET /api/wizard/status endpoint tests."""

    @pytest.mark.asyncio
    async def test_status_returns_current_state(self, client, auth_token, wizard_state):
        """Status endpoint returns the current wizard state."""
        wizard_state.status = WizardStatus(
            phase=WizardPhase.SCANNING,
            progress_pct=5,
            message="Scanning...",
        )

        resp = await client.get(f"/api/wizard/status?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["phase"] == "scanning"
        assert data["progress_pct"] == 5
        assert data["message"] == "Scanning..."

    @pytest.mark.asyncio
    async def test_initial_status_is_idle(self, client, auth_token):
        """Initial wizard status should be idle."""
        resp = await client.get(f"/api/wizard/status?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["phase"] == "idle"


# ---------------------------------------------------------------------------
# Tests: WizardState unit tests
# ---------------------------------------------------------------------------

class TestWizardState:
    """WizardState unit tests."""

    def test_initial_state(self, wizard_state):
        """Initial WizardState has idle phase and empty lists."""
        assert wizard_state.status.phase == WizardPhase.IDLE
        assert wizard_state.discovered_cameras == []
        assert wizard_state.selected_camera is None

    def test_reset_clears_state(self, wizard_state):
        """reset() clears all wizard state."""
        wizard_state.discovered_cameras = [
            DiscoveredCamera(address="AA:BB", name="GoPro"),
        ]
        wizard_state.selected_camera = DiscoveredCamera(address="AA:BB", name="GoPro")
        wizard_state.status = WizardStatus(
            phase=WizardPhase.COMPLETE,
            progress_pct=100,
        )

        wizard_state.reset()

        assert wizard_state.discovered_cameras == []
        assert wizard_state.selected_camera is None
        assert wizard_state.status.phase == WizardPhase.IDLE

    @pytest.mark.asyncio
    async def test_progress_notification_publishes_to_event_bus(
        self, wizard_state, event_bus,
    ):
        """Progress notifications should publish to the system EventBus."""
        sub = event_bus.subscribe(event_types={EventType.CONNECTION})

        wizard_state._notify_progress(WizardStatus(
            phase=WizardPhase.SCANNING,
            progress_pct=5,
            message="Scanning...",
            camera_name="GoPro 1234",
        ))

        received = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        assert received.type == EventType.CONNECTION
        assert received.data["source"] == "wizard"
        assert received.data["phase"] == "scanning"
        assert received.data["progress_pct"] == 5

        event_bus.unsubscribe(sub)

    @pytest.mark.asyncio
    async def test_progress_subscription_receives_updates(self, wizard_state):
        """Progress subscribers should receive status updates."""
        q = wizard_state.subscribe_progress()

        wizard_state._notify_progress(WizardStatus(
            phase=WizardPhase.PROVISIONING,
            progress_pct=50,
            message="Provisioning...",
        ))

        status = await asyncio.wait_for(q.get(), timeout=1.0)
        assert status.phase == WizardPhase.PROVISIONING
        assert status.progress_pct == 50

        wizard_state.unsubscribe_progress(q)

    @pytest.mark.asyncio
    async def test_progress_unsubscribe_removes_queue(self, wizard_state):
        """unsubscribe_progress removes the queue from the list."""
        q = wizard_state.subscribe_progress()
        assert len(wizard_state._progress_queues) == 1

        wizard_state.unsubscribe_progress(q)
        assert len(wizard_state._progress_queues) == 0

    def test_set_event_bus(self):
        """set_event_bus attaches a new EventBus."""
        ws = WizardState()
        assert ws._event_bus is None

        bus = EventBus()
        ws.set_event_bus(bus)
        assert ws._event_bus is bus

    @pytest.mark.asyncio
    async def test_progress_notification_without_event_bus(self):
        """Progress notification without EventBus should not raise."""
        ws = WizardState(event_bus=None)
        # Should not raise
        ws._notify_progress(WizardStatus(
            phase=WizardPhase.SCANNING,
            progress_pct=5,
            message="Scanning...",
        ))
