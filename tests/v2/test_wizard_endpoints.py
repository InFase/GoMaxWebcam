"""
tests/v2/test_wizard_endpoints.py — Tests for BLE setup wizard endpoints.

Tests the FastAPI wizard endpoints for the 4-step COHN provisioning wizard:
  Step 1: POST /api/wizard/scan — BLE scan via bleak (not hand-rolled ble/)
  Step 2: POST /api/wizard/select — Camera selection/pairing
  Step 3: POST /api/wizard/provision — COHN provisioning start
  Step 4: POST /api/wizard/confirm — Credential persistence verification
  Supporting: cancel, status, progress SSE, auth, validation

All tests mock BLE/COHN at the import boundary — no real hardware needed.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from gomaxwebcam.dashboard.app import create_app
from gomaxwebcam.dashboard.wizard import WizardState
from gomaxwebcam.dashboard.wizard_models import (
    BLEScanRequest,
    BLEScanResponse,
    CameraSelectResponse,
    ConfirmResponse,
    DiscoveredCamera,
    ProvisionRequest,
    ProvisionResponse,
    WizardCancelResponse,
    WizardPhase,
    WizardStatus,
)
from gomaxwebcam.dashboard.status_tracker import CameraStatusTracker
from gomaxwebcam.events import EventBus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wizard_state():
    """Create a fresh WizardState with EventBus."""
    bus = EventBus()
    return WizardState(event_bus=bus)


@pytest.fixture
def auth_token():
    return "test-wizard-token-123"


@pytest.fixture
def app(wizard_state, auth_token):
    """Create a FastAPI app with wizard state and known auth token."""
    return create_app(
        tracker=CameraStatusTracker(),
        event_bus=EventBus(),
        auth_token=auth_token,
        wizard_state=wizard_state,
    )


@pytest_asyncio.fixture
async def client(app):
    """Create an async HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Helper: make a mock bleak BLEDevice (for scan endpoint)
# ---------------------------------------------------------------------------


def _mock_ble_device(address="AA:BB:CC:DD:EE:FF", name="GoPro 1234", rssi=-50):
    """Create a mock bleak BLEDevice for scan testing.

    The scan endpoint now uses bleak.BleakScanner directly, not the
    dead ble/scanner.py module. Mock bleak device objects.
    """
    mock = MagicMock()
    mock.address = address
    mock.name = name
    mock.rssi = rssi
    mock.metadata = {"uuids": ["0000fea6-0000-1000-8000-00805f9b34fb"]}
    return mock


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


class TestWizardAuth:
    """All wizard endpoints should require auth."""

    @pytest.mark.asyncio
    async def test_scan_without_auth_returns_401(self, client):
        resp = await client.post("/api/wizard/scan", json={"timeout_s": 5})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_select_without_auth_returns_401(self, client):
        resp = await client.post(
            "/api/wizard/select",
            json={"address": "AA:BB:CC:DD:EE:FF"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_provision_without_auth_returns_401(self, client):
        resp = await client.post(
            "/api/wizard/provision",
            json={"wifi_ssid": "test", "wifi_password": "pass"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_cancel_without_auth_returns_401(self, client):
        resp = await client.post("/api/wizard/cancel")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_status_without_auth_returns_401(self, client):
        resp = await client.get("/api/wizard/status")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_progress_without_auth_returns_401(self, client):
        resp = await client.get("/api/wizard/progress")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# BLE Scan tests
# ---------------------------------------------------------------------------


class TestWizardScan:
    """POST /api/wizard/scan — Step 1: BLE scan via bleak."""

    @pytest.mark.asyncio
    async def test_scan_returns_discovered_cameras(self, client, auth_token):
        """Scan should return discovered cameras from bleak BLE scanner."""
        mock_devices = [
            _mock_ble_device("AA:BB:CC:DD:EE:01", "GoPro 1234", -45),
            _mock_ble_device("AA:BB:CC:DD:EE:02", "GoPro 5678", -60),
        ]

        mock_scanner = MagicMock()
        mock_scanner.discover = AsyncMock(return_value=mock_devices)

        with patch(
            "bleak.BleakScanner",
            return_value=mock_scanner,
        ):
            resp = await client.post(
                f"/api/wizard/scan?token={auth_token}",
                json={"timeout_s": 5.0},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["cameras"]) == 2
        # Sorted by RSSI (strongest first)
        assert data["cameras"][0]["rssi"] == -45
        assert data["cameras"][0]["name"] == "GoPro 1234"
        assert data["cameras"][1]["rssi"] == -60
        assert data["scan_duration_s"] >= 0  # Mocked scan runs instantly
        assert data["error"] == ""

    @pytest.mark.asyncio
    async def test_scan_empty_result(self, client, auth_token):
        """Scan with no cameras found should return empty list."""
        mock_scanner = MagicMock()
        mock_scanner.discover = AsyncMock(return_value=[])

        with patch(
            "bleak.BleakScanner",
            return_value=mock_scanner,
        ):
            resp = await client.post(
                f"/api/wizard/scan?token={auth_token}",
                json={"timeout_s": 3.0},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["cameras"] == []
        assert data["error"] == ""

    @pytest.mark.asyncio
    async def test_scan_filters_non_gopro_devices(self, client, auth_token):
        """Scan should filter out non-GoPro BLE devices."""
        mock_devices = [
            _mock_ble_device("AA:BB:CC:DD:EE:01", "GoPro 1234", -45),
            _mock_ble_device("AA:BB:CC:DD:EE:02", "Random Speaker", -30),
            _mock_ble_device("AA:BB:CC:DD:EE:03", "GP-5678", -55),
        ]
        # Non-GoPro device has no matching service UUID
        mock_devices[1].metadata = {"uuids": []}
        mock_devices[1].name = "Random Speaker"

        mock_scanner = MagicMock()
        mock_scanner.discover = AsyncMock(return_value=mock_devices)

        with patch(
            "bleak.BleakScanner",
            return_value=mock_scanner,
        ):
            resp = await client.post(
                f"/api/wizard/scan?token={auth_token}",
                json={"timeout_s": 5.0},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["cameras"]) == 2
        names = [c["name"] for c in data["cameras"]]
        assert "GoPro 1234" in names
        assert "GP-5678" in names
        assert "Random Speaker" not in names

    @pytest.mark.asyncio
    async def test_scan_import_error(self, client, auth_token):
        """Scan should handle missing bleak gracefully."""
        import builtins
        _orig_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "bleak":
                raise ImportError("No module named 'bleak'")
            return _orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            resp = await client.post(
                f"/api/wizard/scan?token={auth_token}",
                json={"timeout_s": 3.0},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["cameras"] == []
        assert "bleak" in data["error"].lower() or "BLE" in data["error"]

    @pytest.mark.asyncio
    async def test_scan_exception_returns_error(self, client, auth_token):
        """Scan should handle scanner exceptions gracefully."""
        mock_scanner = MagicMock()
        mock_scanner.discover = AsyncMock(
            side_effect=RuntimeError("BLE adapter not found"),
        )

        with patch(
            "bleak.BleakScanner",
            return_value=mock_scanner,
        ):
            resp = await client.post(
                f"/api/wizard/scan?token={auth_token}",
                json={"timeout_s": 3.0},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["cameras"] == []
        assert "BLE adapter not found" in data["error"]

    @pytest.mark.asyncio
    async def test_scan_updates_wizard_state(self, client, auth_token, wizard_state):
        """Scan should update wizard state with discovered cameras."""
        mock_devices = [
            _mock_ble_device("AA:BB:CC:DD:EE:01", "GoPro 1234", -45),
        ]

        mock_scanner = MagicMock()
        mock_scanner.discover = AsyncMock(return_value=mock_devices)

        with patch(
            "bleak.BleakScanner",
            return_value=mock_scanner,
        ):
            resp = await client.post(
                f"/api/wizard/scan?token={auth_token}",
                json={"timeout_s": 5.0},
            )

        assert resp.status_code == 200
        assert len(wizard_state.discovered_cameras) == 1
        assert wizard_state.discovered_cameras[0].address == "AA:BB:CC:DD:EE:01"

    @pytest.mark.asyncio
    async def test_scan_extracts_serial_suffix(self, client, auth_token):
        """Scan should extract serial suffix from GoPro name."""
        mock_devices = [
            _mock_ble_device("AA:BB:CC:DD:EE:01", "GoPro 1234", -45),
            _mock_ble_device("AA:BB:CC:DD:EE:02", "GP-5678", -50),
        ]

        mock_scanner = MagicMock()
        mock_scanner.discover = AsyncMock(return_value=mock_devices)

        with patch(
            "bleak.BleakScanner",
            return_value=mock_scanner,
        ):
            resp = await client.post(
                f"/api/wizard/scan?token={auth_token}",
                json={"timeout_s": 5.0},
            )

        data = resp.json()
        assert data["cameras"][0]["serial_suffix"] == "1234"
        assert data["cameras"][1]["serial_suffix"] == "5678"

    @pytest.mark.asyncio
    async def test_scan_validation_timeout_bounds(self, client, auth_token):
        """Scan timeout should be validated (1-30 seconds)."""
        # Too low
        resp = await client.post(
            f"/api/wizard/scan?token={auth_token}",
            json={"timeout_s": 0.5},
        )
        assert resp.status_code == 422  # Validation error

        # Too high
        resp = await client.post(
            f"/api/wizard/scan?token={auth_token}",
            json={"timeout_s": 60.0},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_scan_default_timeout(self, client, auth_token):
        """Scan with default timeout should work."""
        mock_scanner = MagicMock()
        mock_scanner.discover = AsyncMock(return_value=[])

        with patch(
            "bleak.BleakScanner",
            return_value=mock_scanner,
        ):
            resp = await client.post(
                f"/api/wizard/scan?token={auth_token}",
                json={},
            )

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_scan_publishes_to_event_bus(self, client, auth_token, wizard_state):
        """Scan should publish progress to the EventBus."""
        mock_scanner = MagicMock()
        mock_scanner.discover = AsyncMock(return_value=[])

        event_bus: EventBus = wizard_state._event_bus
        initial_count = event_bus.event_count if event_bus else 0

        with patch(
            "bleak.BleakScanner",
            return_value=mock_scanner,
        ):
            await client.post(
                f"/api/wizard/scan?token={auth_token}",
                json={"timeout_s": 3.0},
            )

        # EventBus should have received at least scanning + scan_complete events
        if event_bus:
            assert event_bus.event_count > initial_count


# ---------------------------------------------------------------------------
# Camera Select tests
# ---------------------------------------------------------------------------


class TestWizardSelect:
    """POST /api/wizard/select endpoint tests."""

    @pytest.mark.asyncio
    async def test_select_from_discovered(self, client, auth_token, wizard_state):
        """Select should work for a camera that was discovered in scan."""
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
        assert data["error"] == ""

    @pytest.mark.asyncio
    async def test_select_by_address_without_scan(self, client, auth_token, wizard_state):
        """Select should allow direct address entry without prior scan."""
        resp = await client.post(
            f"/api/wizard/select?token={auth_token}",
            json={
                "address": "11:22:33:44:55:66",
                "name": "My GoPro",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["selected"] is True
        assert data["address"] == "11:22:33:44:55:66"
        assert data["name"] == "My GoPro"

    @pytest.mark.asyncio
    async def test_select_empty_address_fails(self, client, auth_token):
        """Select with empty address should fail."""
        resp = await client.post(
            f"/api/wizard/select?token={auth_token}",
            json={"address": ""},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["selected"] is False
        assert data["error"] != ""

    @pytest.mark.asyncio
    async def test_select_updates_wizard_state(self, client, auth_token, wizard_state):
        """Select should update wizard_state.selected_camera."""
        wizard_state.discovered_cameras = [
            DiscoveredCamera(
                address="AA:BB:CC:DD:EE:01",
                name="GoPro 1234",
                rssi=-45,
            ),
        ]

        await client.post(
            f"/api/wizard/select?token={auth_token}",
            json={"address": "AA:BB:CC:DD:EE:01"},
        )

        assert wizard_state.selected_camera is not None
        assert wizard_state.selected_camera.address == "AA:BB:CC:DD:EE:01"


# ---------------------------------------------------------------------------
# Provision tests
# ---------------------------------------------------------------------------


class TestWizardProvision:
    """POST /api/wizard/provision — Step 3: COHN provisioning."""

    @pytest.mark.asyncio
    async def test_provision_starts_background_task(self, client, auth_token, wizard_state):
        """Provision should start and return immediately."""
        wizard_state.selected_camera = DiscoveredCamera(
            address="AA:BB:CC:DD:EE:01",
            name="GoPro 1234",
        )

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator"
        ) as MockOrch:
            instance = MockOrch.return_value
            # Make provision_and_connect hang (simulating long operation)
            provision_future = asyncio.Future()
            instance.provision_and_connect = AsyncMock(return_value=provision_future)

            resp = await client.post(
                f"/api/wizard/provision?token={auth_token}",
                json={
                    "wifi_ssid": "HomeWiFi",
                    "wifi_password": "secret123",
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
    async def test_provision_without_camera_fails(self, client, auth_token, wizard_state):
        """Provision without a selected camera should fail."""
        resp = await client.post(
            f"/api/wizard/provision?token={auth_token}",
            json={
                "wifi_ssid": "HomeWiFi",
                "wifi_password": "secret123",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["started"] is False
        assert "No camera selected" in data["error"]

    @pytest.mark.asyncio
    async def test_provision_with_explicit_address(self, client, auth_token, wizard_state):
        """Provision with camera_address should work without prior select."""
        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator"
        ) as MockOrch:
            instance = MockOrch.return_value
            instance.provision_and_connect = AsyncMock(return_value=None)

            resp = await client.post(
                f"/api/wizard/provision?token={auth_token}",
                json={
                    "wifi_ssid": "HomeWiFi",
                    "wifi_password": "secret123",
                    "camera_address": "AA:BB:CC:DD:EE:01",
                    "camera_name": "GoPro 1234",
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

    @pytest.mark.asyncio
    async def test_provision_validation_ssid_required(self, client, auth_token):
        """Provision requires wifi_ssid."""
        resp = await client.post(
            f"/api/wizard/provision?token={auth_token}",
            json={"wifi_password": "secret123"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_provision_validation_password_required(self, client, auth_token):
        """Provision requires wifi_password."""
        resp = await client.post(
            f"/api/wizard/provision?token={auth_token}",
            json={"wifi_ssid": "HomeWiFi"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_provision_double_start_rejected(self, client, auth_token, wizard_state):
        """Starting provision while one is in progress should be rejected."""
        wizard_state.selected_camera = DiscoveredCamera(
            address="AA:BB:CC:DD:EE:01",
            name="GoPro 1234",
        )

        # Create a fake running task
        async def _hang():
            await asyncio.sleep(100)

        wizard_state._provision_task = asyncio.create_task(_hang())

        try:
            resp = await client.post(
                f"/api/wizard/provision?token={auth_token}",
                json={
                    "wifi_ssid": "HomeWiFi",
                    "wifi_password": "secret123",
                },
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["started"] is False
            assert "already in progress" in data["error"]
        finally:
            wizard_state._provision_task.cancel()
            try:
                await wizard_state._provision_task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# Cancel tests
# ---------------------------------------------------------------------------


class TestWizardCancel:
    """POST /api/wizard/cancel endpoint tests."""

    @pytest.mark.asyncio
    async def test_cancel_no_provisioning(self, client, auth_token):
        """Cancel with nothing running should indicate nothing to cancel."""
        resp = await client.post(f"/api/wizard/cancel?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["cancelled"] is False
        assert "No provisioning" in data["message"]

    @pytest.mark.asyncio
    async def test_cancel_with_orchestrator(self, client, auth_token, wizard_state):
        """Cancel should call orchestrator.cancel()."""
        mock_orch = MagicMock()
        mock_orch.cancel = MagicMock()
        wizard_state._orchestrator = mock_orch

        resp = await client.post(f"/api/wizard/cancel?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["cancelled"] is True
        mock_orch.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_running_task(self, client, auth_token, wizard_state):
        """Cancel should cancel the running provision task."""
        async def _hang():
            await asyncio.sleep(100)

        wizard_state._provision_task = asyncio.create_task(_hang())
        wizard_state._orchestrator = None  # No orchestrator, just task

        try:
            resp = await client.post(f"/api/wizard/cancel?token={auth_token}")

            assert resp.status_code == 200
            data = resp.json()
            assert data["cancelled"] is True
        finally:
            if not wizard_state._provision_task.done():
                wizard_state._provision_task.cancel()
                try:
                    await wizard_state._provision_task
                except (asyncio.CancelledError, Exception):
                    pass


# ---------------------------------------------------------------------------
# Status tests
# ---------------------------------------------------------------------------


class TestWizardStatus:
    """GET /api/wizard/status endpoint tests."""

    @pytest.mark.asyncio
    async def test_initial_status_is_idle(self, client, auth_token):
        """Initial wizard status should be IDLE."""
        resp = await client.get(f"/api/wizard/status?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["phase"] == "idle"
        assert data["progress_pct"] == 0
        assert data["cohn_provisioned"] is False

    @pytest.mark.asyncio
    async def test_status_reflects_wizard_state(self, client, auth_token, wizard_state):
        """Status endpoint should reflect current wizard state."""
        wizard_state.status = WizardStatus(
            phase=WizardPhase.PROVISIONING,
            progress_pct=45,
            message="Connecting to WiFi...",
            camera_name="GoPro 1234",
            wifi_ssid="HomeWiFi",
        )

        resp = await client.get(f"/api/wizard/status?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["phase"] == "provisioning"
        assert data["progress_pct"] == 45
        assert data["message"] == "Connecting to WiFi..."
        assert data["camera_name"] == "GoPro 1234"
        assert data["wifi_ssid"] == "HomeWiFi"

    @pytest.mark.asyncio
    async def test_status_shows_completion(self, client, auth_token, wizard_state):
        """Status should show completed provisioning."""
        wizard_state.status = WizardStatus(
            phase=WizardPhase.COMPLETE,
            progress_pct=100,
            message="COHN provisioned!",
            camera_ip="192.168.1.100",
            cohn_provisioned=True,
            cohn_username="gopro",
        )

        resp = await client.get(f"/api/wizard/status?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["phase"] == "complete"
        assert data["cohn_provisioned"] is True
        assert data["camera_ip"] == "192.168.1.100"
        assert data["cohn_username"] == "gopro"


# ---------------------------------------------------------------------------
# Progress SSE tests
# ---------------------------------------------------------------------------


class TestWizardProgress:
    """GET /api/wizard/progress SSE endpoint tests."""

    @pytest.mark.asyncio
    async def test_progress_route_exists(self, app):
        """Progress SSE route should be registered."""
        routes = [
            r for r in app.routes
            if hasattr(r, "path") and r.path == "/api/wizard/progress"
        ]
        assert len(routes) == 1

    @pytest.mark.asyncio
    async def test_progress_subscription(self, wizard_state):
        """WizardState progress subscription should deliver updates."""
        q = wizard_state.subscribe_progress()

        wizard_state._notify_progress(WizardStatus(
            phase=WizardPhase.SCANNING,
            progress_pct=10,
            message="Scanning...",
        ))

        status = await asyncio.wait_for(q.get(), timeout=1.0)
        assert status.phase == WizardPhase.SCANNING
        assert status.progress_pct == 10

        wizard_state.unsubscribe_progress(q)

    @pytest.mark.asyncio
    async def test_progress_multiple_subscribers(self, wizard_state):
        """Multiple subscribers should all receive updates."""
        q1 = wizard_state.subscribe_progress()
        q2 = wizard_state.subscribe_progress()

        wizard_state._notify_progress(WizardStatus(
            phase=WizardPhase.PROVISIONING,
            progress_pct=50,
        ))

        s1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        s2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert s1.progress_pct == 50
        assert s2.progress_pct == 50

        wizard_state.unsubscribe_progress(q1)
        wizard_state.unsubscribe_progress(q2)


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestWizardModels:
    """Pydantic request/response model validation tests."""

    def test_ble_scan_request_defaults(self):
        req = BLEScanRequest()
        assert req.timeout_s == 10.0

    def test_ble_scan_request_custom_timeout(self):
        req = BLEScanRequest(timeout_s=15.0)
        assert req.timeout_s == 15.0

    def test_ble_scan_request_validation_min(self):
        with pytest.raises(Exception):
            BLEScanRequest(timeout_s=0.1)

    def test_ble_scan_request_validation_max(self):
        with pytest.raises(Exception):
            BLEScanRequest(timeout_s=100.0)

    def test_discovered_camera_model(self):
        cam = DiscoveredCamera(
            address="AA:BB:CC:DD:EE:FF",
            name="GoPro 1234",
            rssi=-45,
            serial_suffix="1234",
        )
        assert cam.address == "AA:BB:CC:DD:EE:FF"
        assert cam.rssi == -45

    def test_discovered_camera_serialization(self):
        cam = DiscoveredCamera(
            address="AA:BB:CC:DD:EE:FF",
            name="GoPro 1234",
        )
        data = cam.model_dump()
        assert data["address"] == "AA:BB:CC:DD:EE:FF"
        assert data["name"] == "GoPro 1234"
        assert data["rssi"] == -100  # default

    def test_provision_request_requires_ssid(self):
        with pytest.raises(Exception):
            ProvisionRequest(wifi_password="pass")

    def test_provision_request_requires_password(self):
        with pytest.raises(Exception):
            ProvisionRequest(wifi_ssid="net")

    def test_provision_request_valid(self):
        req = ProvisionRequest(
            wifi_ssid="HomeWiFi",
            wifi_password="secret123",
            camera_address="AA:BB:CC:DD:EE:FF",
        )
        assert req.wifi_ssid == "HomeWiFi"
        assert req.camera_address == "AA:BB:CC:DD:EE:FF"

    def test_wizard_status_defaults(self):
        status = WizardStatus()
        assert status.phase == WizardPhase.IDLE
        assert status.progress_pct == 0
        assert status.cohn_provisioned is False

    def test_wizard_status_serialization(self):
        status = WizardStatus(
            phase=WizardPhase.COMPLETE,
            progress_pct=100,
            camera_ip="192.168.1.100",
            cohn_provisioned=True,
        )
        data = status.model_dump()
        assert data["phase"] == "complete"
        assert data["progress_pct"] == 100
        assert data["cohn_provisioned"] is True

    def test_wizard_phase_enum_values(self):
        assert WizardPhase.IDLE == "idle"
        assert WizardPhase.SCANNING == "scanning"
        assert WizardPhase.PROVISIONING == "provisioning"
        assert WizardPhase.COMPLETE == "complete"
        assert WizardPhase.FAILED == "failed"
        assert WizardPhase.CANCELLED == "cancelled"

    def test_wizard_cancel_response(self):
        resp = WizardCancelResponse(cancelled=True, message="Done")
        assert resp.cancelled is True
        assert resp.message == "Done"

    def test_ble_scan_response_serialization(self):
        resp = BLEScanResponse(
            cameras=[
                DiscoveredCamera(
                    address="AA:BB:CC:DD:EE:FF",
                    name="GoPro 1234",
                    rssi=-45,
                ),
            ],
            scan_duration_s=5.2,
        )
        data = resp.model_dump()
        assert len(data["cameras"]) == 1
        assert data["scan_duration_s"] == 5.2


# ---------------------------------------------------------------------------
# WizardState unit tests
# ---------------------------------------------------------------------------


class TestWizardState:
    """Unit tests for WizardState management."""

    def test_initial_state(self, wizard_state):
        assert wizard_state.status.phase == WizardPhase.IDLE
        assert wizard_state.discovered_cameras == []
        assert wizard_state.selected_camera is None

    def test_reset(self, wizard_state):
        wizard_state.status = WizardStatus(phase=WizardPhase.COMPLETE)
        wizard_state.discovered_cameras = [
            DiscoveredCamera(address="A", name="B"),
        ]
        wizard_state.reset()

        assert wizard_state.status.phase == WizardPhase.IDLE
        assert wizard_state.discovered_cameras == []
        assert wizard_state.selected_camera is None

    @pytest.mark.asyncio
    async def test_notify_progress_updates_status(self, wizard_state):
        new_status = WizardStatus(
            phase=WizardPhase.SCANNING,
            progress_pct=10,
        )
        wizard_state._notify_progress(new_status)
        assert wizard_state.status.phase == WizardPhase.SCANNING

    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe(self, wizard_state):
        q = wizard_state.subscribe_progress()
        assert len(wizard_state._progress_queues) == 1

        wizard_state.unsubscribe_progress(q)
        assert len(wizard_state._progress_queues) == 0

    @pytest.mark.asyncio
    async def test_backpressure_handling(self, wizard_state):
        """Queue should handle backpressure by dropping oldest without crashing."""
        q = wizard_state.subscribe_progress()

        # Fill the queue well beyond capacity (max is 50)
        for i in range(55):
            wizard_state._notify_progress(WizardStatus(progress_pct=i))

        # Should have at most 50 items (oldest were dropped)
        assert q.qsize() <= 50
        # The latest items should be present (not the earliest ones)
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        # Last item should have progress_pct=54 (the final push)
        assert items[-1].progress_pct == 54
        wizard_state.unsubscribe_progress(q)

    def test_event_bus_wired(self, wizard_state):
        """WizardState should have EventBus reference when created by app."""
        # The app fixture creates WizardState with default (no bus),
        # but create_app wires it up
        bus = EventBus()
        wizard_state.set_event_bus(bus)
        assert wizard_state._event_bus is bus


# ---------------------------------------------------------------------------
# Step 4: Confirm endpoint tests
# ---------------------------------------------------------------------------


class TestWizardConfirm:
    """POST /api/wizard/confirm — Step 4: Credential verification."""

    @pytest.mark.asyncio
    async def test_confirm_before_provisioning_fails(self, client, auth_token):
        """Confirm before provisioning completes should fail."""
        resp = await client.post(f"/api/wizard/confirm?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["confirmed"] is False
        assert "not completed" in data["error"].lower() or "idle" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_confirm_with_cached_credentials(self, client, auth_token, wizard_state):
        """Confirm should return cached credentials when available."""
        # Simulate completed provisioning
        wizard_state.status = WizardStatus(
            phase=WizardPhase.COMPLETE,
            progress_pct=100,
            cohn_provisioned=True,
            camera_ip="192.168.1.100",
        )

        # Inject cached credentials
        mock_creds = MagicMock()
        mock_creds.provisioned = True
        mock_creds.ip_address = "192.168.1.100"
        mock_creds.username = "gopro"
        mock_creds.camera_serial = "1234"
        wizard_state._last_credentials = mock_creds

        resp = await client.post(f"/api/wizard/confirm?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["confirmed"] is True
        assert data["camera_ip"] == "192.168.1.100"
        assert data["cohn_username"] == "gopro"
        assert data["camera_serial"] == "1234"

    @pytest.mark.asyncio
    async def test_confirm_reads_cohn_db(self, client, auth_token, wizard_state):
        """Confirm should fall back to reading from open-gopro's cohn_db."""
        wizard_state.status = WizardStatus(
            phase=WizardPhase.COMPLETE,
            progress_pct=100,
            cohn_provisioned=True,
        )
        wizard_state.selected_camera = DiscoveredCamera(
            address="AA:BB:CC:DD:EE:01",
            name="GoPro 1234",
            serial_suffix="1234",
        )
        wizard_state._last_credentials = None  # No cached creds

        mock_stored = MagicMock()
        mock_stored.provisioned = True
        mock_stored.ip_address = "192.168.1.200"
        mock_stored.username = "gopro_user"
        mock_stored.camera_serial = "1234"

        with patch(
            "gomaxwebcam.provisioning.BLEProvisioningService"
        ) as MockSvc:
            MockSvc.return_value.get_stored_credentials.return_value = mock_stored

            resp = await client.post(f"/api/wizard/confirm?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["confirmed"] is True
        assert data["camera_ip"] == "192.168.1.200"
        assert data["cohn_username"] == "gopro_user"

    @pytest.mark.asyncio
    async def test_confirm_falls_back_to_status(self, client, auth_token, wizard_state):
        """Confirm should fall back to wizard status if cohn_db read fails."""
        wizard_state.status = WizardStatus(
            phase=WizardPhase.COMPLETE,
            progress_pct=100,
            cohn_provisioned=True,
            camera_ip="10.0.0.5",
            cohn_username="admin",
        )
        wizard_state._last_credentials = None

        with patch(
            "gomaxwebcam.provisioning.BLEProvisioningService",
            side_effect=Exception("TinyDB not available"),
        ):
            resp = await client.post(f"/api/wizard/confirm?token={auth_token}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["confirmed"] is True
        assert data["camera_ip"] == "10.0.0.5"

    @pytest.mark.asyncio
    async def test_confirm_auth_required(self, client):
        """Confirm endpoint should require auth token."""
        resp = await client.post("/api/wizard/confirm")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_confirm_route_exists(self, app):
        """Confirm route should be registered."""
        routes = [
            r for r in app.routes
            if hasattr(r, "path") and r.path == "/api/wizard/confirm"
        ]
        assert len(routes) == 1


# ---------------------------------------------------------------------------
# EventBus integration tests
# ---------------------------------------------------------------------------


class TestWizardEventBus:
    """Wizard progress should publish to the system EventBus."""

    @pytest.mark.asyncio
    async def test_notify_progress_publishes_to_event_bus(self):
        """_notify_progress should publish events to the EventBus."""
        bus = EventBus()
        wizard = WizardState(event_bus=bus)

        wizard._notify_progress(WizardStatus(
            phase=WizardPhase.SCANNING,
            progress_pct=10,
            message="Scanning...",
        ))

        assert bus.event_count >= 1
        last_event = bus.recent_events[-1]
        assert last_event.data["source"] == "wizard"
        assert last_event.data["phase"] == "scanning"
        assert last_event.data["progress_pct"] == 10

    @pytest.mark.asyncio
    async def test_event_bus_none_does_not_crash(self):
        """If EventBus is None, _notify_progress should not crash."""
        wizard = WizardState(event_bus=None)

        # Should not raise
        wizard._notify_progress(WizardStatus(
            phase=WizardPhase.PROVISIONING,
            progress_pct=50,
            message="Provisioning...",
        ))

        assert wizard.status.phase == WizardPhase.PROVISIONING

    @pytest.mark.asyncio
    async def test_set_event_bus(self):
        """set_event_bus should attach EventBus post-construction."""
        wizard = WizardState()
        assert wizard._event_bus is None

        bus = EventBus()
        wizard.set_event_bus(bus)
        assert wizard._event_bus is bus

        wizard._notify_progress(WizardStatus(
            phase=WizardPhase.COMPLETE,
            progress_pct=100,
        ))
        assert bus.event_count >= 1


# ---------------------------------------------------------------------------
# ConfirmResponse model tests
# ---------------------------------------------------------------------------


class TestConfirmResponseModel:
    """ConfirmResponse Pydantic model tests."""

    def test_defaults(self):
        resp = ConfirmResponse()
        assert resp.confirmed is False
        assert resp.camera_ip == ""
        assert resp.cohn_username == ""
        assert resp.camera_serial == ""

    def test_confirmed_response(self):
        resp = ConfirmResponse(
            confirmed=True,
            camera_ip="192.168.1.100",
            cohn_username="gopro",
            camera_serial="1234",
            message="Credentials confirmed",
        )
        data = resp.model_dump()
        assert data["confirmed"] is True
        assert data["camera_ip"] == "192.168.1.100"
        assert data["cohn_username"] == "gopro"

    def test_failed_response(self):
        resp = ConfirmResponse(
            confirmed=False,
            error="Not found",
            message="Try reprovisioning",
        )
        assert resp.confirmed is False
        assert resp.error == "Not found"
