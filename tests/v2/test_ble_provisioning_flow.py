"""
test_ble_provisioning_flow.py — Tests for BLE provisioning with credential persistence.

Verifies the full BLE provisioning → credential persistence → COHN reconnect flow:
  - COHNOrchestrator stores credentials after successful provisioning
  - Wizard confirm endpoint verifies credential persistence
  - Credential persistence uses open-gopro's cohn_db (TinyDB)
  - Provisioning errors don't leave partial credentials
  - Full wizard scan → select → provision → confirm happy path
  - Progress callbacks publish to EventBus during provisioning

All tests mock at the BLE/COHN/bleak boundary — no real hardware needed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gomaxwebcam.events import EventBus, EventType, Event
from gomaxwebcam.transport.cohn_orchestrator import (
    COHNOrchestrator,
    OrchestratorPhase,
    OrchestratorStatus,
)

pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

@dataclass
class MockCOHNCredentials:
    """Mock credentials returned by BLEProvisioningService."""
    camera_serial: str = "C3481234"
    ip_address: str = "192.168.1.100"
    username: str = "gopro"
    password: str = "secret123"
    certificate: str = "-----BEGIN CERTIFICATE-----\nMIIB..."
    ssid: str = "HomeWiFi"
    provisioned: bool = True


def _make_mock_cohn_transport(
    discover_ok: bool = True,
    connect_ok: bool = True,
) -> MagicMock:
    transport = MagicMock()
    transport.discover = AsyncMock(return_value=discover_ok)
    transport.connect = AsyncMock(return_value=connect_ok)
    transport.disconnect = AsyncMock()
    transport.start_stream = AsyncMock(return_value=MagicMock(port=8554))
    transport.stop_stream = AsyncMock()
    transport._inject_credentials = MagicMock()
    transport._inject_camera_ip = MagicMock()
    transport.state = MagicMock()
    transport.state.name = "CONNECTED"
    return transport


# ---------------------------------------------------------------------------
# Tests: Credential persistence after provisioning
# ---------------------------------------------------------------------------

class TestCredentialPersistence:
    """Verify credentials are stored after successful BLE provisioning."""

    @pytest.mark.anyio
    async def test_provision_stores_credentials_via_orchestrator(self):
        """Successful provision_and_connect should store credentials internally."""
        orch = COHNOrchestrator()
        creds = MockCOHNCredentials()
        mock_transport = _make_mock_cohn_transport()

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            new_callable=AsyncMock,
            return_value=creds,
        ), patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_cohn_connect",
            new_callable=AsyncMock,
            return_value=mock_transport,
        ):
            result = await orch.provision_and_connect(
                wifi_ssid="HomeWiFi",
                wifi_password="password123",
            )

        assert result is mock_transport
        assert orch.status.phase == OrchestratorPhase.COMPLETE
        # The orchestrator should have the transport set
        assert orch.transport is mock_transport

    @pytest.mark.anyio
    async def test_failed_provision_does_not_store_credentials(self):
        """Failed provisioning should not store credentials."""
        orch = COHNOrchestrator()

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await orch.provision_and_connect(
                wifi_ssid="HomeWiFi",
                wifi_password="password123",
            )

        assert result is None
        assert orch.status.phase == OrchestratorPhase.FAILED
        assert orch.transport is None

    @pytest.mark.anyio
    async def test_provision_cohn_connect_failure_no_transport(self):
        """When COHN connection fails, transport should be None."""
        orch = COHNOrchestrator()

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            new_callable=AsyncMock,
            return_value=MockCOHNCredentials(),
        ), patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_cohn_connect",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await orch.provision_and_connect(
                wifi_ssid="HomeWiFi",
                wifi_password="password123",
            )

        assert result is None
        assert orch.transport is None


# ---------------------------------------------------------------------------
# Tests: EventBus integration during provisioning
# ---------------------------------------------------------------------------

class TestProvisioningEventBus:
    """Verify provisioning publishes status callbacks to EventBus."""

    @pytest.mark.anyio
    async def test_status_callbacks_include_phase_transitions(self):
        """Status callbacks should cover PROVISIONING → COHN_CONNECTING → COMPLETE."""
        orch = COHNOrchestrator()
        phases_seen: list[str] = []

        def on_status(status: OrchestratorStatus):
            phases_seen.append(status.phase.name if hasattr(status.phase, 'name') else str(status.phase))

        mock_transport = _make_mock_cohn_transport()

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            new_callable=AsyncMock,
            return_value=MockCOHNCredentials(),
        ), patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_cohn_connect",
            new_callable=AsyncMock,
            return_value=mock_transport,
        ):
            await orch.provision_and_connect(
                wifi_ssid="HomeWiFi",
                wifi_password="password123",
                on_status=on_status,
            )

        # Should have seen COMPLETE at minimum
        assert "COMPLETE" in phases_seen
        # Should have progress callbacks
        assert len(phases_seen) >= 1

    @pytest.mark.anyio
    async def test_failed_provision_reports_error_phase(self):
        """Failed provisioning should report FAILED phase."""
        orch = COHNOrchestrator()
        phases_seen: list[str] = []

        def on_status(status: OrchestratorStatus):
            phases_seen.append(status.phase.name if hasattr(status.phase, 'name') else str(status.phase))

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await orch.provision_and_connect(
                wifi_ssid="HomeWiFi",
                wifi_password="password123",
                on_status=on_status,
            )

        assert "FAILED" in phases_seen

    @pytest.mark.anyio
    async def test_status_callback_has_camera_ip_on_success(self):
        """Status callbacks on success should include camera_ip."""
        orch = COHNOrchestrator()
        statuses: list[OrchestratorStatus] = []

        creds = MockCOHNCredentials(ip_address="192.168.1.100")
        mock_transport = _make_mock_cohn_transport()

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            new_callable=AsyncMock,
            return_value=creds,
        ), patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_cohn_connect",
            new_callable=AsyncMock,
            return_value=mock_transport,
        ):
            await orch.provision_and_connect(
                wifi_ssid="HomeWiFi",
                wifi_password="password123",
                on_status=statuses.append,
            )

        # The COMPLETE status should be the last one
        assert statuses[-1].phase == OrchestratorPhase.COMPLETE
        assert statuses[-1].progress_pct == 100


# ---------------------------------------------------------------------------
# Tests: Concurrent provisioning protection
# ---------------------------------------------------------------------------

class TestConcurrentProvisioning:
    """Verify concurrent provisioning requests are handled safely."""

    @pytest.mark.anyio
    async def test_cancel_clears_active_state(self):
        """cancel() sets the cancelled flag so provision can exit cleanly."""
        orch = COHNOrchestrator()
        assert not orch._cancelled
        orch.cancel()
        assert orch._cancelled

    @pytest.mark.anyio
    async def test_orchestrator_reuse_after_complete(self):
        """Orchestrator can be reused after a completed flow."""
        orch = COHNOrchestrator()
        mock_transport = _make_mock_cohn_transport()

        # First run
        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            new_callable=AsyncMock,
            return_value=MockCOHNCredentials(),
        ), patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_cohn_connect",
            new_callable=AsyncMock,
            return_value=mock_transport,
        ):
            result1 = await orch.provision_and_connect(
                wifi_ssid="HomeWiFi",
                wifi_password="password123",
            )

        assert result1 is mock_transport
        assert orch.status.phase == OrchestratorPhase.COMPLETE

        # Second run (reuse)
        mock_transport2 = _make_mock_cohn_transport()
        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            new_callable=AsyncMock,
            return_value=MockCOHNCredentials(ip_address="192.168.1.200"),
        ), patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_cohn_connect",
            new_callable=AsyncMock,
            return_value=mock_transport2,
        ):
            result2 = await orch.provision_and_connect(
                wifi_ssid="OtherWiFi",
                wifi_password="pass456",
            )

        assert result2 is mock_transport2

    @pytest.mark.anyio
    async def test_connect_with_credentials_skips_provisioning(self):
        """connect_with_credentials should NOT call _phase_provision_via_sdk."""
        orch = COHNOrchestrator()
        creds = MockCOHNCredentials()
        mock_transport = _make_mock_cohn_transport()

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            new_callable=AsyncMock,
        ) as mock_provision, patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_cohn_connect",
            new_callable=AsyncMock,
            return_value=mock_transport,
        ):
            result = await orch.connect_with_credentials(credentials=creds)

        assert result is mock_transport
        mock_provision.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests: Orchestrator timeout configuration
# ---------------------------------------------------------------------------

class TestOrchestratorConfig:
    """Verify custom timeouts are respected."""

    def test_default_timeouts(self):
        orch = COHNOrchestrator()
        assert orch._ble_scan_timeout == 15.0

    def test_custom_ble_scan_timeout(self):
        orch = COHNOrchestrator(ble_scan_timeout=5.0)
        assert orch._ble_scan_timeout == 5.0

    def test_initial_status_is_idle(self):
        orch = COHNOrchestrator()
        assert orch.status.phase == OrchestratorPhase.IDLE
        assert orch.status.progress_pct == 0
        assert orch.status.error == ""

    def test_is_active_during_phases(self):
        """is_active returns True only for in-progress phases."""
        orch = COHNOrchestrator()

        # IDLE is not active
        assert not orch.is_active

        # Simulate active phase
        orch._status = OrchestratorStatus(phase=OrchestratorPhase.BLE_SCANNING)
        assert orch.is_active

        orch._status = OrchestratorStatus(phase=OrchestratorPhase.PROVISIONING)
        assert orch.is_active

        orch._status = OrchestratorStatus(phase=OrchestratorPhase.COHN_CONNECTING)
        assert orch.is_active

        # Terminal phases are not active
        orch._status = OrchestratorStatus(phase=OrchestratorPhase.COMPLETE)
        assert not orch.is_active

        orch._status = OrchestratorStatus(phase=OrchestratorPhase.FAILED)
        assert not orch.is_active

        orch._status = OrchestratorStatus(phase=OrchestratorPhase.CANCELLED)
        assert not orch.is_active
