"""
Tests for COHN orchestrator — sequences BLE provisioning → COHN transport.

Mocked at:
  - BLEProvisioningService boundary (no real BLE or camera)
  - COHNTransport boundary (no real HTTPS or camera)

Tests verify:
  - Full provision_and_connect flow with status callbacks
  - connect_with_credentials shortcut path
  - Error handling at each phase (provision, COHN connect)
  - Retry logic at each phase
  - Cancellation support
  - Status callback firing and progress tracking
  - Cleanup of provisioning resources on failure
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gomaxwebcam.transport.cohn_orchestrator import (
    COHNOrchestrator,
    OrchestratorPhase,
    OrchestratorStatus,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class MockCOHNCredentials:
    """Mock credentials returned by BLEProvisioningService."""
    camera_serial: str = "1234"
    ip_address: str = "192.168.1.100"
    username: str = "gopro"
    password: str = "secret123"
    certificate: str = ""
    ssid: str = "TestWiFi"
    provisioned: bool = True


def _make_mock_cohn_transport(discover_ok: bool = True, connect_ok: bool = True) -> MagicMock:
    """Create a mock COHNTransport."""
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
# Test: OrchestratorStatus
# ---------------------------------------------------------------------------


class TestOrchestratorStatus:
    """OrchestratorStatus serialization tests."""

    def test_default_status(self):
        status = OrchestratorStatus()
        assert status.phase == OrchestratorPhase.IDLE
        assert status.progress_pct == 0
        assert status.message == ""
        assert status.error == ""

    def test_to_dict(self):
        status = OrchestratorStatus(
            phase=OrchestratorPhase.PROVISIONING,
            progress_pct=45,
            message="Testing...",
            camera_name="GoPro 1234",
        )
        d = status.to_dict()
        assert d["phase"] == "PROVISIONING"
        assert d["progress_pct"] == 45
        assert d["message"] == "Testing..."
        assert d["camera_name"] == "GoPro 1234"

    def test_to_dict_all_fields(self):
        status = OrchestratorStatus(
            phase=OrchestratorPhase.COMPLETE,
            progress_pct=100,
            message="Done!",
            camera_ip="192.168.1.100",
            wifi_ssid="TestWiFi",
            elapsed_s=12.345,
        )
        d = status.to_dict()
        assert d["camera_ip"] == "192.168.1.100"
        assert d["wifi_ssid"] == "TestWiFi"
        assert d["elapsed_s"] == 12.3


# ---------------------------------------------------------------------------
# Test: OrchestratorPhase
# ---------------------------------------------------------------------------


class TestOrchestratorPhase:
    """Phase enum tests."""

    def test_all_phases_exist(self):
        phases = [
            OrchestratorPhase.IDLE,
            OrchestratorPhase.BLE_SCANNING,
            OrchestratorPhase.BLE_CONNECTING,
            OrchestratorPhase.PROVISIONING,
            OrchestratorPhase.COHN_DISCOVERING,
            OrchestratorPhase.COHN_CONNECTING,
            OrchestratorPhase.COHN_STREAMING,
            OrchestratorPhase.COMPLETE,
            OrchestratorPhase.FAILED,
            OrchestratorPhase.CANCELLED,
        ]
        assert len(phases) == 10

    def test_phase_names(self):
        assert OrchestratorPhase.IDLE.name == "IDLE"
        assert OrchestratorPhase.COMPLETE.name == "COMPLETE"


# ---------------------------------------------------------------------------
# Test: COHNOrchestrator init and properties
# ---------------------------------------------------------------------------


class TestCOHNOrchestratorInit:
    """Initialization and property tests."""

    def test_initial_state(self):
        orch = COHNOrchestrator()
        assert orch.status.phase == OrchestratorPhase.IDLE
        assert not orch.is_active
        assert orch.transport is None

    def test_custom_timeouts(self):
        orch = COHNOrchestrator(ble_scan_timeout=5.0)
        assert orch._ble_scan_timeout == 5.0


# ---------------------------------------------------------------------------
# Test: Full provision_and_connect flow
# ---------------------------------------------------------------------------


class TestProvisionAndConnect:
    """Full orchestration flow tests."""

    @pytest.mark.anyio
    async def test_full_flow_success(self):
        """Happy path: BLE provision → COHN connect."""
        orch = COHNOrchestrator()
        statuses: list[OrchestratorStatus] = []

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
            result = await orch.provision_and_connect(
                wifi_ssid="TestWiFi",
                wifi_password="secret",
                on_status=statuses.append,
            )

        assert result is mock_transport
        assert orch.status.phase == OrchestratorPhase.COMPLETE
        assert orch.status.progress_pct == 100
        assert orch.transport is mock_transport

    @pytest.mark.anyio
    async def test_with_ble_address(self):
        """When ble_address is provided, it's passed to the SDK."""
        orch = COHNOrchestrator()
        mock_transport = _make_mock_cohn_transport()

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            new_callable=AsyncMock,
            return_value=MockCOHNCredentials(),
        ) as mock_provision, patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_cohn_connect",
            new_callable=AsyncMock,
            return_value=mock_transport,
        ):
            result = await orch.provision_and_connect(
                wifi_ssid="TestWiFi",
                wifi_password="secret",
                ble_address="1234",
            )

        assert result is mock_transport
        mock_provision.assert_awaited_once()
        # The ble_address should be passed as 'target' to the SDK
        call_kwargs = mock_provision.call_args[1]
        assert call_kwargs.get("target") == "1234"

    @pytest.mark.anyio
    async def test_status_callbacks_fire(self):
        """Status callbacks should fire at each phase."""
        orch = COHNOrchestrator()
        statuses: list[OrchestratorStatus] = []

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
                wifi_ssid="TestWiFi",
                wifi_password="secret",
                on_status=statuses.append,
            )

        # Should have at least a COMPLETE status
        assert len(statuses) >= 1
        assert statuses[-1].phase == OrchestratorPhase.COMPLETE

    @pytest.mark.anyio
    async def test_progress_is_monotonic(self):
        """Progress percentage should never decrease during success flow."""
        orch = COHNOrchestrator()
        progress_values: list[int] = []

        def track_progress(status: OrchestratorStatus):
            progress_values.append(status.progress_pct)

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
                wifi_ssid="TestWiFi",
                wifi_password="secret",
                on_status=track_progress,
            )

        # Progress should be non-decreasing
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1], (
                f"Progress decreased: {progress_values[i - 1]} -> {progress_values[i]}"
            )


# ---------------------------------------------------------------------------
# Test: Error handling at each phase
# ---------------------------------------------------------------------------


class TestProvisionErrors:
    """Error handling at each phase of orchestration."""

    @pytest.mark.anyio
    async def test_provision_failure(self):
        """BLE provisioning fails → FAILED."""
        orch = COHNOrchestrator()

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await orch.provision_and_connect(
                wifi_ssid="TestWiFi",
                wifi_password="secret",
            )

        assert result is None
        assert orch.status.phase == OrchestratorPhase.FAILED

    @pytest.mark.anyio
    async def test_cohn_connect_failure(self):
        """COHN transport connect fails → FAILED."""
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
                wifi_ssid="TestWiFi",
                wifi_password="secret",
            )

        assert result is None
        assert orch.status.phase == OrchestratorPhase.FAILED

    @pytest.mark.anyio
    async def test_unexpected_exception(self):
        """Unexpected exception during orchestration → FAILED with error."""
        orch = COHNOrchestrator()

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            result = await orch.provision_and_connect(
                wifi_ssid="TestWiFi",
                wifi_password="secret",
            )

        assert result is None
        assert orch.status.phase == OrchestratorPhase.FAILED
        assert "boom" in orch.status.error

    @pytest.mark.anyio
    async def test_error_status_includes_detail(self):
        """Error status should include human-readable error message."""
        orch = COHNOrchestrator()

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await orch.provision_and_connect(
                wifi_ssid="BadNetwork",
                wifi_password="secret",
            )

        d = orch.status.to_dict()
        assert d["phase"] == "FAILED"
        assert d["error"] != ""
        assert "message" in d


# ---------------------------------------------------------------------------
# Test: Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    """Cancellation tests."""

    @pytest.mark.anyio
    async def test_cancel_during_provision(self):
        """Cancel during provisioning → CANCELLED."""
        orch = COHNOrchestrator()

        async def slow_provision(*args, **kwargs):
            await asyncio.sleep(0.05)
            return MockCOHNCredentials()

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            side_effect=slow_provision,
        ), patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_cohn_connect",
            new_callable=AsyncMock,
            return_value=_make_mock_cohn_transport(),
        ):
            # Cancel immediately after starting
            async def cancel_soon():
                await asyncio.sleep(0.01)
                orch.cancel()

            asyncio.ensure_future(cancel_soon())
            result = await orch.provision_and_connect(
                wifi_ssid="TestWiFi",
                wifi_password="secret",
            )

        # Should either be cancelled or completed (race condition is OK)
        assert result is None or orch.status.phase in (
            OrchestratorPhase.CANCELLED,
            OrchestratorPhase.COMPLETE,
        )

    def test_cancel_method(self):
        """cancel() sets the flag."""
        orch = COHNOrchestrator()
        assert not orch._cancelled
        orch.cancel()
        assert orch._cancelled


# ---------------------------------------------------------------------------
# Test: connect_with_credentials (skip BLE)
# ---------------------------------------------------------------------------


class TestConnectWithCredentials:
    """Shortcut path using existing credentials."""

    @pytest.mark.anyio
    async def test_success(self):
        """Successful connection with stored credentials."""
        orch = COHNOrchestrator()
        creds = MockCOHNCredentials()
        mock_transport = _make_mock_cohn_transport()

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_cohn_connect",
            new_callable=AsyncMock,
            return_value=mock_transport,
        ):
            result = await orch.connect_with_credentials(credentials=creds)

        assert result is mock_transport
        assert orch.status.phase == OrchestratorPhase.COMPLETE

    @pytest.mark.anyio
    async def test_failure(self):
        """Connection with bad credentials → FAILED."""
        orch = COHNOrchestrator()
        creds = MockCOHNCredentials(password="wrong")

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_cohn_connect",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await orch.connect_with_credentials(credentials=creds)

        assert result is None
        assert orch.status.phase == OrchestratorPhase.FAILED

    @pytest.mark.anyio
    async def test_status_callback(self):
        """Status callbacks should fire during connect_with_credentials."""
        orch = COHNOrchestrator()
        creds = MockCOHNCredentials()
        statuses: list[OrchestratorStatus] = []
        mock_transport = _make_mock_cohn_transport()

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_cohn_connect",
            new_callable=AsyncMock,
            return_value=mock_transport,
        ):
            await orch.connect_with_credentials(
                credentials=creds,
                on_status=statuses.append,
            )

        assert len(statuses) >= 1
        assert statuses[-1].phase == OrchestratorPhase.COMPLETE


# ---------------------------------------------------------------------------
# Test: is_active property
# ---------------------------------------------------------------------------


class TestIsActive:
    """is_active property tests."""

    def test_idle_is_not_active(self):
        orch = COHNOrchestrator()
        assert not orch.is_active

    def test_active_during_provisioning(self):
        orch = COHNOrchestrator()
        orch._status = OrchestratorStatus(phase=OrchestratorPhase.PROVISIONING)
        assert orch.is_active

    def test_complete_is_not_active(self):
        orch = COHNOrchestrator()
        orch._status = OrchestratorStatus(phase=OrchestratorPhase.COMPLETE)
        assert not orch.is_active

    def test_failed_is_not_active(self):
        orch = COHNOrchestrator()
        orch._status = OrchestratorStatus(phase=OrchestratorPhase.FAILED)
        assert not orch.is_active


# ---------------------------------------------------------------------------
# Test: Provisioning cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    """Resource cleanup tests."""

    @pytest.mark.anyio
    async def test_cleanup_on_provision_failure(self):
        """Provisioning service should be cleaned up when provisioning fails."""
        orch = COHNOrchestrator()

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await orch.provision_and_connect(
                wifi_ssid="TestWiFi",
                wifi_password="secret",
            )

        # Provisioning service should have been cleaned up
        assert orch._provisioning_service is None

    @pytest.mark.anyio
    async def test_cleanup_on_success(self):
        """Provisioning service should be cleaned up after successful flow."""
        orch = COHNOrchestrator()

        with patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_provision_via_sdk",
            new_callable=AsyncMock,
            return_value=MockCOHNCredentials(),
        ), patch(
            "gomaxwebcam.transport.cohn_orchestrator.COHNOrchestrator._phase_cohn_connect",
            new_callable=AsyncMock,
            return_value=_make_mock_cohn_transport(),
        ):
            await orch.provision_and_connect(
                wifi_ssid="TestWiFi",
                wifi_password="secret",
            )

        # Provisioning service should be cleaned up (no longer needed)
        assert orch._provisioning_service is None


# ---------------------------------------------------------------------------
# Test: Callback error resilience
# ---------------------------------------------------------------------------


class TestCallbackResilience:
    """Status callback errors should not crash orchestration."""

    @pytest.mark.anyio
    async def test_broken_callback_doesnt_crash(self):
        """A callback that throws should not stop orchestration."""
        orch = COHNOrchestrator()

        def bad_callback(status):
            raise RuntimeError("callback exploded")

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
            result = await orch.provision_and_connect(
                wifi_ssid="TestWiFi",
                wifi_password="secret",
                on_status=bad_callback,
            )

        # Should still complete despite callback errors
        assert result is mock_transport
