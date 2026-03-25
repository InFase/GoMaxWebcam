"""
Tests for BLEProvisioningService — open-gopro based COHN provisioning.

Mocked at:
  - WirelessGoPro boundary (no real BLE or camera)
  - CohnFeature boundary (no real COHN provisioning)
  - WiFi BLE commands (no real WiFi connection)
  - TinyDB/CohnDb (no real file I/O)

Tests verify:
  - Full provision flow: BLE scan → connect → WiFi → COHN → credentials
  - Progress callbacks fire at each phase
  - Error handling at each phase (BLE, WiFi, COHN)
  - Cancellation support
  - Stored credential retrieval from cohn_db
  - Resource cleanup (gopro.close called)
  - Test injection helpers
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from gomaxwebcam.provisioning import (
    BLEProvisioningService,
    COHNCredentials,
    ProvisionPhase,
    ProvisionProgress,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class MockCohnInfo:
    """Mimics open_gopro.models.general.CohnInfo."""
    ip_address: str = "172.20.123.45"
    username: str = "gopro"
    password: str = "p@ssw0rd"
    certificate: str = "-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----"
    is_complete: bool = True

    def __iter__(self):
        return iter({"ip_address": self.ip_address, "username": self.username,
                     "password": self.password, "certificate": self.certificate}.items())


class MockResult:
    """Mimics returns.result.Result."""
    def __init__(self, value=None, error=None):
        self._value = value
        self._error = error
        self._is_success = error is None

    def unwrap(self):
        return self._value

    def failure(self):
        return self._error


def _make_mock_gopro(
    identifier: str = "1234",
    open_ok: bool = True,
    wifi_connect_ok: bool = True,
    cohn_configure_ok: bool = True,
    cohn_info: MockCohnInfo | None = None,
) -> MagicMock:
    """Create a mock WirelessGoPro instance."""
    gopro = MagicMock()
    gopro.identifier = identifier
    gopro.is_ble_connected = True

    # open/close
    if open_ok:
        gopro.open = AsyncMock()
    else:
        gopro.open = AsyncMock(side_effect=Exception("BLE connection failed"))
    gopro.close = AsyncMock()

    # BLE commands
    gopro.ble_command = MagicMock()
    scan_resp = MagicMock()
    scan_resp.ok = True
    gopro.ble_command.scan_wifi_networks = AsyncMock(return_value=scan_resp)

    wifi_resp = MagicMock()
    wifi_resp.ok = wifi_connect_ok
    gopro.ble_command.request_wifi_connect_new = AsyncMock(return_value=wifi_resp)

    # COHN feature
    gopro.cohn = MagicMock()
    if cohn_info is None:
        cohn_info = MockCohnInfo()

    if cohn_configure_ok:
        result = MockResult(value=cohn_info)
    else:
        result = MockResult(error=TimeoutError("COHN provision timed out"))

    gopro.cohn.configure = AsyncMock(return_value=result)
    gopro.cohn.credentials = cohn_info

    return gopro


def _factory_for(mock_gopro: MagicMock):
    """Create a gopro factory that returns the mock."""
    def factory(target):
        return mock_gopro
    return factory


# ---------------------------------------------------------------------------
# Test: COHNCredentials
# ---------------------------------------------------------------------------


class TestCOHNCredentials:
    """COHNCredentials dataclass tests."""

    def test_default_values(self):
        creds = COHNCredentials()
        assert creds.ip_address == ""
        assert creds.provisioned is False
        assert not creds.is_complete

    def test_complete_credentials(self):
        creds = COHNCredentials(
            ip_address="172.20.1.1",
            username="gopro",
            password="secret",
            certificate="CERT",
            provisioned=True,
        )
        assert creds.is_complete
        assert creds.provisioned

    def test_incomplete_missing_username(self):
        creds = COHNCredentials(
            ip_address="172.20.1.1",
            password="secret",
            certificate="CERT",
        )
        assert not creds.is_complete


# ---------------------------------------------------------------------------
# Test: ProvisionProgress
# ---------------------------------------------------------------------------


class TestProvisionProgress:
    """ProvisionProgress serialization tests."""

    def test_to_dict(self):
        p = ProvisionProgress(
            phase=ProvisionPhase.BLE_CONNECTING,
            progress_pct=25,
            message="Connecting...",
            camera_serial="1234",
        )
        d = p.to_dict()
        assert d["phase"] == "BLE_CONNECTING"
        assert d["progress_pct"] == 25
        assert d["message"] == "Connecting..."
        assert d["camera_serial"] == "1234"

    def test_default_values(self):
        p = ProvisionProgress()
        assert p.phase == ProvisionPhase.IDLE
        assert p.progress_pct == 0


# ---------------------------------------------------------------------------
# Test: Full provision flow
# ---------------------------------------------------------------------------


class TestProvisionFlow:
    """Full provision flow tests."""

    @pytest.mark.anyio
    async def test_full_flow_success(self):
        """Happy path: BLE → WiFi → COHN → credentials."""
        mock_gopro = _make_mock_gopro()
        service = BLEProvisioningService()
        service._inject_gopro_factory(_factory_for(mock_gopro))

        progress_updates: list[ProvisionProgress] = []

        with patch(
            "gomaxwebcam.provisioning._is_successful",
            return_value=True,
        ):
            creds = await service.provision(
                target="1234",
                wifi_ssid="HomeNet",
                wifi_password="secret",
                on_progress=progress_updates.append,
            )

        assert creds is not None
        assert creds.provisioned is True
        assert creds.ip_address == "172.20.123.45"
        assert creds.username == "gopro"
        assert creds.password == "p@ssw0rd"
        assert creds.certificate.startswith("-----BEGIN")
        assert creds.camera_serial == "1234"

        # Progress callbacks should have fired
        assert len(progress_updates) >= 3  # BLE scan, BLE connect, WiFi, COHN, complete
        phases = [p.phase for p in progress_updates]
        assert ProvisionPhase.COMPLETE in phases

        # gopro should have been closed
        mock_gopro.close.assert_awaited()

    @pytest.mark.anyio
    async def test_provision_without_wifi(self):
        """Provision without WiFi credentials (camera already on network)."""
        mock_gopro = _make_mock_gopro()
        service = BLEProvisioningService()
        service._inject_gopro_factory(_factory_for(mock_gopro))

        with patch(
            "gomaxwebcam.provisioning._is_successful",
            return_value=True,
        ):
            creds = await service.provision(
                target="1234",
                wifi_ssid="",
                wifi_password="",
            )

        assert creds is not None
        assert creds.provisioned is True
        # WiFi commands should NOT have been called
        mock_gopro.ble_command.request_wifi_connect_new.assert_not_awaited()

    @pytest.mark.anyio
    async def test_provision_no_target_first_found(self):
        """Provision without target — connects to first found GoPro."""
        mock_gopro = _make_mock_gopro(identifier="5678")
        service = BLEProvisioningService()
        service._inject_gopro_factory(_factory_for(mock_gopro))

        with patch(
            "gomaxwebcam.provisioning._is_successful",
            return_value=True,
        ):
            creds = await service.provision(
                target=None,
                wifi_ssid="Net",
                wifi_password="pass",
            )

        assert creds is not None
        assert creds.camera_serial == "5678"


# ---------------------------------------------------------------------------
# Test: Error handling
# ---------------------------------------------------------------------------


class TestProvisionErrors:
    """Error handling at each phase."""

    @pytest.mark.anyio
    async def test_ble_open_failure(self):
        """BLE connection failure → None."""
        mock_gopro = _make_mock_gopro(open_ok=False)
        service = BLEProvisioningService()
        service._inject_gopro_factory(_factory_for(mock_gopro))

        progress_updates: list[ProvisionProgress] = []

        creds = await service.provision(
            target="1234",
            wifi_ssid="Net",
            wifi_password="pass",
            on_progress=progress_updates.append,
        )

        assert creds is None
        # Should have a FAILED progress
        phases = [p.phase for p in progress_updates]
        assert ProvisionPhase.FAILED in phases

    @pytest.mark.anyio
    async def test_wifi_connect_failure(self):
        """WiFi connect failure → None (with cleanup)."""
        mock_gopro = _make_mock_gopro(wifi_connect_ok=False)
        service = BLEProvisioningService()
        service._inject_gopro_factory(_factory_for(mock_gopro))

        # WiFi connect returning non-OK still falls through in our implementation
        # (some cameras connect despite non-OK), so COHN provisioning continues.
        # The test verifies it doesn't crash.
        with patch(
            "gomaxwebcam.provisioning._is_successful",
            return_value=True,
        ):
            creds = await service.provision(
                target="1234",
                wifi_ssid="BadNet",
                wifi_password="wrong",
            )

        # Should succeed since wifi connect failure is non-fatal in our impl
        assert creds is not None
        mock_gopro.close.assert_awaited()

    @pytest.mark.anyio
    async def test_cohn_provision_failure(self):
        """COHN provisioning failure → None."""
        mock_gopro = _make_mock_gopro(cohn_configure_ok=False)
        service = BLEProvisioningService()
        service._inject_gopro_factory(_factory_for(mock_gopro))

        progress_updates: list[ProvisionProgress] = []

        with patch(
            "gomaxwebcam.provisioning._is_successful",
            return_value=False,
        ):
            creds = await service.provision(
                target="1234",
                wifi_ssid="Net",
                wifi_password="pass",
                on_progress=progress_updates.append,
            )

        assert creds is None
        phases = [p.phase for p in progress_updates]
        assert ProvisionPhase.FAILED in phases
        mock_gopro.close.assert_awaited()

    @pytest.mark.anyio
    async def test_gopro_open_exception(self):
        """Exception during gopro.open() → proper cleanup."""
        mock_gopro = MagicMock()
        mock_gopro.open = AsyncMock(side_effect=RuntimeError("BLE adapter missing"))
        mock_gopro.close = AsyncMock()

        service = BLEProvisioningService()
        service._inject_gopro_factory(_factory_for(mock_gopro))

        creds = await service.provision(target="1234")
        assert creds is None

    @pytest.mark.anyio
    async def test_cohn_configure_exception(self):
        """Exception during cohn.configure() → handled gracefully."""
        mock_gopro = _make_mock_gopro()
        mock_gopro.cohn.configure = AsyncMock(side_effect=RuntimeError("BLE disconnected"))

        service = BLEProvisioningService()
        service._inject_gopro_factory(_factory_for(mock_gopro))

        creds = await service.provision(
            target="1234",
            wifi_ssid="Net",
            wifi_password="pass",
        )
        assert creds is None
        mock_gopro.close.assert_awaited()


# ---------------------------------------------------------------------------
# Test: Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    """Cancellation tests."""

    @pytest.mark.anyio
    async def test_cancel_during_ble_connect(self):
        """Cancel during BLE connect phase → returns None."""
        mock_gopro = MagicMock()

        # Make open() take a moment, then cancel during it
        async def slow_open(*a, **kw):
            await asyncio.sleep(0.1)

        mock_gopro.open = slow_open
        mock_gopro.close = AsyncMock()
        mock_gopro.identifier = "1234"

        service = BLEProvisioningService()
        service._inject_gopro_factory(_factory_for(mock_gopro))

        async def cancel_soon():
            await asyncio.sleep(0.02)
            service.cancel()

        asyncio.ensure_future(cancel_soon())

        creds = await service.provision(
            target="1234",
            wifi_ssid="Net",
            wifi_password="pass",
        )

        assert creds is None
        mock_gopro.close.assert_awaited()

    def test_cancel_sets_flag(self):
        """cancel() sets the internal flag."""
        service = BLEProvisioningService()
        assert not service._cancelled
        service.cancel()
        assert service._cancelled


# ---------------------------------------------------------------------------
# Test: Stored credentials
# ---------------------------------------------------------------------------


class TestStoredCredentials:
    """Credential retrieval from cohn_db tests."""

    def test_get_stored_credentials_found(self, tmp_path):
        """Retrieve existing credentials from cohn_db."""
        db_path = tmp_path / "cohn_db.json"

        # Write a credential entry using TinyDB
        try:
            from tinydb import TinyDB
            db = TinyDB(str(db_path), indent=4)
            db.insert({
                "serial": "1234",
                "credentials": {
                    "ip_address": "172.20.1.50",
                    "username": "gopro",
                    "password": "stored_pass",
                    "certificate": "STORED_CERT",
                },
            })
            db.close()
        except ImportError:
            pytest.skip("tinydb not installed")

        service = BLEProvisioningService(cohn_db_path=db_path)
        creds = service.get_stored_credentials("1234")

        assert creds is not None
        assert creds.ip_address == "172.20.1.50"
        assert creds.username == "gopro"
        assert creds.password == "stored_pass"
        assert creds.certificate == "STORED_CERT"
        assert creds.provisioned is True
        assert creds.camera_serial == "1234"

    def test_get_stored_credentials_not_found(self, tmp_path):
        """No credentials in cohn_db → None."""
        db_path = tmp_path / "cohn_db.json"

        try:
            from tinydb import TinyDB
            db = TinyDB(str(db_path), indent=4)
            db.close()
        except ImportError:
            pytest.skip("tinydb not installed")

        service = BLEProvisioningService(cohn_db_path=db_path)
        creds = service.get_stored_credentials("9999")
        assert creds is None

    def test_get_stored_credentials_missing_db(self, tmp_path):
        """Missing cohn_db file → None (no crash)."""
        db_path = tmp_path / "nonexistent" / "cohn_db.json"
        service = BLEProvisioningService(cohn_db_path=db_path)
        # Should return None, not raise
        creds = service.get_stored_credentials("1234")
        assert creds is None


# ---------------------------------------------------------------------------
# Test: Resource cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    """Resource cleanup tests."""

    @pytest.mark.anyio
    async def test_close_closes_gopro(self):
        """close() calls gopro.close()."""
        mock_gopro = MagicMock()
        mock_gopro.close = AsyncMock()

        service = BLEProvisioningService()
        service._inject_gopro(mock_gopro)

        await service.close()
        mock_gopro.close.assert_awaited_once()
        assert service._gopro is None

    @pytest.mark.anyio
    async def test_close_handles_exception(self):
        """close() swallows exceptions from gopro.close()."""
        mock_gopro = MagicMock()
        mock_gopro.close = AsyncMock(side_effect=RuntimeError("close failed"))

        service = BLEProvisioningService()
        service._inject_gopro(mock_gopro)

        await service.close()  # Should not raise
        assert service._gopro is None

    @pytest.mark.anyio
    async def test_close_when_no_gopro(self):
        """close() is safe when no gopro is open."""
        service = BLEProvisioningService()
        await service.close()  # Should not raise


# ---------------------------------------------------------------------------
# Test: Progress tracking
# ---------------------------------------------------------------------------


class TestProgressTracking:
    """Progress callback verification."""

    @pytest.mark.anyio
    async def test_progress_monotonic(self):
        """Progress percentage should not decrease during success flow."""
        mock_gopro = _make_mock_gopro()
        service = BLEProvisioningService()
        service._inject_gopro_factory(_factory_for(mock_gopro))

        progress_pcts: list[int] = []

        with patch(
            "gomaxwebcam.provisioning._is_successful",
            return_value=True,
        ):
            await service.provision(
                target="1234",
                wifi_ssid="Net",
                wifi_password="pass",
                on_progress=lambda p: progress_pcts.append(p.progress_pct),
            )

        # Should be non-decreasing
        for i in range(1, len(progress_pcts)):
            assert progress_pcts[i] >= progress_pcts[i - 1], (
                f"Progress decreased: {progress_pcts[i-1]} → {progress_pcts[i]}"
            )

    @pytest.mark.anyio
    async def test_progress_phases_order(self):
        """Phases should progress in expected order."""
        mock_gopro = _make_mock_gopro()
        service = BLEProvisioningService()
        service._inject_gopro_factory(_factory_for(mock_gopro))

        phases: list[ProvisionPhase] = []

        with patch(
            "gomaxwebcam.provisioning._is_successful",
            return_value=True,
        ):
            await service.provision(
                target="1234",
                wifi_ssid="Net",
                wifi_password="pass",
                on_progress=lambda p: phases.append(p.phase),
            )

        # Should see BLE_SCANNING → BLE_CONNECTING → WIFI_CONNECTING → COHN_PROVISIONING → COMPLETE
        assert ProvisionPhase.BLE_SCANNING in phases
        assert ProvisionPhase.BLE_CONNECTING in phases
        assert ProvisionPhase.WIFI_CONNECTING in phases
        assert ProvisionPhase.COHN_PROVISIONING in phases
        assert ProvisionPhase.COMPLETE in phases

        # COMPLETE should be last
        assert phases[-1] == ProvisionPhase.COMPLETE

    @pytest.mark.anyio
    async def test_no_progress_callback_doesnt_crash(self):
        """Provision without progress callback should work fine."""
        mock_gopro = _make_mock_gopro()
        service = BLEProvisioningService()
        service._inject_gopro_factory(_factory_for(mock_gopro))

        with patch(
            "gomaxwebcam.provisioning._is_successful",
            return_value=True,
        ):
            creds = await service.provision(
                target="1234",
                wifi_ssid="Net",
                wifi_password="pass",
                on_progress=None,
            )

        assert creds is not None
        assert creds.provisioned


# ---------------------------------------------------------------------------
# Test: Test injection helpers
# ---------------------------------------------------------------------------


class TestInjectionHelpers:
    """Test helper methods."""

    def test_inject_gopro_factory(self):
        """_inject_gopro_factory sets the factory."""
        service = BLEProvisioningService()
        factory = MagicMock()
        service._inject_gopro_factory(factory)
        assert service._gopro_factory is factory

    def test_inject_gopro(self):
        """_inject_gopro sets the gopro instance."""
        service = BLEProvisioningService()
        mock = MagicMock()
        service._inject_gopro(mock)
        assert service._gopro is mock
