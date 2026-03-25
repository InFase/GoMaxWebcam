"""
Tests for COHN credential caching after BLE provisioning.

Verifies that:
  - BLEProvisioningService._persist_credentials caches to CohnCredentialStore
  - COHNOrchestrator._cache_credentials caches after provisioning
  - Incomplete credentials are not cached
  - Persistence errors are handled gracefully (non-fatal)

All tests use mocked TinyDB/open-gopro — no real hardware or file I/O
beyond temporary directories.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# All tests in this module are mocked — no GoPro hardware needed.
pytestmark = pytest.mark.no_gopro_needed

from gomaxwebcam.provisioning import (
    BLEProvisioningService,
    COHNCredentials,
    ProvisionPhase,
    ProvisionProgress,
)

# Patch target for the local import inside _persist_credentials
_PERSISTENCE_MODULE = "gomaxwebcam.transport.cohn_persistence"
_ORCH_MODULE = "gomaxwebcam.transport.cohn_orchestrator"


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


class MockResult:
    """Mimics returns.result.Result."""
    def __init__(self, value=None, error=None):
        self._value = value
        self._error = error

    def unwrap(self):
        return self._value

    def failure(self):
        return self._error


def _make_mock_gopro(
    identifier: str = "1234",
    cohn_info: MockCohnInfo | None = None,
) -> MagicMock:
    """Create a mock WirelessGoPro instance for provisioning."""
    gopro = MagicMock()
    gopro.identifier = identifier
    gopro.is_ble_connected = True
    gopro.open = AsyncMock()
    gopro.close = AsyncMock()

    # BLE commands
    scan_resp = MagicMock(ok=True)
    gopro.ble_command = MagicMock()
    gopro.ble_command.scan_wifi_networks = AsyncMock(return_value=scan_resp)

    wifi_resp = MagicMock(ok=True)
    gopro.ble_command.request_wifi_connect_new = AsyncMock(return_value=wifi_resp)

    # COHN feature
    if cohn_info is None:
        cohn_info = MockCohnInfo()
    gopro.cohn = MagicMock()
    gopro.cohn.configure = AsyncMock(return_value=MockResult(value=cohn_info))

    return gopro


# ---------------------------------------------------------------------------
# Test: BLEProvisioningService._persist_credentials
# ---------------------------------------------------------------------------


class TestPersistCredentials:
    """BLEProvisioningService._persist_credentials tests."""

    def test_persist_complete_credentials(self, tmp_path: Path):
        """Complete credentials are persisted via CohnCredentialStore."""
        db_path = tmp_path / "cohn_db.json"
        service = BLEProvisioningService(cohn_db_path=db_path)

        creds = COHNCredentials(
            ip_address="172.20.1.100",
            username="gopro",
            password="s3cret",
            certificate="-----BEGIN CERTIFICATE-----\nTEST",
            camera_serial="1234",
            provisioned=True,
        )

        mock_store_instance = MagicMock()
        mock_store_instance.store_credentials.return_value = True
        MockStoreClass = MagicMock(return_value=mock_store_instance)

        with patch(f"{_PERSISTENCE_MODULE}.CohnCredentialStore", MockStoreClass):
            result = service._persist_credentials(creds)

        assert result is True
        MockStoreClass.assert_called_once_with(db_path=db_path)
        mock_store_instance.store_credentials.assert_called_once()
        # Verify stored credential fields
        stored_arg = mock_store_instance.store_credentials.call_args[0][0]
        assert stored_arg.ip_address == "172.20.1.100"
        assert stored_arg.username == "gopro"
        assert stored_arg.password == "s3cret"
        assert stored_arg.camera_serial == "1234"

    def test_persist_incomplete_credentials_skipped(self, tmp_path: Path):
        """Incomplete credentials are not persisted."""
        service = BLEProvisioningService(cohn_db_path=tmp_path / "db.json")

        # Missing password → is_complete is False
        creds = COHNCredentials(
            ip_address="172.20.1.100",
            username="gopro",
            password="",
            certificate="CERT",
            camera_serial="1234",
            provisioned=True,
        )

        result = service._persist_credentials(creds)
        assert result is False

    def test_persist_missing_ip_skipped(self, tmp_path: Path):
        """Credentials without IP are not persisted."""
        service = BLEProvisioningService(cohn_db_path=tmp_path / "db.json")
        creds = COHNCredentials(
            ip_address="",
            username="gopro",
            password="pass",
            certificate="cert",
            camera_serial="1234",
        )
        assert service._persist_credentials(creds) is False

    def test_persist_handles_store_error(self, tmp_path: Path):
        """Persistence errors are handled gracefully (non-fatal)."""
        service = BLEProvisioningService(cohn_db_path=tmp_path / "db.json")

        creds = COHNCredentials(
            ip_address="172.20.1.100",
            username="gopro",
            password="s3cret",
            certificate="CERT",
            camera_serial="1234",
        )

        with patch(
            f"{_PERSISTENCE_MODULE}.CohnCredentialStore",
            side_effect=RuntimeError("DB corrupted"),
        ):
            result = service._persist_credentials(creds)

        assert result is False  # Graceful failure, not an exception

    def test_persist_handles_import_error(self, tmp_path: Path):
        """Import errors in cohn_persistence → graceful failure."""
        service = BLEProvisioningService(cohn_db_path=tmp_path / "db.json")

        creds = COHNCredentials(
            ip_address="172.20.1.100",
            username="gopro",
            password="s3cret",
            certificate="CERT",
            camera_serial="1234",
        )

        # Simulate import failure by making the module import raise
        with patch.dict("sys.modules", {_PERSISTENCE_MODULE: None}):
            result = service._persist_credentials(creds)

        assert result is False


# ---------------------------------------------------------------------------
# Test: Full provision flow with credential caching
# ---------------------------------------------------------------------------


class TestProvisionWithCaching:
    """Verify credentials are cached during full provision flow."""

    @pytest.mark.anyio
    async def test_full_flow_caches_credentials(self):
        """Successful provisioning persists credentials to CohnCredentialStore."""
        mock_gopro = _make_mock_gopro()
        service = BLEProvisioningService()
        service._inject_gopro_factory(lambda target: mock_gopro)

        mock_store_instance = MagicMock()
        mock_store_instance.store_credentials.return_value = True

        with patch("gomaxwebcam.provisioning._is_successful", return_value=True), \
             patch(f"{_PERSISTENCE_MODULE}.CohnCredentialStore", return_value=mock_store_instance):

            creds = await service.provision(
                target="1234",
                wifi_ssid="HomeNet",
                wifi_password="secret",
            )

        assert creds is not None
        assert creds.provisioned is True
        # store_credentials should have been called during _persist_credentials
        mock_store_instance.store_credentials.assert_called_once()

    @pytest.mark.anyio
    async def test_failed_provision_does_not_cache(self):
        """Failed provisioning does not attempt to cache credentials."""
        mock_gopro = _make_mock_gopro()
        mock_gopro.cohn.configure = AsyncMock(
            return_value=MockResult(error=TimeoutError("timeout")),
        )

        service = BLEProvisioningService()
        service._inject_gopro_factory(lambda target: mock_gopro)

        mock_store_cls = MagicMock()

        with patch("gomaxwebcam.provisioning._is_successful", return_value=False), \
             patch(f"{_PERSISTENCE_MODULE}.CohnCredentialStore", mock_store_cls):

            creds = await service.provision(
                target="1234",
                wifi_ssid="Net",
                wifi_password="pass",
            )

        assert creds is None
        # CohnCredentialStore should never have been instantiated
        mock_store_cls.assert_not_called()

    @pytest.mark.anyio
    async def test_cache_failure_does_not_break_provision(self):
        """Credential caching failure does not break the provision flow."""
        mock_gopro = _make_mock_gopro()
        service = BLEProvisioningService()
        service._inject_gopro_factory(lambda target: mock_gopro)

        with patch("gomaxwebcam.provisioning._is_successful", return_value=True), \
             patch(f"{_PERSISTENCE_MODULE}.CohnCredentialStore", side_effect=Exception("DB error")):

            creds = await service.provision(
                target="1234",
                wifi_ssid="HomeNet",
                wifi_password="secret",
            )

        # Provisioning should still succeed even if caching failed
        assert creds is not None
        assert creds.provisioned is True


# ---------------------------------------------------------------------------
# Test: COHNOrchestrator._cache_credentials
# ---------------------------------------------------------------------------


class TestOrchestratorCacheCredentials:
    """COHNOrchestrator._cache_credentials tests."""

    def test_cache_complete_credentials(self, tmp_path: Path):
        """Complete credentials are cached via CohnCredentialStore."""
        from gomaxwebcam.transport.cohn_orchestrator import COHNOrchestrator

        orch = COHNOrchestrator()

        creds = COHNCredentials(
            ip_address="172.20.1.100",
            username="gopro",
            password="s3cret",
            certificate="CERT",
            camera_serial="1234",
            provisioned=True,
        )

        mock_store_instance = MagicMock()
        mock_store_instance.store_credentials.return_value = True

        with patch(
            f"{_PERSISTENCE_MODULE}.CohnCredentialStore",
            return_value=mock_store_instance,
        ), patch.object(orch, "_get_cohn_db_path", return_value=tmp_path / "db.json"):
            result = orch._cache_credentials(creds)

        assert result is True
        mock_store_instance.store_credentials.assert_called_once()

    def test_cache_incomplete_credentials_skipped(self):
        """Incomplete credentials are not cached."""
        from gomaxwebcam.transport.cohn_orchestrator import COHNOrchestrator

        orch = COHNOrchestrator()

        # Missing password
        creds = MagicMock()
        creds.ip_address = "172.20.1.100"
        creds.username = "gopro"
        creds.password = ""
        creds.certificate = "CERT"
        creds.camera_serial = "1234"

        result = orch._cache_credentials(creds)
        assert result is False

    def test_cache_missing_ip_skipped(self):
        """Credentials without IP are not cached."""
        from gomaxwebcam.transport.cohn_orchestrator import COHNOrchestrator

        orch = COHNOrchestrator()

        creds = MagicMock()
        creds.ip_address = ""
        creds.username = "gopro"
        creds.password = "pass"
        creds.certificate = "CERT"
        creds.camera_serial = "1234"

        result = orch._cache_credentials(creds)
        assert result is False

    def test_cache_handles_exception(self, tmp_path: Path):
        """Cache errors are handled gracefully."""
        from gomaxwebcam.transport.cohn_orchestrator import COHNOrchestrator

        orch = COHNOrchestrator()

        creds = COHNCredentials(
            ip_address="172.20.1.100",
            username="gopro",
            password="s3cret",
            certificate="CERT",
            camera_serial="1234",
        )

        with patch(
            f"{_PERSISTENCE_MODULE}.CohnCredentialStore",
            side_effect=RuntimeError("DB error"),
        ), patch.object(orch, "_get_cohn_db_path", return_value=tmp_path / "db.json"):
            result = orch._cache_credentials(creds)

        assert result is False  # Graceful, no exception raised
