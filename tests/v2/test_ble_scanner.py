"""
Tests for BLE scanner — mocked at the bleak boundary.

No real BLE hardware required. These tests verify:
  - GoPro device detection by name prefix and service UUID
  - One-shot scan returns DiscoveredGoPro instances
  - Continuous scan calls back on discovery
  - Serial suffix extraction from camera names
  - Graceful handling of missing bleak dependency
  - Error resilience (scan failures, malformed data)
"""

from __future__ import annotations

import asyncio
import types
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _patch_bleak(scanner_cls=None):
    """Create a mock bleak module with a controllable BleakScanner."""
    mock_bleak = types.ModuleType("bleak")
    mock_bleak.BleakScanner = scanner_cls or MagicMock()
    mock_bleak.BleakClient = MagicMock()
    return patch.dict("sys.modules", {"bleak": mock_bleak})


# ---------------------------------------------------------------------------
# Mock bleak objects
# ---------------------------------------------------------------------------


@dataclass
class MockBLEDevice:
    """Simulates a bleak BLEDevice."""
    address: str = "AA:BB:CC:DD:EE:FF"
    name: str | None = "GoPro 1234"
    rssi: int = -55
    metadata: dict | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {"uuids": []}


@dataclass
class MockAdvertisementData:
    """Simulates bleak AdvertisementData."""
    rssi: int = -55
    local_name: str | None = None
    service_uuids: list[str] | None = None
    manufacturer_data: dict | None = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scanner():
    from gomaxwebcam.ble.scanner import BLEScanner
    return BLEScanner(scan_timeout=5.0)


# ---------------------------------------------------------------------------
# DiscoveredGoPro dataclass tests
# ---------------------------------------------------------------------------


class TestDiscoveredGoPro:
    def test_creation(self):
        from gomaxwebcam.ble.scanner import DiscoveredGoPro
        gp = DiscoveredGoPro(
            address="AA:BB:CC:DD:EE:FF",
            name="GoPro 1234",
            rssi=-50,
            serial_suffix="1234",
        )
        assert gp.address == "AA:BB:CC:DD:EE:FF"
        assert gp.name == "GoPro 1234"
        assert gp.rssi == -50
        assert gp.serial_suffix == "1234"

    def test_frozen(self):
        from gomaxwebcam.ble.scanner import DiscoveredGoPro
        gp = DiscoveredGoPro(address="AA:BB:CC:DD:EE:FF", name="GoPro 1234")
        with pytest.raises(AttributeError):
            gp.name = "changed"  # type: ignore

    def test_defaults(self):
        from gomaxwebcam.ble.scanner import DiscoveredGoPro
        gp = DiscoveredGoPro(address="AA:BB:CC:DD:EE:FF", name="GoPro 1234")
        assert gp.rssi == -100
        assert gp.serial_suffix == ""
        assert gp.model_hint == ""


# ---------------------------------------------------------------------------
# Name matching tests
# ---------------------------------------------------------------------------


class TestIsGoPro:
    def test_gopro_prefix(self, scanner):
        device = MockBLEDevice(name="GoPro 1234")
        assert scanner._is_gopro(device) is True

    def test_gp_prefix(self, scanner):
        device = MockBLEDevice(name="GP-1234")
        assert scanner._is_gopro(device) is True

    def test_non_gopro(self, scanner):
        device = MockBLEDevice(name="Random Device")
        assert scanner._is_gopro(device) is False

    def test_none_name(self, scanner):
        device = MockBLEDevice(name=None)
        assert scanner._is_gopro(device) is False

    def test_empty_name(self, scanner):
        device = MockBLEDevice(name="")
        assert scanner._is_gopro(device) is False

    def test_service_uuid_match(self, scanner):
        from gomaxwebcam.ble.uuids import GOPRO_SERVICE_UUID
        device = MockBLEDevice(
            name="Unknown",
            metadata={"uuids": [GOPRO_SERVICE_UUID]},
        )
        assert scanner._is_gopro(device) is True


# ---------------------------------------------------------------------------
# Serial suffix extraction tests
# ---------------------------------------------------------------------------


class TestMakeDiscovered:
    def test_gopro_prefix_serial(self, scanner):
        device = MockBLEDevice(name="GoPro 1234", rssi=-50)
        gp = scanner._make_discovered(device)
        assert gp.serial_suffix == "1234"
        assert gp.name == "GoPro 1234"

    def test_gp_prefix_serial(self, scanner):
        device = MockBLEDevice(name="GP-5678")
        gp = scanner._make_discovered(device)
        assert gp.serial_suffix == "5678"

    def test_no_name(self, scanner):
        device = MockBLEDevice(name=None)
        gp = scanner._make_discovered(device)
        assert gp.name == "Unknown GoPro"
        assert gp.serial_suffix == ""

    def test_rssi_from_device(self, scanner):
        device = MockBLEDevice(name="GoPro 1234", rssi=-45)
        gp = scanner._make_discovered(device)
        assert gp.rssi == -45

    def test_rssi_from_adv_data(self, scanner):
        device = MockBLEDevice(name="GoPro 1234", rssi=-45)
        adv = MockAdvertisementData(rssi=-60)
        gp = scanner._make_discovered(device, adv)
        assert gp.rssi == -60  # adv_data takes precedence


# ---------------------------------------------------------------------------
# One-shot scan tests
# ---------------------------------------------------------------------------


class TestScanOnce:
    @pytest.mark.anyio
    async def test_finds_gopro_devices(self, scanner):
        mock_devices = [
            MockBLEDevice(address="AA:BB:CC:DD:EE:01", name="GoPro 1234", rssi=-50),
            MockBLEDevice(address="AA:BB:CC:DD:EE:02", name="Random Speaker", rssi=-30),
            MockBLEDevice(address="AA:BB:CC:DD:EE:03", name="GoPro 5678", rssi=-60),
        ]

        mock_instance = MagicMock()
        mock_instance.discover = AsyncMock(return_value=mock_devices)
        mock_cls = MagicMock(return_value=mock_instance)

        with _patch_bleak(scanner_cls=mock_cls):
            results = await scanner.scan_once(timeout=3.0)

        assert len(results) == 2
        assert results[0].name == "GoPro 1234"
        assert results[1].name == "GoPro 5678"

    @pytest.mark.anyio
    async def test_empty_scan(self, scanner):
        mock_instance = MagicMock()
        mock_instance.discover = AsyncMock(return_value=[])
        mock_cls = MagicMock(return_value=mock_instance)

        with _patch_bleak(scanner_cls=mock_cls):
            results = await scanner.scan_once()

        assert results == []

    @pytest.mark.anyio
    async def test_bleak_not_installed(self, scanner):
        """Should return empty list if bleak is not installed."""
        with patch.dict("sys.modules", {"bleak": None}):
            results = await scanner.scan_once()
            assert results == []

    @pytest.mark.anyio
    async def test_scan_exception(self, scanner):
        """Should return empty list on scan failure."""
        mock_instance = MagicMock()
        mock_instance.discover = AsyncMock(side_effect=OSError("BLE adapter not found"))
        mock_cls = MagicMock(return_value=mock_instance)

        with _patch_bleak(scanner_cls=mock_cls):
            results = await scanner.scan_once()

        assert results == []

    @pytest.mark.anyio
    async def test_uses_custom_timeout(self, scanner):
        mock_instance = MagicMock()
        mock_instance.discover = AsyncMock(return_value=[])
        mock_cls = MagicMock(return_value=mock_instance)

        with _patch_bleak(scanner_cls=mock_cls):
            await scanner.scan_once(timeout=7.5)

        mock_instance.discover.assert_awaited_once_with(timeout=7.5)


# ---------------------------------------------------------------------------
# Continuous scan tests
# ---------------------------------------------------------------------------


class TestScanContinuous:
    @pytest.mark.anyio
    async def test_continuous_calls_back(self, scanner):
        found: list = []
        stop = asyncio.Event()

        mock_scanner_instance = MagicMock()
        mock_scanner_instance.start = AsyncMock()
        mock_scanner_instance.stop = AsyncMock()

        # Capture the detection callback
        detection_cb = None

        def capture_scanner(**kwargs):
            nonlocal detection_cb
            detection_cb = kwargs.get("detection_callback")
            return mock_scanner_instance

        with _patch_bleak(scanner_cls=capture_scanner):
            async def run_scan():
                await scanner.scan_continuous(
                    callback=lambda gp: found.append(gp),
                    duration=0.5,
                    stop_event=stop,
                )

            task = asyncio.create_task(run_scan())

            # Give scanner time to start
            await asyncio.sleep(0.1)

            # Simulate device discovery
            if detection_cb:
                device = MockBLEDevice(
                    address="AA:BB:CC:DD:EE:01",
                    name="GoPro 1234",
                )
                detection_cb(device, MockAdvertisementData(rssi=-50))

            # Stop and wait
            stop.set()
            await task

        assert len(found) == 1
        assert found[0].name == "GoPro 1234"

    @pytest.mark.anyio
    async def test_continuous_deduplicates(self, scanner):
        found: list = []
        stop = asyncio.Event()

        mock_scanner_instance = MagicMock()
        mock_scanner_instance.start = AsyncMock()
        mock_scanner_instance.stop = AsyncMock()

        detection_cb = None

        def capture_scanner(**kwargs):
            nonlocal detection_cb
            detection_cb = kwargs.get("detection_callback")
            return mock_scanner_instance

        with _patch_bleak(scanner_cls=capture_scanner):
            async def run_scan():
                await scanner.scan_continuous(
                    callback=lambda gp: found.append(gp),
                    duration=0.5,
                    stop_event=stop,
                )

            task = asyncio.create_task(run_scan())
            await asyncio.sleep(0.1)

            # Same device seen twice
            if detection_cb:
                device = MockBLEDevice(address="AA:BB:CC:DD:EE:01", name="GoPro 1234")
                detection_cb(device, MockAdvertisementData())
                detection_cb(device, MockAdvertisementData())

            stop.set()
            await task

        # Should only have one callback (deduplication by address)
        assert len(found) == 1

    @pytest.mark.anyio
    async def test_stop_method(self, scanner):
        mock_scanner_instance = MagicMock()
        mock_scanner_instance.stop = AsyncMock()
        scanner._inject_scanner(mock_scanner_instance)

        await scanner.stop()
        mock_scanner_instance.stop.assert_awaited_once()
        assert scanner._scanner is None
