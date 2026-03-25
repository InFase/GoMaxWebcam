"""
Tests for USB device discovery and enumeration (v2 async module).

Tests cover:
  - GoProDeviceInfo dataclass construction and properties
  - Serial → IP computation (Open GoPro spec)
  - Serial suffix extraction
  - VID:PID parsing from USB device ID strings
  - Serial extraction from composite device entries
  - WMI enumeration with mocked PowerShell output
  - pnputil enumeration with mocked output
  - Async enumerate_usb_gopro_devices with fallback chain
  - Async find_gopro_device convenience function
  - Integration with USBTransport.discover() auto-serial detection
  - Non-Windows platform returns empty
  - Error handling (subprocess failures, malformed output)

No real hardware required — all OS calls are mocked.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from gomaxwebcam.discovery import (
    GOPRO_VENDOR_ID,
    GOPRO_VENDOR_IDS,
    GOPRO_KNOWN_PIDS,
    GoProDeviceInfo,
    DiscoveryMethod,
    compute_ip_from_serial,
    compute_serial_suffix,
    parse_vid_pid,
    extract_serial_from_device_id,
    _enumerate_via_wmi_sync,
    _enumerate_via_pnputil_sync,
    enumerate_usb_gopro_devices,
    find_gopro_device,
    enumerate_usb_gopro_devices_sync,
)


# ---------------------------------------------------------------------------
# GoProDeviceInfo tests
# ---------------------------------------------------------------------------


class TestGoProDeviceInfo:
    def test_defaults(self):
        info = GoProDeviceInfo()
        assert info.vendor_id == GOPRO_VENDOR_ID
        assert info.product_id == 0
        assert info.description == "GoPro Device"
        assert info.serial_number is None
        assert info.camera_ip is None

    def test_usb_id_str(self):
        info = GoProDeviceInfo(vendor_id=0x2672, product_id=0x0059)
        assert info.usb_id_str == "2672:0059"

    def test_is_known_model(self):
        info = GoProDeviceInfo(product_id=0x0059)
        assert info.is_known_model is True

    def test_is_unknown_model(self):
        info = GoProDeviceInfo(product_id=0xFFFF)
        assert info.is_known_model is False

    def test_str_minimal(self):
        info = GoProDeviceInfo()
        s = str(info)
        assert "GoPro" in s
        assert "2672:0000" in s

    def test_str_full(self):
        info = GoProDeviceInfo(
            vendor_id=0x2672,
            product_id=0x0059,
            description="GoPro Hero 13",
            serial_number="C3531350067212",
            camera_ip="172.22.112.51",
            discovery_method=DiscoveryMethod.WMI,
        )
        s = str(info)
        assert "C3531350067212" in s
        assert "172.22.112.51" in s
        assert "WMI" in s


# ---------------------------------------------------------------------------
# Serial → IP computation
# ---------------------------------------------------------------------------


class TestComputeIpFromSerial:
    def test_hero13_serial(self):
        # Serial ending "212" → 172.22.112.51
        assert compute_ip_from_serial("C3531350067212") == "172.22.112.51"

    def test_serial_000(self):
        assert compute_ip_from_serial("ABC000") == "172.20.100.51"

    def test_serial_999(self):
        assert compute_ip_from_serial("XYZ999") == "172.29.199.51"

    def test_serial_555(self):
        assert compute_ip_from_serial("TEST555") == "172.25.155.51"

    def test_short_serial(self):
        assert compute_ip_from_serial("12") is None

    def test_empty_serial(self):
        assert compute_ip_from_serial("") is None

    def test_none_serial(self):
        assert compute_ip_from_serial(None) is None  # type: ignore

    def test_non_digit_ending(self):
        assert compute_ip_from_serial("ABCXYZ") is None

    def test_exactly_3_digits(self):
        assert compute_ip_from_serial("123") == "172.21.123.51"


class TestComputeSerialSuffix:
    def test_normal(self):
        assert compute_serial_suffix("C3531350067212") == "212"

    def test_short(self):
        assert compute_serial_suffix("12") is None

    def test_non_digit(self):
        assert compute_serial_suffix("ABCXYZ") is None

    def test_none(self):
        assert compute_serial_suffix(None) is None  # type: ignore


# ---------------------------------------------------------------------------
# VID:PID parsing
# ---------------------------------------------------------------------------


class TestParseVidPid:
    def test_standard_format(self):
        vid, pid = parse_vid_pid(r"USB\VID_2672&PID_0059\C3531350067212")
        assert vid == 0x2672
        assert pid == 0x0059

    def test_with_mi(self):
        vid, pid = parse_vid_pid(r"USB\VID_2672&PID_0059&MI_00\...")
        assert vid == 0x2672
        assert pid == 0x0059

    def test_lowercase(self):
        vid, pid = parse_vid_pid(r"usb\vid_2672&pid_0059\serial")
        assert vid == 0x2672
        assert pid == 0x0059

    def test_no_match(self):
        vid, pid = parse_vid_pid("random string")
        assert vid == 0
        assert pid == 0

    def test_only_vid(self):
        vid, pid = parse_vid_pid(r"USB\VID_2672\serial")
        assert vid == 0x2672
        assert pid == 0

    def test_empty_string(self):
        vid, pid = parse_vid_pid("")
        assert vid == 0
        assert pid == 0


# ---------------------------------------------------------------------------
# Serial extraction
# ---------------------------------------------------------------------------


class TestExtractSerial:
    def test_composite_device(self):
        serial = extract_serial_from_device_id(
            r"USB\VID_2672&PID_0059\C3531350067212"
        )
        assert serial == "C3531350067212"

    def test_child_interface_excluded(self):
        serial = extract_serial_from_device_id(
            r"USB\VID_2672&PID_0059&MI_00\C3531350067212"
        )
        assert serial is None

    def test_short_serial_excluded(self):
        serial = extract_serial_from_device_id(
            r"USB\VID_2672&PID_0059\SHORT"
        )
        assert serial is None

    def test_double_backslash(self):
        serial = extract_serial_from_device_id(
            "USB\\\\VID_2672&PID_0059\\\\C3531350067212"
        )
        assert serial == "C3531350067212"

    def test_too_few_parts(self):
        serial = extract_serial_from_device_id("USB")
        assert serial is None


# ---------------------------------------------------------------------------
# WMI enumeration (mocked)
# ---------------------------------------------------------------------------


class TestEnumerateViaWmi:
    """Tests for _enumerate_via_wmi_sync with mocked subprocess."""

    @pytest.fixture
    def wmi_output_single(self):
        """Simulated PowerShell output for a single GoPro device."""
        return json.dumps({
            "DeviceID": r"USB\VID_2672&PID_0059\C3531350067212",
            "Name": "GoPro Hero 13 Black",
            "Description": "GoPro Hero 13 Black",
        })

    @pytest.fixture
    def wmi_output_multi(self):
        """Simulated output with composite + child interface entries."""
        return json.dumps([
            {
                "DeviceID": r"USB\VID_2672&PID_0059\C3531350067212",
                "Name": "GoPro Hero 13 Black",
                "Description": "GoPro Hero 13 Black",
            },
            {
                "DeviceID": r"USB\VID_2672&PID_0059&MI_00\7&1234&0&0000",
                "Name": "USB Composite Device",
                "Description": "USB Composite Device",
            },
        ])

    @patch("sys.platform", "win32")
    @patch("gomaxwebcam.discovery.subprocess.run")
    def test_single_device(self, mock_run, wmi_output_single):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=wmi_output_single
        )
        devices = _enumerate_via_wmi_sync()
        assert len(devices) == 1
        assert devices[0].vendor_id == 0x2672
        assert devices[0].product_id == 0x0059
        assert devices[0].serial_number == "C3531350067212"
        assert devices[0].camera_ip == "172.22.112.51"
        assert devices[0].serial_suffix == "212"
        assert devices[0].description == "GoPro Hero 13 Black"
        assert devices[0].discovery_method == DiscoveryMethod.WMI

    @patch("sys.platform", "win32")
    @patch("gomaxwebcam.discovery.subprocess.run")
    def test_multi_entries_all_get_serial(self, mock_run, wmi_output_multi):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=wmi_output_multi
        )
        devices = _enumerate_via_wmi_sync()
        assert len(devices) == 2
        # Both should have the serial from the composite entry
        for d in devices:
            assert d.serial_number == "C3531350067212"
            assert d.camera_ip == "172.22.112.51"

    @patch("sys.platform", "win32")
    @patch("gomaxwebcam.discovery.subprocess.run")
    def test_empty_output(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        devices = _enumerate_via_wmi_sync()
        assert devices == []

    @patch("sys.platform", "win32")
    @patch("gomaxwebcam.discovery.subprocess.run")
    def test_nonzero_returncode(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        devices = _enumerate_via_wmi_sync()
        assert devices == []

    @patch("sys.platform", "win32")
    @patch("gomaxwebcam.discovery.subprocess.run")
    def test_malformed_json(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="{not valid json"
        )
        devices = _enumerate_via_wmi_sync()
        assert devices == []

    @patch("sys.platform", "win32")
    @patch("gomaxwebcam.discovery.subprocess.run")
    def test_subprocess_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="powershell", timeout=10)
        devices = _enumerate_via_wmi_sync()
        assert devices == []

    @patch("sys.platform", "win32")
    @patch("gomaxwebcam.discovery.subprocess.run")
    def test_powershell_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("powershell not found")
        devices = _enumerate_via_wmi_sync()
        assert devices == []

    @patch("sys.platform", "linux")
    def test_non_windows_returns_empty(self):
        devices = _enumerate_via_wmi_sync()
        assert devices == []


# ---------------------------------------------------------------------------
# pnputil enumeration (mocked)
# ---------------------------------------------------------------------------


class TestEnumerateViaPnputil:

    @pytest.fixture
    def pnputil_output(self):
        return (
            "Instance ID:    USB\\VID_2672&PID_0059\\C3531350067212\r\n"
            "Device Description:    GoPro Hero 13 Black\r\n"
            "Class Name:            USB\r\n"
            "\r\n"
            "Instance ID:    USB\\VID_2672&PID_0059&MI_00\\7&1234&0&0000\r\n"
            "Device Description:    USB Composite Device\r\n"
        )

    @patch("sys.platform", "win32")
    @patch("gomaxwebcam.discovery.subprocess.run")
    def test_parses_devices(self, mock_run, pnputil_output):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=pnputil_output
        )
        devices = _enumerate_via_pnputil_sync()
        assert len(devices) == 2
        assert devices[0].vendor_id == 0x2672
        assert devices[0].serial_number == "C3531350067212"
        assert devices[0].camera_ip == "172.22.112.51"
        assert devices[0].discovery_method == DiscoveryMethod.PNPUTIL

    @patch("sys.platform", "win32")
    @patch("gomaxwebcam.discovery.subprocess.run")
    def test_empty_output(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        devices = _enumerate_via_pnputil_sync()
        assert devices == []

    @patch("sys.platform", "linux")
    def test_non_windows_returns_empty(self):
        devices = _enumerate_via_pnputil_sync()
        assert devices == []


# ---------------------------------------------------------------------------
# Async API tests
# ---------------------------------------------------------------------------


class TestEnumerateAsync:

    @pytest.mark.anyio
    @patch("sys.platform", "win32")
    async def test_returns_wmi_results(self):
        device = GoProDeviceInfo(
            vendor_id=0x2672,
            product_id=0x0059,
            description="GoPro Hero 13 Black",
            serial_number="C3531350067212",
            serial_suffix="212",
            camera_ip="172.22.112.51",
            discovery_method=DiscoveryMethod.WMI,
        )
        with patch(
            "gomaxwebcam.discovery._enumerate_via_wmi_sync",
            return_value=[device],
        ):
            result = await enumerate_usb_gopro_devices()
        assert len(result) == 1
        assert result[0].serial_number == "C3531350067212"

    @pytest.mark.anyio
    @patch("sys.platform", "win32")
    async def test_falls_back_to_pnputil(self):
        device = GoProDeviceInfo(
            vendor_id=0x2672,
            product_id=0x0059,
            description="GoPro",
            discovery_method=DiscoveryMethod.PNPUTIL,
        )
        with patch(
            "gomaxwebcam.discovery._enumerate_via_wmi_sync",
            return_value=[],
        ), patch(
            "gomaxwebcam.discovery._enumerate_via_pnputil_sync",
            return_value=[device],
        ):
            result = await enumerate_usb_gopro_devices()
        assert len(result) == 1
        assert result[0].discovery_method == DiscoveryMethod.PNPUTIL

    @pytest.mark.anyio
    @patch("sys.platform", "win32")
    async def test_both_empty(self):
        with patch(
            "gomaxwebcam.discovery._enumerate_via_wmi_sync",
            return_value=[],
        ), patch(
            "gomaxwebcam.discovery._enumerate_via_pnputil_sync",
            return_value=[],
        ):
            result = await enumerate_usb_gopro_devices()
        assert result == []

    @pytest.mark.anyio
    @patch("sys.platform", "linux")
    async def test_non_windows_empty(self):
        result = await enumerate_usb_gopro_devices()
        assert result == []

    @pytest.mark.anyio
    @patch("sys.platform", "win32")
    async def test_wmi_exception_falls_through(self):
        device = GoProDeviceInfo(vendor_id=0x2672, product_id=0x0059)
        with patch(
            "gomaxwebcam.discovery._enumerate_via_wmi_sync",
            side_effect=RuntimeError("WMI boom"),
        ), patch(
            "gomaxwebcam.discovery._enumerate_via_pnputil_sync",
            return_value=[device],
        ):
            result = await enumerate_usb_gopro_devices()
        assert len(result) == 1


class TestFindGoProDevice:

    @pytest.mark.anyio
    async def test_prefers_device_with_serial(self):
        d1 = GoProDeviceInfo(product_id=0x0059, description="Interface")
        d2 = GoProDeviceInfo(
            product_id=0x0059,
            description="Composite",
            serial_number="C3531350067212",
        )
        with patch(
            "gomaxwebcam.discovery.enumerate_usb_gopro_devices",
            new_callable=AsyncMock,
            return_value=[d1, d2],
        ):
            result = await find_gopro_device()
        assert result is d2

    @pytest.mark.anyio
    async def test_returns_first_if_no_serial(self):
        d1 = GoProDeviceInfo(product_id=0x0059, description="First")
        d2 = GoProDeviceInfo(product_id=0x0043, description="Second")
        with patch(
            "gomaxwebcam.discovery.enumerate_usb_gopro_devices",
            new_callable=AsyncMock,
            return_value=[d1, d2],
        ):
            result = await find_gopro_device()
        assert result is d1

    @pytest.mark.anyio
    async def test_returns_none_when_empty(self):
        with patch(
            "gomaxwebcam.discovery.enumerate_usb_gopro_devices",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await find_gopro_device()
        assert result is None


# ---------------------------------------------------------------------------
# USBTransport.discover() integration (auto-serial from enumeration)
# ---------------------------------------------------------------------------


class TestUSBTransportDiscoverWithEnumeration:
    """Test that USBTransport.discover() uses USB enumeration to auto-detect serial."""

    @pytest.mark.anyio
    async def test_auto_serial_from_enumeration(self):
        """discover() should pick up serial suffix from USB enumeration."""
        from gomaxwebcam.transport.usb import USBTransport

        transport = USBTransport(serial=None)
        assert transport.serial is None

        device = GoProDeviceInfo(
            vendor_id=0x2672,
            product_id=0x0059,
            description="GoPro Hero 13 Black",
            serial_number="C3531350067212",
            serial_suffix="212",
            camera_ip="172.22.112.51",
        )

        mock_wired = MagicMock()
        with patch(
            "gomaxwebcam.discovery.find_gopro_device",
            new_callable=AsyncMock,
            return_value=device,
        ), patch.dict("sys.modules", {"open_gopro": MagicMock(WiredGoPro=mock_wired)}):
            result = await transport.discover()

        assert result is True
        assert transport.serial == "212"
        assert transport.stats.camera_serial == "C3531350067212"
        assert transport.stats.camera_model == "GoPro Hero 13 Black"
        assert transport.device_info is device

    @pytest.mark.anyio
    async def test_preserves_explicit_serial(self):
        """If serial was given at construction, enumeration shouldn't override it."""
        from gomaxwebcam.transport.usb import USBTransport

        transport = USBTransport(serial="999")

        device = GoProDeviceInfo(
            serial_number="C3531350067212",
            serial_suffix="212",
            camera_ip="172.22.112.51",
        )

        mock_wired = MagicMock()
        with patch(
            "gomaxwebcam.discovery.find_gopro_device",
            new_callable=AsyncMock,
            return_value=device,
        ), patch.dict("sys.modules", {"open_gopro": MagicMock(WiredGoPro=mock_wired)}):
            result = await transport.discover()

        assert result is True
        # Serial should NOT be overridden
        assert transport.serial == "999"

    @pytest.mark.anyio
    async def test_discover_works_without_enumeration(self):
        """discover() should succeed even if USB enumeration finds nothing."""
        from gomaxwebcam.transport.usb import USBTransport

        transport = USBTransport(serial="123")

        mock_wired = MagicMock()
        with patch(
            "gomaxwebcam.discovery.find_gopro_device",
            new_callable=AsyncMock,
            return_value=None,
        ), patch.dict("sys.modules", {"open_gopro": MagicMock(WiredGoPro=mock_wired)}):
            result = await transport.discover()

        assert result is True
        assert transport.device_info is None

    @pytest.mark.anyio
    async def test_discover_survives_enumeration_error(self):
        """discover() should not fail if USB enumeration throws."""
        from gomaxwebcam.transport.usb import USBTransport

        transport = USBTransport(serial="123")

        mock_wired = MagicMock()
        with patch(
            "gomaxwebcam.discovery.find_gopro_device",
            new_callable=AsyncMock,
            side_effect=RuntimeError("enumeration crash"),
        ), patch.dict("sys.modules", {"open_gopro": MagicMock(WiredGoPro=mock_wired)}):
            result = await transport.discover()

        assert result is True  # Should still succeed via SDK
