"""
Tests for USB device enumeration and GoPro discovery.

These tests mock system calls (WMI, pnputil, ipconfig) so they run
on any machine -- no GoPro hardware needed.
"""

import json
import socket
import struct
import subprocess
import unittest
from unittest.mock import patch, MagicMock, call

from src.discovery import (
    GOPRO_VENDOR_ID,
    GOPRO_KNOWN_PIDS,
    GOPRO_SECOND_OCTET_RANGE,
    GOPRO_THIRD_OCTET_RANGE,
    GOPRO_CAMERA_HOST,
    GOPRO_CANDIDATE_HOSTS,
    GOPRO_API_PORT,
    GOPRO_STATUS_ENDPOINT,
    IP_RANGE_SCAN_TIMEOUT,
    IP_RANGE_SCAN_MAX_WORKERS,
    GOPRO_MDNS_SERVICE_TYPE,
    GOPRO_MDNS_ALT_SERVICE_TYPES,
    MDNS_PORT,
    MDNS_ADDR,
    DiscoveryMethod,
    GoProDevice,
    _parse_vid_pid,
    _enumerate_via_wmi,
    _enumerate_via_pnputil,
    _discover_via_ipconfig,
    _discover_via_ip_range_scan,
    _discover_via_mdns,
    _build_mdns_query,
    _parse_mdns_response,
    _skip_dns_name,
    _generate_gopro_candidate_ips,
    _probe_gopro_http,
    _verify_gopro_endpoint,
    enumerate_usb_gopro_devices,
    discover_gopro_ip,
    discover_gopro_ip_chain,
    full_discovery,
)

import pytest

pytestmark = pytest.mark.no_gopro_needed


class TestParseVidPid(unittest.TestCase):
    """Test VID/PID extraction from USB device ID strings."""

    def test_standard_format(self):
        vid, pid = _parse_vid_pid(r"USB\VID_0A70&PID_000D\5&3A2A7C5E&0&1")
        self.assertEqual(vid, 0x0A70)
        self.assertEqual(pid, 0x000D)

    def test_lowercase(self):
        vid, pid = _parse_vid_pid(r"usb\vid_0a70&pid_000d\serial123")
        self.assertEqual(vid, 0x0A70)
        self.assertEqual(pid, 0x000D)

    def test_no_match(self):
        vid, pid = _parse_vid_pid("some random string")
        self.assertEqual(vid, 0)
        self.assertEqual(pid, 0)

    def test_vid_only(self):
        vid, pid = _parse_vid_pid(r"USB\VID_0A70\stuff")
        self.assertEqual(vid, 0x0A70)
        self.assertEqual(pid, 0)

    def test_alternate_pid(self):
        vid, pid = _parse_vid_pid(r"USB\VID_0A70&PID_0011\xxx")
        self.assertEqual(vid, 0x0A70)
        self.assertEqual(pid, 0x0011)


class TestWmiEnumeration(unittest.TestCase):
    """Test GoPro detection via WMI/PowerShell."""

    @patch("src.discovery.subprocess.run")
    def test_single_gopro_found(self, mock_run):
        """WMI returns a single GoPro device."""
        wmi_output = json.dumps({
            "DeviceID": r"USB\VID_0A70&PID_000D\5&3A2A7C5E&0&1",
            "Name": "GoPro Hero 12 RNDIS",
            "Description": "Remote NDIS Compatible Device",
        })
        mock_run.return_value = MagicMock(returncode=0, stdout=wmi_output)

        devices = _enumerate_via_wmi()

        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].vendor_id, 0x0A70)
        self.assertEqual(devices[0].product_id, 0x000D)
        self.assertIn("GoPro", devices[0].description)

    @patch("src.discovery.subprocess.run")
    def test_multiple_gopro_entries(self, mock_run):
        """WMI returns multiple GoPro-related entries (composite device)."""
        wmi_output = json.dumps([
            {
                "DeviceID": r"USB\VID_0A70&PID_000D\serial1",
                "Name": "GoPro RNDIS",
                "Description": "RNDIS device",
            },
            {
                "DeviceID": r"USB\VID_0A70&PID_0043\serial2",
                "Name": "GoPro MTP",
                "Description": "MTP device",
            },
        ])
        mock_run.return_value = MagicMock(returncode=0, stdout=wmi_output)

        devices = _enumerate_via_wmi()
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0].product_id, 0x000D)
        self.assertEqual(devices[1].product_id, 0x0043)

    @patch("src.discovery.subprocess.run")
    def test_no_gopro(self, mock_run):
        """WMI returns empty output (no GoPro connected)."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        devices = _enumerate_via_wmi()
        self.assertEqual(devices, [])

    @patch("src.discovery.subprocess.run")
    def test_powershell_failure(self, mock_run):
        """PowerShell command fails."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")

        devices = _enumerate_via_wmi()
        self.assertEqual(devices, [])

    @patch("src.discovery.subprocess.run")
    def test_invalid_json(self, mock_run):
        """WMI returns invalid JSON."""
        mock_run.return_value = MagicMock(returncode=0, stdout="not json {{{")

        devices = _enumerate_via_wmi()
        self.assertEqual(devices, [])


class TestPnputilEnumeration(unittest.TestCase):
    """Test GoPro detection via pnputil fallback."""

    @patch("src.discovery.subprocess.run")
    def test_gopro_found(self, mock_run):
        pnputil_output = """Instance ID:    USB\\VID_0A70&PID_000D\\5&3A2A7C5E
Device Description:    Remote NDIS Compatible Device
Class Name:    Net
Status:        Started

Instance ID:    USB\\VID_1234&PID_5678\\other
Device Description:    Some Other Device
Class Name:    USB
Status:        Started

"""
        mock_run.return_value = MagicMock(returncode=0, stdout=pnputil_output)

        devices = _enumerate_via_pnputil()
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].vendor_id, 0x0A70)
        self.assertEqual(devices[0].product_id, 0x000D)

    @patch("src.discovery.subprocess.run")
    def test_no_gopro(self, mock_run):
        pnputil_output = """Instance ID:    USB\\VID_1234&PID_5678\\other
Device Description:    Some Other Device
Class Name:    USB
Status:        Started

"""
        mock_run.return_value = MagicMock(returncode=0, stdout=pnputil_output)

        devices = _enumerate_via_pnputil()
        self.assertEqual(devices, [])


class TestIpconfigDiscovery(unittest.TestCase):
    """Test GoPro IP discovery from ipconfig output."""

    @patch("src.discovery.subprocess.run")
    def test_gopro_adapter_found(self, mock_run):
        ipconfig_output = """
Ethernet adapter Ethernet:

   IPv4 Address. . . . . . . . . . . : 192.168.1.100
   Subnet Mask . . . . . . . . . . . : 255.255.255.0
   Default Gateway . . . . . . . . . : 192.168.1.1

Ethernet adapter GoPro Network:

   IPv4 Address. . . . . . . . . . . : 172.20.145.50
   Subnet Mask . . . . . . . . . . . : 255.255.255.0
   Default Gateway . . . . . . . . . : 172.20.145.51

Wireless LAN adapter Wi-Fi:

   IPv4 Address. . . . . . . . . . . : 192.168.0.5
"""
        mock_run.return_value = MagicMock(returncode=0, stdout=ipconfig_output)

        ip = _discover_via_ipconfig()
        self.assertEqual(ip, "172.20.145.51")

    @patch("src.discovery.subprocess.run")
    def test_rndis_adapter_found(self, mock_run):
        """Detect via 'Remote NDIS' adapter name."""
        ipconfig_output = """
Ethernet adapter Remote NDIS based Internet Sharing Device:

   IPv4 Address. . . . . . . . . . . : 172.25.110.50
   Subnet Mask . . . . . . . . . . . : 255.255.255.0
   Default Gateway . . . . . . . . . : 172.25.110.51
"""
        mock_run.return_value = MagicMock(returncode=0, stdout=ipconfig_output)

        ip = _discover_via_ipconfig()
        self.assertEqual(ip, "172.25.110.51")

    @patch("src.discovery.subprocess.run")
    def test_no_gopro_adapter(self, mock_run):
        ipconfig_output = """
Ethernet adapter Ethernet:

   IPv4 Address. . . . . . . . . . . : 192.168.1.100
   Default Gateway . . . . . . . . . : 192.168.1.1
"""
        mock_run.return_value = MagicMock(returncode=0, stdout=ipconfig_output)

        ip = _discover_via_ipconfig()
        self.assertIsNone(ip)


class TestProbeGoPro(unittest.TestCase):
    """Test TCP probe for GoPro HTTP API."""

    @patch("src.discovery.socket.socket")
    def test_probe_success(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 0
        mock_socket_cls.return_value = mock_sock

        result = _probe_gopro_http("172.20.145.51")
        self.assertTrue(result)
        mock_sock.connect_ex.assert_called_once_with(("172.20.145.51", 8080))

    @patch("src.discovery.socket.socket")
    def test_probe_failure(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 1  # Connection refused
        mock_socket_cls.return_value = mock_sock

        result = _probe_gopro_http("172.20.145.51")
        self.assertFalse(result)

    @patch("src.discovery.socket.socket")
    def test_probe_timeout(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.connect_ex.side_effect = socket.timeout("timed out")
        mock_socket_cls.return_value = mock_sock

        result = _probe_gopro_http("172.20.145.51")
        self.assertFalse(result)


# ============================================================================
# mDNS tests
# ============================================================================

class TestBuildMdnsQuery(unittest.TestCase):
    """Test mDNS query packet construction."""

    def test_query_is_bytes(self):
        """Query should be a bytes object."""
        query = _build_mdns_query("_gopro-web._tcp.local.")
        self.assertIsInstance(query, bytes)

    def test_query_has_dns_header(self):
        """Query should start with 12-byte DNS header."""
        query = _build_mdns_query("_gopro-web._tcp.local.")
        self.assertGreaterEqual(len(query), 12)
        # ID=0, flags=0, QDCOUNT=1, ANCOUNT=0, NSCOUNT=0, ARCOUNT=0
        _id, flags, qdcount, ancount, nscount, arcount = struct.unpack(
            "!HHHHHH", query[:12]
        )
        self.assertEqual(_id, 0)
        self.assertEqual(flags, 0)
        self.assertEqual(qdcount, 1)
        self.assertEqual(ancount, 0)

    def test_query_contains_service_name(self):
        """Query should contain the encoded service name labels."""
        query = _build_mdns_query("_gopro-web._tcp.local.")
        # The label "_gopro-web" should appear in the packet
        self.assertIn(b"_gopro-web", query)
        self.assertIn(b"_tcp", query)
        self.assertIn(b"local", query)

    def test_query_ends_with_ptr_type(self):
        """Query should end with QTYPE=PTR(12), QCLASS=IN with unicast bit."""
        query = _build_mdns_query("_gopro-web._tcp.local.")
        # Last 4 bytes: QTYPE + QCLASS
        qtype, qclass = struct.unpack("!HH", query[-4:])
        self.assertEqual(qtype, 12)  # PTR
        self.assertEqual(qclass, 0x8001)  # IN with unicast response bit


class TestParseMdnsResponse(unittest.TestCase):
    """Test mDNS response parsing."""

    def _make_a_record_response(self, ip_str: str) -> bytes:
        """Build a minimal mDNS response with one A record."""
        ip_bytes = socket.inet_aton(ip_str)

        # Header: ID=0, flags=0x8400 (response), QDCOUNT=0, ANCOUNT=1
        header = struct.pack("!HHHHHH", 0, 0x8400, 0, 1, 0, 0)

        # Name: simple label "gopro" + root
        name = bytes([5]) + b"gopro" + b"\x00"

        # A record: type=1, class=IN(1), TTL=120, RDLENGTH=4
        record = struct.pack("!HHIH", 1, 1, 120, 4) + ip_bytes

        return header + name + record

    def test_extracts_gopro_ip(self):
        """Should extract 172.x.x.x A record IPs."""
        data = self._make_a_record_response("172.20.145.51")
        ips = _parse_mdns_response(data)
        self.assertEqual(ips, ["172.20.145.51"])

    def test_ignores_non_gopro_ip(self):
        """Should ignore IPs outside 172.x.x.x range."""
        data = self._make_a_record_response("192.168.1.100")
        ips = _parse_mdns_response(data)
        self.assertEqual(ips, [])

    def test_handles_empty_data(self):
        """Should return empty for too-short data."""
        ips = _parse_mdns_response(b"")
        self.assertEqual(ips, [])

    def test_handles_truncated_data(self):
        """Should return empty for data shorter than DNS header."""
        ips = _parse_mdns_response(b"\x00" * 5)
        self.assertEqual(ips, [])

    def test_multiple_a_records(self):
        """Should extract multiple A records."""
        ip1 = socket.inet_aton("172.20.145.51")
        ip2 = socket.inet_aton("172.25.110.51")

        # Header with 2 answers
        header = struct.pack("!HHHHHH", 0, 0x8400, 0, 2, 0, 0)

        # Record 1
        name1 = bytes([5]) + b"gopro" + b"\x00"
        rec1 = struct.pack("!HHIH", 1, 1, 120, 4) + ip1

        # Record 2
        name2 = bytes([5]) + b"gopro" + b"\x00"
        rec2 = struct.pack("!HHIH", 1, 1, 120, 4) + ip2

        data = header + name1 + rec1 + name2 + rec2
        ips = _parse_mdns_response(data)
        self.assertIn("172.20.145.51", ips)
        self.assertIn("172.25.110.51", ips)


class TestSkipDnsName(unittest.TestCase):
    """Test DNS name skipping in mDNS packets."""

    def test_skip_simple_label(self):
        """Skip a simple one-label name ending with root."""
        # "test" + root
        data = bytes([4]) + b"test" + b"\x00"
        offset = _skip_dns_name(data, 0)
        self.assertEqual(offset, 6)  # 1(len) + 4(test) + 1(root)

    def test_skip_multi_label(self):
        """Skip a multi-label name like _gopro-web._tcp.local."""
        data = bytes([10]) + b"_gopro-web" + bytes([4]) + b"_tcp" + bytes([5]) + b"local" + b"\x00"
        offset = _skip_dns_name(data, 0)
        # 1+10 + 1+4 + 1+5 + 1 = 23
        self.assertEqual(offset, 23)

    def test_skip_pointer(self):
        """Skip a DNS pointer (compression)."""
        # Pointer: 0xC0 0x0C (points to offset 12)
        data = b"\xC0\x0C"
        offset = _skip_dns_name(data, 0)
        self.assertEqual(offset, 2)

    def test_empty_data_returns_none(self):
        """Empty data returns None."""
        result = _skip_dns_name(b"", 0)
        self.assertIsNone(result)

    def test_offset_beyond_data_returns_none(self):
        """Offset past data returns None."""
        result = _skip_dns_name(b"\x00", 5)
        self.assertIsNone(result)


class TestDiscoverViaMdns(unittest.TestCase):
    """Test mDNS discovery function."""

    @patch("src.discovery._probe_gopro_http")
    @patch("src.discovery.socket.socket")
    def test_mdns_finds_gopro(self, mock_socket_cls, mock_probe):
        """mDNS should find GoPro when response contains valid A record."""
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        # Build a fake mDNS response with GoPro IP
        ip_bytes = socket.inet_aton("172.20.145.51")
        header = struct.pack("!HHHHHH", 0, 0x8400, 0, 1, 0, 0)
        name = bytes([5]) + b"gopro" + b"\x00"
        record = struct.pack("!HHIH", 1, 1, 120, 4) + ip_bytes
        response_data = header + name + record

        mock_sock.recvfrom.side_effect = [
            (response_data, ("172.20.145.51", 5353)),
            socket.timeout("done"),
        ]
        mock_probe.return_value = True

        result = _discover_via_mdns(timeout=0.1)
        self.assertEqual(result, "172.20.145.51")

    @patch("src.discovery.socket.socket")
    def test_mdns_timeout_returns_none(self, mock_socket_cls):
        """mDNS returns None when no response before timeout."""
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.recvfrom.side_effect = socket.timeout("timed out")

        result = _discover_via_mdns(timeout=0.1)
        self.assertIsNone(result)

    @patch("src.discovery.socket.socket")
    def test_mdns_socket_error_returns_none(self, mock_socket_cls):
        """mDNS returns None on socket error."""
        mock_socket_cls.side_effect = OSError("no network")

        result = _discover_via_mdns(timeout=0.1)
        self.assertIsNone(result)

    @patch("src.discovery._probe_gopro_http")
    @patch("src.discovery.socket.socket")
    def test_mdns_probe_failure_continues(self, mock_socket_cls, mock_probe):
        """mDNS should continue listening if probe fails on candidate IP."""
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        # Build response
        ip_bytes = socket.inet_aton("172.20.145.51")
        header = struct.pack("!HHHHHH", 0, 0x8400, 0, 1, 0, 0)
        name = bytes([5]) + b"gopro" + b"\x00"
        record = struct.pack("!HHIH", 1, 1, 120, 4) + ip_bytes
        response_data = header + name + record

        mock_sock.recvfrom.side_effect = [
            (response_data, ("172.20.145.51", 5353)),
            socket.timeout("done"),
        ]
        mock_probe.return_value = False  # Probe fails

        result = _discover_via_mdns(timeout=0.1)
        self.assertIsNone(result)


# ============================================================================
# Fallback Chain Orchestrator tests
# ============================================================================

class TestDiscoverGoProIpChain(unittest.TestCase):
    """Test the fallback chain orchestrator."""

    @patch("src.discovery._discover_via_mdns")
    @patch("src.discovery._discover_via_ip_range_scan")
    @patch("src.discovery.discover_gopro_ip")
    def test_usb_scan_succeeds_first(self, mock_usb, mock_ip_scan, mock_mdns):
        """When USB scan succeeds, IP scan and mDNS are not called."""
        mock_usb.return_value = "172.20.145.51"

        ip, method = discover_gopro_ip_chain()

        self.assertEqual(ip, "172.20.145.51")
        self.assertEqual(method, DiscoveryMethod.USB_SCAN)
        mock_ip_scan.assert_not_called()
        mock_mdns.assert_not_called()

    @patch("src.discovery._discover_via_mdns")
    @patch("src.discovery._discover_via_ip_range_scan")
    @patch("src.discovery.discover_gopro_ip")
    def test_ip_scan_succeeds_after_usb_fails(self, mock_usb, mock_ip_scan, mock_mdns):
        """When USB scan fails, IP scan is tried and succeeds."""
        mock_usb.return_value = None
        mock_ip_scan.return_value = "172.22.150.51"

        ip, method = discover_gopro_ip_chain()

        self.assertEqual(ip, "172.22.150.51")
        self.assertEqual(method, DiscoveryMethod.IP_SCAN)
        mock_mdns.assert_not_called()

    @patch("src.discovery._discover_via_mdns")
    @patch("src.discovery._discover_via_ip_range_scan")
    @patch("src.discovery.discover_gopro_ip")
    def test_mdns_succeeds_after_both_fail(self, mock_usb, mock_ip_scan, mock_mdns):
        """When USB and IP scans fail, mDNS is tried and succeeds."""
        mock_usb.return_value = None
        mock_ip_scan.return_value = None
        mock_mdns.return_value = "172.28.180.51"

        ip, method = discover_gopro_ip_chain()

        self.assertEqual(ip, "172.28.180.51")
        self.assertEqual(method, DiscoveryMethod.MDNS)

    @patch("src.discovery._discover_via_mdns")
    @patch("src.discovery._discover_via_ip_range_scan")
    @patch("src.discovery.discover_gopro_ip")
    def test_all_methods_fail_returns_none(self, mock_usb, mock_ip_scan, mock_mdns):
        """When all 3 methods fail, returns (None, None)."""
        mock_usb.return_value = None
        mock_ip_scan.return_value = None
        mock_mdns.return_value = None

        ip, method = discover_gopro_ip_chain()

        self.assertIsNone(ip)
        self.assertIsNone(method)

    @patch("src.discovery._discover_via_mdns")
    @patch("src.discovery._discover_via_ip_range_scan")
    @patch("src.discovery.discover_gopro_ip")
    def test_usb_scan_exception_continues_to_ip_scan(self, mock_usb, mock_ip_scan, mock_mdns):
        """Exception in USB scan should not prevent IP scan from running."""
        mock_usb.side_effect = RuntimeError("ipconfig crashed")
        mock_ip_scan.return_value = "172.22.150.51"

        ip, method = discover_gopro_ip_chain()

        self.assertEqual(ip, "172.22.150.51")
        self.assertEqual(method, DiscoveryMethod.IP_SCAN)

    @patch("src.discovery._discover_via_mdns")
    @patch("src.discovery._discover_via_ip_range_scan")
    @patch("src.discovery.discover_gopro_ip")
    def test_ip_scan_exception_continues_to_mdns(self, mock_usb, mock_ip_scan, mock_mdns):
        """Exception in IP scan should not prevent mDNS from running."""
        mock_usb.return_value = None
        mock_ip_scan.side_effect = OSError("thread pool error")
        mock_mdns.return_value = "172.28.180.51"

        ip, method = discover_gopro_ip_chain()

        self.assertEqual(ip, "172.28.180.51")
        self.assertEqual(method, DiscoveryMethod.MDNS)

    @patch("src.discovery._discover_via_mdns")
    @patch("src.discovery._discover_via_ip_range_scan")
    @patch("src.discovery.discover_gopro_ip")
    def test_all_methods_raise_returns_none(self, mock_usb, mock_ip_scan, mock_mdns):
        """When all methods raise exceptions, returns (None, None)."""
        mock_usb.side_effect = RuntimeError("fail 1")
        mock_ip_scan.side_effect = RuntimeError("fail 2")
        mock_mdns.side_effect = RuntimeError("fail 3")

        ip, method = discover_gopro_ip_chain()

        self.assertIsNone(ip)
        self.assertIsNone(method)

    @patch("src.discovery._discover_via_mdns")
    @patch("src.discovery._discover_via_ip_range_scan")
    @patch("src.discovery.discover_gopro_ip")
    def test_custom_timeouts_passed(self, mock_usb, mock_ip_scan, mock_mdns):
        """Custom timeouts should be passed to underlying methods."""
        mock_usb.return_value = None
        mock_ip_scan.return_value = None
        mock_mdns.return_value = None

        discover_gopro_ip_chain(
            probe_timeout=1.5,
            mdns_timeout=5.0,
            ip_scan_max_workers=10,
        )

        mock_ip_scan.assert_called_once_with(timeout=1.5, max_workers=10)
        mock_mdns.assert_called_once_with(timeout=5.0)


# ============================================================================
# Full Discovery tests (updated for chain orchestrator)
# ============================================================================

class TestFullDiscovery(unittest.TestCase):
    """Test the complete discovery flow with fallback chain."""

    @patch("src.discovery.discover_gopro_ip_chain")
    @patch("src.discovery.enumerate_usb_gopro_devices")
    def test_full_success_usb_scan(self, mock_enum, mock_chain):
        """USB device found + USB scan resolves IP."""
        mock_enum.return_value = [GoProDevice(
            vendor_id=0x0A70,
            product_id=0x000D,
            description="GoPro Hero 12 RNDIS",
        )]
        mock_chain.return_value = ("172.20.145.51", DiscoveryMethod.USB_SCAN)

        device = full_discovery()

        self.assertIsNotNone(device)
        self.assertEqual(device.vendor_id, 0x0A70)
        self.assertEqual(device.camera_ip, "172.20.145.51")
        self.assertEqual(device.discovery_method, DiscoveryMethod.USB_SCAN)

    @patch("src.discovery.discover_gopro_ip_chain")
    @patch("src.discovery.enumerate_usb_gopro_devices")
    def test_full_success_ip_scan(self, mock_enum, mock_chain):
        """USB device found + IP scan resolves IP (usb scan failed)."""
        mock_enum.return_value = [GoProDevice(
            vendor_id=0x0A70,
            product_id=0x000D,
            description="GoPro Hero 12",
        )]
        mock_chain.return_value = ("172.22.150.51", DiscoveryMethod.IP_SCAN)

        device = full_discovery()

        self.assertIsNotNone(device)
        self.assertEqual(device.camera_ip, "172.22.150.51")
        self.assertEqual(device.discovery_method, DiscoveryMethod.IP_SCAN)

    @patch("src.discovery.discover_gopro_ip_chain")
    @patch("src.discovery.enumerate_usb_gopro_devices")
    def test_full_success_mdns(self, mock_enum, mock_chain):
        """USB device found + mDNS resolves IP."""
        mock_enum.return_value = [GoProDevice(
            vendor_id=0x0A70,
            product_id=0x000D,
            description="GoPro Hero 12",
        )]
        mock_chain.return_value = ("172.28.180.51", DiscoveryMethod.MDNS)

        device = full_discovery()

        self.assertIsNotNone(device)
        self.assertEqual(device.camera_ip, "172.28.180.51")
        self.assertEqual(device.discovery_method, DiscoveryMethod.MDNS)

    @patch("src.discovery.discover_gopro_ip_chain")
    @patch("src.discovery.enumerate_usb_gopro_devices")
    def test_usb_found_no_ip(self, mock_enum, mock_chain):
        """USB device found but all IP methods fail."""
        mock_enum.return_value = [GoProDevice(
            vendor_id=0x0A70,
            product_id=0x000D,
            description="GoPro Hero 12",
        )]
        mock_chain.return_value = (None, None)

        device = full_discovery()

        # Should still return the device (USB confirmed) but no IP
        self.assertIsNotNone(device)
        self.assertEqual(device.vendor_id, 0x0A70)
        self.assertIsNone(device.camera_ip)

    @patch("src.discovery.discover_gopro_ip_chain")
    @patch("src.discovery.enumerate_usb_gopro_devices")
    def test_no_usb_but_chain_finds_ip(self, mock_enum, mock_chain):
        """No USB device found, but IP chain still finds the camera."""
        mock_enum.return_value = []
        mock_chain.return_value = ("172.20.145.51", DiscoveryMethod.IP_SCAN)

        device = full_discovery()

        # Should create a device from network discovery
        self.assertIsNotNone(device)
        self.assertEqual(device.camera_ip, "172.20.145.51")
        self.assertEqual(device.vendor_id, GOPRO_VENDOR_ID)
        self.assertEqual(device.product_id, 0)
        self.assertIn("network", device.description.lower())
        self.assertEqual(device.discovery_method, DiscoveryMethod.IP_SCAN)

    @patch("src.discovery.discover_gopro_ip_chain")
    @patch("src.discovery.enumerate_usb_gopro_devices")
    def test_no_usb_no_ip(self, mock_enum, mock_chain):
        """No USB device, no IP found = None."""
        mock_enum.return_value = []
        mock_chain.return_value = (None, None)

        device = full_discovery()
        self.assertIsNone(device)

    def test_gopro_device_str(self):
        """Test GoProDevice string representation."""
        device = GoProDevice(
            vendor_id=0x0A70,
            product_id=0x000D,
            description="GoPro Hero 12",
            camera_ip="172.20.145.51",
        )
        s = str(device)
        self.assertIn("0A70", s)
        self.assertIn("000D", s)
        self.assertIn("172.20.145.51", s)

    def test_gopro_device_str_with_method(self):
        """String representation includes discovery method."""
        device = GoProDevice(
            vendor_id=0x0A70,
            product_id=0x000D,
            description="GoPro Hero 12",
            camera_ip="172.20.145.51",
            discovery_method=DiscoveryMethod.USB_SCAN,
        )
        s = str(device)
        self.assertIn("usb_scan", s)

    def test_gopro_device_usb_id_str(self):
        device = GoProDevice(vendor_id=0x0A70, product_id=0x000D, description="test")
        self.assertEqual(device.usb_id_str, "0A70:000D")


class TestDiscoveryMethod(unittest.TestCase):
    """Test DiscoveryMethod enum."""

    def test_values(self):
        self.assertEqual(DiscoveryMethod.USB_SCAN.value, "usb_scan")
        self.assertEqual(DiscoveryMethod.IP_SCAN.value, "ip_scan")
        self.assertEqual(DiscoveryMethod.MDNS.value, "mdns")

    def test_all_methods_present(self):
        methods = list(DiscoveryMethod)
        self.assertEqual(len(methods), 3)


# ============================================================================
# Existing tests kept (USB enumeration, constants, IP range scan)
# ============================================================================

class TestEnumerateUsbGoProDevices(unittest.TestCase):
    """Test the top-level enumerate_usb_gopro_devices fallback chain."""

    @patch("src.discovery._enumerate_via_pnputil")
    @patch("src.discovery._enumerate_via_wmi")
    def test_wmi_success_skips_pnputil(self, mock_wmi, mock_pnp):
        mock_wmi.return_value = [GoProDevice(
            vendor_id=0x0A70, product_id=0x000D, description="GoPro via WMI",
        )]
        result = enumerate_usb_gopro_devices()
        self.assertEqual(len(result), 1)
        mock_pnp.assert_not_called()

    @patch("src.discovery._enumerate_via_pnputil")
    @patch("src.discovery._enumerate_via_wmi")
    def test_wmi_empty_falls_back_to_pnputil(self, mock_wmi, mock_pnp):
        mock_wmi.return_value = []
        mock_pnp.return_value = [GoProDevice(
            vendor_id=0x0A70, product_id=0x0011, description="GoPro via pnputil",
        )]
        result = enumerate_usb_gopro_devices()
        self.assertEqual(len(result), 1)

    @patch("src.discovery._enumerate_via_pnputil")
    @patch("src.discovery._enumerate_via_wmi")
    def test_both_methods_fail_returns_empty(self, mock_wmi, mock_pnp):
        mock_wmi.side_effect = Exception("WMI failed")
        mock_pnp.side_effect = Exception("pnputil failed")
        result = enumerate_usb_gopro_devices()
        self.assertEqual(result, [])


class TestConstants(unittest.TestCase):
    """Verify critical constants match the spec."""

    def test_vendor_id(self):
        self.assertEqual(GOPRO_VENDOR_ID, 0x2672)
        self.assertEqual(GOPRO_VENDOR_ID, 9842)

    def test_vendor_ids_set_includes_both_known_vids(self):
        """GOPRO_VENDOR_IDS must include both 0x2672 (Hero 12+) and 0x0A70 (older)."""
        from src.discovery import GOPRO_VENDOR_IDS
        self.assertIn(0x2672, GOPRO_VENDOR_IDS)
        self.assertIn(0x0A70, GOPRO_VENDOR_IDS)
        self.assertEqual(len(GOPRO_VENDOR_IDS), 2)

    def test_known_pids_is_set(self):
        self.assertIsInstance(GOPRO_KNOWN_PIDS, set)
        self.assertTrue(len(GOPRO_KNOWN_PIDS) > 0)

    def test_mdns_service_type(self):
        self.assertEqual(GOPRO_MDNS_SERVICE_TYPE, "_gopro-web._tcp.local.")

    def test_mdns_port(self):
        self.assertEqual(MDNS_PORT, 5353)

    def test_mdns_addr(self):
        self.assertEqual(MDNS_ADDR, "224.0.0.251")

    def test_gopro_api_port(self):
        """GoPro HTTP API port is 8080."""
        self.assertEqual(GOPRO_API_PORT, 8080)

    def test_gopro_status_endpoint(self):
        """Status endpoint matches Open GoPro API."""
        self.assertEqual(GOPRO_STATUS_ENDPOINT, "/gopro/webcam/status")

    def test_candidate_hosts_includes_51(self):
        """Candidate hosts includes the primary .51 address."""
        self.assertIn(51, GOPRO_CANDIDATE_HOSTS)

    def test_candidate_hosts_51_is_first(self):
        """Primary host .51 is first in the list for priority probing."""
        self.assertEqual(GOPRO_CANDIDATE_HOSTS[0], 51)

    def test_candidate_hosts_are_valid_octets(self):
        """All candidate host octets are valid (1-254)."""
        for host in GOPRO_CANDIDATE_HOSTS:
            self.assertGreaterEqual(host, 1)
            self.assertLessEqual(host, 254)

    def test_ip_range_scan_covers_expected_subnets(self):
        """Second octet 20-29 and third octet 100-199 = 1000 /24 subnets."""
        subnet_count = len(GOPRO_SECOND_OCTET_RANGE) * len(GOPRO_THIRD_OCTET_RANGE)
        self.assertEqual(subnet_count, 1000)


class TestGenerateGoProCandidateIps(unittest.TestCase):
    """Test candidate IP generation for range scan."""

    def test_generates_correct_count_with_default_hosts(self):
        """Default generates IPs for all candidate hosts per subnet."""
        candidates = _generate_gopro_candidate_ips()
        expected = (
            len(GOPRO_SECOND_OCTET_RANGE)
            * len(GOPRO_THIRD_OCTET_RANGE)
            * len(GOPRO_CANDIDATE_HOSTS)
        )
        self.assertEqual(len(candidates), expected)
        # 10 * 100 * 3 = 3000 with default [51, 1, 101]
        self.assertEqual(len(candidates), 3000)

    def test_generates_correct_count_single_host(self):
        """With a single host octet, generates one IP per /24 subnet."""
        candidates = _generate_gopro_candidate_ips(host_octets=[51])
        expected = len(GOPRO_SECOND_OCTET_RANGE) * len(GOPRO_THIRD_OCTET_RANGE)
        self.assertEqual(len(candidates), expected)
        self.assertEqual(len(candidates), 1000)

    def test_all_ips_in_correct_range(self):
        candidates = _generate_gopro_candidate_ips()
        for ip in candidates:
            parts = ip.split(".")
            self.assertEqual(parts[0], "172")
            self.assertIn(int(parts[1]), GOPRO_SECOND_OCTET_RANGE)
            self.assertIn(int(parts[2]), GOPRO_THIRD_OCTET_RANGE)
            self.assertIn(int(parts[3]), GOPRO_CANDIDATE_HOSTS)

    def test_primary_host_listed_first_per_subnet(self):
        """The primary host (.51) should be first for each subnet."""
        candidates = _generate_gopro_candidate_ips()
        # First IP should be 172.20.100.51 (primary host)
        self.assertEqual(candidates[0], "172.20.100.51")
        # Next should be .1 and .101 for the same subnet
        self.assertEqual(candidates[1], "172.20.100.1")
        self.assertEqual(candidates[2], "172.20.100.101")
        # Then next subnet starts with .51
        self.assertEqual(candidates[3], "172.20.101.51")

    def test_first_and_last_ip(self):
        candidates = _generate_gopro_candidate_ips()
        self.assertEqual(candidates[0], "172.20.100.51")
        # Last: 172.29.199.{last host in GOPRO_CANDIDATE_HOSTS}
        self.assertEqual(candidates[-1], f"172.29.199.{GOPRO_CANDIDATE_HOSTS[-1]}")

    def test_no_duplicates(self):
        candidates = _generate_gopro_candidate_ips()
        self.assertEqual(len(candidates), len(set(candidates)))

    def test_custom_host_octets(self):
        """Custom host octets override the default list."""
        candidates = _generate_gopro_candidate_ips(host_octets=[42, 99])
        for ip in candidates:
            parts = ip.split(".")
            self.assertIn(int(parts[3]), [42, 99])
        expected = len(GOPRO_SECOND_OCTET_RANGE) * len(GOPRO_THIRD_OCTET_RANGE) * 2
        self.assertEqual(len(candidates), expected)


class TestIpRangeScan(unittest.TestCase):
    """Test IP range scan discovery method."""

    @patch("src.discovery._verify_gopro_endpoint", return_value=True)
    @patch("src.discovery._probe_gopro_http")
    def test_finds_gopro_in_range(self, mock_probe, mock_verify):
        """IP range scan finds a GoPro at the target IP."""
        target_ip = "172.22.145.51"

        def probe_side_effect(ip, port=8080, timeout=2.0):
            return ip == target_ip

        mock_probe.side_effect = probe_side_effect

        result = _discover_via_ip_range_scan(timeout=0.1, max_workers=10)
        self.assertEqual(result, target_ip)

    @patch("src.discovery._probe_gopro_http")
    def test_no_gopro_found(self, mock_probe):
        """Returns None when no IP responds on port 8080."""
        mock_probe.return_value = False
        result = _discover_via_ip_range_scan(timeout=0.1, max_workers=10)
        self.assertIsNone(result)

    @patch("src.discovery._probe_gopro_http")
    def test_probe_exception_handled(self, mock_probe):
        """Exceptions from probe are caught gracefully."""
        mock_probe.side_effect = OSError("Network unreachable")
        result = _discover_via_ip_range_scan(timeout=0.1, max_workers=5)
        self.assertIsNone(result)

    @patch("src.discovery._verify_gopro_endpoint", return_value=True)
    @patch("src.discovery._probe_gopro_http")
    def test_finds_non_standard_host(self, mock_probe, mock_verify):
        """IP range scan finds a GoPro at a non-.51 host address."""
        target_ip = "172.25.150.1"  # Non-standard .1 host

        def probe_side_effect(ip, port=8080, timeout=2.0):
            return ip == target_ip

        mock_probe.side_effect = probe_side_effect

        result = _discover_via_ip_range_scan(timeout=0.1, max_workers=10)
        self.assertEqual(result, target_ip)

    @patch("src.discovery._verify_gopro_endpoint")
    @patch("src.discovery._probe_gopro_http")
    def test_verify_rejects_non_gopro(self, mock_probe, mock_verify):
        """TCP-open but non-GoPro IPs are rejected by HTTP verification."""
        # TCP probe succeeds for all IPs, but verify rejects them
        mock_probe.return_value = True
        mock_verify.return_value = False

        result = _discover_via_ip_range_scan(timeout=0.1, max_workers=5)
        self.assertIsNone(result)
        # verify should have been called for at least some probed IPs
        self.assertTrue(mock_verify.called)

    @patch("src.discovery._probe_gopro_http")
    def test_skip_verification(self, mock_probe):
        """When verify_endpoint=False, skips HTTP verification."""
        target_ip = "172.22.145.51"

        def probe_side_effect(ip, port=8080, timeout=2.0):
            return ip == target_ip

        mock_probe.side_effect = probe_side_effect

        # Should find it without HTTP verification
        result = _discover_via_ip_range_scan(
            timeout=0.1, max_workers=10, verify_endpoint=False
        )
        self.assertEqual(result, target_ip)

    @patch("src.discovery._verify_gopro_endpoint")
    @patch("src.discovery._probe_gopro_http")
    def test_verify_passes_for_real_gopro(self, mock_probe, mock_verify):
        """When TCP probe and HTTP verify both pass, IP is returned."""
        target_ip = "172.23.120.51"
        mock_probe.side_effect = lambda ip, port=8080, timeout=2.0: ip == target_ip
        mock_verify.return_value = True

        result = _discover_via_ip_range_scan(timeout=0.1, max_workers=10)
        self.assertEqual(result, target_ip)
        mock_verify.assert_called_with(target_ip, GOPRO_API_PORT, timeout=2.0)

    @patch("src.discovery._verify_gopro_endpoint", return_value=True)
    @patch("src.discovery._probe_gopro_http")
    def test_uses_gopro_api_port(self, mock_probe, mock_verify):
        """Probes use the correct GOPRO_API_PORT (8080)."""
        mock_probe.return_value = False

        _discover_via_ip_range_scan(timeout=0.1, max_workers=5)

        # All calls should use GOPRO_API_PORT
        for call_args in mock_probe.call_args_list:
            args, kwargs = call_args
            # port is positional arg 2 or keyword
            port = args[1] if len(args) > 1 else kwargs.get("port", 8080)
            self.assertEqual(port, GOPRO_API_PORT)


class TestDiscoverGoProIpNoRangeScan(unittest.TestCase):
    """Test that discover_gopro_ip only does ipconfig + interface scan (no range scan)."""

    @patch("src.discovery._discover_via_interface_scan")
    @patch("src.discovery._discover_via_ipconfig")
    def test_ipconfig_success(self, mock_ipconfig, mock_iface):
        mock_ipconfig.return_value = "172.20.145.51"

        result = discover_gopro_ip()

        self.assertEqual(result, "172.20.145.51")
        mock_iface.assert_not_called()

    @patch("src.discovery._discover_via_interface_scan")
    @patch("src.discovery._discover_via_ipconfig")
    def test_interface_scan_success(self, mock_ipconfig, mock_iface):
        mock_ipconfig.return_value = None
        mock_iface.return_value = "172.25.110.51"

        result = discover_gopro_ip()

        self.assertEqual(result, "172.25.110.51")

    @patch("src.discovery._discover_via_interface_scan")
    @patch("src.discovery._discover_via_ipconfig")
    def test_both_fail(self, mock_ipconfig, mock_iface):
        mock_ipconfig.return_value = None
        mock_iface.return_value = None

        result = discover_gopro_ip()

        self.assertIsNone(result)


class TestVerifyGoProEndpoint(unittest.TestCase):
    """Test HTTP endpoint verification for GoPro identity confirmation."""

    def _mock_http_response(self, body: bytes, status: int = 200):
        """Create a mock urllib response context manager."""
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = status
        mock_resp.read.return_value = body
        return mock_resp

    @patch("urllib.request.urlopen")
    def test_valid_gopro_status_response(self, mock_urlopen):
        """Returns True when endpoint returns JSON with 'status' field."""
        mock_urlopen.return_value = self._mock_http_response(b'{"status": 0}')

        result = _verify_gopro_endpoint("172.20.145.51")
        self.assertTrue(result)

    @patch("urllib.request.urlopen")
    def test_gopro_error_response_still_valid(self, mock_urlopen):
        """Returns True when endpoint returns JSON with 'error' field (still GoPro)."""
        mock_urlopen.return_value = self._mock_http_response(
            b'{"error": "not in webcam mode"}'
        )

        result = _verify_gopro_endpoint("172.20.145.51")
        self.assertTrue(result)

    @patch("urllib.request.urlopen")
    def test_http_500_is_still_gopro(self, mock_urlopen):
        """HTTP 500 from GoPro means it's there but webcam mode not active."""
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            url="http://172.20.145.51:8080/gopro/webcam/status",
            code=500, msg="Internal Server Error",
            hdrs=None, fp=None,
        )

        result = _verify_gopro_endpoint("172.20.145.51")
        self.assertTrue(result)

    @patch("urllib.request.urlopen")
    def test_non_json_response_rejected(self, mock_urlopen):
        """Returns False for non-JSON response (not a GoPro)."""
        mock_urlopen.return_value = self._mock_http_response(
            b"<html>Not a GoPro</html>"
        )

        result = _verify_gopro_endpoint("172.20.145.51")
        self.assertFalse(result)

    @patch("urllib.request.urlopen")
    def test_connection_refused_returns_false(self, mock_urlopen):
        """Returns False when connection is refused."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")

        result = _verify_gopro_endpoint("172.20.145.51")
        self.assertFalse(result)

    @patch("urllib.request.urlopen")
    def test_timeout_returns_false(self, mock_urlopen):
        """Returns False on timeout."""
        mock_urlopen.side_effect = socket.timeout("timed out")

        result = _verify_gopro_endpoint("172.20.145.51")
        self.assertFalse(result)

    @patch("urllib.request.urlopen")
    def test_json_without_status_rejected(self, mock_urlopen):
        """JSON without 'status' or 'error' key is rejected (not GoPro)."""
        mock_urlopen.return_value = self._mock_http_response(
            b'{"name": "some other service"}'
        )

        result = _verify_gopro_endpoint("172.20.145.51")
        self.assertFalse(result)

    @patch("urllib.request.urlopen")
    def test_webcam_active_status(self, mock_urlopen):
        """Returns True for GoPro webcam active status (status=2)."""
        mock_urlopen.return_value = self._mock_http_response(b'{"status": 2}')

        result = _verify_gopro_endpoint("172.20.145.51")
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
