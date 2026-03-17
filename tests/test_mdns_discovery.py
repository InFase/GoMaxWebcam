"""
Tests for mDNS discovery of GoPro cameras.

Tests cover:
- mDNS query packet building (_build_mdns_query)
- mDNS response packet parsing (_parse_mdns_response)
- DNS name skipping (_skip_dns_name)
- Raw socket mDNS discovery (_discover_via_mdns)
- Integration with discover_gopro_ip_chain
- Constants validation

All tests mock network I/O so they run without hardware.
"""

import socket
import struct
import unittest
from unittest.mock import patch, MagicMock

from src.discovery import (

    GOPRO_MDNS_SERVICE_TYPE,
    GOPRO_MDNS_ALT_SERVICE_TYPES,
    MDNS_DEFAULT_TIMEOUT,
    MDNS_PORT,
    MDNS_ADDR,
    DiscoveryMethod,
    _build_mdns_query,
    _parse_mdns_response,
    _skip_dns_name,
    _discover_via_mdns,
    discover_gopro_ip_chain,
)
import pytest

pytestmark = pytest.mark.no_gopro_needed



# --- Helper: build synthetic mDNS response packets ---

def _make_a_record_response(ip: str, name: str = "gopro._gopro-web._tcp.local",
                            qdcount: int = 0) -> bytes:
    """Build a minimal mDNS response with a single A record.

    Args:
        ip: IPv4 address string (e.g. '172.20.145.51').
        name: DNS name for the A record.
        qdcount: Number of questions (usually 0 for responses).

    Returns:
        Raw DNS response bytes.
    """
    flags = 0x8400  # QR=1, AA=1
    header = struct.pack("!HHHHHH", 0, flags, qdcount, 1, 0, 0)

    # Encode name as DNS labels
    qname = b""
    for label in name.split("."):
        encoded = label.encode("utf-8")
        qname += struct.pack("B", len(encoded)) + encoded
    qname += b"\x00"

    # A record: TYPE=1, CLASS=IN(1), TTL=120, RDLENGTH=4, RDATA=IP
    ip_bytes = socket.inet_aton(ip)
    rr = qname + struct.pack("!HHIH", 1, 1, 120, 4) + ip_bytes

    return header + rr


def _make_response_with_question_and_answer(ip: str) -> bytes:
    """Build an mDNS response that includes 1 question + 1 A record answer."""
    flags = 0x8400
    header = struct.pack("!HHHHHH", 0, flags, 1, 1, 0, 0)

    # Question section: _gopro-web._tcp.local. PTR IN
    name = "_gopro-web._tcp.local"
    qname = b""
    for label in name.split("."):
        encoded = label.encode("utf-8")
        qname += struct.pack("B", len(encoded)) + encoded
    qname += b"\x00"
    question = qname + struct.pack("!HH", 12, 1)  # PTR, IN

    # Answer: A record
    a_name = "gopro._gopro-web._tcp.local"
    a_qname = b""
    for label in a_name.split("."):
        encoded = label.encode("utf-8")
        a_qname += struct.pack("B", len(encoded)) + encoded
    a_qname += b"\x00"

    ip_bytes = socket.inet_aton(ip)
    answer = a_qname + struct.pack("!HHIH", 1, 1, 120, 4) + ip_bytes

    return header + question + answer


class TestMdnsConstants(unittest.TestCase):
    """Verify mDNS-related constants."""

    def test_service_type(self):
        self.assertEqual(GOPRO_MDNS_SERVICE_TYPE, "_gopro-web._tcp.local.")

    def test_alt_service_types_includes_primary(self):
        self.assertIn(GOPRO_MDNS_SERVICE_TYPE, GOPRO_MDNS_ALT_SERVICE_TYPES)

    def test_alt_service_types_includes_http(self):
        self.assertIn("_http._tcp.local.", GOPRO_MDNS_ALT_SERVICE_TYPES)

    def test_default_timeout(self):
        self.assertEqual(MDNS_DEFAULT_TIMEOUT, 3.0)

    def test_mdns_port(self):
        self.assertEqual(MDNS_PORT, 5353)

    def test_mdns_addr(self):
        self.assertEqual(MDNS_ADDR, "224.0.0.251")


class TestBuildMdnsQuery(unittest.TestCase):
    """Test mDNS query packet construction."""

    def test_query_is_bytes(self):
        query = _build_mdns_query(GOPRO_MDNS_SERVICE_TYPE)
        self.assertIsInstance(query, bytes)

    def test_query_header(self):
        query = _build_mdns_query(GOPRO_MDNS_SERVICE_TYPE)
        _id, flags, qdcount, ancount, nscount, arcount = struct.unpack(
            "!HHHHHH", query[:12]
        )
        self.assertEqual(_id, 0)
        self.assertEqual(flags, 0)  # Standard query
        self.assertEqual(qdcount, 1)
        self.assertEqual(ancount, 0)
        self.assertEqual(nscount, 0)
        self.assertEqual(arcount, 0)

    def test_query_contains_service_labels(self):
        query = _build_mdns_query("_gopro-web._tcp.local.")
        self.assertIn(b"\x0a_gopro-web", query)  # 10-char label
        self.assertIn(b"\x04_tcp", query)          # 4-char label
        self.assertIn(b"\x05local", query)         # 5-char label

    def test_query_ends_with_ptr_type(self):
        query = _build_mdns_query(GOPRO_MDNS_SERVICE_TYPE)
        qtype, qclass = struct.unpack("!HH", query[-4:])
        self.assertEqual(qtype, 12)  # PTR
        self.assertEqual(qclass, 0x8001)  # IN with unicast response bit

    def test_trailing_dot_stripped(self):
        q1 = _build_mdns_query("_gopro-web._tcp.local.")
        q2 = _build_mdns_query("_gopro-web._tcp.local")
        self.assertEqual(q1, q2)


class TestParseMdnsResponse(unittest.TestCase):
    """Test mDNS response packet parsing.

    _parse_mdns_response returns a list of IPs matching 172.x.x.x pattern.
    """

    def test_extract_a_record_ip(self):
        data = _make_a_record_response("172.20.145.51")
        ips = _parse_mdns_response(data)
        self.assertEqual(ips, ["172.20.145.51"])

    def test_different_ip(self):
        data = _make_a_record_response("172.25.110.51")
        ips = _parse_mdns_response(data)
        self.assertEqual(ips, ["172.25.110.51"])

    def test_non_gopro_ip_filtered(self):
        """Non-172.x.x.x IPs are filtered out by the parser."""
        data = _make_a_record_response("192.168.1.100")
        ips = _parse_mdns_response(data)
        self.assertEqual(ips, [])

    def test_too_short_packet(self):
        self.assertEqual(_parse_mdns_response(b"\x00" * 5), [])

    def test_empty_bytes(self):
        self.assertEqual(_parse_mdns_response(b""), [])

    def test_no_answer_records(self):
        """Response with 0 answer/ns/ar records returns empty list."""
        header = struct.pack("!HHHHHH", 0, 0x8400, 0, 0, 0, 0)
        self.assertEqual(_parse_mdns_response(header), [])

    def test_response_with_question_section(self):
        """Response that includes a question section is parsed correctly."""
        data = _make_response_with_question_and_answer("172.20.145.51")
        ips = _parse_mdns_response(data)
        self.assertEqual(ips, ["172.20.145.51"])

    def test_truncated_rdata(self):
        """Truncated A record RDATA should return empty list."""
        data = _make_a_record_response("172.20.145.51")
        self.assertEqual(_parse_mdns_response(data[:-2]), [])

    def test_non_a_record_skipped(self):
        """Non-A records (e.g., AAAA type=28) should be skipped."""
        flags = 0x8400
        header = struct.pack("!HHHHHH", 0, flags, 0, 1, 0, 0)
        name = b"\x05gopro\x00"
        rr = name + struct.pack("!HHIH", 28, 1, 120, 16) + b"\x00" * 16
        data = header + rr
        self.assertEqual(_parse_mdns_response(data), [])


class TestSkipDnsName(unittest.TestCase):
    """Test DNS name skipping logic."""

    def test_simple_name(self):
        data = b"\x05hello\x05local\x00EXTRA"
        offset = _skip_dns_name(data, 0)
        self.assertEqual(offset, 13)  # 1+5+1+5+1 = 13

    def test_compression_pointer(self):
        data = b"\x00" * 5 + b"\xC0\x0C"
        offset = _skip_dns_name(data, 5)
        self.assertEqual(offset, 7)  # 2-byte pointer

    def test_empty_name(self):
        data = b"\x00"
        offset = _skip_dns_name(data, 0)
        self.assertEqual(offset, 1)

    def test_offset_out_of_bounds(self):
        data = b"\x05hello"
        self.assertIsNone(_skip_dns_name(data, 100))

    def test_empty_data(self):
        self.assertIsNone(_skip_dns_name(b"", 0))


class TestDiscoverViaMdns(unittest.TestCase):
    """Test the mDNS discovery function (raw socket implementation)."""

    @patch("src.discovery._probe_gopro_http")
    @patch("src.discovery.socket.socket")
    def test_mdns_finds_gopro(self, mock_socket_cls, mock_probe):
        """mDNS finds GoPro when response contains valid A record."""
        mock_probe.return_value = True

        response = _make_a_record_response("172.20.145.51")
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.recvfrom.side_effect = [
            (response, ("172.20.145.51", 5353)),
            socket.timeout("done"),
        ]

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
        """Socket creation error returns None gracefully."""
        mock_socket_cls.side_effect = OSError("No network")

        result = _discover_via_mdns(timeout=0.1)
        self.assertIsNone(result)

    @patch("src.discovery.socket.socket")
    def test_mdns_bind_fallback(self, mock_socket_cls):
        """When bind to port 5353 fails, falls back to any port."""
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        call_count = [0]

        def bind_side_effect(addr):
            call_count[0] += 1
            if call_count[0] == 1 and addr[1] == 5353:
                raise OSError("Address already in use")

        mock_sock.bind.side_effect = bind_side_effect
        mock_sock.recvfrom.side_effect = socket.timeout("timed out")

        result = _discover_via_mdns(timeout=0.1)
        self.assertIsNone(result)
        # Should have attempted bind twice (5353 failed, then 0)
        self.assertEqual(mock_sock.bind.call_count, 2)

    @patch("src.discovery._probe_gopro_http")
    @patch("src.discovery.socket.socket")
    def test_mdns_probe_failure_skips_candidate(self, mock_socket_cls, mock_probe):
        """When HTTP probe fails on mDNS candidate, continues listening."""
        mock_probe.return_value = False

        response = _make_a_record_response("172.25.110.51")
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.recvfrom.side_effect = [
            (response, ("172.25.110.51", 5353)),
            socket.timeout("done"),
        ]

        result = _discover_via_mdns(timeout=0.1)
        self.assertIsNone(result)

    @patch("src.discovery._probe_gopro_http")
    @patch("src.discovery.socket.socket")
    def test_mdns_sends_queries_for_all_service_types(self, mock_socket_cls, mock_probe):
        """mDNS should send queries for all configured service types."""
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.recvfrom.side_effect = socket.timeout("timed out")

        _discover_via_mdns(timeout=0.1)

        # Should have sent one query per service type
        sendto_calls = mock_sock.sendto.call_args_list
        self.assertEqual(len(sendto_calls), len(GOPRO_MDNS_ALT_SERVICE_TYPES))
        for call_obj in sendto_calls:
            args, kwargs = call_obj
            # Each sendto should target the mDNS multicast address
            self.assertEqual(args[1], (MDNS_ADDR, MDNS_PORT))


class TestMdnsInDiscoveryChain(unittest.TestCase):
    """Test that mDNS is integrated into the discover_gopro_ip_chain."""

    @patch("src.discovery._discover_via_mdns")
    @patch("src.discovery._discover_via_ip_range_scan")
    @patch("src.discovery.discover_gopro_ip")
    def test_mdns_used_when_others_fail(self, mock_usb, mock_ip_scan, mock_mdns):
        """mDNS is tried when USB scan and IP scan both fail."""
        mock_usb.return_value = None
        mock_ip_scan.return_value = None
        mock_mdns.return_value = "172.20.145.51"

        ip, method = discover_gopro_ip_chain()

        self.assertEqual(ip, "172.20.145.51")
        self.assertEqual(method, DiscoveryMethod.MDNS)
        mock_mdns.assert_called_once()

    @patch("src.discovery._discover_via_mdns")
    @patch("src.discovery._discover_via_ip_range_scan")
    @patch("src.discovery.discover_gopro_ip")
    def test_mdns_not_called_when_usb_succeeds(self, mock_usb, mock_ip_scan, mock_mdns):
        """mDNS is not called when USB scan succeeds first."""
        mock_usb.return_value = "172.20.145.51"

        ip, method = discover_gopro_ip_chain()

        self.assertEqual(ip, "172.20.145.51")
        self.assertEqual(method, DiscoveryMethod.USB_SCAN)
        mock_mdns.assert_not_called()

    @patch("src.discovery._discover_via_mdns")
    @patch("src.discovery._discover_via_ip_range_scan")
    @patch("src.discovery.discover_gopro_ip")
    def test_mdns_not_called_when_ip_scan_succeeds(self, mock_usb, mock_ip_scan, mock_mdns):
        """mDNS is not called when IP scan succeeds."""
        mock_usb.return_value = None
        mock_ip_scan.return_value = "172.22.150.51"

        ip, method = discover_gopro_ip_chain()

        self.assertEqual(ip, "172.22.150.51")
        self.assertEqual(method, DiscoveryMethod.IP_SCAN)
        mock_mdns.assert_not_called()


if __name__ == "__main__":
    unittest.main()
