"""
Tests for COHN provisioner — mocked at the BLE GATT client boundary.

No real BLE or camera hardware required. These tests verify:
  - WiFi scan request encoding and result decoding
  - WiFi connect flow with SSID/password provisioning
  - Certificate creation and decoding
  - COHN status query and credential extraction
  - Full provisioning flow (WiFi → cert → status)
  - Protobuf encoding/decoding correctness
  - State machine transitions
  - Error handling (timeouts, failed responses)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gomaxwebcam.ble.cohn import COHNProvisioner, COHNCredentials, COHNState, WiFiNetwork
from gomaxwebcam.ble.uuids import COHN_FEATURE_ID, WIFI_CONNECT, COHN_GET_STATUS


# ---------------------------------------------------------------------------
# Mock BLE client
# ---------------------------------------------------------------------------


class MockBLEClient:
    """Simulates GoProBLEClient for testing the provisioner."""

    def __init__(self):
        self.write_network_management = AsyncMock(return_value=None)
        self.write_command = AsyncMock(return_value=None)
        self.read_wifi_ssid = AsyncMock(return_value="GoPro-1234")
        self.read_wifi_password = AsyncMock(return_value="abc123")
        self.is_connected = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ble():
    return MockBLEClient()


@pytest.fixture
def provisioner(mock_ble):
    return COHNProvisioner(
        ble_client=mock_ble,
        wifi_connect_timeout=2.0,  # Short for tests
        cert_create_timeout=2.0,
    )


# ---------------------------------------------------------------------------
# Protobuf encoding tests
# ---------------------------------------------------------------------------


class TestProtobufEncoding:
    def test_encode_varint_small(self):
        assert COHNProvisioner._encode_varint(0) == b"\x00"
        assert COHNProvisioner._encode_varint(1) == b"\x01"
        assert COHNProvisioner._encode_varint(127) == b"\x7f"

    def test_encode_varint_multi_byte(self):
        # 128 = 0x80 -> 0x80 0x01 in varint
        result = COHNProvisioner._encode_varint(128)
        assert result == bytes([0x80, 0x01])

    def test_encode_varint_large(self):
        # 300 = 0x012C -> 0xAC 0x02
        result = COHNProvisioner._encode_varint(300)
        assert result == bytes([0xAC, 0x02])

    def test_encode_length_delimited(self):
        result = COHNProvisioner._encode_length_delimited(1, b"test")
        # field 1, wire type 2 -> tag = (1 << 3) | 2 = 0x0A
        # length = 4 -> 0x04
        assert result[0] == 0x0A
        assert result[1] == 0x04
        assert result[2:] == b"test"

    def test_encode_varint_field(self):
        result = COHNProvisioner._encode_varint_field(1, 42)
        # field 1, wire type 0 -> tag = (1 << 3) | 0 = 0x08
        # value = 42 -> 0x2A
        assert result[0] == 0x08
        assert result[1] == 42

    def test_decode_varint_roundtrip(self):
        for val in [0, 1, 127, 128, 255, 300, 16384, 100000]:
            encoded = COHNProvisioner._encode_varint(val)
            decoded, offset = COHNProvisioner._decode_varint(encoded)
            assert decoded == val
            assert offset == len(encoded)

    def test_decode_field_varint(self):
        data = COHNProvisioner._encode_varint_field(1, 42)
        fn, wt, value, offset = COHNProvisioner._decode_field(data, 0)
        assert fn == 1
        assert wt == 0
        assert value == 42

    def test_decode_field_length_delimited(self):
        data = COHNProvisioner._encode_length_delimited(2, b"hello")
        fn, wt, value, offset = COHNProvisioner._decode_field(data, 0)
        assert fn == 2
        assert wt == 2
        assert value == b"hello"


# ---------------------------------------------------------------------------
# WiFi connect encoding tests
# ---------------------------------------------------------------------------


class TestWiFiConnectEncoding:
    def test_encode_wifi_connect(self, provisioner):
        request = provisioner._encode_wifi_connect_request("MyWiFi", "password123")

        # Should start with feature_id and action_id
        assert request[0] == COHN_FEATURE_ID
        assert request[1] == WIFI_CONNECT

        # Rest should contain protobuf-encoded SSID and password
        payload = request[2:]
        assert len(payload) > 0

        # Decode and verify
        offset = 0
        fields = {}
        while offset < len(payload):
            fn, wt, value, offset = COHNProvisioner._decode_field(payload, offset)
            if value is not None:
                fields[fn] = value

        assert fields[1] == b"MyWiFi"
        assert fields[2] == b"password123"

    def test_encode_wifi_connect_unicode(self, provisioner):
        """Should handle Unicode SSIDs."""
        request = provisioner._encode_wifi_connect_request("Café_WiFi", "pässwörd")
        assert len(request) > 4  # feature_id + action_id + protobuf


# ---------------------------------------------------------------------------
# WiFi scan result decoding tests
# ---------------------------------------------------------------------------


class TestWiFiScanDecoding:
    def test_decode_empty_response(self, provisioner):
        networks = provisioner._decode_wifi_scan_results(b"")
        assert networks == []

    def test_decode_short_response(self, provisioner):
        networks = provisioner._decode_wifi_scan_results(bytes([0xF5, 0x03]))
        assert networks == []

    def test_decode_single_network(self, provisioner):
        # Build a response with one network entry
        # Network is field 1 (length-delimited) containing:
        #   field 1 = SSID, field 2 = signal, field 3 = security
        inner = (
            COHNProvisioner._encode_length_delimited(1, b"TestNetwork")
            + COHNProvisioner._encode_varint_field(2, 75)
            + COHNProvisioner._encode_length_delimited(3, b"WPA2")
        )
        response = bytes([0xF5, 0x03]) + COHNProvisioner._encode_length_delimited(1, inner)

        networks = provisioner._decode_wifi_scan_results(response)
        assert len(networks) == 1
        assert networks[0].ssid == "TestNetwork"
        assert networks[0].signal_strength == 75
        assert networks[0].security == "WPA2"


# ---------------------------------------------------------------------------
# WiFi status decoding tests
# ---------------------------------------------------------------------------


class TestWiFiStatusDecoding:
    def test_decode_connected(self, provisioner):
        response = (
            bytes([0xF5, 0x05])
            + COHNProvisioner._encode_varint_field(1, 1)
            + COHNProvisioner._encode_length_delimited(2, b"192.168.1.100")
        )

        connected, ip = provisioner._decode_wifi_status(response)
        assert connected is True
        assert ip == "192.168.1.100"

    def test_decode_not_connected(self, provisioner):
        response = (
            bytes([0xF5, 0x05])
            + COHNProvisioner._encode_varint_field(1, 0)
        )

        connected, ip = provisioner._decode_wifi_status(response)
        assert connected is False
        assert ip == ""

    def test_decode_empty(self, provisioner):
        connected, ip = provisioner._decode_wifi_status(b"")
        assert connected is False


# ---------------------------------------------------------------------------
# Certificate decoding tests
# ---------------------------------------------------------------------------


class TestCertificateDecoding:
    def test_decode_certificate(self, provisioner):
        pem = b"-----BEGIN CERTIFICATE-----\nMIIBkTCC...\n-----END CERTIFICATE-----"
        response = bytes([0xF5, 0x03]) + COHNProvisioner._encode_length_delimited(1, pem)

        cert = provisioner._decode_certificate_response(response)
        assert "BEGIN CERTIFICATE" in cert
        assert "END CERTIFICATE" in cert

    def test_decode_empty_cert(self, provisioner):
        cert = provisioner._decode_certificate_response(b"")
        assert cert == ""

    def test_decode_missing_cert(self, provisioner):
        # Response with no cert field
        response = bytes([0xF5, 0x03]) + COHNProvisioner._encode_varint_field(2, 0)
        cert = provisioner._decode_certificate_response(response)
        assert cert == ""


# ---------------------------------------------------------------------------
# COHN status decoding tests
# ---------------------------------------------------------------------------


class TestCOHNStatusDecoding:
    def test_decode_full_status(self, provisioner):
        response = (
            bytes([0xF5, 0x01])
            + COHNProvisioner._encode_varint_field(1, 1)  # enabled
            + COHNProvisioner._encode_length_delimited(2, b"192.168.1.50")  # IP
            + COHNProvisioner._encode_length_delimited(3, b"gopro")  # username
            + COHNProvisioner._encode_length_delimited(4, b"CamPass123")  # password
            + COHNProvisioner._encode_length_delimited(5, b"HomeWiFi")  # SSID
        )

        provisioner._decode_cohn_status(response)

        creds = provisioner.credentials
        assert creds.provisioned is True
        assert creds.ip_address == "192.168.1.50"
        assert creds.username == "gopro"
        assert creds.password == "CamPass123"
        assert creds.ssid == "HomeWiFi"

    def test_decode_partial_status(self, provisioner):
        response = (
            bytes([0xF5, 0x01])
            + COHNProvisioner._encode_varint_field(1, 0)  # not enabled
        )

        provisioner._decode_cohn_status(response)
        assert provisioner.credentials.provisioned is False


# ---------------------------------------------------------------------------
# State machine tests
# ---------------------------------------------------------------------------


class TestCOHNProvisionerState:
    def test_initial_state(self, provisioner):
        assert provisioner.state == COHNState.IDLE

    def test_state_listener(self, provisioner):
        transitions: list = []
        provisioner.add_state_listener(lambda old, new: transitions.append((old, new)))

        provisioner._set_state(COHNState.SCANNING_WIFI)
        assert (COHNState.IDLE, COHNState.SCANNING_WIFI) in transitions

    def test_no_duplicate_state(self, provisioner):
        transitions: list = []
        provisioner.add_state_listener(lambda old, new: transitions.append((old, new)))

        provisioner._set_state(COHNState.IDLE)  # Same as current
        assert len(transitions) == 0

    def test_listener_exception_doesnt_crash(self, provisioner):
        def bad_listener(old, new):
            raise RuntimeError("boom")

        provisioner.add_state_listener(bad_listener)
        # Should not raise
        provisioner._set_state(COHNState.SCANNING_WIFI)


# ---------------------------------------------------------------------------
# WiFi scan flow tests
# ---------------------------------------------------------------------------


class TestWiFiScanFlow:
    @pytest.mark.anyio
    async def test_scan_wifi_success(self, provisioner, mock_ble):
        # First call: scan start, returns ack
        # Second call: scan results, returns network data
        inner = COHNProvisioner._encode_length_delimited(1, b"HomeNet")
        results_response = bytes([0xF5, 0x03]) + COHNProvisioner._encode_length_delimited(1, inner)

        mock_ble.write_network_management = AsyncMock(
            side_effect=[
                b"\xF5\x02\x00",  # scan start ack
                results_response,  # scan results
            ]
        )

        networks = await provisioner.scan_wifi(timeout=3.0)

        assert len(networks) == 1
        assert networks[0].ssid == "HomeNet"
        assert provisioner.state == COHNState.IDLE

    @pytest.mark.anyio
    async def test_scan_wifi_no_response(self, provisioner, mock_ble):
        mock_ble.write_network_management = AsyncMock(return_value=None)

        networks = await provisioner.scan_wifi()

        assert networks == []
        assert provisioner.state == COHNState.ERROR

    @pytest.mark.anyio
    async def test_scan_wifi_error(self, provisioner, mock_ble):
        mock_ble.write_network_management = AsyncMock(
            side_effect=Exception("BLE error")
        )

        networks = await provisioner.scan_wifi()

        assert networks == []
        assert provisioner.state == COHNState.ERROR


# ---------------------------------------------------------------------------
# Provisioning flow tests
# ---------------------------------------------------------------------------


class TestProvisionFlow:
    @pytest.mark.anyio
    async def test_provision_wifi_connect_failure(self, provisioner, mock_ble):
        """If WiFi connection fails, provisioning should fail."""
        mock_ble.write_network_management = AsyncMock(return_value=None)

        creds = await provisioner.provision(ssid="TestNet", password="pass123")

        assert creds.provisioned is False
        assert provisioner.state == COHNState.ERROR

    @pytest.mark.anyio
    async def test_provision_full_success(self, provisioner, mock_ble):
        """Full provisioning flow: WiFi → cert → status."""
        # Build responses for each step
        wifi_connect_ack = b"\xF5\x04\x00"
        wifi_status_connected = (
            bytes([0xF5, 0x05])
            + COHNProvisioner._encode_varint_field(1, 1)
            + COHNProvisioner._encode_length_delimited(2, b"192.168.1.100")
        )
        cert_response = (
            bytes([0xF5, 0x03])
            + COHNProvisioner._encode_length_delimited(1, b"-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----")
        )
        cohn_status = (
            bytes([0xF5, 0x01])
            + COHNProvisioner._encode_varint_field(1, 1)
            + COHNProvisioner._encode_length_delimited(2, b"192.168.1.100")
            + COHNProvisioner._encode_length_delimited(3, b"gopro")
            + COHNProvisioner._encode_length_delimited(4, b"CamPass")
            + COHNProvisioner._encode_length_delimited(5, b"TestNet")
        )

        # The provisioner calls write_network_management multiple times:
        # 1. WiFi connect
        # 2. WiFi status poll(s) - we return connected on first poll
        # 3. Create certificate
        # 4. Get COHN status
        mock_ble.write_network_management = AsyncMock(
            side_effect=[
                wifi_connect_ack,
                wifi_status_connected,
                cert_response,
                cohn_status,
            ]
        )

        creds = await provisioner.provision(
            ssid="TestNet",
            password="pass123",
            camera_serial="1234",
        )

        assert creds.provisioned is True
        assert creds.ip_address == "192.168.1.100"
        assert creds.username == "gopro"
        assert creds.password == "CamPass"
        assert creds.ssid == "TestNet"
        assert creds.camera_serial == "1234"
        assert "CERTIFICATE" in creds.certificate
        assert provisioner.state == COHNState.PROVISIONED

    @pytest.mark.anyio
    async def test_provision_cert_failure(self, provisioner, mock_ble):
        """If certificate creation fails, provisioning should fail."""
        wifi_connect_ack = b"\xF5\x04\x00"
        wifi_status_connected = (
            bytes([0xF5, 0x05])
            + COHNProvisioner._encode_varint_field(1, 1)
            + COHNProvisioner._encode_length_delimited(2, b"192.168.1.100")
        )

        mock_ble.write_network_management = AsyncMock(
            side_effect=[
                wifi_connect_ack,
                wifi_status_connected,
                None,  # cert creation fails
            ]
        )

        creds = await provisioner.provision(ssid="TestNet", password="pass123")

        assert creds.provisioned is False


# ---------------------------------------------------------------------------
# COHN status query tests
# ---------------------------------------------------------------------------


class TestGetStatus:
    @pytest.mark.anyio
    async def test_get_status_success(self, provisioner, mock_ble):
        status_response = (
            bytes([0xF5, 0x01])
            + COHNProvisioner._encode_varint_field(1, 1)
            + COHNProvisioner._encode_length_delimited(2, b"10.0.0.50")
        )
        mock_ble.write_network_management = AsyncMock(return_value=status_response)

        creds = await provisioner.get_status()

        assert creds.ip_address == "10.0.0.50"
        assert creds.provisioned is True

    @pytest.mark.anyio
    async def test_get_status_no_response(self, provisioner, mock_ble):
        mock_ble.write_network_management = AsyncMock(return_value=None)

        creds = await provisioner.get_status()
        # Should not crash, returns default credentials
        assert isinstance(creds, COHNCredentials)


# ---------------------------------------------------------------------------
# Clear certificate tests
# ---------------------------------------------------------------------------


class TestClearCertificate:
    @pytest.mark.anyio
    async def test_clear_cert_success(self, provisioner, mock_ble):
        mock_ble.write_network_management = AsyncMock(return_value=b"\xF5\x04\x00")

        result = await provisioner.clear_certificate()
        assert result is True

    @pytest.mark.anyio
    async def test_clear_cert_no_response(self, provisioner, mock_ble):
        mock_ble.write_network_management = AsyncMock(return_value=None)

        result = await provisioner.clear_certificate()
        assert result is False

    @pytest.mark.anyio
    async def test_clear_cert_error(self, provisioner, mock_ble):
        mock_ble.write_network_management = AsyncMock(
            side_effect=Exception("BLE error")
        )

        result = await provisioner.clear_certificate()
        assert result is False


# ---------------------------------------------------------------------------
# COHNCredentials dataclass tests
# ---------------------------------------------------------------------------


class TestCOHNCredentials:
    def test_defaults(self):
        creds = COHNCredentials()
        assert creds.camera_serial == ""
        assert creds.ip_address == ""
        assert creds.username == "gopro"
        assert creds.password == ""
        assert creds.certificate == ""
        assert creds.ssid == ""
        assert creds.provisioned is False

    def test_custom_values(self):
        creds = COHNCredentials(
            camera_serial="1234",
            ip_address="192.168.1.50",
            username="gopro",
            password="secret",
            certificate="PEM...",
            ssid="HomeWiFi",
            provisioned=True,
        )
        assert creds.provisioned is True
        assert creds.ip_address == "192.168.1.50"


# ---------------------------------------------------------------------------
# WiFiNetwork dataclass tests
# ---------------------------------------------------------------------------


class TestWiFiNetwork:
    def test_defaults(self):
        net = WiFiNetwork()
        assert net.ssid == ""
        assert net.signal_strength == 0
        assert net.security == ""
        assert net.frequency_mhz == 0
