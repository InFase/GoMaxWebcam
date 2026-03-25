"""
ble/cohn.py — COHN (Camera On Home Network) provisioning via BLE.

COHN allows the GoPro to connect to a home WiFi network in station mode,
enabling HTTP API access over the local network without USB or WiFi AP.

The provisioning flow:
  1. Connect to camera via BLE (scanner.py + gatt_client.py)
  2. Send WiFi SSID and password via Network Management GATT characteristic
  3. Camera connects to the WiFi network
  4. Exchange/create TLS certificate for secure HTTPS communication
  5. Camera reports its IP address on the home network
  6. Store COHN credentials (IP, username, password, cert) in keyring

The COHNProvisioner orchestrates this flow using the GoProBLEClient for
the actual BLE I/O.

Protobuf encoding follows the open-gopro BLE protocol specification.
We use minimal hand-rolled protobuf encoding to avoid a protoc dependency.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from gomaxwebcam.ble.gatt_client import GoProBLEClient
from gomaxwebcam.ble.uuids import (
    NETWORK_MGMT_REQUEST_UUID,
    NETWORK_MGMT_RESPONSE_UUID,
    COHN_FEATURE_ID,
    COHN_GET_STATUS,
    COHN_SET_SETTING,
    COHN_CREATE_CERT,
    COHN_CLEAR_CERT,
    COHN_GET_CERT,
    WIFI_SCAN_START,
    WIFI_SCAN_RESULTS,
    WIFI_CONNECT,
    WIFI_GET_STATUS,
)

log = logging.getLogger("gomaxwebcam.ble.cohn")


class COHNState(Enum):
    """State of the COHN provisioning process."""
    IDLE = auto()
    SCANNING_WIFI = auto()
    CONNECTING_WIFI = auto()
    CREATING_CERT = auto()
    PROVISIONED = auto()
    ERROR = auto()


@dataclass
class COHNCredentials:
    """Credentials for accessing a COHN-provisioned GoPro camera.

    These are the result of a successful COHN provisioning flow and
    are stored securely in the system keyring.

    Attributes:
        camera_serial: Camera serial number / BLE name suffix.
        ip_address: Camera's IP on the home WiFi network.
        username: COHN HTTP basic auth username (typically "gopro").
        password: COHN HTTP basic auth password (camera-generated).
        certificate: PEM-encoded TLS certificate for HTTPS verification.
        ssid: WiFi network the camera is connected to.
        provisioned: Whether provisioning completed successfully.
    """
    camera_serial: str = ""
    ip_address: str = ""
    username: str = "gopro"
    password: str = ""
    certificate: str = ""
    ssid: str = ""
    provisioned: bool = False


@dataclass
class WiFiNetwork:
    """A WiFi network found during camera's WiFi scan."""
    ssid: str = ""
    signal_strength: int = 0  # dBm
    security: str = ""  # "WPA2", "WPA3", "OPEN", etc.
    frequency_mhz: int = 0


class COHNProvisioner:
    """Orchestrates COHN provisioning for a GoPro camera via BLE.

    Usage:
        provisioner = COHNProvisioner(ble_client)
        networks = await provisioner.scan_wifi()
        credentials = await provisioner.provision(ssid="MyWiFi", password="secret")
        # credentials now contains the camera's COHN IP, username, password, cert

    Args:
        ble_client: Connected GoProBLEClient instance.
        wifi_connect_timeout: Timeout for camera to connect to WiFi (seconds).
        cert_create_timeout: Timeout for certificate creation (seconds).
    """

    def __init__(
        self,
        ble_client: GoProBLEClient,
        wifi_connect_timeout: float = 30.0,
        cert_create_timeout: float = 15.0,
    ):
        self._ble = ble_client
        self._wifi_connect_timeout = wifi_connect_timeout
        self._cert_create_timeout = cert_create_timeout
        self._state = COHNState.IDLE
        self._credentials = COHNCredentials()
        self._state_listeners: list = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> COHNState:
        return self._state

    @property
    def credentials(self) -> COHNCredentials:
        return self._credentials

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _set_state(self, new_state: COHNState) -> None:
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        log.info("COHN state: %s -> %s", old.name, new_state.name)
        for listener in self._state_listeners:
            try:
                listener(old, new_state)
            except Exception:
                log.exception("Error in COHN state listener")

    def add_state_listener(self, callback) -> None:
        """Register a callback(old_state, new_state) for state transitions."""
        self._state_listeners.append(callback)

    # ------------------------------------------------------------------
    # WiFi scanning
    # ------------------------------------------------------------------

    async def scan_wifi(self, timeout: float = 15.0) -> list[WiFiNetwork]:
        """Ask the camera to scan for WiFi networks.

        The camera performs the scan and returns results over BLE.

        Returns:
            List of WiFiNetwork instances found by the camera.
        """
        self._set_state(COHNState.SCANNING_WIFI)
        log.info("Requesting WiFi scan from camera")

        try:
            # Send WiFi scan start command
            request = self._encode_wifi_scan_request()
            response = await self._ble.write_network_management(request)

            if response is None:
                log.error("No response to WiFi scan request")
                self._set_state(COHNState.ERROR)
                return []

            # Wait a moment for the scan to complete on the camera
            await asyncio.sleep(2.0)

            # Request scan results
            results_request = self._encode_wifi_scan_results_request()
            results_response = await self._ble.write_network_management(results_request)

            if results_response is None:
                log.error("No response to WiFi scan results request")
                self._set_state(COHNState.ERROR)
                return []

            networks = self._decode_wifi_scan_results(results_response)
            log.info("Camera found %d WiFi networks", len(networks))

            self._set_state(COHNState.IDLE)
            return networks

        except Exception as e:
            log.error("WiFi scan failed: %s", e)
            self._set_state(COHNState.ERROR)
            return []

    # ------------------------------------------------------------------
    # COHN provisioning
    # ------------------------------------------------------------------

    async def provision(
        self,
        ssid: str,
        password: str,
        camera_serial: str = "",
    ) -> COHNCredentials:
        """Provision COHN on the camera.

        Full provisioning flow:
          1. Connect camera to WiFi network (SSID + password)
          2. Wait for camera to get IP address
          3. Create/exchange TLS certificate
          4. Read COHN credentials from camera
          5. Return COHNCredentials for secure HTTP access

        Args:
            ssid: WiFi network SSID to connect the camera to.
            password: WiFi network password.
            camera_serial: Camera serial for credential storage.

        Returns:
            COHNCredentials with provisioned access info.
            Check .provisioned field for success.
        """
        self._credentials = COHNCredentials(
            camera_serial=camera_serial,
            ssid=ssid,
        )

        # Step 1: Connect camera to WiFi
        wifi_ok = await self._connect_camera_to_wifi(ssid, password)
        if not wifi_ok:
            self._credentials.provisioned = False
            return self._credentials

        # Step 2: Create TLS certificate
        cert_ok = await self._create_cohn_certificate()
        if not cert_ok:
            self._credentials.provisioned = False
            return self._credentials

        # Step 3: Get COHN status (IP, credentials)
        status_ok = await self._get_cohn_status()
        if not status_ok:
            self._credentials.provisioned = False
            return self._credentials

        self._credentials.provisioned = True
        self._set_state(COHNState.PROVISIONED)
        log.info(
            "COHN provisioned: camera at %s on '%s'",
            self._credentials.ip_address,
            self._credentials.ssid,
        )
        return self._credentials

    async def get_status(self) -> COHNCredentials:
        """Query current COHN status from the camera.

        Useful for checking if COHN is already provisioned.
        Returns COHNCredentials with whatever info the camera reports.
        """
        log.info("Querying COHN status from camera")

        try:
            request = self._encode_cohn_command(COHN_GET_STATUS)
            response = await self._ble.write_network_management(request)

            if response is None:
                log.warning("No response to COHN status query")
                return self._credentials

            self._decode_cohn_status(response)
            return self._credentials

        except Exception as e:
            log.error("COHN status query failed: %s", e)
            return self._credentials

    async def clear_certificate(self) -> bool:
        """Clear the COHN TLS certificate on the camera.

        Use this before re-provisioning if the existing cert is invalid.
        """
        log.info("Clearing COHN certificate on camera")

        try:
            request = self._encode_cohn_command(COHN_CLEAR_CERT)
            response = await self._ble.write_network_management(request)
            if response is not None:
                log.info("COHN certificate cleared")
                return True
            return False

        except Exception as e:
            log.error("Failed to clear COHN cert: %s", e)
            return False

    # ------------------------------------------------------------------
    # Internal: WiFi connection
    # ------------------------------------------------------------------

    async def _connect_camera_to_wifi(
        self,
        ssid: str,
        password: str,
    ) -> bool:
        """Send WiFi credentials to the camera and wait for connection."""
        self._set_state(COHNState.CONNECTING_WIFI)
        log.info("Connecting camera to WiFi network: '%s'", ssid)

        try:
            # Encode WiFi connect request with SSID and password
            request = self._encode_wifi_connect_request(ssid, password)
            response = await self._ble.write_network_management(request)

            if response is None:
                log.error("No response to WiFi connect request")
                self._set_state(COHNState.ERROR)
                return False

            # Poll WiFi status until connected or timeout
            connected = await self._poll_wifi_status(
                timeout=self._wifi_connect_timeout,
            )

            if not connected:
                log.error(
                    "Camera failed to connect to WiFi '%s' within %.0fs",
                    ssid, self._wifi_connect_timeout,
                )
                self._set_state(COHNState.ERROR)
                return False

            log.info("Camera connected to WiFi '%s'", ssid)
            return True

        except Exception as e:
            log.error("WiFi connection failed: %s", e)
            self._set_state(COHNState.ERROR)
            return False

    async def _poll_wifi_status(self, timeout: float) -> bool:
        """Poll the camera's WiFi status until connected or timeout."""
        import time

        deadline = time.monotonic() + timeout
        poll_interval = 2.0

        while time.monotonic() < deadline:
            try:
                request = self._encode_wifi_status_request()
                response = await self._ble.write_network_management(request)

                if response is not None:
                    connected, ip = self._decode_wifi_status(response)
                    if connected:
                        if ip:
                            self._credentials.ip_address = ip
                        return True

            except Exception as e:
                log.debug("WiFi status poll error: %s", e)

            await asyncio.sleep(poll_interval)

        return False

    # ------------------------------------------------------------------
    # Internal: Certificate management
    # ------------------------------------------------------------------

    async def _create_cohn_certificate(self) -> bool:
        """Create a TLS certificate for COHN HTTPS access."""
        self._set_state(COHNState.CREATING_CERT)
        log.info("Creating COHN TLS certificate")

        try:
            request = self._encode_cohn_command(COHN_CREATE_CERT)
            response = await self._ble.write_network_management(request)

            if response is None:
                log.error("No response to certificate creation request")
                self._set_state(COHNState.ERROR)
                return False

            # Decode certificate from response
            cert = self._decode_certificate_response(response)
            if cert:
                self._credentials.certificate = cert
                log.info("COHN certificate created (%d bytes)", len(cert))
                return True
            else:
                log.error("Failed to decode certificate from response")
                self._set_state(COHNState.ERROR)
                return False

        except Exception as e:
            log.error("Certificate creation failed: %s", e)
            self._set_state(COHNState.ERROR)
            return False

    async def _get_cohn_status(self) -> bool:
        """Query COHN status to get IP, credentials, etc."""
        log.info("Querying COHN status")

        try:
            request = self._encode_cohn_command(COHN_GET_STATUS)
            response = await self._ble.write_network_management(request)

            if response is None:
                log.warning("No COHN status response")
                return False

            self._decode_cohn_status(response)

            # COHN status should now have IP and password
            if self._credentials.ip_address:
                return True
            else:
                log.warning("COHN status missing IP address")
                return False

        except Exception as e:
            log.error("COHN status query failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Protobuf encoding helpers (minimal hand-rolled, no protoc needed)
    # ------------------------------------------------------------------
    #
    # GoPro BLE network management uses a protobuf-like TLV encoding.
    # Each request is: [feature_id, action_id, ...protobuf_fields]
    #
    # Protobuf field encoding:
    #   field_tag = (field_number << 3) | wire_type
    #   wire_type 0 = varint, 2 = length-delimited (string/bytes)
    #
    # We encode only the fields needed for COHN provisioning.
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_varint(value: int) -> bytes:
        """Encode an integer as a protobuf varint."""
        result = bytearray()
        while value > 0x7F:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value & 0x7F)
        return bytes(result)

    @staticmethod
    def _encode_length_delimited(field_number: int, data: bytes) -> bytes:
        """Encode a protobuf length-delimited field (wire type 2)."""
        tag = (field_number << 3) | 2
        return (
            COHNProvisioner._encode_varint(tag)
            + COHNProvisioner._encode_varint(len(data))
            + data
        )

    @staticmethod
    def _encode_varint_field(field_number: int, value: int) -> bytes:
        """Encode a protobuf varint field (wire type 0)."""
        tag = (field_number << 3) | 0
        return (
            COHNProvisioner._encode_varint(tag)
            + COHNProvisioner._encode_varint(value)
        )

    def _encode_cohn_command(self, action_id: int) -> bytes:
        """Encode a COHN command request.

        Format: [feature_id, action_id]
        """
        return bytes([COHN_FEATURE_ID, action_id])

    def _encode_wifi_scan_request(self) -> bytes:
        """Encode a WiFi scan start request."""
        return bytes([COHN_FEATURE_ID, WIFI_SCAN_START])

    def _encode_wifi_scan_results_request(self) -> bytes:
        """Encode a WiFi scan results request."""
        return bytes([COHN_FEATURE_ID, WIFI_SCAN_RESULTS])

    def _encode_wifi_connect_request(self, ssid: str, password: str) -> bytes:
        """Encode a WiFi connect request with SSID and password.

        Protobuf fields:
          field 1 (string): SSID
          field 2 (string): password
        """
        payload = (
            self._encode_length_delimited(1, ssid.encode("utf-8"))
            + self._encode_length_delimited(2, password.encode("utf-8"))
        )
        return bytes([COHN_FEATURE_ID, WIFI_CONNECT]) + payload

    def _encode_wifi_status_request(self) -> bytes:
        """Encode a WiFi status query request."""
        return bytes([COHN_FEATURE_ID, WIFI_GET_STATUS])

    # ------------------------------------------------------------------
    # Protobuf decoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_varint(data: bytes, offset: int = 0) -> tuple[int, int]:
        """Decode a protobuf varint. Returns (value, new_offset)."""
        result = 0
        shift = 0
        while offset < len(data):
            byte = data[offset]
            result |= (byte & 0x7F) << shift
            offset += 1
            if not (byte & 0x80):
                break
            shift += 7
        return result, offset

    @staticmethod
    def _decode_field(data: bytes, offset: int) -> tuple[int, int, Any, int]:
        """Decode a single protobuf field.

        Returns (field_number, wire_type, value, new_offset).
        For wire_type 0: value is int.
        For wire_type 2: value is bytes.
        """
        if offset >= len(data):
            return 0, 0, None, offset

        tag, offset = COHNProvisioner._decode_varint(data, offset)
        field_number = tag >> 3
        wire_type = tag & 0x07

        if wire_type == 0:  # Varint
            value, offset = COHNProvisioner._decode_varint(data, offset)
            return field_number, wire_type, value, offset
        elif wire_type == 2:  # Length-delimited
            length, offset = COHNProvisioner._decode_varint(data, offset)
            value = data[offset:offset + length]
            return field_number, wire_type, bytes(value), offset + length
        else:
            # Skip unknown wire types
            log.debug("Unknown wire type %d for field %d", wire_type, field_number)
            return field_number, wire_type, None, offset + 1

    def _decode_wifi_scan_results(self, response: bytes) -> list[WiFiNetwork]:
        """Decode WiFi scan results from a network management response."""
        networks: list[WiFiNetwork] = []

        if len(response) < 2:
            return networks

        # Skip feature_id and action_id header
        offset = 2
        while offset < len(response):
            try:
                fn, wt, value, offset = self._decode_field(response, offset)
                if wt == 2 and isinstance(value, bytes):
                    # Each network is a nested protobuf message
                    network = self._decode_wifi_network(value)
                    if network.ssid:
                        networks.append(network)
            except Exception:
                break

        return networks

    @staticmethod
    def _decode_wifi_network(data: bytes) -> WiFiNetwork:
        """Decode a single WiFi network from nested protobuf."""
        network = WiFiNetwork()
        offset = 0
        while offset < len(data):
            try:
                fn, wt, value, offset = COHNProvisioner._decode_field(data, offset)
                if fn == 1 and isinstance(value, bytes):
                    network.ssid = value.decode("utf-8", errors="replace")
                elif fn == 2 and isinstance(value, int):
                    network.signal_strength = value
                elif fn == 3 and isinstance(value, bytes):
                    network.security = value.decode("utf-8", errors="replace")
                elif fn == 4 and isinstance(value, int):
                    network.frequency_mhz = value
            except Exception:
                break
        return network

    def _decode_wifi_status(self, response: bytes) -> tuple[bool, str]:
        """Decode WiFi connection status.

        Returns (connected: bool, ip_address: str).
        """
        connected = False
        ip_address = ""

        if len(response) < 2:
            return connected, ip_address

        offset = 2  # Skip header
        while offset < len(response):
            try:
                fn, wt, value, offset = self._decode_field(response, offset)
                if fn == 1 and isinstance(value, int):
                    # Status field: 1 = connected
                    connected = value == 1
                elif fn == 2 and isinstance(value, bytes):
                    # IP address field
                    ip_address = value.decode("utf-8", errors="replace")
            except Exception:
                break

        return connected, ip_address

    def _decode_certificate_response(self, response: bytes) -> str:
        """Decode a certificate creation response.

        Returns the PEM certificate string, or empty string on failure.
        """
        if len(response) < 2:
            return ""

        offset = 2  # Skip header
        while offset < len(response):
            try:
                fn, wt, value, offset = self._decode_field(response, offset)
                if fn == 1 and isinstance(value, bytes):
                    # Certificate PEM data
                    return value.decode("utf-8", errors="replace")
            except Exception:
                break

        return ""

    def _decode_cohn_status(self, response: bytes) -> None:
        """Decode COHN status response and update credentials."""
        if len(response) < 2:
            return

        offset = 2  # Skip header
        while offset < len(response):
            try:
                fn, wt, value, offset = self._decode_field(response, offset)
                if fn == 1 and isinstance(value, int):
                    # COHN enabled: 1 = yes
                    self._credentials.provisioned = value == 1
                elif fn == 2 and isinstance(value, bytes):
                    # IP address
                    self._credentials.ip_address = value.decode("utf-8", errors="replace")
                elif fn == 3 and isinstance(value, bytes):
                    # Username
                    self._credentials.username = value.decode("utf-8", errors="replace")
                elif fn == 4 and isinstance(value, bytes):
                    # Password
                    self._credentials.password = value.decode("utf-8", errors="replace")
                elif fn == 5 and isinstance(value, bytes):
                    # SSID the camera is connected to
                    self._credentials.ssid = value.decode("utf-8", errors="replace")
                elif fn == 6 and isinstance(value, bytes):
                    # Certificate
                    self._credentials.certificate = value.decode("utf-8", errors="replace")
            except Exception:
                break

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def _inject_ble_client(self, mock_client: GoProBLEClient) -> None:
        """For testing: inject a mock BLE client."""
        self._ble = mock_client
        log.debug("Injected mock BLE client")
