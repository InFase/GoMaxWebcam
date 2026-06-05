"""
ble/gatt_client.py — BLE GATT client for GoPro camera communication.

Connects to a discovered GoPro camera via BLE and provides methods to
read/write GATT characteristics. This is the low-level BLE transport
used by the COHN provisioner and WiFi AP provisioner.

The GATT client handles:
  1. BLE connection to a specific GoPro by address
  2. Service and characteristic discovery
  3. Writing command/network-management requests
  4. Subscribing to notification characteristics for responses
  5. Request-response correlation with fragmented TLV reassembly
  6. Graceful disconnect

Uses bleak for cross-platform BLE GATT operations.
All methods are async.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from gomaxwebcam.ble.uuids import (
    GOPRO_SERVICE_UUID,
    COMMAND_REQUEST_UUID,
    COMMAND_RESPONSE_UUID,
    SETTINGS_REQUEST_UUID,
    SETTINGS_RESPONSE_UUID,
    QUERY_REQUEST_UUID,
    QUERY_RESPONSE_UUID,
    NETWORK_MGMT_REQUEST_UUID,
    NETWORK_MGMT_RESPONSE_UUID,
    WIFI_AP_SSID_UUID,
    WIFI_AP_PASSWORD_UUID,
)

log = logging.getLogger("gomaxwebcam.ble.gatt_client")

# BLE MTU — GoPro uses extended MTU, but we chunk to be safe
DEFAULT_MTU = 256

# Timeout for individual GATT operations
GATT_TIMEOUT_S = 10.0

# Response wait timeout
RESPONSE_TIMEOUT_S = 15.0


class GoProBLEClient:
    """BLE GATT client for communicating with a GoPro camera.

    Handles BLE connection, characteristic I/O, and response notification
    handling. Higher-level provisioning logic lives in cohn.py.

    Args:
        address: BLE address of the GoPro (MAC or UUID).
        mtu: Maximum write size per BLE packet.
    """

    def __init__(self, address: str, mtu: int = DEFAULT_MTU):
        self._address = address
        self._mtu = mtu
        self._client: Any = None  # BleakClient instance
        self._connected = False

        # Response accumulators keyed by characteristic UUID
        self._response_events: dict[str, asyncio.Event] = {}
        self._response_buffers: dict[str, bytearray] = {}
        self._response_expected_len: dict[str, int] = {}
        self._notification_handlers: dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def address(self) -> str:
        return self._address

    @property
    def is_connected(self) -> bool:
        if self._client is not None:
            return getattr(self._client, "is_connected", self._connected)
        return self._connected

    @property
    def mtu(self) -> int:
        return self._mtu

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, timeout: float = GATT_TIMEOUT_S) -> bool:
        """Connect to the GoPro camera via BLE.

        Returns True on success. Sets up notification handlers for
        all response characteristics.
        """
        log.info("Connecting to GoPro BLE device: %s", self._address)

        try:
            from bleak import BleakClient

            self._client = BleakClient(
                self._address,
                timeout=timeout,
            )

            await self._client.connect()
            self._connected = True

            # Pair/bond with the camera — required for encrypted characteristics
            try:
                if hasattr(self._client, "pair"):
                    paired = await self._client.pair()
                    log.info("BLE pairing result: %s", paired)
            except Exception as e:
                log.debug("BLE pairing attempt: %s (may already be paired)", e)

            # Negotiate MTU if possible
            if hasattr(self._client, "mtu_size"):
                actual_mtu = self._client.mtu_size
                log.info("BLE MTU: %d (requested %d)", actual_mtu, self._mtu)
                self._mtu = min(self._mtu, actual_mtu - 3)  # ATT header

            # Subscribe to notification characteristics
            await self._setup_notifications()

            log.info("BLE connected to %s", self._address)
            return True

        except ImportError:
            log.error("bleak not installed — BLE connection unavailable")
            return False
        except Exception as e:
            log.error("BLE connection failed: %s", e)
            self._client = None
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from the GoPro camera."""
        if self._client is not None:
            try:
                if self.is_connected:
                    await self._client.disconnect()
                    log.info("BLE disconnected from %s", self._address)
            except Exception as e:
                log.warning("Error during BLE disconnect: %s", e)
            finally:
                self._client = None
                self._connected = False
                self._response_events.clear()
                self._response_buffers.clear()

    # ------------------------------------------------------------------
    # Read characteristics
    # ------------------------------------------------------------------

    async def read_wifi_ssid(self) -> str | None:
        """Read the camera's WiFi AP SSID from the BLE characteristic."""
        return await self._read_string(WIFI_AP_SSID_UUID)

    async def read_wifi_password(self) -> str | None:
        """Read the camera's WiFi AP password from the BLE characteristic."""
        return await self._read_string(WIFI_AP_PASSWORD_UUID)

    async def read_characteristic(self, uuid: str) -> bytes | None:
        """Read a raw characteristic value by UUID."""
        if not self.is_connected or self._client is None:
            log.error("Cannot read: not connected")
            return None

        try:
            data = await asyncio.wait_for(
                self._client.read_gatt_char(uuid),
                timeout=GATT_TIMEOUT_S,
            )
            log.debug("Read %s: %d bytes", uuid[-8:], len(data))
            return bytes(data)
        except asyncio.TimeoutError:
            log.error("Read timeout for %s", uuid[-8:])
            return None
        except Exception as e:
            log.error("Read failed for %s: %s", uuid[-8:], e)
            return None

    # ------------------------------------------------------------------
    # Write characteristics
    # ------------------------------------------------------------------

    async def write_command(self, data: bytes) -> bytes | None:
        """Write a command to the GoPro and wait for the response.

        Uses the Command Request/Response characteristic pair.
        Handles fragmentation for payloads larger than MTU.

        Returns the response bytes, or None on failure.
        """
        return await self._write_and_wait(
            request_uuid=COMMAND_REQUEST_UUID,
            response_uuid=COMMAND_RESPONSE_UUID,
            data=data,
        )

    async def write_query(self, data: bytes) -> bytes | None:
        """Write a query to the GoPro and wait for the response.

        Uses the Query Request/Response characteristic pair.
        COHN status queries use this path.
        """
        return await self._write_and_wait(
            request_uuid=QUERY_REQUEST_UUID,
            response_uuid=QUERY_RESPONSE_UUID,
            data=data,
        )

    async def write_setting(self, data: bytes) -> bytes | None:
        """Write a setting change and wait for the response.

        Uses the Settings Request/Response characteristic pair.
        """
        return await self._write_and_wait(
            request_uuid=SETTINGS_REQUEST_UUID,
            response_uuid=SETTINGS_RESPONSE_UUID,
            data=data,
        )

    async def write_network_management(self, data: bytes) -> bytes | None:
        """Write a network management request and wait for the response.

        Used for WiFi provisioning and COHN configuration.
        Uses the Network Management Request/Response characteristic pair.
        """
        return await self._write_and_wait(
            request_uuid=NETWORK_MGMT_REQUEST_UUID,
            response_uuid=NETWORK_MGMT_RESPONSE_UUID,
            data=data,
        )

    async def write_raw(self, uuid: str, data: bytes, response: bool = True) -> None:
        """Write raw bytes to a characteristic.

        Args:
            uuid: Characteristic UUID to write to.
            data: Bytes to write.
            response: Whether to request a BLE write-with-response.
        """
        if not self.is_connected or self._client is None:
            log.error("Cannot write: not connected")
            return

        try:
            # Fragment if needed
            for chunk in self._fragment(data):
                await asyncio.wait_for(
                    self._client.write_gatt_char(uuid, chunk, response=response),
                    timeout=GATT_TIMEOUT_S,
                )
            log.debug("Wrote %d bytes to %s", len(data), uuid[-8:])
        except asyncio.TimeoutError:
            log.error("Write timeout for %s", uuid[-8:])
        except Exception as e:
            log.error("Write failed for %s: %s", uuid[-8:], e)

    # ------------------------------------------------------------------
    # Notification handling
    # ------------------------------------------------------------------

    def register_notification_handler(
        self,
        uuid: str,
        handler: Callable[[bytes], None],
    ) -> None:
        """Register a custom notification handler for a characteristic.

        The handler receives the raw notification bytes. This is useful
        for streaming notification data (e.g., status updates).
        """
        self._notification_handlers[uuid] = handler

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    async def _setup_notifications(self) -> None:
        """Subscribe to all GoPro response/notification characteristics."""
        notify_uuids = [
            COMMAND_RESPONSE_UUID,
            SETTINGS_RESPONSE_UUID,
            QUERY_RESPONSE_UUID,
            NETWORK_MGMT_RESPONSE_UUID,
        ]

        for uuid in notify_uuids:
            try:
                # Initialize response tracking for this UUID
                self._response_events[uuid] = asyncio.Event()
                self._response_buffers[uuid] = bytearray()

                await self._client.start_notify(
                    uuid,
                    lambda _, data, u=uuid: self._on_notification(u, data),
                )
                log.debug("Subscribed to notifications: %s", uuid[-8:])
            except Exception as e:
                log.warning("Could not subscribe to %s: %s", uuid[-8:], e)

    def _on_notification(self, uuid: str, data: bytearray) -> None:
        """Handle an incoming BLE notification with proper fragmentation reassembly.

        GoPro BLE fragmentation protocol:
          Start packet:  header byte encodes total message length
            - Bit 5 (0x20) clear: lower 5 bits = total length, payload at byte 1
            - Bit 5 (0x20) set:   bytes 1-2 = 16-bit big-endian length, payload at byte 3
          Continuation:  bit 7 (0x80) set, payload at byte 1

        We track expected length and signal completion only when all bytes arrive.
        """
        if uuid in self._notification_handlers:
            try:
                self._notification_handlers[uuid](bytes(data))
            except Exception as e:
                log.warning("Notification handler error for %s: %s", uuid[-8:], e)

        if uuid not in self._response_buffers:
            self._response_buffers[uuid] = bytearray()
            self._response_events[uuid] = asyncio.Event()
            self._response_expected_len[uuid] = 0

        header = data[0] if data else 0
        is_continuation = bool(header & 0x80)

        if not is_continuation:
            self._response_buffers[uuid] = bytearray()
            if header & 0x20:
                # Extended length: high bits in header[0:4], low byte in data[1]
                expected = ((header & 0x1F) << 8) | (data[1] & 0xFF) if len(data) > 1 else 0
                self._response_expected_len[uuid] = expected
                if len(data) > 2:
                    self._response_buffers[uuid].extend(data[2:])
            else:
                expected = header & 0x1F
                self._response_expected_len[uuid] = expected
                if len(data) > 1:
                    self._response_buffers[uuid].extend(data[1:])
        else:
            if len(data) > 1:
                self._response_buffers[uuid].extend(data[1:])

        expected = self._response_expected_len.get(uuid, 0)
        received = len(self._response_buffers.get(uuid, b""))

        if received >= expected and expected > 0:
            event = self._response_events.get(uuid)
            if event is not None:
                event.set()

    async def _write_and_wait(
        self,
        request_uuid: str,
        response_uuid: str,
        data: bytes,
        timeout: float = RESPONSE_TIMEOUT_S,
    ) -> bytes | None:
        """Write a request and wait for the response notification.

        Fragments the write if needed, then waits for the response
        characteristic to receive notification data.
        """
        if not self.is_connected or self._client is None:
            log.error("Cannot write: not connected")
            return None

        # Clear previous response
        self._response_buffers[response_uuid] = bytearray()
        event = asyncio.Event()
        self._response_events[response_uuid] = event

        try:
            # Write request (fragmented if needed)
            for chunk in self._fragment(data):
                await asyncio.wait_for(
                    self._client.write_gatt_char(
                        request_uuid, chunk, response=True,
                    ),
                    timeout=GATT_TIMEOUT_S,
                )

            # Wait for complete response (signalled when all bytes received)
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                log.warning(
                    "Response timeout waiting for %s (wrote %d bytes to %s)",
                    response_uuid[-8:], len(data), request_uuid[-8:],
                )
                buf = self._response_buffers.get(response_uuid, bytearray())
                return bytes(buf) if buf else None

            # Return accumulated response
            buf = self._response_buffers.get(response_uuid, bytearray())
            log.debug(
                "Response from %s: %d bytes",
                response_uuid[-8:], len(buf),
            )
            return bytes(buf)

        except Exception as e:
            log.error("Write-and-wait failed: %s", e)
            return None

    async def _read_string(self, uuid: str) -> str | None:
        """Read a characteristic and decode as UTF-8 string."""
        data = await self.read_characteristic(uuid)
        if data is None:
            return None
        try:
            return data.decode("utf-8").rstrip("\x00")
        except UnicodeDecodeError:
            log.warning("Non-UTF8 data from %s", uuid[-8:])
            return data.hex()

    def _fragment(self, data: bytes) -> list[bytes]:
        """Fragment data into MTU-sized BLE write chunks.

        GoPro BLE fragmentation protocol:
          - First chunk: header byte + payload
            - If total length fits in 5 bits: header = length, payload follows
            - If total length > 31: header bit 5 set, next 2 bytes = length
          - Continuation chunks: header byte (0x80) + payload
        """
        if not data:
            return [b"\x00"]

        total_len = len(data)
        chunks: list[bytes] = []

        # Build first chunk with length header
        if total_len <= 0x1F:
            # Short header: length in lower 5 bits
            header = bytes([total_len & 0x1F])
            max_first_payload = self._mtu - 1
        else:
            # Extended header: bit 5 set, 2-byte big-endian length
            header = bytes([
                0x20 | ((total_len >> 8) & 0x1F),
                total_len & 0xFF,
            ])
            max_first_payload = self._mtu - 2  # Adjusted for the extra length byte

        first_payload = data[:max_first_payload]
        chunks.append(header + first_payload)
        offset = len(first_payload)

        # Continuation chunks
        max_cont_payload = self._mtu - 1  # 1 byte for continuation header
        while offset < total_len:
            chunk_data = data[offset:offset + max_cont_payload]
            chunks.append(bytes([0x80]) + chunk_data)
            offset += len(chunk_data)

        return chunks

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def _inject_client(self, mock_client: Any) -> None:
        """For testing: inject a mock BleakClient."""
        self._client = mock_client
        self._connected = True
        log.debug("Injected mock BLE client for %s", self._address)

    def _simulate_notification(self, uuid: str, data: bytes) -> None:
        """For testing: simulate receiving a BLE notification."""
        self._on_notification(uuid, bytearray(data))
