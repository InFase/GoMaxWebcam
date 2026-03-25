"""
Tests for BLE GATT client — mocked at the bleak boundary.

No real BLE hardware required. These tests verify:
  - BLE connection and disconnection lifecycle
  - Reading WiFi AP SSID and password characteristics
  - Writing commands with fragmentation
  - Notification handling and response reassembly
  - Write-and-wait request-response pattern
  - Error handling (timeouts, disconnects)
  - Fragmentation protocol correctness
"""

from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest


def _patch_bleak(client_cls=None):
    """Create a mock bleak module with a controllable BleakClient."""
    mock_bleak = types.ModuleType("bleak")
    mock_bleak.BleakClient = client_cls or MagicMock()
    mock_bleak.BleakScanner = MagicMock()
    return patch.dict("sys.modules", {"bleak": mock_bleak})


# ---------------------------------------------------------------------------
# Mock bleak objects
# ---------------------------------------------------------------------------


class MockBleakClient:
    """Simulates a bleak BleakClient with configurable behavior."""

    def __init__(self, address: str = "AA:BB:CC:DD:EE:FF", **kwargs):
        self.address = address
        self.is_connected = True  # Default to True since _inject_client assumes connected
        self.mtu_size = 256

        # Track operations
        self.connect = AsyncMock(side_effect=self._do_connect)
        self.disconnect = AsyncMock(side_effect=self._do_disconnect)
        self.read_gatt_char = AsyncMock(return_value=b"test_data")
        self.write_gatt_char = AsyncMock()
        self.start_notify = AsyncMock()

        # Track notification callbacks by UUID
        self._notify_callbacks: dict = {}

    async def _do_connect(self):
        self.is_connected = True

    async def _do_disconnect(self):
        self.is_connected = False

    async def _start_notify(self, uuid, callback):
        self._notify_callbacks[uuid] = callback


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_bleak_client():
    return MockBleakClient()


@pytest.fixture
def gatt_client():
    from gomaxwebcam.ble.gatt_client import GoProBLEClient
    return GoProBLEClient(address="AA:BB:CC:DD:EE:FF")


@pytest.fixture
def connected_client(gatt_client, mock_bleak_client):
    """A GoProBLEClient with an injected mock BleakClient."""
    gatt_client._inject_client(mock_bleak_client)
    return gatt_client


# ---------------------------------------------------------------------------
# Connection lifecycle tests
# ---------------------------------------------------------------------------


class TestGATTClientConnection:
    def test_initial_state(self, gatt_client):
        assert gatt_client.address == "AA:BB:CC:DD:EE:FF"
        assert not gatt_client.is_connected
        assert gatt_client.mtu == 256

    @pytest.mark.anyio
    async def test_connect_success(self, gatt_client):
        mock_client = MockBleakClient()

        with _patch_bleak(client_cls=lambda *a, **kw: mock_client):
            result = await gatt_client.connect(timeout=5.0)

        assert result is True
        assert gatt_client.is_connected
        mock_client.connect.assert_awaited_once()

    @pytest.mark.anyio
    async def test_connect_bleak_not_installed(self, gatt_client):
        with patch.dict("sys.modules", {"bleak": None}):
            result = await gatt_client.connect()
            assert result is False

    @pytest.mark.anyio
    async def test_connect_failure(self, gatt_client):
        mock_client = MockBleakClient()
        mock_client.connect = AsyncMock(side_effect=OSError("Connection refused"))

        with _patch_bleak(client_cls=lambda *a, **kw: mock_client):
            result = await gatt_client.connect()

        assert result is False
        assert not gatt_client.is_connected

    @pytest.mark.anyio
    async def test_disconnect(self, connected_client, mock_bleak_client):
        assert connected_client.is_connected
        await connected_client.disconnect()

        assert not connected_client.is_connected

    @pytest.mark.anyio
    async def test_disconnect_clears_state(self, connected_client):
        await connected_client.disconnect()

        assert connected_client._client is None
        assert len(connected_client._response_events) == 0

    @pytest.mark.anyio
    async def test_disconnect_when_not_connected(self, gatt_client):
        # Should not raise
        await gatt_client.disconnect()


# ---------------------------------------------------------------------------
# Read characteristic tests
# ---------------------------------------------------------------------------


class TestGATTClientRead:
    @pytest.mark.anyio
    async def test_read_wifi_ssid(self, connected_client, mock_bleak_client):
        mock_bleak_client.read_gatt_char = AsyncMock(return_value=b"MyGoProWiFi")

        ssid = await connected_client.read_wifi_ssid()

        assert ssid == "MyGoProWiFi"
        mock_bleak_client.read_gatt_char.assert_awaited_once()

    @pytest.mark.anyio
    async def test_read_wifi_password(self, connected_client, mock_bleak_client):
        mock_bleak_client.read_gatt_char = AsyncMock(return_value=b"s3cretPass!")

        password = await connected_client.read_wifi_password()

        assert password == "s3cretPass!"

    @pytest.mark.anyio
    async def test_read_strips_null_bytes(self, connected_client, mock_bleak_client):
        mock_bleak_client.read_gatt_char = AsyncMock(return_value=b"SSID\x00\x00")

        ssid = await connected_client.read_wifi_ssid()

        assert ssid == "SSID"

    @pytest.mark.anyio
    async def test_read_not_connected(self, gatt_client):
        result = await gatt_client.read_characteristic("some-uuid")
        assert result is None

    @pytest.mark.anyio
    async def test_read_timeout(self, connected_client, mock_bleak_client):
        mock_bleak_client.read_gatt_char = AsyncMock(
            side_effect=asyncio.TimeoutError
        )

        result = await connected_client.read_characteristic("some-uuid")
        assert result is None

    @pytest.mark.anyio
    async def test_read_error(self, connected_client, mock_bleak_client):
        mock_bleak_client.read_gatt_char = AsyncMock(
            side_effect=OSError("Read failed")
        )

        result = await connected_client.read_characteristic("some-uuid")
        assert result is None


# ---------------------------------------------------------------------------
# Fragmentation tests
# ---------------------------------------------------------------------------


class TestFragmentation:
    def test_empty_data(self, gatt_client):
        chunks = gatt_client._fragment(b"")
        assert chunks == [b"\x00"]

    def test_short_payload_single_chunk(self, gatt_client):
        """Payload <= 31 bytes should fit in a single chunk."""
        data = b"\x01\x02\x03"
        chunks = gatt_client._fragment(data)
        assert len(chunks) == 1
        # First byte is length header (3), then the payload
        assert chunks[0][0] == 3
        assert chunks[0][1:] == data

    def test_max_short_header(self, gatt_client):
        """31 bytes is the max for short header (5-bit length)."""
        data = bytes(range(31))
        chunks = gatt_client._fragment(data)
        assert len(chunks) == 1
        assert chunks[0][0] == 31

    def test_extended_header_for_large_payload(self, gatt_client):
        """Payloads > 31 bytes use extended 2-byte length header."""
        data = bytes(range(50))
        chunks = gatt_client._fragment(data)

        # First chunk should have extended header (bit 5 set)
        first_header = chunks[0][0]
        assert first_header & 0x20  # Extended header bit

    def test_large_payload_fragments(self, gatt_client):
        """Large payloads should be split into multiple chunks."""
        gatt_client._mtu = 20  # Small MTU to force fragmentation
        data = bytes(range(100))
        chunks = gatt_client._fragment(data)

        # Should need multiple chunks
        assert len(chunks) > 1

        # Continuation chunks should have header byte 0x80
        for chunk in chunks[1:]:
            assert chunk[0] & 0x80

    def test_fragment_reassembly_roundtrip(self, gatt_client):
        """Verify that fragmented data can be reassembled."""
        gatt_client._mtu = 20
        original = bytes(range(100))
        chunks = gatt_client._fragment(original)

        # Reassemble following the GoPro BLE fragmentation protocol
        reassembled = bytearray()
        for i, chunk in enumerate(chunks):
            if i == 0:
                header = chunk[0]
                if header & 0x20:  # Extended header: 2 header bytes
                    reassembled.extend(chunk[2:])
                else:
                    reassembled.extend(chunk[1:])
            else:
                # Skip continuation header byte
                reassembled.extend(chunk[1:])

        assert bytes(reassembled) == original


# ---------------------------------------------------------------------------
# Notification handling tests
# ---------------------------------------------------------------------------


class TestNotificationHandling:
    def test_simple_notification(self, connected_client):
        from gomaxwebcam.ble.uuids import COMMAND_RESPONSE_UUID

        # Initialize response tracking
        connected_client._response_events[COMMAND_RESPONSE_UUID] = asyncio.Event()
        connected_client._response_buffers[COMMAND_RESPONSE_UUID] = bytearray()

        # Simulate a short notification (length=3, payload=0x01,0x02,0x03)
        connected_client._simulate_notification(
            COMMAND_RESPONSE_UUID,
            bytes([3, 0x01, 0x02, 0x03]),
        )

        buf = connected_client._response_buffers[COMMAND_RESPONSE_UUID]
        assert bytes(buf) == bytes([0x01, 0x02, 0x03])

    def test_notification_sets_event(self, connected_client):
        from gomaxwebcam.ble.uuids import COMMAND_RESPONSE_UUID

        event = asyncio.Event()
        connected_client._response_events[COMMAND_RESPONSE_UUID] = event
        connected_client._response_buffers[COMMAND_RESPONSE_UUID] = bytearray()

        connected_client._simulate_notification(
            COMMAND_RESPONSE_UUID,
            bytes([1, 0xFF]),
        )

        assert event.is_set()

    def test_continuation_packet_appends(self, connected_client):
        from gomaxwebcam.ble.uuids import COMMAND_RESPONSE_UUID

        connected_client._response_events[COMMAND_RESPONSE_UUID] = asyncio.Event()
        connected_client._response_buffers[COMMAND_RESPONSE_UUID] = bytearray()

        # First packet (start)
        connected_client._simulate_notification(
            COMMAND_RESPONSE_UUID,
            bytes([2, 0xAA, 0xBB]),
        )

        # Continuation packet
        connected_client._simulate_notification(
            COMMAND_RESPONSE_UUID,
            bytes([0x80, 0xCC, 0xDD]),
        )

        buf = connected_client._response_buffers[COMMAND_RESPONSE_UUID]
        assert bytes(buf) == bytes([0xAA, 0xBB, 0xCC, 0xDD])

    def test_custom_notification_handler(self, connected_client):
        from gomaxwebcam.ble.uuids import COMMAND_RESPONSE_UUID

        received: list[bytes] = []
        connected_client.register_notification_handler(
            COMMAND_RESPONSE_UUID,
            lambda data: received.append(data),
        )

        connected_client._response_events[COMMAND_RESPONSE_UUID] = asyncio.Event()
        connected_client._response_buffers[COMMAND_RESPONSE_UUID] = bytearray()

        connected_client._simulate_notification(
            COMMAND_RESPONSE_UUID,
            bytes([1, 0xFF]),
        )

        assert len(received) == 1
        assert received[0] == bytes([1, 0xFF])


# ---------------------------------------------------------------------------
# Write-and-wait tests
# ---------------------------------------------------------------------------


class TestWriteAndWait:
    @pytest.mark.anyio
    async def test_write_command_sends_and_waits(self, connected_client, mock_bleak_client):
        from gomaxwebcam.ble.uuids import COMMAND_RESPONSE_UUID

        # Set up mock to trigger notification after write
        async def fake_write(uuid, data, response=True):
            # Simulate camera response via notification
            await asyncio.sleep(0.01)
            connected_client._simulate_notification(
                COMMAND_RESPONSE_UUID,
                bytes([2, 0x01, 0x00]),  # success response
            )

        mock_bleak_client.write_gatt_char = AsyncMock(side_effect=fake_write)

        result = await connected_client.write_command(bytes([0x01, 0x02]))

        assert result is not None
        assert len(result) > 0

    @pytest.mark.anyio
    async def test_write_not_connected(self, gatt_client):
        result = await gatt_client.write_command(b"\x01")
        assert result is None

    @pytest.mark.anyio
    async def test_write_network_management(self, connected_client, mock_bleak_client):
        from gomaxwebcam.ble.uuids import NETWORK_MGMT_RESPONSE_UUID

        async def fake_write(uuid, data, response=True):
            await asyncio.sleep(0.01)
            connected_client._simulate_notification(
                NETWORK_MGMT_RESPONSE_UUID,
                bytes([3, 0xF5, 0x01, 0x00]),
            )

        mock_bleak_client.write_gatt_char = AsyncMock(side_effect=fake_write)

        result = await connected_client.write_network_management(bytes([0xF5, 0x01]))

        assert result is not None

    @pytest.mark.anyio
    async def test_write_raw(self, connected_client, mock_bleak_client):
        await connected_client.write_raw("some-uuid", b"\x01\x02\x03")
        mock_bleak_client.write_gatt_char.assert_awaited()

    @pytest.mark.anyio
    async def test_write_raw_not_connected(self, gatt_client):
        # Should not raise
        await gatt_client.write_raw("some-uuid", b"\x01")
