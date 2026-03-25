"""
tests/v2/conftest.py — Shared fixtures for v2 unit tests.

Provides:
  1. Mock open-gopro SDK classes (WiredGoPro, WirelessGoPro, GoProResp)
  2. Mock transport fixtures (USBTransport, COHNTransport)
  3. Mock pipeline fixtures (decoder, virtual camera sink)
  4. Mock orchestrator component fixtures (EventBus, CameraManager, etc.)
  5. RAM limit introspection (inherited from tests/conftest.py)

All fixtures mock at the open-gopro SDK boundary — no real hardware,
network, or BLE is ever used. Tests marked ``no_gopro_needed`` are
never skipped by the GoPro detection guard in the parent conftest.

The parent tests/conftest.py applies the OS-level Job Object memory
limit at import time. This conftest supplements it with v2-specific
fixtures but does NOT duplicate the RAM enforcement.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mark all tests in tests/v2/ as no_gopro_needed by default.
# Individual test files can override with @pytest.mark.hardware.
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.no_gopro_needed


# ===========================================================================
# open-gopro SDK mock classes
# ===========================================================================


class MockWebcamResponse:
    """Simulates open_gopro WebcamResponse model (webcam_status data)."""

    def __init__(self, status: int | None = None, error: int = 0):
        self.status = MagicMock(value=status) if status is not None else None
        self.error = MagicMock(value=error)


class MockGoProResp:
    """Simulates open_gopro GoProResp — the standard API response wrapper."""

    def __init__(
        self,
        ok: bool = True,
        status: int | None = None,
        error: int = 0,
        data: Any = None,
    ):
        self.ok = ok
        if data is not None:
            self.data = data
        else:
            self.data = MockWebcamResponse(status=status, error=error)


class MockHttpCommand:
    """Simulates gopro.http_command with webcam methods (USB + COHN)."""

    def __init__(
        self,
        webcam_status_val: int = 2,  # HIGH_POWER_PREVIEW
        webcam_error: int = 0,
        webcam_start_ok: bool = True,
    ):
        self._webcam_status_val = webcam_status_val
        self._webcam_error = webcam_error
        self._webcam_start_ok = webcam_start_ok

    async def webcam_status(self) -> MockGoProResp:
        return MockGoProResp(
            ok=True,
            status=self._webcam_status_val,
            error=self._webcam_error,
        )

    async def webcam_start(self, **kwargs: Any) -> MockGoProResp:
        return MockGoProResp(
            ok=self._webcam_start_ok,
            status=1,  # IDLE initially
            error=self._webcam_error,
        )

    async def webcam_stop(self) -> MockGoProResp:
        return MockGoProResp(ok=True, status=0, error=0)

    async def webcam_exit(self) -> MockGoProResp:
        return MockGoProResp(ok=True, status=0, error=0)


class MockWiredGoPro:
    """Simulates open-gopro's WiredGoPro for USB connections.

    Supports the _inject pattern used by USBTransport tests.
    """

    def __init__(
        self,
        is_http_connected: bool = True,
        webcam_status_val: int = 2,
        webcam_error: int = 0,
        webcam_start_ok: bool = True,
        serial: str = "C3501324500721",
    ):
        self._is_http_connected = is_http_connected
        self._serial = serial
        self.http_command = MockHttpCommand(
            webcam_status_val=webcam_status_val,
            webcam_error=webcam_error,
            webcam_start_ok=webcam_start_ok,
        )
        self.opened = False
        self.closed = False

    @property
    def is_http_connected(self) -> bool:
        return self._is_http_connected

    @property
    def identifier(self) -> str:
        return self._serial

    async def open(self, **kwargs: Any) -> None:
        self.opened = True

    async def close(self) -> None:
        self.closed = True


class MockWirelessGoPro:
    """Simulates open-gopro's WirelessGoPro for COHN/BLE connections.

    Supports the _inject pattern used by COHNTransport tests.
    """

    def __init__(
        self,
        is_http_connected: bool = True,
        webcam_status_val: int = 2,
        webcam_error: int = 0,
        webcam_start_ok: bool = True,
    ):
        self._is_http_connected = is_http_connected
        self.http_command = MockHttpCommand(
            webcam_status_val=webcam_status_val,
            webcam_error=webcam_error,
            webcam_start_ok=webcam_start_ok,
        )
        self.opened = False
        self.closed = False
        # BLE provisioning mocks
        self.ble = MagicMock()
        self.ble.scan = AsyncMock(return_value=[])

    @property
    def is_http_connected(self) -> bool:
        return self._is_http_connected

    async def open(self, **kwargs: Any) -> None:
        self.opened = True

    async def close(self) -> None:
        self.closed = True


class MockBLEDevice:
    """Simulates a BLE device returned by open-gopro BLE scan."""

    def __init__(
        self,
        name: str = "GoPro 0721",
        address: str = "AA:BB:CC:DD:EE:FF",
        rssi: int = -55,
    ):
        self.name = name
        self.address = address
        self.rssi = rssi


@dataclass
class MockCOHNCredentials:
    """Simulates COHN credentials from open-gopro provisioning."""
    ip_address: str = "172.20.123.51"
    username: str = "gopro"
    password: str = "p@ss-from-ble-provision"
    certificate: str = "-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----"


# ===========================================================================
# Factory helpers
# ===========================================================================


def make_mock_wired_gopro(**kwargs: Any) -> MockWiredGoPro:
    """Create a MockWiredGoPro with sensible defaults. Override via kwargs."""
    return MockWiredGoPro(**kwargs)


def make_mock_wireless_gopro(**kwargs: Any) -> MockWirelessGoPro:
    """Create a MockWirelessGoPro with sensible defaults. Override via kwargs."""
    return MockWirelessGoPro(**kwargs)


def make_mock_gopro_resp(
    ok: bool = True, status: int | None = None, error: int = 0, data: Any = None,
) -> MockGoProResp:
    """Create a MockGoProResp."""
    return MockGoProResp(ok=ok, status=status, error=error, data=data)


def make_test_frame(
    width: int = 1920, height: int = 1080, fill: int = 128,
) -> np.ndarray:
    """Create a small numpy RGB24 frame for testing.

    Returns an ndarray with shape (height, width, 3), dtype=uint8.
    """
    return np.full((height, width, 3), fill, dtype=np.uint8)


# ===========================================================================
# open-gopro SDK fixtures
# ===========================================================================


@pytest.fixture
def mock_wired_gopro() -> MockWiredGoPro:
    """A MockWiredGoPro with default (healthy) responses."""
    return make_mock_wired_gopro()


@pytest.fixture
def mock_wireless_gopro() -> MockWirelessGoPro:
    """A MockWirelessGoPro with default (healthy) responses."""
    return make_mock_wireless_gopro()


@pytest.fixture
def mock_ble_device() -> MockBLEDevice:
    """A MockBLEDevice representing a discovered GoPro."""
    return MockBLEDevice()


@pytest.fixture
def mock_cohn_credentials() -> MockCOHNCredentials:
    """Mock COHN credentials from open-gopro provisioning."""
    return MockCOHNCredentials()


# ===========================================================================
# Transport fixtures
# ===========================================================================


@pytest.fixture
def usb_transport():
    """Create a USBTransport without triggering any real USB/network calls."""
    from gomaxwebcam.transport.usb import USBTransport
    return USBTransport()


@pytest.fixture
def connected_usb_transport(usb_transport, mock_wired_gopro):
    """USBTransport already in CONNECTED state with a mock WiredGoPro."""
    from gomaxwebcam.transport.base import TransportState
    usb_transport._inject_gopro(mock_wired_gopro)
    usb_transport._inject_camera_ip("172.20.123.51")
    usb_transport._set_state(TransportState.CONNECTED)
    usb_transport._connected_at = time.monotonic()
    return usb_transport


@pytest.fixture
def cohn_transport():
    """Create a COHNTransport with test credentials (no network calls)."""
    from gomaxwebcam.transport.cohn import COHNTransport
    return COHNTransport(
        ip_address="192.168.1.100",
        username="gopro",
        password="test-password-123",
        udp_port=8554,
    )


@pytest.fixture
def connected_cohn_transport(cohn_transport, mock_wireless_gopro):
    """COHNTransport already in CONNECTED state with a mock WirelessGoPro."""
    from gomaxwebcam.transport.base import TransportState
    cohn_transport._inject_camera_ip("192.168.1.100")
    cohn_transport._inject_gopro(mock_wireless_gopro)
    cohn_transport._set_state(TransportState.CONNECTED)
    cohn_transport._connected_at = time.monotonic()
    return cohn_transport


# ===========================================================================
# EventBus fixture
# ===========================================================================


@pytest.fixture
def mock_event_bus():
    """A mock EventBus with all methods stubbed."""
    bus = MagicMock()
    bus.set_loop = MagicMock()
    bus.shutdown = AsyncMock()
    bus.subscribe = MagicMock()
    bus.publish = MagicMock()
    bus.recent_events = []
    return bus


# ===========================================================================
# Component mock fixtures (for AppOrchestrator tests)
# ===========================================================================


@pytest.fixture
def mock_camera_manager():
    """A mock CameraManager with async start/stop."""
    cm = MagicMock()
    cm.start = AsyncMock()
    cm.stop = AsyncMock()
    cm.set_transport = MagicMock()
    cm.set_pipeline = MagicMock()
    cm.transport = None
    return cm


@pytest.fixture
def mock_transport_manager():
    """A mock TransportManager with callback slots."""
    tm = MagicMock()
    tm.start = AsyncMock(return_value=True)
    tm.stop = AsyncMock()
    tm.on_transport_switch = None
    tm.on_freeze = None
    tm.on_unfreeze = None
    return tm


@pytest.fixture
def mock_status_tracker():
    """A mock CameraStatusTracker."""
    st = MagicMock()
    st.start = AsyncMock()
    st.stop = AsyncMock()
    st.set_transport = MagicMock()
    st.set_pipeline = MagicMock()
    return st


@pytest.fixture
def mock_pipeline():
    """A mock FramePipeline with async start/stop."""
    pipeline = AsyncMock()
    pipeline.start = AsyncMock(return_value=True)
    pipeline.stop = AsyncMock()
    pipeline.enable_freeze_frame = MagicMock()
    pipeline.disable_freeze_frame = MagicMock()
    pipeline.is_running = False
    pipeline.stats = MagicMock()
    return pipeline


# ===========================================================================
# Frame / decoder helpers
# ===========================================================================


@pytest.fixture
def test_frame():
    """A 1920×1080 RGB24 test frame (gray, fill=128)."""
    return make_test_frame()


@pytest.fixture
def small_test_frame():
    """A small 320×240 RGB24 test frame for fast tests."""
    return make_test_frame(width=320, height=240, fill=64)


# ===========================================================================
# StreamInfo fixture
# ===========================================================================


@pytest.fixture
def stream_info():
    """A standard StreamInfo for testing pipeline integration."""
    from gomaxwebcam.transport.base import StreamInfo
    return StreamInfo(
        protocol="udp",
        host="0.0.0.0",
        port=8554,
        width=1920,
        height=1080,
        fps=30,
        codec="h264",
    )


# ===========================================================================
# AppOrchestrator fixture
# ===========================================================================


@pytest.fixture
def orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager, mock_status_tracker):
    """An AppOrchestrator wired to all mock components."""
    from gomaxwebcam.orchestrator import AppOrchestrator
    return AppOrchestrator(
        event_bus=mock_event_bus,
        camera_manager=mock_camera_manager,
        transport_manager=mock_transport_manager,
        status_tracker=mock_status_tracker,
    )
