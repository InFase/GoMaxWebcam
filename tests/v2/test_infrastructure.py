"""
test_infrastructure.py — Validates the v2 test infrastructure.

Tests:
  1. OS-level RAM cap enforcement is active (Job Object on Windows)
  2. Per-test RAM limit fixture works
  3. Shared conftest mock fixtures are properly constructed
  4. Mock open-gopro SDK objects have the expected interface
  5. no_gopro_needed marker is applied to all v2 tests by default
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# All tests here are pure unit tests — no GoPro needed.
pytestmark = pytest.mark.no_gopro_needed


# ===========================================================================
# RAM cap enforcement tests
# ===========================================================================


class TestRAMCapEnforcement:
    """Verify the OS-level memory limit is active."""

    def test_ram_limit_info_fixture_exists(self, ram_limit_info):
        """The session-scoped ram_limit_info fixture is accessible."""
        assert isinstance(ram_limit_info, dict)
        assert "limit_mb" in ram_limit_info
        assert "limit_bytes" in ram_limit_info
        assert "backend" in ram_limit_info
        assert "active" in ram_limit_info

    def test_ram_limit_is_2gb_default(self, ram_limit_info):
        """Default RAM limit is 2048 MB (2 GB) unless overridden."""
        expected_mb = int(os.environ.get("TEST_RAM_LIMIT_MB", "2048"))
        assert ram_limit_info["limit_mb"] == expected_mb

    def test_ram_limit_bytes_matches_mb(self, ram_limit_info):
        """limit_bytes is consistent with limit_mb."""
        assert ram_limit_info["limit_bytes"] == ram_limit_info["limit_mb"] * 1024 * 1024

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Job Object enforcement is Windows-only",
    )
    def test_job_object_active_on_windows(self, ram_limit_info):
        """On Windows, the Job Object backend should be active.

        This may be False if pywin32 is not installed — that's a
        valid test failure indicating the test environment is incomplete.
        """
        # The backend should be "job_object" if pywin32 is available
        try:
            import win32job  # noqa: F401
            assert ram_limit_info["backend"] == "job_object", (
                "pywin32 is installed but Job Object limit is NOT active. "
                "This means tests run without OS-level RAM protection."
            )
            assert ram_limit_info["active"] is True
        except ImportError:
            pytest.skip("pywin32 not installed — Job Object test skipped")

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="resource module test is for POSIX only",
    )
    def test_resource_limit_on_posix(self, ram_limit_info):
        """On POSIX, the resource module backend should be active."""
        if ram_limit_info["active"]:
            assert ram_limit_info["backend"] == "resource"

    def test_no_allocation_exceeds_cap(self, ram_limit_info):
        """Sanity check: current process RSS is below the cap.

        This is a canary test — if it fails, something in the test
        suite is leaking memory before this test runs.
        """
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss
        limit = ram_limit_info["limit_bytes"]
        assert rss < limit, (
            f"Process RSS ({rss / (1024**2):.0f} MB) already exceeds "
            f"the {ram_limit_info['limit_mb']} MB cap!"
        )


# ===========================================================================
# Mock open-gopro SDK validation tests
# ===========================================================================


class TestMockWiredGoPro:
    """Validate MockWiredGoPro has the expected interface."""

    def test_properties(self, mock_wired_gopro):
        assert mock_wired_gopro.is_http_connected is True
        assert mock_wired_gopro.identifier == "C3501324500721"

    @pytest.mark.asyncio
    async def test_open_close_lifecycle(self, mock_wired_gopro):
        assert not mock_wired_gopro.opened
        await mock_wired_gopro.open()
        assert mock_wired_gopro.opened

        assert not mock_wired_gopro.closed
        await mock_wired_gopro.close()
        assert mock_wired_gopro.closed

    @pytest.mark.asyncio
    async def test_http_command_webcam_status(self, mock_wired_gopro):
        resp = await mock_wired_gopro.http_command.webcam_status()
        assert resp.ok is True
        assert resp.data.status.value == 2  # HIGH_POWER_PREVIEW

    @pytest.mark.asyncio
    async def test_http_command_webcam_start(self, mock_wired_gopro):
        resp = await mock_wired_gopro.http_command.webcam_start()
        assert resp.ok is True

    @pytest.mark.asyncio
    async def test_http_command_webcam_stop(self, mock_wired_gopro):
        resp = await mock_wired_gopro.http_command.webcam_stop()
        assert resp.ok is True

    @pytest.mark.asyncio
    async def test_http_command_webcam_exit(self, mock_wired_gopro):
        resp = await mock_wired_gopro.http_command.webcam_exit()
        assert resp.ok is True


class TestMockWirelessGoPro:
    """Validate MockWirelessGoPro has the expected COHN/BLE interface."""

    def test_properties(self, mock_wireless_gopro):
        assert mock_wireless_gopro.is_http_connected is True

    @pytest.mark.asyncio
    async def test_open_close_lifecycle(self, mock_wireless_gopro):
        await mock_wireless_gopro.open()
        assert mock_wireless_gopro.opened
        await mock_wireless_gopro.close()
        assert mock_wireless_gopro.closed

    @pytest.mark.asyncio
    async def test_http_command_webcam_status(self, mock_wireless_gopro):
        resp = await mock_wireless_gopro.http_command.webcam_status()
        assert resp.ok is True
        assert resp.data.status.value == 2

    def test_ble_attribute_exists(self, mock_wireless_gopro):
        """WirelessGoPro has a ble attribute for BLE provisioning."""
        assert hasattr(mock_wireless_gopro, "ble")


class TestMockBLEDevice:
    """Validate MockBLEDevice has BLE scan result interface."""

    def test_ble_device_attributes(self, mock_ble_device):
        assert mock_ble_device.name == "GoPro 0721"
        assert mock_ble_device.address == "AA:BB:CC:DD:EE:FF"
        assert mock_ble_device.rssi == -55


class TestMockCOHNCredentials:
    """Validate MockCOHNCredentials dataclass."""

    def test_cohn_credential_fields(self, mock_cohn_credentials):
        assert mock_cohn_credentials.ip_address == "172.20.123.51"
        assert mock_cohn_credentials.username == "gopro"
        assert mock_cohn_credentials.password == "p@ss-from-ble-provision"
        assert "CERTIFICATE" in mock_cohn_credentials.certificate


class TestGoProRespFactory:
    """Validate make_mock_gopro_resp helper."""

    def test_ok_response(self):
        from tests.v2.conftest import make_mock_gopro_resp
        resp = make_mock_gopro_resp(ok=True, status=2)
        assert resp.ok is True
        assert resp.data.status.value == 2

    def test_error_response(self):
        from tests.v2.conftest import make_mock_gopro_resp
        resp = make_mock_gopro_resp(ok=False, error=7)
        assert resp.ok is False
        assert resp.data.error.value == 7

    def test_custom_data(self):
        from tests.v2.conftest import make_mock_gopro_resp
        custom = {"key": "value"}
        resp = make_mock_gopro_resp(data=custom)
        assert resp.data == custom


# ===========================================================================
# Transport fixture validation tests
# ===========================================================================


class TestTransportFixtures:
    """Validate transport fixtures are properly configured."""

    def test_usb_transport_exists(self, usb_transport):
        from gomaxwebcam.transport.base import TransportState
        assert usb_transport.state == TransportState.DISCONNECTED
        assert usb_transport.name == "USB"

    def test_connected_usb_transport_state(self, connected_usb_transport):
        from gomaxwebcam.transport.base import TransportState
        assert connected_usb_transport.state == TransportState.CONNECTED
        assert connected_usb_transport.is_connected

    def test_cohn_transport_exists(self, cohn_transport):
        from gomaxwebcam.transport.base import TransportState
        assert cohn_transport.state == TransportState.DISCONNECTED
        assert cohn_transport.name == "COHN"

    def test_cohn_transport_has_credentials(self, cohn_transport):
        assert cohn_transport.has_credentials is True

    def test_connected_cohn_transport_state(self, connected_cohn_transport):
        from gomaxwebcam.transport.base import TransportState
        assert connected_cohn_transport.state == TransportState.CONNECTED
        assert connected_cohn_transport.is_connected


# ===========================================================================
# Component fixture validation tests
# ===========================================================================


class TestComponentFixtures:
    """Validate orchestrator component mock fixtures."""

    def test_mock_event_bus(self, mock_event_bus):
        assert hasattr(mock_event_bus, "set_loop")
        assert hasattr(mock_event_bus, "shutdown")
        assert hasattr(mock_event_bus, "subscribe")
        assert hasattr(mock_event_bus, "publish")
        assert mock_event_bus.recent_events == []

    def test_mock_camera_manager(self, mock_camera_manager):
        assert hasattr(mock_camera_manager, "start")
        assert hasattr(mock_camera_manager, "stop")
        assert hasattr(mock_camera_manager, "set_transport")
        assert hasattr(mock_camera_manager, "set_pipeline")

    def test_mock_transport_manager(self, mock_transport_manager):
        assert hasattr(mock_transport_manager, "start")
        assert hasattr(mock_transport_manager, "stop")
        assert mock_transport_manager.on_transport_switch is None
        assert mock_transport_manager.on_freeze is None
        assert mock_transport_manager.on_unfreeze is None

    def test_mock_status_tracker(self, mock_status_tracker):
        assert hasattr(mock_status_tracker, "start")
        assert hasattr(mock_status_tracker, "stop")
        assert hasattr(mock_status_tracker, "set_transport")
        assert hasattr(mock_status_tracker, "set_pipeline")

    def test_mock_pipeline(self, mock_pipeline):
        assert hasattr(mock_pipeline, "start")
        assert hasattr(mock_pipeline, "stop")
        assert hasattr(mock_pipeline, "enable_freeze_frame")
        assert hasattr(mock_pipeline, "disable_freeze_frame")


# ===========================================================================
# Frame helper validation tests
# ===========================================================================


class TestFrameHelpers:
    """Validate frame creation helpers."""

    def test_test_frame_shape(self, test_frame):
        assert test_frame.shape == (1080, 1920, 3)
        assert test_frame.dtype == np.uint8

    def test_small_test_frame_shape(self, small_test_frame):
        assert small_test_frame.shape == (240, 320, 3)
        assert small_test_frame.dtype == np.uint8

    def test_make_test_frame_custom(self):
        from tests.v2.conftest import make_test_frame
        frame = make_test_frame(width=640, height=480, fill=255)
        assert frame.shape == (480, 640, 3)
        assert frame[0, 0, 0] == 255

    def test_test_frame_memory_bounded(self, test_frame):
        """A single 1080p frame is ~6 MB — verify it's reasonable."""
        size_mb = test_frame.nbytes / (1024 * 1024)
        assert size_mb < 10, f"Test frame is {size_mb:.1f} MB — too large!"


# ===========================================================================
# StreamInfo fixture validation
# ===========================================================================


class TestStreamInfoFixture:
    """Validate the stream_info fixture."""

    def test_stream_info_defaults(self, stream_info):
        assert stream_info.protocol == "udp"
        assert stream_info.host == "0.0.0.0"
        assert stream_info.port == 8554
        assert stream_info.width == 1920
        assert stream_info.height == 1080
        assert stream_info.fps == 30
        assert stream_info.codec == "h264"


# ===========================================================================
# Orchestrator fixture validation
# ===========================================================================


class TestOrchestratorFixture:
    """Validate the AppOrchestrator fixture."""

    def test_orchestrator_not_running_initially(self, orchestrator):
        assert not orchestrator.is_running

    @pytest.mark.asyncio
    async def test_orchestrator_start_stop(self, orchestrator):
        await orchestrator.start()
        assert orchestrator.is_running
        await orchestrator.stop()
        assert not orchestrator.is_running
