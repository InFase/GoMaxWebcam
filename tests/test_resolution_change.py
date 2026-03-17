"""
Tests for mid-session resolution change detection and API commands.
(Sub-AC 1 of AC 16)

Tests cover:
  - RESOLUTION_MAP and VALID_RESOLUTIONS constants
  - GoProConnection.needs_resolution_change() detection logic
  - GoProConnection.change_resolution() API command (stop + start cycle)
  - GoProConnection resolution tracking (_current_resolution, _current_fov)
  - GoProConnection.start_webcam() resolution validation and tracking
  - AppController.request_resolution_change() detection-layer orchestration
  - AppController.change_resolution() freeze-frame and fallback logic
  - Invalid resolution code rejection
  - No-op when already at requested resolution
"""

import sys
import os
import unittest
from dataclasses import dataclass
from unittest.mock import patch, MagicMock, call

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gopro_connection import (
    GoProConnection,
    WebcamStatus,
    ConnectionState,
    RESOLUTION_MAP,
    VALID_RESOLUTIONS,
)

import pytest

pytestmark = pytest.mark.no_gopro_needed


@dataclass
class MockConfig:
    """Minimal config for testing resolution changes."""
    resolution: int = 4
    fov: int = 4
    discovery_timeout: float = 1.0
    discovery_overall_timeout: float = 5.0
    idle_reset_delay: float = 0.01  # Fast for tests
    keepalive_interval: float = 0.1
    reconnect_delay: float = 0.01
    ncm_adapter_wait: float = 0.01
    discovery_retry_interval: float = 0.01
    discovery_max_retries: int = 3
    stream_width: int = 1920
    stream_height: int = 1080
    stream_fps: int = 30
    udp_port: int = 8554
    ffmpeg_path: str = "ffmpeg"
    virtual_camera_name: str = "GoPro Webcam"
    auto_start_webcam: bool = True
    auto_connect_on_launch: bool = True
    health_check_interval: float = 5.0
    reconnect_max_retries: int = 0
    log_level: str = "INFO"
    log_max_session_files: int = 5
    log_max_total_bytes: int = 50 * 1024 * 1024
    log_max_file_bytes: int = 10 * 1024 * 1024
    _config_path: str = ""

    def save(self):
        pass


def make_mock_response(json_data, status_code=200):
    """Create a mock requests.Response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status.return_value = None
    return mock


# ── Constants ──────────────────────────────────────────────────


class TestResolutionConstants(unittest.TestCase):
    """Test RESOLUTION_MAP and VALID_RESOLUTIONS constants."""

    def test_resolution_map_has_all_codes(self):
        """RESOLUTION_MAP contains 720p, 1080p, and 4K."""
        self.assertIn(7, RESOLUTION_MAP)    # 720p
        self.assertIn(4, RESOLUTION_MAP)    # 1080p
        self.assertIn(12, RESOLUTION_MAP)   # 4K

    def test_resolution_map_dimensions(self):
        """Each resolution maps to correct (width, height, label)."""
        self.assertEqual(RESOLUTION_MAP[7], (1280, 720, "720p"))
        self.assertEqual(RESOLUTION_MAP[4], (1920, 1080, "1080p"))
        self.assertEqual(RESOLUTION_MAP[12], (3840, 2160, "4K"))

    def test_valid_resolutions_matches_map(self):
        """VALID_RESOLUTIONS matches RESOLUTION_MAP keys."""
        self.assertEqual(VALID_RESOLUTIONS, set(RESOLUTION_MAP.keys()))


# ── GoProConnection.needs_resolution_change() ──────────────────


class TestNeedsResolutionChange(unittest.TestCase):
    """Test GoProConnection.needs_resolution_change() detection."""

    def setUp(self):
        self.config = MockConfig()
        self.gopro = GoProConnection(self.config)

    def test_no_change_needed_when_not_streaming(self):
        """When not streaming (_current_resolution is None), no change needed."""
        self.assertIsNone(self.gopro._current_resolution)
        self.assertFalse(self.gopro.needs_resolution_change(4))

    def test_change_needed_when_different_resolution(self):
        """Detects when requested resolution differs from active."""
        self.gopro._current_resolution = 4  # 1080p
        self.gopro._current_fov = 4
        self.assertTrue(self.gopro.needs_resolution_change(7))   # Want 720p

    def test_no_change_when_same_resolution(self):
        """No change needed when already at requested resolution."""
        self.gopro._current_resolution = 4
        self.gopro._current_fov = 4
        self.assertFalse(self.gopro.needs_resolution_change(4))

    def test_change_needed_when_different_fov(self):
        """Detects when requested FOV differs from active."""
        self.gopro._current_resolution = 4
        self.gopro._current_fov = 4  # linear
        self.assertTrue(self.gopro.needs_resolution_change(4, fov=0))  # wide

    def test_no_change_when_same_resolution_and_fov(self):
        """No change needed when both resolution and FOV match."""
        self.gopro._current_resolution = 4
        self.gopro._current_fov = 4
        self.assertFalse(self.gopro.needs_resolution_change(4, fov=4))

    def test_fov_none_means_no_fov_check(self):
        """When fov=None, only resolution is checked."""
        self.gopro._current_resolution = 4
        self.gopro._current_fov = 4
        self.assertFalse(self.gopro.needs_resolution_change(4, fov=None))

    def test_change_needed_4k_to_720p(self):
        """4K to 720p is detected as a change."""
        self.gopro._current_resolution = 12
        self.gopro._current_fov = 0
        self.assertTrue(self.gopro.needs_resolution_change(7))


# ── GoProConnection.change_resolution() ────────────────────────


class TestChangeResolution(unittest.TestCase):
    """Test GoProConnection.change_resolution() API command."""

    def setUp(self):
        self.config = MockConfig()
        self.gopro = GoProConnection(self.config)
        self.gopro.ip = "172.20.123.51"
        self.gopro.base_url = "http://172.20.123.51:8080"
        self.gopro._connected = True
        self.gopro._current_resolution = 4
        self.gopro._current_fov = 4

    def test_invalid_resolution_rejected(self):
        """Invalid resolution codes are rejected immediately."""
        self.assertFalse(self.gopro.change_resolution(99))

    @patch("gopro_connection.requests.get")
    def test_change_resolution_calls_stop_then_start(self, mock_get):
        """change_resolution stops webcam then starts with new resolution."""
        mock_get.side_effect = [
            make_mock_response({"error": 0}),       # stop
            make_mock_response({"status": 0}),       # status -> OFF
            make_mock_response({"error": 0}),        # start res=7
            make_mock_response({"status": 3}),       # polling -> STREAMING
        ]

        result = self.gopro.change_resolution(7)  # Switch to 720p
        self.assertTrue(result)

        calls = [c[0][0] for c in mock_get.call_args_list]
        self.assertTrue(any("webcam/stop" in c for c in calls))
        self.assertTrue(any("res=7" in c for c in calls))

    @patch("gopro_connection.requests.get")
    def test_change_resolution_tracks_new_resolution(self, mock_get):
        """After successful change, _current_resolution is updated."""
        mock_get.side_effect = [
            make_mock_response({"error": 0}),       # stop
            make_mock_response({"status": 0}),       # status -> OFF
            make_mock_response({"error": 0}),        # start res=7
            make_mock_response({"status": 3}),       # polling -> STREAMING
        ]

        self.gopro.change_resolution(7, fov=0)
        self.assertEqual(self.gopro._current_resolution, 7)
        self.assertEqual(self.gopro._current_fov, 0)

    @patch("gopro_connection.requests.get")
    def test_change_resolution_preserves_fov_when_none(self, mock_get):
        """When fov=None, uses current FOV."""
        self.gopro._current_fov = 2  # narrow
        mock_get.side_effect = [
            make_mock_response({"error": 0}),       # stop
            make_mock_response({"status": 0}),       # status -> OFF
            make_mock_response({"error": 0}),        # start
            make_mock_response({"status": 3}),       # polling -> STREAMING
        ]

        self.gopro.change_resolution(12)  # 4K, keep FOV
        calls = [c[0][0] for c in mock_get.call_args_list]
        start_call = [c for c in calls if "webcam/start" in c][0]
        self.assertIn("fov=2", start_call)

    @patch("gopro_connection.requests.get")
    def test_change_resolution_returns_false_on_failure(self, mock_get):
        """Returns False when camera rejects the new resolution."""
        mock_get.side_effect = [
            make_mock_response({"error": 0}),       # stop
            make_mock_response({"status": 0}),       # status -> OFF
            make_mock_response({"error": 1}),        # start FAILS
            # IDLE workaround retry
            make_mock_response({"error": 0}),        # reset start
            make_mock_response({"error": 0}),        # reset stop
            make_mock_response({"error": 1}),        # retry start FAILS
            make_mock_response({"error": 1}),        # final retry FAILS
        ]

        result = self.gopro.change_resolution(7)
        self.assertFalse(result)


# ── Resolution tracking in start_webcam() ──────────────────────


class TestResolutionTracking(unittest.TestCase):
    """Test that start_webcam tracks active resolution."""

    def setUp(self):
        self.config = MockConfig()
        self.gopro = GoProConnection(self.config)
        self.gopro.ip = "172.20.123.51"
        self.gopro.base_url = "http://172.20.123.51:8080"

    @patch("gopro_connection.requests.get")
    def test_start_webcam_tracks_resolution(self, mock_get):
        """start_webcam sets _current_resolution on success."""
        mock_get.side_effect = [
            make_mock_response({"status": 0}),       # webcam_status -> OFF
            make_mock_response({"error": 0}),        # webcam/start
            make_mock_response({"status": 3}),       # polling -> STREAMING
        ]

        result = self.gopro.start_webcam(resolution=7, fov=0)
        self.assertTrue(result)
        self.assertEqual(self.gopro._current_resolution, 7)
        self.assertEqual(self.gopro._current_fov, 0)

    @patch("gopro_connection.requests.get")
    def test_start_webcam_rejects_invalid_resolution(self, mock_get):
        """start_webcam rejects invalid resolution codes."""
        result = self.gopro.start_webcam(resolution=99)
        self.assertFalse(result)
        mock_get.assert_not_called()

    def test_current_resolution_property(self):
        """current_resolution property is thread-safe."""
        self.assertIsNone(self.gopro.current_resolution)
        self.gopro._current_resolution = 4
        self.assertEqual(self.gopro.current_resolution, 4)

    def test_current_fov_property(self):
        """current_fov property is thread-safe."""
        self.assertIsNone(self.gopro.current_fov)
        self.gopro._current_fov = 2
        self.assertEqual(self.gopro.current_fov, 2)

    @patch("gopro_connection.requests.get")
    def test_resolution_not_tracked_on_failure(self, mock_get):
        """Failed start_webcam doesn't update resolution tracking."""
        mock_get.side_effect = [
            make_mock_response({"status": 0}),       # webcam_status -> OFF
            make_mock_response({"error": 1}),        # webcam/start FAILS
            # IDLE workaround attempts
            make_mock_response({"error": 0}),        # reset start
            make_mock_response({"error": 0}),        # reset stop
            make_mock_response({"error": 1}),        # retry FAILS
        ]

        self.gopro._current_resolution = None
        result = self.gopro.start_webcam(resolution=7, fov=0)
        self.assertFalse(result)
        self.assertIsNone(self.gopro._current_resolution)


# ── AppController.request_resolution_change() ──────────────────


class TestAppControllerRequestResolutionChange(unittest.TestCase):
    """Test AppController.request_resolution_change() detection layer.

    IMPORTANT: USBEventListener uses raw Win32 APIs that crash in pytest.
    All tests must mock usb_event_listener.USBEventListener.
    """

    def _make_controller(self):
        """Create an AppController with mocked USBEventListener."""
        from app_controller import AppController
        with patch("usb_event_listener.USBEventListener"):
            controller = AppController(config=MockConfig())
        return controller

    @patch("usb_event_listener.USBEventListener")
    def test_invalid_resolution_rejected(self, mock_usb):
        """Invalid resolution is rejected without touching the camera."""
        controller = self._make_controller()
        result = controller.request_resolution_change(99)
        self.assertFalse(result)

    @patch("usb_event_listener.USBEventListener")
    @patch("gopro_connection.requests.get")
    def test_no_change_when_already_at_target(self, mock_get, mock_usb):
        """No-op when already streaming at requested resolution."""
        controller = self._make_controller()
        controller.gopro._current_resolution = 4
        controller.gopro._current_fov = 4

        result = controller.request_resolution_change(4)
        self.assertTrue(result)
        mock_get.assert_not_called()

    @patch("usb_event_listener.USBEventListener")
    def test_delegates_to_change_resolution_when_needed(self, mock_usb):
        """Delegates to change_resolution() when a change is detected."""
        controller = self._make_controller()
        controller.gopro._current_resolution = 4  # Currently 1080p
        controller.gopro._current_fov = 4

        controller.change_resolution = MagicMock(return_value=True)

        result = controller.request_resolution_change(7)  # Change to 720p
        self.assertTrue(result)
        controller.change_resolution.assert_called_once_with(7, None)

    @patch("usb_event_listener.USBEventListener")
    def test_delegates_with_fov(self, mock_usb):
        """Passes fov to change_resolution."""
        controller = self._make_controller()
        controller.gopro._current_resolution = 4
        controller.gopro._current_fov = 4

        controller.change_resolution = MagicMock(return_value=True)

        result = controller.request_resolution_change(12, fov=0)
        self.assertTrue(result)
        controller.change_resolution.assert_called_once_with(12, 0)

    @patch("usb_event_listener.USBEventListener")
    def test_returns_false_on_change_failure(self, mock_usb):
        """Returns False when change_resolution fails."""
        controller = self._make_controller()
        controller.gopro._current_resolution = 4
        controller.gopro._current_fov = 4

        controller.change_resolution = MagicMock(return_value=False)

        result = controller.request_resolution_change(7)
        self.assertFalse(result)


# ── AppController.change_resolution() ──────────────────────────


class TestAppControllerChangeResolution(unittest.TestCase):
    """Test AppController.change_resolution() orchestration.

    These tests mock _apply_resolution_on_camera and
    _resume_stream_after_resolution_change to isolate the
    change_resolution orchestration logic.
    """

    def _make_controller(self):
        from app_controller import AppController
        with patch("usb_event_listener.USBEventListener"):
            controller = AppController(config=MockConfig())
        return controller

    @patch("usb_event_listener.USBEventListener")
    def test_enters_freeze_frame(self, mock_usb):
        """change_resolution enters freeze-frame before camera change."""
        controller = self._make_controller()

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        controller._frame_pipeline = mock_pipeline

        controller._apply_resolution_on_camera = MagicMock(return_value=True)
        controller._resume_stream_after_resolution_change = MagicMock(return_value=True)

        controller.config.resolution = 4
        controller.config.fov = 4

        controller.change_resolution(12)

        mock_pipeline.enter_freeze_frame.assert_called_once()

    @patch("usb_event_listener.USBEventListener")
    def test_fallback_on_camera_failure(self, mock_usb):
        """Falls back to previous resolution when camera rejects new one."""
        controller = self._make_controller()
        controller.config.resolution = 4
        controller.config.fov = 4

        # First call (new res) fails, fallback (old res) succeeds
        controller._apply_resolution_on_camera = MagicMock(
            side_effect=[False, True]
        )
        controller._resume_stream_after_resolution_change = MagicMock(
            return_value=True
        )

        result = controller.change_resolution(7)

        calls = controller._apply_resolution_on_camera.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], call(7, 4))   # New resolution
        self.assertEqual(calls[1], call(4, 4))   # Fallback to old

    @patch("usb_event_listener.USBEventListener")
    def test_noop_when_same_resolution(self, mock_usb):
        """No-op when resolution and FOV match current config."""
        controller = self._make_controller()
        controller.config.resolution = 4
        controller.config.fov = 4

        result = controller.change_resolution(4)
        self.assertTrue(result)

    @patch("usb_event_listener.USBEventListener")
    def test_invalid_resolution_returns_false(self, mock_usb):
        """Invalid resolution code returns False."""
        controller = self._make_controller()
        result = controller.change_resolution(99)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
