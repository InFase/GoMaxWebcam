"""
Tests for USB mode detection logic (detect_camera_mode).

Verifies that GoProConnection.detect_camera_mode() correctly determines
the GoPro's operating mode by combining /gopro/webcam/status and
/gopro/camera/state queries. All HTTP calls are mocked.

Test matrix:
  - Webcam STREAMING  -> CameraMode.WEBCAM
  - Webcam READY      -> CameraMode.WEBCAM
  - Webcam IDLE       -> CameraMode.WEBCAM_IDLE
  - Webcam UNAVAILABLE -> CameraMode.UNAVAILABLE
  - Webcam OFF + preset group 0 -> CameraMode.VIDEO
  - Webcam OFF + preset group 1 -> CameraMode.PHOTO
  - Webcam OFF + preset group 2 -> CameraMode.TIMELAPSE
  - Webcam OFF + no camera state -> CameraMode.VIDEO (fallback)
  - Webcam OFF + unknown preset  -> CameraMode.VIDEO (fallback)
  - Camera unreachable -> CameraMode.UNKNOWN
  - CameraMode enum properties (is_webcam_active, needs_webcam_start, etc.)
"""

import sys
import os
import unittest
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gopro_connection import (
    GoProConnection,
    CameraMode,
    WebcamStatus,
    ConnectionState,
    GOPRO_API_PORT,
    _STATUS_ID_PRESET_GROUP,
    _STATUS_ID_PRESET_GROUP_INT,
)

import pytest

pytestmark = pytest.mark.no_gopro_needed


@dataclass
class MockConfig:
    """Minimal config for testing."""
    resolution: int = 4
    fov: int = 4
    discovery_timeout: float = 1.0
    discovery_overall_timeout: float = 5.0
    idle_reset_delay: float = 0.01
    keepalive_interval: float = 0.1
    reconnect_delay: float = 0.01
    ncm_adapter_wait: float = 0.01
    discovery_retry_interval: float = 0.01
    discovery_max_retries: int = 3


def make_mock_response(json_data, status_code=200):
    """Create a mock requests.Response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status.return_value = None
    return mock


class TestCameraModeEnum(unittest.TestCase):
    """Test CameraMode enum values and properties."""

    def test_enum_values(self):
        self.assertEqual(CameraMode.UNKNOWN, -1)
        self.assertEqual(CameraMode.VIDEO, 0)
        self.assertEqual(CameraMode.PHOTO, 1)
        self.assertEqual(CameraMode.TIMELAPSE, 2)
        self.assertEqual(CameraMode.WEBCAM, 10)
        self.assertEqual(CameraMode.WEBCAM_IDLE, 11)
        self.assertEqual(CameraMode.WEBCAM_STARTING, 12)
        self.assertEqual(CameraMode.UNAVAILABLE, 20)

    def test_is_webcam_active(self):
        """is_webcam_active should be True for all webcam-related modes."""
        self.assertTrue(CameraMode.WEBCAM.is_webcam_active)
        self.assertTrue(CameraMode.WEBCAM_IDLE.is_webcam_active)
        self.assertTrue(CameraMode.WEBCAM_STARTING.is_webcam_active)
        # Non-webcam modes
        self.assertFalse(CameraMode.VIDEO.is_webcam_active)
        self.assertFalse(CameraMode.PHOTO.is_webcam_active)
        self.assertFalse(CameraMode.TIMELAPSE.is_webcam_active)
        self.assertFalse(CameraMode.UNKNOWN.is_webcam_active)
        self.assertFalse(CameraMode.UNAVAILABLE.is_webcam_active)

    def test_is_ready_to_stream(self):
        """Only WEBCAM mode is truly ready to stream."""
        self.assertTrue(CameraMode.WEBCAM.is_ready_to_stream)
        self.assertFalse(CameraMode.WEBCAM_IDLE.is_ready_to_stream)
        self.assertFalse(CameraMode.VIDEO.is_ready_to_stream)
        self.assertFalse(CameraMode.UNKNOWN.is_ready_to_stream)

    def test_needs_webcam_start(self):
        """Modes that require webcam/start to begin streaming."""
        self.assertTrue(CameraMode.VIDEO.needs_webcam_start)
        self.assertTrue(CameraMode.PHOTO.needs_webcam_start)
        self.assertTrue(CameraMode.TIMELAPSE.needs_webcam_start)
        self.assertTrue(CameraMode.WEBCAM_IDLE.needs_webcam_start)
        self.assertTrue(CameraMode.UNKNOWN.needs_webcam_start)
        # Already streaming — no start needed
        self.assertFalse(CameraMode.WEBCAM.needs_webcam_start)
        self.assertFalse(CameraMode.UNAVAILABLE.needs_webcam_start)

    def test_label_strings(self):
        """Every mode should have a human-readable label."""
        for mode in CameraMode:
            label = mode.label
            self.assertIsInstance(label, str)
            self.assertTrue(len(label) > 0, f"Empty label for {mode.name}")

    def test_label_specific_values(self):
        self.assertEqual(CameraMode.VIDEO.label, "Video Mode")
        self.assertEqual(CameraMode.PHOTO.label, "Photo Mode")
        self.assertEqual(CameraMode.WEBCAM.label, "Webcam Mode (Active)")
        self.assertEqual(CameraMode.WEBCAM_IDLE.label, "Webcam Mode (Idle)")
        self.assertEqual(CameraMode.UNAVAILABLE.label, "Unavailable")


class TestDetectCameraMode(unittest.TestCase):
    """Test GoProConnection.detect_camera_mode() with mocked API responses."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)
        self.conn.ip = "172.20.145.51"
        self.conn.base_url = f"http://172.20.145.51:{GOPRO_API_PORT}"

    @patch("gopro_connection.requests.get")
    def test_streaming_returns_webcam(self, mock_get):
        """Webcam status STREAMING -> CameraMode.WEBCAM."""
        mock_get.return_value = make_mock_response({"status": 3, "error": 0})

        mode = self.conn.detect_camera_mode()

        self.assertEqual(mode, CameraMode.WEBCAM)
        self.assertTrue(mode.is_ready_to_stream)

    @patch("gopro_connection.requests.get")
    def test_ready_returns_webcam(self, mock_get):
        """Webcam status READY -> CameraMode.WEBCAM."""
        mock_get.return_value = make_mock_response({"status": 2, "error": 0})

        mode = self.conn.detect_camera_mode()

        self.assertEqual(mode, CameraMode.WEBCAM)

    @patch("gopro_connection.requests.get")
    def test_idle_returns_webcam_idle(self, mock_get):
        """Webcam status IDLE -> CameraMode.WEBCAM_IDLE."""
        mock_get.return_value = make_mock_response({"status": 1, "error": 0})

        mode = self.conn.detect_camera_mode()

        self.assertEqual(mode, CameraMode.WEBCAM_IDLE)
        self.assertTrue(mode.is_webcam_active)
        self.assertTrue(mode.needs_webcam_start)

    @patch("gopro_connection.requests.get")
    def test_unavailable_returns_unavailable(self, mock_get):
        """Webcam status UNAVAILABLE -> CameraMode.UNAVAILABLE."""
        mock_get.return_value = make_mock_response({"status": 4, "error": 0})

        mode = self.conn.detect_camera_mode()

        self.assertEqual(mode, CameraMode.UNAVAILABLE)
        self.assertFalse(mode.needs_webcam_start)

    @patch("gopro_connection.requests.get")
    def test_off_video_preset(self, mock_get):
        """Webcam OFF + preset group 0 -> CameraMode.VIDEO."""
        mock_get.side_effect = [
            # First call: webcam/status returns OFF
            make_mock_response({"status": 0, "error": 0}),
            # Second call: camera/state returns preset group 0 (video)
            make_mock_response({"status": {"89": 0}}),
        ]

        mode = self.conn.detect_camera_mode()

        self.assertEqual(mode, CameraMode.VIDEO)
        self.assertTrue(mode.needs_webcam_start)
        self.assertFalse(mode.is_webcam_active)

    @patch("gopro_connection.requests.get")
    def test_off_photo_preset(self, mock_get):
        """Webcam OFF + preset group 1 -> CameraMode.PHOTO."""
        mock_get.side_effect = [
            make_mock_response({"status": 0, "error": 0}),
            make_mock_response({"status": {"89": 1}}),
        ]

        mode = self.conn.detect_camera_mode()

        self.assertEqual(mode, CameraMode.PHOTO)

    @patch("gopro_connection.requests.get")
    def test_off_timelapse_preset(self, mock_get):
        """Webcam OFF + preset group 2 -> CameraMode.TIMELAPSE."""
        mock_get.side_effect = [
            make_mock_response({"status": 0, "error": 0}),
            make_mock_response({"status": {"89": 2}}),
        ]

        mode = self.conn.detect_camera_mode()

        self.assertEqual(mode, CameraMode.TIMELAPSE)

    @patch("gopro_connection.requests.get")
    def test_off_with_integer_key(self, mock_get):
        """Camera state response may use integer keys instead of string keys."""
        mock_get.side_effect = [
            make_mock_response({"status": 0, "error": 0}),
            # Integer key 89 instead of string "89"
            make_mock_response({"status": {89: 1}}),
        ]

        mode = self.conn.detect_camera_mode()

        self.assertEqual(mode, CameraMode.PHOTO)

    @patch("gopro_connection.requests.get")
    def test_off_camera_state_unreachable(self, mock_get):
        """Webcam OFF + camera/state fails -> CameraMode.VIDEO (fallback)."""
        import requests
        mock_get.side_effect = [
            # webcam/status returns OFF
            make_mock_response({"status": 0, "error": 0}),
            # camera/state fails with ConnectionError
            requests.ConnectionError("connection lost"),
        ]

        mode = self.conn.detect_camera_mode()

        # Falls back to VIDEO when camera state is unavailable
        self.assertEqual(mode, CameraMode.VIDEO)

    @patch("gopro_connection.requests.get")
    def test_off_unknown_preset_group(self, mock_get):
        """Webcam OFF + unknown preset group -> CameraMode.VIDEO (fallback)."""
        mock_get.side_effect = [
            make_mock_response({"status": 0, "error": 0}),
            make_mock_response({"status": {"89": 99}}),
        ]

        mode = self.conn.detect_camera_mode()

        self.assertEqual(mode, CameraMode.VIDEO)

    @patch("gopro_connection.requests.get")
    def test_off_no_preset_in_state(self, mock_get):
        """Webcam OFF + camera state has no preset group -> VIDEO fallback."""
        mock_get.side_effect = [
            make_mock_response({"status": 0, "error": 0}),
            # camera/state response without status ID 89
            make_mock_response({"status": {"2": 85, "70": True}}),
        ]

        mode = self.conn.detect_camera_mode()

        self.assertEqual(mode, CameraMode.VIDEO)

    @patch("gopro_connection.requests.get")
    def test_camera_unreachable(self, mock_get):
        """Camera completely unreachable -> CameraMode.UNKNOWN."""
        import requests
        mock_get.side_effect = requests.ConnectionError("no GoPro")

        mode = self.conn.detect_camera_mode()

        self.assertEqual(mode, CameraMode.UNKNOWN)

    @patch("gopro_connection.requests.get")
    def test_off_invalid_preset_group_value(self, mock_get):
        """Webcam OFF + non-numeric preset group -> VIDEO fallback."""
        mock_get.side_effect = [
            make_mock_response({"status": 0, "error": 0}),
            make_mock_response({"status": {"89": "not_a_number"}}),
        ]

        mode = self.conn.detect_camera_mode()

        # Invalid preset group falls back to VIDEO
        self.assertEqual(mode, CameraMode.VIDEO)

    @patch("gopro_connection.requests.get")
    def test_off_empty_status_dict(self, mock_get):
        """Webcam OFF + empty camera state status -> VIDEO fallback."""
        mock_get.side_effect = [
            make_mock_response({"status": 0, "error": 0}),
            make_mock_response({"status": {}}),
        ]

        mode = self.conn.detect_camera_mode()

        self.assertEqual(mode, CameraMode.VIDEO)

    @patch("gopro_connection.requests.get")
    def test_queries_correct_endpoints(self, mock_get):
        """Verify detect_camera_mode calls the right API endpoints."""
        mock_get.side_effect = [
            make_mock_response({"status": 0, "error": 0}),
            make_mock_response({"status": {"89": 0}}),
        ]

        self.conn.detect_camera_mode()

        # Should have made exactly 2 calls
        self.assertEqual(mock_get.call_count, 2)
        # First call: webcam/status
        first_url = mock_get.call_args_list[0][0][0]
        self.assertIn("/gopro/webcam/status", first_url)
        # Second call: camera/state
        second_url = mock_get.call_args_list[1][0][0]
        self.assertIn("/gopro/camera/state", second_url)

    @patch("gopro_connection.requests.get")
    def test_streaming_skips_camera_state_query(self, mock_get):
        """When webcam is STREAMING, no need to query camera/state."""
        mock_get.return_value = make_mock_response({"status": 3, "error": 0})

        self.conn.detect_camera_mode()

        # Should only call webcam/status (1 call), not camera/state
        self.assertEqual(mock_get.call_count, 1)

    @patch("gopro_connection.requests.get")
    def test_idle_skips_camera_state_query(self, mock_get):
        """When webcam is IDLE, no need to query camera/state."""
        mock_get.return_value = make_mock_response({"status": 1, "error": 0})

        self.conn.detect_camera_mode()

        self.assertEqual(mock_get.call_count, 1)

    @patch("gopro_connection.requests.get")
    def test_no_base_url_returns_unknown(self, mock_get):
        """If no base_url is set, API calls return None -> UNKNOWN."""
        self.conn.base_url = None

        mode = self.conn.detect_camera_mode()

        self.assertEqual(mode, CameraMode.UNKNOWN)
        # No HTTP requests should have been made
        mock_get.assert_not_called()


class TestCameraModeIntegrationWithStartWebcam(unittest.TestCase):
    """Test that detect_camera_mode results align with start_webcam logic."""

    def test_webcam_mode_means_no_start_needed(self):
        """WEBCAM mode should not require webcam/start."""
        self.assertFalse(CameraMode.WEBCAM.needs_webcam_start)

    def test_video_mode_means_start_needed(self):
        """VIDEO mode requires webcam/start."""
        self.assertTrue(CameraMode.VIDEO.needs_webcam_start)

    def test_idle_mode_means_start_needed_after_reset(self):
        """WEBCAM_IDLE needs start (after IDLE workaround)."""
        self.assertTrue(CameraMode.WEBCAM_IDLE.needs_webcam_start)
        self.assertTrue(CameraMode.WEBCAM_IDLE.is_webcam_active)

    def test_unavailable_mode_blocks_start(self):
        """UNAVAILABLE should not attempt webcam/start."""
        self.assertFalse(CameraMode.UNAVAILABLE.needs_webcam_start)
        self.assertFalse(CameraMode.UNAVAILABLE.is_webcam_active)


if __name__ == "__main__":
    unittest.main()
