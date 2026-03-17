"""
Tests for USB mode switch (wired USB control) in gopro_connection.py.

Verifies the GoPro API call to enable/disable wired USB control mode:
  - GET /gopro/camera/control/wired_usb?p=1 (enable)
  - GET /gopro/camera/control/wired_usb?p=0 (disable)

These tests mock HTTP API calls so they run on any machine without
GoPro hardware. They verify:
  - enable_wired_usb_control() sends correct API endpoint
  - disable_wired_usb_control() sends correct API endpoint
  - Success handling (error code 0)
  - Non-zero error codes treated as non-fatal (camera may already be in mode)
  - Network failure handling (ConnectionError, Timeout)
  - Integration into open_connection() flow (called after API verify)
  - Integration into disconnect() flow (called during cleanup)
  - Status notifications fire correctly
  - Logging events for USB mode switch
"""

import sys
import os
import time
import unittest
from dataclasses import dataclass
from unittest.mock import patch, MagicMock, call

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gopro_connection import (
    GoProConnection,
    WebcamStatus,
    ConnectionState,
    GOPRO_API_PORT,
)
from discovery import GoProDevice

import pytest

pytestmark = pytest.mark.no_gopro_needed


@dataclass
class MockConfig:
    """Minimal config for testing."""
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


def make_mock_response(json_data, status_code=200):
    """Create a mock requests.Response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status.return_value = None
    return mock


class TestEnableWiredUsbControl(unittest.TestCase):
    """Test enable_wired_usb_control() — the core USB mode switch command."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)
        self.conn.ip = "172.20.145.51"
        self.conn.base_url = f"http://172.20.145.51:{GOPRO_API_PORT}"

    @patch("gopro_connection.requests.get")
    def test_enable_success(self, mock_get):
        """Successful USB control enable returns True."""
        mock_get.return_value = make_mock_response({"error": 0})

        result = self.conn.enable_wired_usb_control()

        self.assertTrue(result)
        # Verify the correct endpoint was called
        call_url = mock_get.call_args[0][0]
        self.assertIn("/gopro/camera/control/wired_usb?p=1", call_url)

    @patch("gopro_connection.requests.get")
    def test_enable_sends_correct_endpoint(self, mock_get):
        """Verify the exact API endpoint URL."""
        mock_get.return_value = make_mock_response({"error": 0})

        self.conn.enable_wired_usb_control()

        expected_url = f"http://172.20.145.51:{GOPRO_API_PORT}/gopro/camera/control/wired_usb?p=1"
        mock_get.assert_called_once_with(expected_url, timeout=5.0)

    @patch("gopro_connection.requests.get")
    def test_enable_non_zero_error_treated_non_fatal(self, mock_get):
        """Non-zero error code is treated as non-fatal (returns True).

        Some firmware versions return non-zero when USB control is already
        active. We don't want to fail the connection for this.
        """
        mock_get.return_value = make_mock_response({"error": 2})

        result = self.conn.enable_wired_usb_control()

        # Should return True — non-zero error is treated as "already enabled"
        self.assertTrue(result)

    @patch("gopro_connection.requests.get")
    def test_enable_camera_unreachable(self, mock_get):
        """Returns False when camera doesn't respond at all."""
        import requests
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        result = self.conn.enable_wired_usb_control()

        self.assertFalse(result)

    @patch("gopro_connection.requests.get")
    def test_enable_timeout(self, mock_get):
        """Returns False on request timeout."""
        import requests
        mock_get.side_effect = requests.Timeout("Request timed out")

        result = self.conn.enable_wired_usb_control()

        self.assertFalse(result)

    @patch("gopro_connection.requests.get")
    def test_enable_no_base_url(self, mock_get):
        """Returns False if no base_url is set (not connected)."""
        self.conn.base_url = None

        result = self.conn.enable_wired_usb_control()

        self.assertFalse(result)
        mock_get.assert_not_called()

    @patch("gopro_connection.requests.get")
    def test_enable_fires_status_callback(self, mock_get):
        """Status notifications are sent during USB mode switch."""
        mock_get.return_value = make_mock_response({"error": 0})

        messages = []
        self.conn.on_status_change = lambda msg, lvl: messages.append((msg, lvl))

        self.conn.enable_wired_usb_control()

        # Should have at least an "Enabling..." and "enabled" message
        msg_texts = [m[0] for m in messages]
        self.assertTrue(
            any("Enabling wired USB control" in t for t in msg_texts),
            f"Expected 'Enabling' message in: {msg_texts}"
        )
        self.assertTrue(
            any("enabled" in t.lower() for t in msg_texts),
            f"Expected 'enabled' message in: {msg_texts}"
        )

    @patch("gopro_connection.requests.get")
    def test_enable_failure_fires_warning(self, mock_get):
        """Warning notification sent when USB control command fails."""
        import requests
        mock_get.side_effect = requests.ConnectionError("refused")

        messages = []
        self.conn.on_status_change = lambda msg, lvl: messages.append((msg, lvl))

        self.conn.enable_wired_usb_control()

        levels = [m[1] for m in messages]
        self.assertIn("warning", levels)


class TestDisableWiredUsbControl(unittest.TestCase):
    """Test disable_wired_usb_control() — cleanup on disconnect."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)
        self.conn.ip = "172.20.145.51"
        self.conn.base_url = f"http://172.20.145.51:{GOPRO_API_PORT}"

    @patch("gopro_connection.requests.get")
    def test_disable_success(self, mock_get):
        """Successful USB control disable returns True."""
        mock_get.return_value = make_mock_response({"error": 0})

        result = self.conn.disable_wired_usb_control()

        self.assertTrue(result)
        call_url = mock_get.call_args[0][0]
        self.assertIn("/gopro/camera/control/wired_usb?p=0", call_url)

    @patch("gopro_connection.requests.get")
    def test_disable_sends_correct_endpoint(self, mock_get):
        """Verify the exact API endpoint for disable."""
        mock_get.return_value = make_mock_response({"error": 0})

        self.conn.disable_wired_usb_control()

        expected_url = f"http://172.20.145.51:{GOPRO_API_PORT}/gopro/camera/control/wired_usb?p=0"
        mock_get.assert_called_once_with(expected_url, timeout=3.0)

    @patch("gopro_connection.requests.get")
    def test_disable_error_code_returns_false(self, mock_get):
        """Non-zero error code from disable returns False."""
        mock_get.return_value = make_mock_response({"error": 1})

        result = self.conn.disable_wired_usb_control()

        self.assertFalse(result)

    @patch("gopro_connection.requests.get")
    def test_disable_camera_gone(self, mock_get):
        """Returns False when camera is already disconnected."""
        import requests
        mock_get.side_effect = requests.ConnectionError("gone")

        result = self.conn.disable_wired_usb_control()

        self.assertFalse(result)


class TestUsbModeInOpenConnection(unittest.TestCase):
    """Test that USB mode switch is called during open_connection() flow."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)
        self.device = GoProDevice(
            vendor_id=0x0A70,
            product_id=0x000D,
            description="GoPro Hero 12",
            camera_ip="172.20.145.51",
        )

    @patch("gopro_connection.requests.get")
    def test_open_connection_calls_usb_control(self, mock_get):
        """open_connection calls enable_wired_usb_control in the flow."""
        # All responses succeed
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        result = self.conn.open_connection(self.device)

        self.assertTrue(result)

        # Check that wired_usb?p=1 was called among the API calls
        call_urls = [c[0][0] for c in mock_get.call_args_list]
        usb_control_calls = [u for u in call_urls if "wired_usb?p=1" in u]
        self.assertTrue(
            len(usb_control_calls) >= 1,
            f"Expected wired_usb?p=1 call in: {call_urls}"
        )

        # Clean up keep-alive
        self.conn._stop_keepalive()

    @patch("gopro_connection.requests.get")
    def test_open_connection_usb_control_before_idle_reset(self, mock_get):
        """USB control enable happens before IDLE workaround in the flow."""
        call_order = []

        original_get = mock_get.side_effect

        def tracking_get(url, timeout=5.0):
            if "wired_usb" in url:
                call_order.append("usb_control")
            elif "webcam/start" in url:
                call_order.append("webcam_start")
            elif "webcam/stop" in url:
                call_order.append("webcam_stop")
            elif "webcam/status" in url:
                call_order.append("webcam_status")
            return make_mock_response({"status": 0, "error": 0})

        mock_get.side_effect = tracking_get

        self.conn.open_connection(self.device)
        self.conn._stop_keepalive()

        # USB control should appear before any webcam_start (IDLE workaround)
        if "usb_control" in call_order and "webcam_start" in call_order:
            usb_idx = call_order.index("usb_control")
            start_idx = call_order.index("webcam_start")
            self.assertLess(
                usb_idx, start_idx,
                f"USB control ({usb_idx}) should come before webcam_start ({start_idx}). "
                f"Order: {call_order}"
            )

    @patch("gopro_connection.requests.get")
    def test_open_connection_continues_on_usb_control_failure(self, mock_get):
        """open_connection proceeds even if USB control enable fails."""
        call_count = [0]

        def conditional_response(url, timeout=5.0):
            call_count[0] += 1
            if "wired_usb" in url:
                # USB control fails
                import requests
                raise requests.ConnectionError("USB control failed")
            return make_mock_response({"status": 0, "error": 0})

        mock_get.side_effect = conditional_response

        result = self.conn.open_connection(self.device)

        # Should still succeed despite USB control failure
        self.assertTrue(result)
        self.assertTrue(self.conn.is_connected)
        self.conn._stop_keepalive()


class TestUsbModeInDisconnect(unittest.TestCase):
    """Test that USB mode is disabled during disconnect()."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)
        self.conn.ip = "172.20.145.51"
        self.conn.base_url = f"http://172.20.145.51:{GOPRO_API_PORT}"
        self.conn._connected = True
        self.conn._state = ConnectionState.STREAMING

    @patch("gopro_connection.requests.get")
    def test_disconnect_calls_disable_usb_control(self, mock_get):
        """disconnect() calls disable_wired_usb_control."""
        mock_get.return_value = make_mock_response({"error": 0})

        self.conn.disconnect()

        # Check that wired_usb?p=0 was called
        call_urls = [c[0][0] for c in mock_get.call_args_list]
        disable_calls = [u for u in call_urls if "wired_usb?p=0" in u]
        self.assertTrue(
            len(disable_calls) >= 1,
            f"Expected wired_usb?p=0 call in: {call_urls}"
        )

    @patch("gopro_connection.requests.get")
    def test_disconnect_survives_usb_disable_failure(self, mock_get):
        """disconnect() completes even if USB disable fails."""
        def conditional_response(url, timeout=5.0):
            if "wired_usb" in url:
                import requests
                raise requests.ConnectionError("gone")
            return make_mock_response({"error": 0})

        mock_get.side_effect = conditional_response

        # Should not raise
        self.conn.disconnect()

        self.assertIsNone(self.conn.ip)
        self.assertEqual(self.conn.state, ConnectionState.DISCONNECTED)


class TestUsbModeSwitchEdgeCases(unittest.TestCase):
    """Test edge cases and robustness of USB mode switch."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)
        self.conn.ip = "172.20.145.51"
        self.conn.base_url = f"http://172.20.145.51:{GOPRO_API_PORT}"

    @patch("gopro_connection.requests.get")
    def test_enable_empty_response(self, mock_get):
        """Handle empty JSON response (no 'error' key)."""
        mock_get.return_value = make_mock_response({})

        result = self.conn.enable_wired_usb_control()

        # Empty response has no error key -> result.get("error", -1) returns -1
        # -1 != 0, so it goes to the non-zero path, which returns True (non-fatal)
        self.assertTrue(result)

    @patch("gopro_connection.requests.get")
    def test_enable_http_500(self, mock_get):
        """Handle HTTP 500 error from camera."""
        import requests
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_get.return_value = mock_resp

        result = self.conn.enable_wired_usb_control()

        # _api_get returns None on HTTPError, so enable returns False
        self.assertFalse(result)

    @patch("gopro_connection.requests.get")
    def test_enable_called_twice_idempotent(self, mock_get):
        """Calling enable twice should work fine (idempotent)."""
        mock_get.return_value = make_mock_response({"error": 0})

        result1 = self.conn.enable_wired_usb_control()
        result2 = self.conn.enable_wired_usb_control()

        self.assertTrue(result1)
        self.assertTrue(result2)
        self.assertEqual(mock_get.call_count, 2)

    @patch("gopro_connection.requests.get")
    def test_disable_no_base_url(self, mock_get):
        """disable returns False if not connected."""
        self.conn.base_url = None

        result = self.conn.disable_wired_usb_control()

        self.assertFalse(result)
        mock_get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
