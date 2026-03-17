"""
Tests for automatic GoPro USB control connection (Sub-AC 2 of AC 1).

Verifies that once a GoPro device is discovered (by discovery.py), the
system automatically opens the correct USB endpoint/interface:
  1. open_connection() establishes HTTP control at {camera_ip}:8080
  2. enable_wired_usb_control() sends wired_usb?p=1
  3. verify_usb_control_connection() confirms state machine is reachable
  4. The full auto-connect flow from discovery to USB control works end-to-end
  5. Error handling: network not ready, API unresponsive, USB control failures
  6. State transitions fire correctly throughout the flow

All tests mock HTTP API calls and discovery — no GoPro hardware required.
"""

import sys
import os
import time
import threading
import unittest
from dataclasses import dataclass
from unittest.mock import patch, MagicMock, call

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gopro_connection import (
    GoProConnection,
    WebcamStatus,
    ConnectionState,
    CameraMode,
    GOPRO_API_PORT,
)
from discovery import GoProDevice, DiscoveryTimeout

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


def make_device(ip="172.20.145.51"):
    """Create a standard test GoProDevice."""
    return GoProDevice(
        vendor_id=0x0A70,
        product_id=0x000D,
        description="GoPro Hero 12 Black",
        camera_ip=ip,
    )


class TestAutoUsbConnectionFlow(unittest.TestCase):
    """Test the full automatic USB control connection flow.

    After discovery finds a GoPro, open_connection() must:
      1. Store device info and set up base_url
      2. Verify the HTTP API responds (with retries)
      3. Enable wired USB control (wired_usb?p=1)
      4. Verify USB control connection is working
      5. Reset webcam state (IDLE workaround)
      6. Start keep-alive thread
      7. Set state to CONNECTED
    """

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)
        self.device = make_device()

    def tearDown(self):
        self.conn._stop_keepalive()

    @patch("gopro_connection.requests.get")
    def test_full_auto_connection_flow(self, mock_get):
        """End-to-end: discovery result → open_connection → CONNECTED state."""
        # All API calls succeed
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        result = self.conn.open_connection(self.device)

        self.assertTrue(result)
        self.assertEqual(self.conn.ip, "172.20.145.51")
        self.assertEqual(
            self.conn.base_url,
            f"http://172.20.145.51:{GOPRO_API_PORT}",
        )
        self.assertTrue(self.conn.is_connected)
        self.assertEqual(self.conn.state, ConnectionState.CONNECTED)
        self.assertIs(self.conn.device_info, self.device)

    @patch("gopro_connection.requests.get")
    def test_connection_calls_usb_control_in_order(self, mock_get):
        """Verify API verify → USB control enable → verify control → IDLE reset order."""
        call_log = []

        def tracking_get(url, timeout=5.0):
            if "wired_usb?p=1" in url:
                call_log.append("enable_usb_control")
            elif "webcam/start" in url:
                call_log.append("webcam_start")
            elif "webcam/stop" in url:
                call_log.append("webcam_stop")
            elif "webcam/status" in url:
                call_log.append("webcam_status")
            return make_mock_response({"status": 0, "error": 0})

        mock_get.side_effect = tracking_get

        self.conn.open_connection(self.device)

        # USB control enable must come after first webcam_status (API verification)
        # and before webcam_start (IDLE workaround)
        self.assertIn("enable_usb_control", call_log)
        self.assertIn("webcam_status", call_log)

        # Find first occurrence of each
        first_status = call_log.index("webcam_status")
        usb_ctrl_idx = call_log.index("enable_usb_control")

        # API verification (webcam_status) must come before USB control enable
        self.assertLess(
            first_status, usb_ctrl_idx,
            f"API verification should precede USB control. Order: {call_log}"
        )

    @patch("gopro_connection.requests.get")
    def test_connection_enables_usb_control_endpoint(self, mock_get):
        """Verify the exact wired_usb?p=1 endpoint is called."""
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        self.conn.open_connection(self.device)

        # Extract all called URLs
        call_urls = [c[0][0] for c in mock_get.call_args_list]
        usb_control_urls = [u for u in call_urls if "wired_usb?p=1" in u]

        self.assertTrue(
            len(usb_control_urls) >= 1,
            f"Expected wired_usb?p=1 call. URLs called: {call_urls}"
        )
        # Verify the full URL format
        expected_prefix = f"http://172.20.145.51:{GOPRO_API_PORT}"
        self.assertTrue(usb_control_urls[0].startswith(expected_prefix))

    @patch("gopro_connection.requests.get")
    def test_connection_state_transitions(self, mock_get):
        """State changes: DISCONNECTED → CONNECTING → CONNECTED."""
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        states = []
        self.conn.on_connection_state = lambda s: states.append(s)

        initial_state = self.conn.state
        self.conn.open_connection(self.device)

        self.assertEqual(initial_state, ConnectionState.DISCONNECTED)
        self.assertIn(ConnectionState.CONNECTING, states)
        self.assertIn(ConnectionState.CONNECTED, states)

        # CONNECTING must come before CONNECTED
        connecting_idx = states.index(ConnectionState.CONNECTING)
        connected_idx = states.index(ConnectionState.CONNECTED)
        self.assertLess(connecting_idx, connected_idx)


class TestAutoConnectFromDiscovery(unittest.TestCase):
    """Test auto_connect() which chains discovery → open_connection."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)

    def tearDown(self):
        self.conn._stop_keepalive()

    @patch("gopro_connection.requests.get")
    @patch("gopro_connection.timed_full_discovery")
    def test_auto_connect_discovery_to_usb_control(self, mock_discovery, mock_get):
        """auto_connect: discovery → API verify → USB control → CONNECTED."""
        mock_discovery.return_value = make_device()
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        result = self.conn.auto_connect()

        self.assertTrue(result)
        self.assertEqual(self.conn.ip, "172.20.145.51")
        self.assertTrue(self.conn.is_connected)

        # Verify wired_usb?p=1 was called
        call_urls = [c[0][0] for c in mock_get.call_args_list]
        self.assertTrue(
            any("wired_usb?p=1" in u for u in call_urls),
            f"Expected USB control enable call. URLs: {call_urls}"
        )

    @patch("gopro_connection.requests.get")
    @patch("gopro_connection.timed_full_discovery")
    def test_auto_connect_stores_device_info(self, mock_discovery, mock_get):
        """auto_connect stores the discovered device info."""
        device = make_device()
        mock_discovery.return_value = device
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        self.conn.auto_connect()

        self.assertIs(self.conn.device_info, device)
        self.assertEqual(self.conn.device_info.vendor_id, 0x0A70)
        self.assertEqual(self.conn.device_info.product_id, 0x000D)
        self.assertEqual(self.conn.device_info.description, "GoPro Hero 12 Black")

    @patch("gopro_connection.timed_full_discovery")
    def test_auto_connect_handles_discovery_timeout(self, mock_discovery):
        """auto_connect handles DiscoveryTimeout gracefully."""
        partial_device = GoProDevice(
            vendor_id=0x0A70, product_id=0x000D,
            description="GoPro", camera_ip=None,
        )
        mock_discovery.side_effect = DiscoveryTimeout(
            device=partial_device, elapsed=30.0, timeout=30.0
        )

        result = self.conn.auto_connect()

        self.assertFalse(result)
        self.assertEqual(self.conn.state, ConnectionState.DISCONNECTED)

    @patch("gopro_connection.timed_full_discovery")
    def test_auto_connect_no_device_found(self, mock_discovery):
        """auto_connect returns False when no USB device found."""
        mock_discovery.return_value = None

        result = self.conn.auto_connect()

        self.assertFalse(result)
        self.assertEqual(self.conn.state, ConnectionState.DISCONNECTED)


class TestVerifyUsbControlConnection(unittest.TestCase):
    """Test verify_usb_control_connection() — post-enable validation."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)
        self.conn.ip = "172.20.145.51"
        self.conn.base_url = f"http://172.20.145.51:{GOPRO_API_PORT}"

    @patch("gopro_connection.requests.get")
    def test_verify_success(self, mock_get):
        """Verification succeeds when webcam status responds."""
        mock_get.return_value = make_mock_response({"status": 0})

        result = self.conn.verify_usb_control_connection()

        self.assertTrue(result)

    @patch("gopro_connection.requests.get")
    def test_verify_with_idle_status(self, mock_get):
        """Verification succeeds even if webcam reports IDLE."""
        mock_get.return_value = make_mock_response({"status": 1})

        result = self.conn.verify_usb_control_connection()

        self.assertTrue(result)

    @patch("gopro_connection.requests.get")
    def test_verify_with_streaming_status(self, mock_get):
        """Verification succeeds when camera is already streaming."""
        mock_get.return_value = make_mock_response({"status": 3})

        result = self.conn.verify_usb_control_connection()

        self.assertTrue(result)

    @patch("gopro_connection.requests.get")
    def test_verify_fails_on_connection_error(self, mock_get):
        """Verification fails when camera is unreachable."""
        import requests
        mock_get.side_effect = requests.ConnectionError("refused")

        result = self.conn.verify_usb_control_connection()

        self.assertFalse(result)

    @patch("gopro_connection.requests.get")
    def test_verify_fails_on_missing_status(self, mock_get):
        """Verification fails if response has no 'status' field."""
        mock_get.return_value = make_mock_response({"error": 0})

        result = self.conn.verify_usb_control_connection()

        self.assertFalse(result)

    @patch("gopro_connection.requests.get")
    def test_verify_no_base_url(self, mock_get):
        """Verification fails if not connected (no base_url)."""
        self.conn.base_url = None

        result = self.conn.verify_usb_control_connection()

        self.assertFalse(result)
        mock_get.assert_not_called()


class TestOpenConnectionErrorHandling(unittest.TestCase):
    """Test error handling in the auto USB connection flow."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)
        self.device = make_device()

    def tearDown(self):
        self.conn._stop_keepalive()

    def test_open_connection_no_ip(self):
        """open_connection fails gracefully when device has no IP."""
        device = GoProDevice(
            vendor_id=0x0A70, product_id=0x000D,
            description="GoPro", camera_ip=None,
        )
        result = self.conn.open_connection(device)

        self.assertFalse(result)
        self.assertEqual(self.conn.state, ConnectionState.DISCONNECTED)

    @patch("gopro_connection.requests.get")
    def test_open_connection_api_never_responds(self, mock_get):
        """open_connection fails after 3 API verification retries."""
        import requests
        mock_get.side_effect = requests.ConnectionError("refused")

        result = self.conn.open_connection(self.device)

        self.assertFalse(result)
        self.assertIsNone(self.conn.ip)
        self.assertIsNone(self.conn.base_url)
        self.assertEqual(self.conn.state, ConnectionState.DISCONNECTED)

    @patch("gopro_connection.requests.get")
    def test_connection_succeeds_despite_usb_control_failure(self, mock_get):
        """Connection proceeds even if USB control enable fails."""
        call_count = [0]

        def conditional_response(url, timeout=5.0):
            call_count[0] += 1
            if "wired_usb" in url:
                import requests
                raise requests.ConnectionError("USB control failed")
            return make_mock_response({"status": 0, "error": 0})

        mock_get.side_effect = conditional_response

        result = self.conn.open_connection(self.device)

        # Should still succeed — USB control failure is non-fatal
        self.assertTrue(result)
        self.assertTrue(self.conn.is_connected)
        self.assertEqual(self.conn.state, ConnectionState.CONNECTED)

    @patch("gopro_connection.requests.get")
    def test_connection_succeeds_despite_verify_control_failure(self, mock_get):
        """Connection proceeds even if USB control verification fails."""
        verify_count = [0]

        def conditional_response(url, timeout=5.0):
            if "wired_usb?p=1" in url:
                return make_mock_response({"error": 0})
            elif "webcam/status" in url:
                verify_count[0] += 1
                if verify_count[0] == 1:
                    # First webcam/status call: API verification succeeds
                    return make_mock_response({"status": 0})
                elif verify_count[0] == 2:
                    # Second webcam/status call: USB control verify fails
                    return make_mock_response({"error": 0})  # No 'status' key
                else:
                    # Subsequent calls (IDLE reset check): OFF
                    return make_mock_response({"status": 0})
            return make_mock_response({"status": 0, "error": 0})

        mock_get.side_effect = conditional_response

        result = self.conn.open_connection(self.device)

        # Should still succeed — verify failure is non-fatal
        self.assertTrue(result)

    @patch("gopro_connection.requests.get")
    def test_connection_status_messages(self, mock_get):
        """Status callback fires with meaningful messages during connection."""
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        messages = []
        self.conn.on_status_change = lambda msg, lvl: messages.append((msg, lvl))

        self.conn.open_connection(self.device)

        msg_texts = [m[0] for m in messages]

        # Should have messages about: opening connection, API verified,
        # USB control, and final connected
        self.assertTrue(
            any("Opening control connection" in t for t in msg_texts),
            f"Expected 'Opening' message in: {msg_texts}"
        )
        self.assertTrue(
            any("API verified" in t or "HTTP API verified" in t for t in msg_texts),
            f"Expected 'API verified' message in: {msg_texts}"
        )
        self.assertTrue(
            any("Connected to GoPro" in t for t in msg_texts),
            f"Expected 'Connected' message in: {msg_texts}"
        )


class TestKeepAliveAfterConnection(unittest.TestCase):
    """Test that keep-alive starts after USB control connection."""

    def setUp(self):
        self.config = MockConfig()
        self.config.keepalive_interval = 0.05
        self.conn = GoProConnection(self.config)
        self.device = make_device()

    def tearDown(self):
        self.conn._stop_keepalive()

    @patch("gopro_connection.requests.get")
    def test_keepalive_starts_after_connection(self, mock_get):
        """Keep-alive thread starts after successful open_connection."""
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        self.conn.open_connection(self.device)

        self.assertIsNotNone(self.conn._keepalive_thread)
        self.assertTrue(self.conn._keepalive_thread.is_alive())

    @patch("gopro_connection.requests.get")
    def test_keepalive_not_started_on_failure(self, mock_get):
        """Keep-alive thread does NOT start if connection fails."""
        import requests
        mock_get.side_effect = requests.ConnectionError("refused")

        self.conn.open_connection(self.device)

        # Thread should not be running (or not exist)
        if self.conn._keepalive_thread is not None:
            self.assertFalse(self.conn._keepalive_thread.is_alive())


class TestAutoConnectWithRetries(unittest.TestCase):
    """Test auto_connect_with_retries() for resilient connection."""

    def setUp(self):
        self.config = MockConfig()
        self.config.discovery_max_retries = 3
        self.conn = GoProConnection(self.config)

    def tearDown(self):
        self.conn._stop_keepalive()

    @patch("gopro_connection.requests.get")
    @patch("gopro_connection.timed_full_discovery")
    def test_retries_until_success(self, mock_discovery, mock_get):
        """auto_connect_with_retries succeeds on second attempt."""
        device = make_device()
        mock_discovery.side_effect = [None, device]
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        result = self.conn.auto_connect_with_retries()

        self.assertTrue(result)
        self.assertEqual(mock_discovery.call_count, 2)

    @patch("gopro_connection.timed_full_discovery")
    def test_gives_up_after_max_retries(self, mock_discovery):
        """auto_connect_with_retries fails after exhausting retries."""
        mock_discovery.return_value = None

        result = self.conn.auto_connect_with_retries()

        self.assertFalse(result)
        self.assertEqual(mock_discovery.call_count, self.config.discovery_max_retries)
        self.assertEqual(self.conn.state, ConnectionState.DISCONNECTED)


class TestConnectionWithDifferentIPs(unittest.TestCase):
    """Test USB control connection works with various GoPro IP ranges."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)

    def tearDown(self):
        self.conn._stop_keepalive()

    @patch("gopro_connection.requests.get")
    def test_connect_172_20_range(self, mock_get):
        """Connection works with 172.20.x.51 IP."""
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})
        device = make_device("172.20.123.51")

        result = self.conn.open_connection(device)

        self.assertTrue(result)
        self.assertEqual(self.conn.ip, "172.20.123.51")

    @patch("gopro_connection.requests.get")
    def test_connect_172_29_range(self, mock_get):
        """Connection works with 172.29.x.51 IP."""
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})
        device = make_device("172.29.199.51")

        result = self.conn.open_connection(device)

        self.assertTrue(result)
        self.assertEqual(self.conn.ip, "172.29.199.51")

    @patch("gopro_connection.requests.get")
    def test_base_url_format(self, mock_get):
        """Base URL is correctly formatted as http://{ip}:8080."""
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})
        device = make_device("172.25.150.51")

        self.conn.open_connection(device)

        self.assertEqual(self.conn.base_url, "http://172.25.150.51:8080")


class TestDiscoverToConnection(unittest.TestCase):
    """Test the backward-compatible discover() → connection path."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)

    @patch("gopro_connection.requests.get")
    @patch("gopro_connection.timed_full_discovery")
    def test_discover_sets_up_connection(self, mock_discovery, mock_get):
        """discover() stores IP and verifies connection."""
        mock_discovery.return_value = make_device()
        mock_get.return_value = make_mock_response({"status": 0})

        result = self.conn.discover()

        self.assertTrue(result)
        self.assertEqual(self.conn.ip, "172.20.145.51")
        self.assertTrue(self.conn.is_connected)

    @patch("gopro_connection.requests.get")
    @patch("gopro_connection.timed_full_discovery")
    def test_discover_no_ip_resolved(self, mock_discovery, mock_get):
        """discover() returns False when USB found but no IP."""
        device = GoProDevice(
            vendor_id=0x0A70, product_id=0x000D,
            description="GoPro", camera_ip=None,
        )
        mock_discovery.return_value = device

        result = self.conn.discover()

        self.assertFalse(result)
        # Device info should still be stored (USB found, just no IP)
        self.assertIsNotNone(self.conn.device_info)

    @patch("gopro_connection.timed_full_discovery")
    def test_discover_handles_timeout_exception(self, mock_discovery):
        """discover() handles DiscoveryTimeout without crashing."""
        partial = GoProDevice(
            vendor_id=0x0A70, product_id=0x000D,
            description="GoPro", camera_ip=None,
        )
        mock_discovery.side_effect = DiscoveryTimeout(
            device=partial, elapsed=10.0, timeout=10.0,
        )

        result = self.conn.discover()

        self.assertFalse(result)
        # Partial device info should be stored
        self.assertIsNotNone(self.conn.device_info)
        self.assertEqual(self.conn.device_info.vendor_id, 0x0A70)


class TestConcurrentConnectionSafety(unittest.TestCase):
    """Test thread safety of the connection state management."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)

    def test_state_read_is_thread_safe(self):
        """Reading state from multiple threads doesn't crash."""
        self.conn.ip = "172.20.145.51"
        self.conn._connected = True

        results = []
        errors = []

        def read_state():
            try:
                for _ in range(100):
                    _ = self.conn.state
                    _ = self.conn.is_connected
                    _ = self.conn.is_streaming
                results.append(True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(len(results), 5)


if __name__ == "__main__":
    unittest.main()
