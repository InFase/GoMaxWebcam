"""
Tests for GoPro USB control connection (gopro_connection.py).

These tests mock HTTP API calls so they run on any machine without
GoPro hardware. They verify:
  - Auto-connect flow (discovery -> verify API -> IDLE workaround -> keep-alive)
  - open_connection() with API verification and retries
  - IDLE state workaround (start/stop cycle)
  - Webcam start/stop with status polling
  - Keep-alive thread and disconnect detection
  - Thread-safe state management
  - Backward-compatible discover() method
"""

import threading
import time
import unittest
from dataclasses import dataclass
from unittest.mock import patch, MagicMock, PropertyMock

import sys
import os

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
    udp_port: int = 8554
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


class TestConnectionState(unittest.TestCase):
    """Test ConnectionState enum values."""

    def test_state_values(self):
        self.assertEqual(ConnectionState.DISCONNECTED, 0)
        self.assertEqual(ConnectionState.CONNECTING, 1)
        self.assertEqual(ConnectionState.CONNECTED, 2)
        self.assertEqual(ConnectionState.STREAMING, 3)
        self.assertEqual(ConnectionState.RECONNECTING, 4)


class TestWebcamStatus(unittest.TestCase):
    """Test WebcamStatus enum."""

    def test_status_values(self):
        self.assertEqual(WebcamStatus.OFF, 0)
        self.assertEqual(WebcamStatus.IDLE, 1)
        self.assertEqual(WebcamStatus.READY, 2)
        self.assertEqual(WebcamStatus.STREAMING, 3)
        self.assertEqual(WebcamStatus.UNKNOWN, -1)


class TestGoProConnectionInit(unittest.TestCase):
    """Test GoProConnection initialization."""

    def test_initial_state(self):
        config = MockConfig()
        conn = GoProConnection(config)

        self.assertIsNone(conn.ip)
        self.assertIsNone(conn.base_url)
        self.assertFalse(conn.is_connected)
        self.assertFalse(conn.is_streaming)
        self.assertEqual(conn.state, ConnectionState.DISCONNECTED)

    def test_callbacks_initially_none(self):
        config = MockConfig()
        conn = GoProConnection(config)
        self.assertIsNone(conn.on_status_change)
        self.assertIsNone(conn.on_connection_state)


class TestOpenConnection(unittest.TestCase):
    """Test open_connection() - the core auto-connect logic."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)
        self.device = GoProDevice(
            vendor_id=0x0A72,
            product_id=0x000D,
            description="GoPro Hero 12",
            camera_ip="172.20.145.51",
        )

    @patch("gopro_connection.requests.get")
    def test_open_connection_success(self, mock_get):
        """Successful connection: API responds, IDLE workaround succeeds."""
        # Mock API responses: verify (OFF), IDLE workaround start, stop, verify
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        result = self.conn.open_connection(self.device)

        self.assertTrue(result)
        self.assertEqual(self.conn.ip, "172.20.145.51")
        self.assertEqual(self.conn.base_url, f"http://172.20.145.51:{GOPRO_API_PORT}")
        self.assertTrue(self.conn.is_connected)
        self.assertEqual(self.conn.state, ConnectionState.CONNECTED)

        # Clean up keep-alive thread
        self.conn._stop_keepalive()

    @patch("gopro_connection.requests.get")
    def test_open_connection_no_ip(self, mock_get):
        """open_connection fails if device has no IP."""
        device = GoProDevice(
            vendor_id=0x0A72, product_id=0x000D,
            description="GoPro", camera_ip=None,
        )
        result = self.conn.open_connection(device)

        self.assertFalse(result)
        self.assertEqual(self.conn.state, ConnectionState.DISCONNECTED)

    @patch("gopro_connection.requests.get")
    def test_open_connection_api_unreachable(self, mock_get):
        """open_connection fails if API never responds."""
        import requests
        mock_get.side_effect = requests.ConnectionError("refused")

        result = self.conn.open_connection(self.device)

        self.assertFalse(result)
        self.assertIsNone(self.conn.ip)
        self.assertEqual(self.conn.state, ConnectionState.DISCONNECTED)

    @patch("gopro_connection.requests.get")
    def test_open_connection_starts_keepalive(self, mock_get):
        """open_connection starts the keep-alive thread."""
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        self.conn.open_connection(self.device)

        # Keep-alive thread should be running
        self.assertIsNotNone(self.conn._keepalive_thread)
        self.assertTrue(self.conn._keepalive_thread.is_alive())

        # Clean up
        self.conn._stop_keepalive()

    @patch("gopro_connection.requests.get")
    def test_open_connection_state_callback(self, mock_get):
        """State change callback is fired during open_connection."""
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        states = []
        self.conn.on_connection_state = lambda s: states.append(s)

        self.conn.open_connection(self.device)
        self.conn._stop_keepalive()

        # Should have seen CONNECTING -> CONNECTED
        self.assertIn(ConnectionState.CONNECTING, states)
        self.assertIn(ConnectionState.CONNECTED, states)


class TestAutoConnect(unittest.TestCase):
    """Test auto_connect() - discovery + connection in one call."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)

    @patch("gopro_connection.requests.get")
    @patch("gopro_connection.timed_full_discovery")
    def test_auto_connect_success(self, mock_discovery, mock_get):
        """auto_connect succeeds when device is found with IP."""
        mock_discovery.return_value = GoProDevice(
            vendor_id=0x0A72, product_id=0x000D,
            description="GoPro Hero 12",
            camera_ip="172.20.145.51",
        )
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        result = self.conn.auto_connect()

        self.assertTrue(result)
        self.assertEqual(self.conn.ip, "172.20.145.51")
        self.assertTrue(self.conn.is_connected)
        self.conn._stop_keepalive()

    @patch("gopro_connection.timed_full_discovery")
    def test_auto_connect_no_device(self, mock_discovery):
        """auto_connect fails when no device found."""
        mock_discovery.return_value = None

        result = self.conn.auto_connect()

        self.assertFalse(result)
        self.assertEqual(self.conn.state, ConnectionState.DISCONNECTED)

    @patch("gopro_connection.requests.get")
    @patch("gopro_connection.timed_full_discovery")
    def test_auto_connect_waits_for_network(self, mock_discovery, mock_get):
        """auto_connect succeeds when timed_full_discovery resolves IP after polling."""
        # timed_full_discovery handles the USB+no-IP polling internally
        # and returns a fully resolved device
        mock_discovery.return_value = GoProDevice(
            vendor_id=0x0A72, product_id=0x000D,
            description="GoPro Hero 12",
            camera_ip="172.20.145.51",
        )
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        result = self.conn.auto_connect()

        self.assertTrue(result)
        self.assertEqual(self.conn.ip, "172.20.145.51")
        self.conn._stop_keepalive()

    @patch("gopro_connection.requests.get")
    @patch("gopro_connection.timed_full_discovery")
    def test_auto_connect_with_retries(self, mock_discovery, mock_get):
        """auto_connect_with_retries retries on failure."""
        # Fail first two attempts, succeed on third
        device = GoProDevice(
            vendor_id=0x0A72, product_id=0x000D,
            description="GoPro Hero 12",
            camera_ip="172.20.145.51",
        )
        mock_discovery.side_effect = [None, None, device]
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        result = self.conn.auto_connect_with_retries()

        self.assertTrue(result)
        self.assertEqual(mock_discovery.call_count, 3)
        self.conn._stop_keepalive()


class TestIdleWorkaround(unittest.TestCase):
    """Test the IDLE state workaround (start/stop cycle)."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)
        self.conn.ip = "172.20.145.51"
        self.conn.base_url = f"http://172.20.145.51:{GOPRO_API_PORT}"

    @patch("gopro_connection.requests.get")
    def test_skip_when_already_off(self, mock_get):
        """IDLE workaround should skip if webcam is already OFF."""
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        result = self.conn.reset_webcam_state()

        self.assertTrue(result)
        # Should only have called webcam_status once (the initial check)
        # No start/stop cycle needed
        self.assertEqual(mock_get.call_count, 1)

    @patch("gopro_connection.requests.get")
    def test_resets_idle_state(self, mock_get):
        """IDLE workaround performs start/stop cycle when in IDLE."""
        # First call: status returns IDLE (1)
        # Second call: start command
        # Third call: stop command
        # Fourth call: status returns OFF (0) - verification
        mock_get.side_effect = [
            make_mock_response({"status": 1, "error": 0}),  # initial status: IDLE
            make_mock_response({"error": 0}),                 # start
            make_mock_response({"error": 0}),                 # stop
            make_mock_response({"status": 0, "error": 0}),   # final status: OFF
        ]

        result = self.conn.reset_webcam_state()

        self.assertTrue(result)
        self.assertEqual(mock_get.call_count, 4)


class TestStartStopWebcam(unittest.TestCase):
    """Test webcam start and stop commands."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)
        self.conn.ip = "172.20.145.51"
        self.conn.base_url = f"http://172.20.145.51:{GOPRO_API_PORT}"
        self.conn._state = ConnectionState.CONNECTED
        self.conn._connected = True

    @patch("gopro_connection.requests.get")
    def test_start_webcam_reaches_streaming(self, mock_get):
        """start_webcam succeeds when camera reaches STREAMING."""
        mock_get.side_effect = [
            make_mock_response({"status": 0}),      # webcam_status: OFF
            make_mock_response({"error": 0}),         # preview warmup (Phase 1.8)
            make_mock_response({"error": 0}),         # start command
            make_mock_response({"status": 3}),        # poll: STREAMING
        ]

        result = self.conn.start_webcam()

        self.assertTrue(result)
        self.assertEqual(self.conn.state, ConnectionState.STREAMING)

    @patch("gopro_connection.requests.get")
    def test_start_webcam_accepts_ready(self, mock_get):
        """start_webcam accepts READY immediately (Phase 1.7)."""
        responses = [
            make_mock_response({"status": 0}),    # webcam_status: OFF
            make_mock_response({"error": 0}),      # preview warmup (Phase 1.8)
            make_mock_response({"error": 0}),      # start command
            make_mock_response({"status": 2}),     # first poll: READY — accepted immediately
        ]

        mock_get.side_effect = responses

        result = self.conn.start_webcam()

        self.assertTrue(result)
        self.assertEqual(self.conn.state, ConnectionState.STREAMING)

    @patch("gopro_connection.requests.get")
    def test_stop_webcam(self, mock_get):
        """stop_webcam sends stop command and updates state."""
        self.conn._state = ConnectionState.STREAMING
        mock_get.return_value = make_mock_response({"error": 0})

        result = self.conn.stop_webcam()

        self.assertTrue(result)
        self.assertEqual(self.conn.state, ConnectionState.CONNECTED)

    @patch("gopro_connection.requests.get")
    def test_start_webcam_handles_idle(self, mock_get):
        """start_webcam auto-resets when camera is in IDLE."""
        mock_get.side_effect = [
            make_mock_response({"status": 1}),    # webcam_status: IDLE
            # IDLE workaround: status check shows IDLE
            make_mock_response({"status": 1}),
            make_mock_response({"error": 0}),      # start (workaround)
            make_mock_response({"error": 0}),      # stop (workaround)
            make_mock_response({"status": 0}),     # verify OFF
            make_mock_response({"status": 0}),     # status after reset: OFF
            make_mock_response({"error": 0}),      # preview warmup (Phase 1.8)
            make_mock_response({"error": 0}),      # actual start command
            make_mock_response({"status": 3}),     # poll: STREAMING
        ]

        result = self.conn.start_webcam()
        self.assertTrue(result)


class TestKeepAlive(unittest.TestCase):
    """Test keep-alive thread behavior."""

    def setUp(self):
        self.config = MockConfig()
        self.config.keepalive_interval = 0.05  # Very fast for testing
        self.conn = GoProConnection(self.config)
        self.conn.ip = "172.20.145.51"
        self.conn.base_url = f"http://172.20.145.51:{GOPRO_API_PORT}"
        self.conn._connected = True

    @patch("gopro_connection.requests.get")
    def test_keep_alive_success(self, mock_get):
        """keep_alive returns True when camera responds."""
        mock_get.return_value = make_mock_response({"status": 3})

        result = self.conn.keep_alive()

        self.assertTrue(result)
        self.assertTrue(self.conn.is_connected)

    @patch("gopro_connection.requests.get")
    def test_keep_alive_failure(self, mock_get):
        """keep_alive returns False and marks disconnected on failure."""
        import requests
        mock_get.side_effect = requests.ConnectionError("gone")

        result = self.conn.keep_alive()

        self.assertFalse(result)
        self.assertFalse(self.conn.is_connected)

    @patch("gopro_connection.requests.get")
    def test_keepalive_thread_detects_disconnect(self, mock_get):
        """Keep-alive thread sets RECONNECTING after 3 failures."""
        import requests as req_lib
        mock_get.side_effect = req_lib.ConnectionError("gone")

        states = []
        self.conn.on_connection_state = lambda s: states.append(s)

        self.conn._start_keepalive()

        # Wait for the keep-alive to detect 3 failures
        time.sleep(0.5)

        self.assertIn(ConnectionState.RECONNECTING, states)
        self.conn._stop_keepalive()


class TestDiscover(unittest.TestCase):
    """Test backward-compatible discover() method."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)

    @patch("gopro_connection.requests.get")
    @patch("gopro_connection.timed_full_discovery")
    def test_discover_success(self, mock_discovery, mock_get):
        """discover() finds camera and verifies API."""
        mock_discovery.return_value = GoProDevice(
            vendor_id=0x0A72, product_id=0x000D,
            description="GoPro Hero 12",
            camera_ip="172.20.145.51",
        )
        mock_get.return_value = make_mock_response({"status": 0})

        result = self.conn.discover()

        self.assertTrue(result)
        self.assertEqual(self.conn.ip, "172.20.145.51")
        self.assertTrue(self.conn.is_connected)

    @patch("gopro_connection.timed_full_discovery")
    def test_discover_no_device(self, mock_discovery):
        """discover() returns False when no GoPro found."""
        mock_discovery.return_value = None

        result = self.conn.discover()
        self.assertFalse(result)

    @patch("gopro_connection.requests.get")
    @patch("gopro_connection.timed_full_discovery")
    def test_discover_waits_for_ip(self, mock_discovery, mock_get):
        """discover() succeeds when timed_full_discovery resolves IP after polling."""
        # timed_full_discovery handles the USB+no-IP polling internally
        mock_discovery.return_value = GoProDevice(
            vendor_id=0x0A72, product_id=0x000D,
            description="GoPro Hero 12",
            camera_ip="172.20.145.51",
        )
        mock_get.return_value = make_mock_response({"status": 0})

        result = self.conn.discover()

        self.assertTrue(result)
        self.assertEqual(self.conn.ip, "172.20.145.51")


class TestDisconnect(unittest.TestCase):
    """Test clean disconnect."""

    def setUp(self):
        self.config = MockConfig()
        self.conn = GoProConnection(self.config)
        self.conn.ip = "172.20.145.51"
        self.conn.base_url = f"http://172.20.145.51:{GOPRO_API_PORT}"
        self.conn._connected = True
        self.conn._state = ConnectionState.STREAMING

    @patch("gopro_connection.requests.get")
    def test_disconnect_resets_state(self, mock_get):
        """disconnect() resets all connection state."""
        mock_get.return_value = make_mock_response({"error": 0})

        self.conn.disconnect()

        self.assertIsNone(self.conn.ip)
        self.assertIsNone(self.conn.base_url)
        self.assertFalse(self.conn.is_connected)
        self.assertEqual(self.conn.state, ConnectionState.DISCONNECTED)

    @patch("gopro_connection.requests.get")
    def test_disconnect_stops_keepalive(self, mock_get):
        """disconnect() stops the keep-alive thread."""
        mock_get.return_value = make_mock_response({"status": 0, "error": 0})

        # Start keep-alive first
        self.conn._start_keepalive()
        self.assertTrue(self.conn._keepalive_thread.is_alive())

        self.conn.disconnect()

        # Thread should have stopped
        self.assertFalse(self.conn._keepalive_thread.is_alive())


class TestNotifyCallback(unittest.TestCase):
    """Test status notification callbacks."""

    def test_notify_fires_callback(self):
        config = MockConfig()
        conn = GoProConnection(config)

        messages = []
        conn.on_status_change = lambda msg, lvl: messages.append((msg, lvl))

        conn._notify("test message", "info")

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0], ("test message", "info"))

    def test_notify_survives_callback_error(self):
        """Callback errors should not crash the connection flow."""
        config = MockConfig()
        conn = GoProConnection(config)

        def bad_callback(msg, lvl):
            raise RuntimeError("callback crash")

        conn.on_status_change = bad_callback

        # Should not raise
        conn._notify("test", "info")


class TestStateCallback(unittest.TestCase):
    """Test connection state change callbacks."""

    def test_state_change_fires_callback(self):
        config = MockConfig()
        conn = GoProConnection(config)

        states = []
        conn.on_connection_state = lambda s: states.append(s)

        conn._set_state(ConnectionState.CONNECTING)
        conn._set_state(ConnectionState.CONNECTED)

        self.assertEqual(states, [ConnectionState.CONNECTING, ConnectionState.CONNECTED])

    def test_duplicate_state_ignored(self):
        config = MockConfig()
        conn = GoProConnection(config)

        states = []
        conn.on_connection_state = lambda s: states.append(s)

        conn._set_state(ConnectionState.CONNECTING)
        conn._set_state(ConnectionState.CONNECTING)  # duplicate

        self.assertEqual(len(states), 1)


class TestResetForRecovery(unittest.TestCase):
    """Tests for the public reset_for_recovery() method."""

    def test_clears_ip_and_url(self):
        """reset_for_recovery() should clear ip and base_url."""
        config = MockConfig()
        conn = GoProConnection(config)
        conn.ip = "172.20.123.51"
        conn.base_url = "http://172.20.123.51:8080"
        conn._connected = True

        conn.reset_for_recovery()

        self.assertIsNone(conn.ip)
        self.assertIsNone(conn.base_url)
        self.assertFalse(conn.is_connected)

    def test_clears_connected_flag(self):
        """reset_for_recovery() should set _connected to False."""
        config = MockConfig()
        conn = GoProConnection(config)
        conn._connected = True

        conn.reset_for_recovery()

        self.assertFalse(conn.is_connected)

    def test_is_thread_safe(self):
        """reset_for_recovery() should be safe to call from any thread."""
        config = MockConfig()
        conn = GoProConnection(config)
        conn.ip = "172.20.123.51"
        conn.base_url = "http://172.20.123.51:8080"
        conn._connected = True

        errors = []

        def reset_from_thread():
            try:
                conn.reset_for_recovery()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reset_from_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        self.assertEqual(errors, [])
        self.assertIsNone(conn.ip)
        self.assertFalse(conn.is_connected)

    def test_idempotent_when_already_reset(self):
        """Calling reset_for_recovery() when already reset should be safe."""
        config = MockConfig()
        conn = GoProConnection(config)
        # Already in default state (ip=None, _connected=False)
        conn.reset_for_recovery()  # Should not raise
        self.assertIsNone(conn.ip)
        self.assertFalse(conn.is_connected)

    def test_does_not_stop_webcam(self):
        """reset_for_recovery() should NOT attempt to stop webcam or call API."""
        config = MockConfig()
        conn = GoProConnection(config)
        conn.ip = "172.20.123.51"
        conn._connected = True

        with patch.object(conn, 'stop_webcam') as mock_stop, \
             patch.object(conn, 'exit_webcam') as mock_exit:
            conn.reset_for_recovery()

        mock_stop.assert_not_called()
        mock_exit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
