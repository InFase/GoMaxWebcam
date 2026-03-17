"""
Tests for the discovery timeout mechanism.

Verifies that timed_full_discovery() correctly:
  - Returns None immediately when no USB device is found
  - Returns a device immediately when full discovery succeeds on first try
  - Polls for IP resolution when USB is detected but no IP
  - Raises DiscoveryTimeout when the deadline expires
  - Respects configurable timeout and poll interval parameters
  - Carries the partial device info in the DiscoveryTimeout exception
"""

import time
import unittest
from unittest.mock import patch, MagicMock, call

from src.discovery import (
    GoProDevice,
    DiscoveryTimeout,
    timed_full_discovery,
    full_discovery,
)

import pytest

pytestmark = pytest.mark.no_gopro_needed


def _make_device(with_ip: bool = False) -> GoProDevice:
    """Create a test GoProDevice."""
    return GoProDevice(
        vendor_id=0x0A70,
        product_id=0x000D,
        description="GoPro Hero 12 RNDIS",
        camera_ip="172.20.145.51" if with_ip else None,
    )


class TestDiscoveryTimeout(unittest.TestCase):
    """Test the DiscoveryTimeout exception class."""

    def test_exception_attributes(self):
        """DiscoveryTimeout carries device, elapsed, and timeout."""
        device = _make_device()
        exc = DiscoveryTimeout(device=device, elapsed=15.2, timeout=30.0)

        self.assertIs(exc.device, device)
        self.assertAlmostEqual(exc.elapsed, 15.2)
        self.assertAlmostEqual(exc.timeout, 30.0)

    def test_exception_message(self):
        """DiscoveryTimeout has a descriptive message."""
        device = _make_device()
        exc = DiscoveryTimeout(device=device, elapsed=10.0, timeout=30.0)

        msg = str(exc)
        self.assertIn("10.0s", msg)
        self.assertIn("30.0s", msg)
        self.assertIn("0A70:000D", msg)
        self.assertIn("IP not resolved", msg)

    def test_exception_is_exception(self):
        """DiscoveryTimeout is a proper Exception subclass."""
        self.assertTrue(issubclass(DiscoveryTimeout, Exception))


class TestTimedFullDiscoveryNoUSB(unittest.TestCase):
    """Test timed_full_discovery when no USB device is found."""

    @patch("src.discovery.full_discovery")
    def test_no_usb_returns_none_immediately(self, mock_fd):
        """When no USB device at all, return None without polling."""
        mock_fd.return_value = None

        result = timed_full_discovery(overall_timeout=5.0)

        self.assertIsNone(result)
        mock_fd.assert_called_once()

    @patch("src.discovery.discover_gopro_ip")
    @patch("src.discovery.full_discovery")
    def test_no_usb_does_not_poll(self, mock_fd, mock_ip):
        """When no USB device, discover_gopro_ip should never be called."""
        mock_fd.return_value = None

        timed_full_discovery(overall_timeout=5.0)

        mock_ip.assert_not_called()


class TestTimedFullDiscoveryImmediateSuccess(unittest.TestCase):
    """Test timed_full_discovery when device + IP found on first try."""

    @patch("src.discovery.full_discovery")
    def test_immediate_success(self, mock_fd):
        """Device with IP on first try returns immediately."""
        device = _make_device(with_ip=True)
        mock_fd.return_value = device

        result = timed_full_discovery(overall_timeout=30.0)

        self.assertIsNotNone(result)
        self.assertEqual(result.camera_ip, "172.20.145.51")
        mock_fd.assert_called_once()

    @patch("src.discovery.discover_gopro_ip")
    @patch("src.discovery.full_discovery")
    def test_immediate_success_no_polling(self, mock_fd, mock_ip):
        """When first try succeeds, no IP polling should happen."""
        mock_fd.return_value = _make_device(with_ip=True)

        timed_full_discovery(overall_timeout=30.0)

        mock_ip.assert_not_called()


class TestTimedFullDiscoveryPolling(unittest.TestCase):
    """Test the polling loop when USB detected but no IP initially."""

    @patch("src.discovery.time.sleep")
    @patch("src.discovery.discover_gopro_ip")
    @patch("src.discovery.full_discovery")
    def test_ip_found_after_polling(self, mock_fd, mock_ip, mock_sleep):
        """IP resolves after a few polls — should succeed."""
        mock_fd.return_value = _make_device(with_ip=False)
        # IP not found on first 2 polls, found on 3rd
        mock_ip.side_effect = [None, None, "172.20.145.51"]

        result = timed_full_discovery(
            overall_timeout=30.0,
            poll_interval=1.0,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.camera_ip, "172.20.145.51")
        self.assertEqual(mock_ip.call_count, 3)

    @patch("src.discovery.time.sleep")
    @patch("src.discovery.discover_gopro_ip")
    @patch("src.discovery.full_discovery")
    def test_sleep_called_between_polls(self, mock_fd, mock_ip, mock_sleep):
        """Sleep is called between each polling attempt."""
        mock_fd.return_value = _make_device(with_ip=False)
        mock_ip.side_effect = [None, "172.20.145.51"]

        timed_full_discovery(
            overall_timeout=30.0,
            poll_interval=2.0,
        )

        # sleep should be called at least once with the poll interval
        self.assertTrue(mock_sleep.called)
        # First sleep call should be for poll_interval (or less if near deadline)
        first_sleep = mock_sleep.call_args_list[0][0][0]
        self.assertLessEqual(first_sleep, 2.0)

    @patch("src.discovery.time.sleep")
    @patch("src.discovery.discover_gopro_ip")
    @patch("src.discovery.full_discovery")
    def test_probe_timeout_passed_through(self, mock_fd, mock_ip, mock_sleep):
        """probe_timeout is passed to full_discovery as timeout parameter."""
        mock_fd.return_value = _make_device(with_ip=True)

        timed_full_discovery(probe_timeout=7.5)

        mock_fd.assert_called_once_with(timeout=7.5)


class TestTimedFullDiscoveryTimeoutRaised(unittest.TestCase):
    """Test that DiscoveryTimeout is raised when deadline expires."""

    @patch("src.discovery.discover_gopro_ip")
    @patch("src.discovery.full_discovery")
    def test_timeout_raised_after_deadline(self, mock_fd, mock_ip):
        """DiscoveryTimeout raised when IP never resolves within deadline."""
        device = _make_device(with_ip=False)
        mock_fd.return_value = device
        mock_ip.return_value = None  # IP never resolves

        with self.assertRaises(DiscoveryTimeout) as ctx:
            timed_full_discovery(overall_timeout=0.15, poll_interval=0.03)

        exc = ctx.exception
        self.assertIs(exc.device, device)
        self.assertAlmostEqual(exc.timeout, 0.15)
        self.assertIn("0A70:000D", str(exc))

    @patch("src.discovery.discover_gopro_ip")
    @patch("src.discovery.full_discovery")
    def test_timeout_carries_partial_device(self, mock_fd, mock_ip):
        """The raised exception carries the USB-only device info."""
        device = _make_device(with_ip=False)
        mock_fd.return_value = device
        mock_ip.return_value = None

        with self.assertRaises(DiscoveryTimeout) as ctx:
            timed_full_discovery(overall_timeout=0.1, poll_interval=0.02)

        self.assertEqual(ctx.exception.device.vendor_id, 0x0A70)
        self.assertEqual(ctx.exception.device.product_id, 0x000D)
        self.assertIsNone(ctx.exception.device.camera_ip)

    @patch("src.discovery.discover_gopro_ip")
    @patch("src.discovery.full_discovery")
    def test_multiple_polls_before_timeout(self, mock_fd, mock_ip):
        """Multiple polling attempts happen before timeout fires."""
        mock_fd.return_value = _make_device(with_ip=False)
        mock_ip.return_value = None

        with self.assertRaises(DiscoveryTimeout):
            timed_full_discovery(overall_timeout=0.2, poll_interval=0.03)

        # With 0.2s timeout and 0.03s interval, should have multiple polls
        self.assertGreater(mock_ip.call_count, 1)


class TestTimedFullDiscoveryWithShortTimeout(unittest.TestCase):
    """Test edge cases with very short timeouts."""

    @patch("src.discovery.time.monotonic")
    @patch("src.discovery.time.sleep")
    @patch("src.discovery.discover_gopro_ip")
    @patch("src.discovery.full_discovery")
    def test_zero_timeout_raises_immediately(self, mock_fd, mock_ip, mock_sleep, mock_mono):
        """With 0s timeout, should raise DiscoveryTimeout without polling."""
        mock_fd.return_value = _make_device(with_ip=False)

        # Start at time 0, deadline is 0 — loop should exit immediately
        mock_mono.side_effect = [
            0.0,   # start_time
            0.0,   # while check: 0 < 0 is False, exits loop
            0.0,   # elapsed calc
        ]

        with self.assertRaises(DiscoveryTimeout):
            timed_full_discovery(overall_timeout=0.0)

        # No IP polling should have happened
        mock_ip.assert_not_called()

    @patch("src.discovery.time.sleep")
    @patch("src.discovery.discover_gopro_ip")
    @patch("src.discovery.full_discovery")
    def test_sleep_capped_to_remaining_time(self, mock_fd, mock_ip, mock_sleep):
        """Sleep duration should not exceed remaining time to deadline."""
        mock_fd.return_value = _make_device(with_ip=False)
        mock_ip.side_effect = [None, "172.20.145.51"]

        # Use real time.monotonic but very short timeout
        result = timed_full_discovery(
            overall_timeout=60.0,
            poll_interval=0.01,
        )

        # Verify sleep was called with values <= poll_interval
        for call_args in mock_sleep.call_args_list:
            sleep_duration = call_args[0][0]
            self.assertLessEqual(sleep_duration, 0.01 + 0.001)  # small float tolerance


class TestTimedFullDiscoveryIntegration(unittest.TestCase):
    """Integration-style tests using real timing (with short intervals)."""

    @patch("src.discovery.discover_gopro_ip")
    @patch("src.discovery.full_discovery")
    def test_real_timing_success(self, mock_fd, mock_ip):
        """Test with real time: IP resolves after a short delay."""
        mock_fd.return_value = _make_device(with_ip=False)
        mock_ip.side_effect = [None, "172.20.145.51"]

        start = time.monotonic()
        result = timed_full_discovery(
            overall_timeout=5.0,
            poll_interval=0.05,
        )
        elapsed = time.monotonic() - start

        self.assertIsNotNone(result)
        self.assertEqual(result.camera_ip, "172.20.145.51")
        # Should complete quickly (2 polls at 0.05s intervals)
        self.assertLess(elapsed, 2.0)

    @patch("src.discovery.discover_gopro_ip")
    @patch("src.discovery.full_discovery")
    def test_real_timing_timeout(self, mock_fd, mock_ip):
        """Test with real time: DiscoveryTimeout raised after deadline."""
        mock_fd.return_value = _make_device(with_ip=False)
        mock_ip.return_value = None  # Never resolves

        start = time.monotonic()
        with self.assertRaises(DiscoveryTimeout) as ctx:
            timed_full_discovery(
                overall_timeout=0.2,
                poll_interval=0.05,
            )
        elapsed = time.monotonic() - start

        # Should timeout close to the overall_timeout
        self.assertGreater(elapsed, 0.15)  # at least most of the timeout
        self.assertLess(elapsed, 1.0)      # but not too much longer
        self.assertAlmostEqual(ctx.exception.timeout, 0.2)


class TestGoProConnectionDiscoveryTimeout(unittest.TestCase):
    """Test that GoProConnection handles DiscoveryTimeout from discover().

    IMPORTANT: gopro_connection.py imports from 'discovery' (not 'src.discovery')
    because conftest.py adds src/ to sys.path. We must use the same DiscoveryTimeout
    class that gopro_connection sees, otherwise except clauses won't match.
    """

    def _make_connection(self):
        """Create a GoProConnection with a mock config."""
        from gopro_connection import GoProConnection
        config = MagicMock()
        config.discovery_overall_timeout = 5.0
        config.discovery_timeout = 3.0
        config.discovery_retry_interval = 1.0
        config.keepalive_interval = 2.5
        config.idle_reset_delay = 1.0
        return GoProConnection(config)

    def _make_timeout_exc(self, device=None):
        """Create a DiscoveryTimeout using the same class gopro_connection uses."""
        from discovery import DiscoveryTimeout as DT, GoProDevice
        dev = device or GoProDevice(
            vendor_id=0x0A70, product_id=0x000D,
            description="GoPro Hero 12 RNDIS",
        )
        return DT(device=dev, elapsed=5.0, timeout=5.0)

    @patch("gopro_connection.timed_full_discovery")
    def test_auto_connect_handles_timeout(self, mock_tfd):
        """auto_connect returns False when DiscoveryTimeout is raised."""
        mock_tfd.side_effect = self._make_timeout_exc()

        conn = self._make_connection()
        result = conn.auto_connect()

        self.assertFalse(result)

    @patch("gopro_connection.timed_full_discovery")
    def test_auto_connect_timeout_sets_disconnected(self, mock_tfd):
        """auto_connect sets state to DISCONNECTED on timeout."""
        from gopro_connection import ConnectionState
        mock_tfd.side_effect = self._make_timeout_exc()

        conn = self._make_connection()
        states = []
        conn.on_connection_state = lambda s: states.append(s)
        conn.auto_connect()

        self.assertIn(ConnectionState.DISCONNECTED, states)

    @patch("gopro_connection.timed_full_discovery")
    def test_auto_connect_timeout_notifies(self, mock_tfd):
        """auto_connect sends error notification on timeout."""
        mock_tfd.side_effect = self._make_timeout_exc()

        conn = self._make_connection()
        messages = []
        conn.on_status_change = lambda msg, lvl: messages.append((msg, lvl))
        conn.auto_connect()

        # Should have at least one error message about timeout
        error_msgs = [m for m, l in messages if l == "error"]
        self.assertTrue(any("timed out" in m.lower() for m in error_msgs))

    @patch("gopro_connection.timed_full_discovery")
    def test_discover_handles_timeout(self, mock_tfd):
        """discover() returns False when DiscoveryTimeout is raised."""
        mock_tfd.side_effect = self._make_timeout_exc()

        conn = self._make_connection()
        result = conn.discover()

        self.assertFalse(result)

    @patch("gopro_connection.timed_full_discovery")
    def test_discover_timeout_preserves_device_info(self, mock_tfd):
        """discover() stores the partial device on timeout for retry context."""
        mock_tfd.side_effect = self._make_timeout_exc()

        conn = self._make_connection()
        conn.discover()

        self.assertIsNotNone(conn.device_info)
        self.assertEqual(conn.device_info.vendor_id, 0x0A70)

    @patch("gopro_connection.timed_full_discovery")
    def test_auto_connect_passes_config_values(self, mock_tfd):
        """auto_connect passes config timeout values to timed_full_discovery."""
        mock_tfd.return_value = None  # No device found

        conn = self._make_connection()
        conn.auto_connect()

        mock_tfd.assert_called_once_with(
            overall_timeout=5.0,
            probe_timeout=3.0,
            poll_interval=1.0,
        )


class TestConfigDiscoveryOverallTimeout(unittest.TestCase):
    """Test that config.discovery_overall_timeout exists and has correct default."""

    def test_default_value(self):
        """Config should have discovery_overall_timeout defaulting to 30.0."""
        from src.config import Config
        config = Config()
        self.assertEqual(config.discovery_overall_timeout, 30.0)

    def test_config_serialization(self):
        """discovery_overall_timeout should be included in config JSON."""
        from src.config import Config
        from dataclasses import asdict
        config = Config()
        data = {k: v for k, v in asdict(config).items() if not k.startswith("_")}
        self.assertIn("discovery_overall_timeout", data)
        self.assertEqual(data["discovery_overall_timeout"], 30.0)


if __name__ == "__main__":
    unittest.main()
