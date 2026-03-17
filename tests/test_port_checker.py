"""
Tests for port_checker.py — UDP port availability checking.

Tests cover:
  - Port available (free port → returns None)
  - Port in use (occupied port → returns PortConflict)
  - PortInUseError is raised by check_port_and_raise
  - PortConflict user_message formatting
  - Process identification (mocked netstat/tasklist)
"""

import socket
import subprocess
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from port_checker import (
    check_udp_port_available,
    check_port_and_raise,
    find_available_port,
    PortConflict,
    PortInUseError,
    _identify_port_owner,
    _get_process_name,
)
import pytest

pytestmark = pytest.mark.no_gopro_needed



class TestCheckUdpPortAvailable(unittest.TestCase):
    """Test the core port-checking function."""

    def test_free_port_returns_none(self):
        """A port nobody is using should return None (available)."""
        # Use a high ephemeral port that's almost certainly free
        result = check_udp_port_available(0)
        # Port 0 tells OS to pick a free port, but our function binds to 0.0.0.0:0
        # which always succeeds. Use a specific free port instead.
        # Find a free port first
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        free_port = sock.getsockname()[1]
        sock.close()

        result = check_udp_port_available(free_port)
        self.assertIsNone(result)

    def test_occupied_port_returns_conflict(self):
        """A port already bound should return a PortConflict."""
        # Bind a port to simulate it being in use
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        occupied_port = sock.getsockname()[1]

        try:
            result = check_udp_port_available(occupied_port)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, PortConflict)
            self.assertEqual(result.port, occupied_port)
            self.assertEqual(result.protocol, "UDP")
        finally:
            sock.close()

    def test_default_port_8554_when_free(self):
        """Port 8554 specifically should be checkable."""
        # We can't guarantee 8554 is free in CI, so just verify the function
        # runs without crashing
        result = check_udp_port_available(8554)
        # Result is either None (free) or PortConflict (in use) — both are valid
        self.assertTrue(result is None or isinstance(result, PortConflict))


class TestCheckPortAndRaise(unittest.TestCase):
    """Test the convenience function that raises on conflict."""

    def test_raises_on_occupied_port(self):
        """Should raise PortInUseError when port is occupied."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        occupied_port = sock.getsockname()[1]

        try:
            with self.assertRaises(PortInUseError) as ctx:
                check_port_and_raise(occupied_port)

            self.assertIsInstance(ctx.exception.conflict, PortConflict)
            self.assertEqual(ctx.exception.conflict.port, occupied_port)
        finally:
            sock.close()

    def test_no_raise_on_free_port(self):
        """Should not raise when port is free."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        free_port = sock.getsockname()[1]
        sock.close()

        # Should not raise
        check_port_and_raise(free_port)


class TestPortConflictUserMessage(unittest.TestCase):
    """Test that user-facing messages are clear and helpful."""

    def test_message_with_process_name_and_pid(self):
        conflict = PortConflict(
            port=8554, protocol="UDP", pid=12345, process_name="gopro_webcam.exe"
        )
        msg = conflict.user_message
        self.assertIn("8554", msg)
        self.assertIn("gopro_webcam.exe", msg)
        self.assertIn("12345", msg)
        self.assertIn("taskkill", msg)

    def test_message_with_pid_only(self):
        conflict = PortConflict(port=8554, protocol="UDP", pid=12345)
        msg = conflict.user_message
        self.assertIn("12345", msg)
        self.assertIn("taskkill", msg)
        self.assertNotIn("None", msg)

    def test_message_with_no_details(self):
        conflict = PortConflict(port=8554, protocol="UDP")
        msg = conflict.user_message
        self.assertIn("8554", msg)
        self.assertIn("GoPro Webcam Utility", msg)
        self.assertIn("ffmpeg", msg)

    def test_port_in_use_error_str(self):
        """PortInUseError str() should contain the user message."""
        conflict = PortConflict(port=8554, protocol="UDP", pid=999, process_name="test.exe")
        err = PortInUseError(conflict)
        self.assertIn("test.exe", str(err))
        self.assertIn("8554", str(err))


class TestIdentifyPortOwner(unittest.TestCase):
    """Test netstat parsing for port owner identification."""

    @patch("port_checker.subprocess.run")
    def test_parses_netstat_output(self, mock_run):
        """Should extract PID from netstat output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "Active Connections\n"
                "\n"
                "  Proto  Local Address          Foreign Address        State           PID\n"
                "  UDP    0.0.0.0:8554           *:*                                    42069\n"
                "  UDP    0.0.0.0:5353           *:*                                    1234\n"
            ),
        )

        conflict = PortConflict(port=8554, protocol="UDP")
        _identify_port_owner(conflict)

        self.assertEqual(conflict.pid, 42069)

    @patch("port_checker.subprocess.run")
    def test_handles_netstat_failure(self, mock_run):
        """Should not crash if netstat fails."""
        mock_run.return_value = MagicMock(returncode=1, stderr="error")

        conflict = PortConflict(port=8554, protocol="UDP")
        _identify_port_owner(conflict)

        self.assertIsNone(conflict.pid)

    @patch("port_checker.subprocess.run")
    def test_handles_netstat_timeout(self, mock_run):
        """Should not crash if netstat times out."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="netstat", timeout=10)

        conflict = PortConflict(port=8554, protocol="UDP")
        _identify_port_owner(conflict)  # Should not raise

        self.assertIsNone(conflict.pid)


class TestGetProcessName(unittest.TestCase):
    """Test tasklist parsing for process name resolution."""

    @patch("port_checker.subprocess.run")
    def test_parses_tasklist_output(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='"ffmpeg.exe","42069","Console","1","15,432 K"\n',
        )

        conflict = PortConflict(port=8554, protocol="UDP", pid=42069)
        _get_process_name(conflict)

        self.assertEqual(conflict.process_name, "ffmpeg.exe")

    @patch("port_checker.subprocess.run")
    def test_handles_no_pid(self, mock_run):
        """Should do nothing if pid is None."""
        conflict = PortConflict(port=8554, protocol="UDP")
        _get_process_name(conflict)

        mock_run.assert_not_called()
        self.assertIsNone(conflict.process_name)

    @patch("port_checker.subprocess.run")
    def test_handles_tasklist_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)

        conflict = PortConflict(port=8554, protocol="UDP", pid=42069)
        _get_process_name(conflict)

        self.assertIsNone(conflict.process_name)


class TestFindAvailablePort(unittest.TestCase):
    """Test ephemeral port auto-selection."""

    def test_returns_preferred_port_when_free(self):
        """Should return the preferred port if it's available."""
        # Find a free port to use as the preferred
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        free_port = sock.getsockname()[1]
        sock.close()

        result = find_available_port(free_port)
        self.assertEqual(result, free_port)

    def test_skips_to_next_port_when_preferred_busy(self):
        """Should try next port when preferred port is occupied."""
        # Bind the preferred port
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        occupied_port = sock.getsockname()[1]

        try:
            result = find_available_port(occupied_port)
            # Should have selected a port after the occupied one
            self.assertGreater(result, occupied_port)
            self.assertLessEqual(result, occupied_port + 10)
        finally:
            sock.close()

    def test_skips_multiple_occupied_ports(self):
        """Should skip multiple occupied ports to find a free one."""
        socks = []
        # Bind two consecutive ports
        sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock1.bind(("0.0.0.0", 0))
        base_port = sock1.getsockname()[1]
        socks.append(sock1)

        sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock2.bind(("0.0.0.0", base_port + 1))
            socks.append(sock2)
        except OSError:
            # If we can't bind base_port+1, the test still validates
            # that find_available_port skips at least one port
            pass

        try:
            result = find_available_port(base_port)
            self.assertGreater(result, base_port)
        finally:
            for s in socks:
                s.close()

    def test_raises_when_all_candidates_occupied(self):
        """Should raise PortInUseError when all ports in range are busy."""
        socks = []
        # Bind a block of consecutive ports
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        base_port = sock.getsockname()[1]
        socks.append(sock)

        # Try to bind the next 2 ports (use max_attempts=3)
        for offset in range(1, 3):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.bind(("0.0.0.0", base_port + offset))
                socks.append(s)
            except OSError:
                s.close()

        try:
            if len(socks) == 3:
                # All 3 ports are occupied — should raise
                with self.assertRaises(PortInUseError):
                    find_available_port(base_port, max_attempts=3)
            else:
                # Couldn't occupy all ports — skip test
                self.skipTest("Could not bind consecutive ports for test")
        finally:
            for s in socks:
                s.close()

    def test_does_not_persist_to_config(self):
        """Auto-selected port should not be saved to config file."""
        # This is a design test — find_available_port returns an int,
        # it never touches config. The caller (AppController) updates
        # config.udp_port in memory but never calls config.save().
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        occupied_port = sock.getsockname()[1]

        try:
            result = find_available_port(occupied_port)
            # Just verify it returns an int (not a config object)
            self.assertIsInstance(result, int)
            self.assertNotEqual(result, occupied_port)
        finally:
            sock.close()


class TestFindAvailablePortEdgeCases(unittest.TestCase):
    """Edge cases for ephemeral port auto-selection."""

    def test_port_65535_is_valid_candidate(self):
        """Port 65535 (max valid port) should be a valid candidate."""
        # Just check that the function doesn't crash with high ports
        with patch('port_checker.check_udp_port_available', return_value=None):
            result = find_available_port(65530, max_attempts=5)
            self.assertEqual(result, 65530)

    def test_port_above_65535_skipped(self):
        """Ports above 65535 should be skipped (invalid port numbers)."""
        # Start at 65534, max_attempts=5 → tries 65534, 65535, then stops
        calls = []

        def mock_check(port):
            calls.append(port)
            return PortConflict(port=port, protocol="UDP")

        with patch('port_checker.check_udp_port_available', side_effect=mock_check):
            try:
                find_available_port(65534, max_attempts=5)
            except PortInUseError:
                pass

        # Should not try ports > 65535
        for p in calls:
            self.assertLessEqual(p, 65535)

    def test_auto_selected_port_is_not_persisted(self):
        """Auto-selected port should NOT modify Config on disk."""
        from config import Config
        config = Config()
        config._config_path = ""  # No disk path
        original_port = config.udp_port

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        occupied_port = sock.getsockname()[1]
        config.udp_port = occupied_port

        try:
            new_port = find_available_port(occupied_port)
            # find_available_port returns an int, it does NOT touch config
            self.assertIsInstance(new_port, int)
            self.assertNotEqual(new_port, occupied_port)
            # Config is not modified by find_available_port
            self.assertEqual(config.udp_port, occupied_port)
        finally:
            sock.close()

    def test_config_save_preserves_preferred_port_not_auto_selected(self):
        """config.save() should write the original preferred port, not the auto-selected one."""
        import json
        import tempfile
        from config import Config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            config = Config()
            config._config_path = tmp_path
            preferred = 8554
            auto_selected = 8556

            # Simulate what app_controller does: save preferred, then overwrite udp_port
            config.udp_port = preferred
            config._preferred_udp_port = preferred
            config.udp_port = auto_selected

            # Runtime should use the auto-selected port
            self.assertEqual(config.udp_port, auto_selected)

            # But save() should write the preferred port to disk
            config.save()
            with open(tmp_path, "r") as f:
                saved = json.load(f)
            self.assertEqual(saved["udp_port"], preferred,
                             "config.save() must persist the user's preferred port, not the auto-selected one")
        finally:
            os.unlink(tmp_path)

    def test_config_save_without_auto_selection_saves_udp_port_normally(self):
        """When no auto-selection occurred, save() should write udp_port as-is."""
        import json
        import tempfile
        from config import Config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            config = Config()
            config._config_path = tmp_path
            config.udp_port = 9999
            # _preferred_udp_port stays 0 (default) — no auto-selection happened

            config.save()
            with open(tmp_path, "r") as f:
                saved = json.load(f)
            self.assertEqual(saved["udp_port"], 9999)
        finally:
            os.unlink(tmp_path)


class TestPortConflictEdgeCases(unittest.TestCase):
    """Edge cases for PortConflict message formatting."""

    def test_user_message_includes_taskkill_command(self):
        """User message with PID should include exact taskkill command."""
        conflict = PortConflict(port=8554, protocol="UDP", pid=9999)
        msg = conflict.user_message
        self.assertIn("taskkill /PID 9999 /F", msg)

    def test_user_message_without_pid_suggests_common_culprits(self):
        """User message without PID should list common culprits."""
        conflict = PortConflict(port=8554, protocol="UDP")
        msg = conflict.user_message
        self.assertIn("GoPro Webcam Utility", msg)
        self.assertIn("ffmpeg", msg)
        self.assertIn("Task Manager", msg)

    def test_port_in_use_error_inherits_from_runtime_error(self):
        """PortInUseError should be a RuntimeError subclass."""
        conflict = PortConflict(port=8554, protocol="UDP")
        err = PortInUseError(conflict)
        self.assertIsInstance(err, RuntimeError)
        self.assertIs(err.conflict, conflict)


class TestPortCheckerRaceCondition(unittest.TestCase):
    """Tests for race condition handling in port checking."""

    def test_find_available_port_race_recovery(self):
        """If port frees up between all-occupied scan and final check, use it."""
        call_count = [0]

        def mock_check(port):
            call_count[0] += 1
            # First scan: all ports busy. Final recheck: port freed up.
            if call_count[0] <= 3:
                return PortConflict(port=port, protocol="UDP")
            return None  # Port freed up

        with patch('port_checker.check_udp_port_available', side_effect=mock_check):
            result = find_available_port(8554, max_attempts=3)
            self.assertEqual(result, 8554)


if __name__ == "__main__":
    unittest.main()
