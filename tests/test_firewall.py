"""
Tests for the firewall module.

These tests mock Windows-specific APIs (netsh, ctypes, ShellExecuteW) so they
can run on any platform without actually modifying firewall rules.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import firewall

import pytest

pytestmark = pytest.mark.no_gopro_needed


class TestIsAdmin(unittest.TestCase):
    """Test the is_admin() function."""

    @mock.patch("ctypes.windll.shell32.IsUserAnAdmin", return_value=1)
    def test_returns_true_when_admin(self, mock_admin):
        self.assertTrue(firewall.is_admin())

    @mock.patch("ctypes.windll.shell32.IsUserAnAdmin", return_value=0)
    def test_returns_false_when_not_admin(self, mock_admin):
        self.assertFalse(firewall.is_admin())

    @mock.patch("ctypes.windll.shell32.IsUserAnAdmin", side_effect=AttributeError)
    def test_returns_false_on_non_windows(self, mock_admin):
        self.assertFalse(firewall.is_admin())


class TestFirewallRuleExists(unittest.TestCase):
    """Test checking if the firewall rule exists."""

    @mock.patch("subprocess.run")
    def test_rule_found(self, mock_run):
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout=f"Rule Name: {firewall.RULE_NAME}\nEnabled: Yes\n",
        )
        self.assertTrue(firewall.firewall_rule_exists())

    @mock.patch("subprocess.run")
    def test_rule_not_found(self, mock_run):
        mock_run.return_value = mock.Mock(
            returncode=1,
            stdout="No rules match the specified criteria.\n",
        )
        self.assertFalse(firewall.firewall_rule_exists())

    @mock.patch("subprocess.run")
    def test_rule_returncode_zero_but_name_missing(self, mock_run):
        """Edge case: netsh returns 0 but the rule name isn't in output."""
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout="Some other rule\n",
        )
        self.assertFalse(firewall.firewall_rule_exists())

    @mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("netsh", 10))
    def test_timeout_returns_false(self, mock_run):
        self.assertFalse(firewall.firewall_rule_exists())

    @mock.patch("subprocess.run", side_effect=FileNotFoundError("netsh not found"))
    def test_netsh_missing_returns_false(self, mock_run):
        self.assertFalse(firewall.firewall_rule_exists())


class TestCreateFirewallRuleDirect(unittest.TestCase):
    """Test direct firewall rule creation (when already admin)."""

    @mock.patch("subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=0, stdout="Ok.\n", stderr="")
        result = firewall._create_firewall_rule_direct()
        self.assertTrue(result)

        # Verify netsh was called with correct args
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "netsh")
        self.assertIn("add", args)
        self.assertIn("rule", args)
        self.assertIn(f"name={firewall.RULE_NAME}", args)
        self.assertIn("protocol=UDP", args)
        self.assertIn(f"localport={firewall.UDP_PORT}", args)
        self.assertIn("dir=in", args)
        self.assertIn("action=allow", args)

    @mock.patch("subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = mock.Mock(
            returncode=1, stdout="", stderr="Access denied."
        )
        result = firewall._create_firewall_rule_direct()
        self.assertFalse(result)

    @mock.patch("subprocess.run", side_effect=OSError("netsh error"))
    def test_oserror(self, mock_run):
        result = firewall._create_firewall_rule_direct()
        self.assertFalse(result)


class TestMarkerFile(unittest.TestCase):
    """Test marker file read/write."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._orig_appdata = firewall.APPDATA_DIR
        self._orig_marker = firewall.FIREWALL_MARKER
        firewall.APPDATA_DIR = self.tmp_dir
        firewall.FIREWALL_MARKER = os.path.join(self.tmp_dir, ".firewall_rule_created")

    def tearDown(self):
        firewall.APPDATA_DIR = self._orig_appdata
        firewall.FIREWALL_MARKER = self._orig_marker
        # Clean up temp files
        marker = os.path.join(self.tmp_dir, ".firewall_rule_created")
        if os.path.exists(marker):
            os.remove(marker)
        os.rmdir(self.tmp_dir)

    def test_write_and_read_marker(self):
        self.assertFalse(firewall._marker_exists())
        firewall._write_marker()
        self.assertTrue(firewall._marker_exists())

        # Verify marker content is valid JSON
        with open(firewall.FIREWALL_MARKER, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["rule_name"], firewall.RULE_NAME)
        self.assertEqual(data["port"], firewall.UDP_PORT)
        self.assertEqual(data["protocol"], "UDP")


class TestEnsureFirewallRule(unittest.TestCase):
    """Test the main ensure_firewall_rule() entry point."""

    @mock.patch("firewall.firewall_rule_exists", return_value=True)
    @mock.patch("firewall._marker_exists", return_value=True)
    def test_rule_already_exists(self, mock_marker, mock_exists):
        """If rule already exists, return True immediately."""
        result = firewall.ensure_firewall_rule()
        self.assertTrue(result)

    @mock.patch("firewall._write_marker")
    @mock.patch("firewall.firewall_rule_exists", return_value=True)
    @mock.patch("firewall._marker_exists", return_value=False)
    def test_rule_exists_but_no_marker_syncs_marker(self, mock_marker_exists, mock_fw_exists, mock_write):
        """If rule exists but marker is missing, sync the marker."""
        result = firewall.ensure_firewall_rule()
        self.assertTrue(result)
        mock_write.assert_called_once()

    @mock.patch("firewall._write_marker")
    @mock.patch("firewall._create_firewall_rule_direct", return_value=True)
    @mock.patch("firewall.is_admin", return_value=True)
    @mock.patch("firewall.firewall_rule_exists", return_value=False)
    def test_creates_rule_when_admin(self, mock_exists, mock_admin, mock_create, mock_write):
        """If rule doesn't exist and we're admin, create it directly."""
        result = firewall.ensure_firewall_rule()
        self.assertTrue(result)
        mock_create.assert_called_once()
        mock_write.assert_called_once()

    @mock.patch("firewall._request_uac_elevation", return_value=True)
    @mock.patch("firewall.is_admin", return_value=False)
    @mock.patch("firewall.firewall_rule_exists", return_value=False)
    def test_requests_uac_when_not_admin(self, mock_exists, mock_admin, mock_uac):
        """If rule doesn't exist and not admin, request UAC elevation."""
        result = firewall.ensure_firewall_rule()
        self.assertTrue(result)
        mock_uac.assert_called_once()

    @mock.patch("firewall._request_uac_elevation", return_value=False)
    @mock.patch("firewall.is_admin", return_value=False)
    @mock.patch("firewall.firewall_rule_exists", return_value=False)
    def test_returns_false_when_uac_declined(self, mock_exists, mock_admin, mock_uac):
        """If user declines UAC, return False."""
        result = firewall.ensure_firewall_rule()
        self.assertFalse(result)

    @mock.patch("firewall._write_marker")
    @mock.patch("firewall._create_firewall_rule_direct", return_value=False)
    @mock.patch("firewall.is_admin", return_value=True)
    @mock.patch("firewall.firewall_rule_exists", return_value=False)
    def test_returns_false_when_admin_create_fails(self, mock_exists, mock_admin, mock_create, mock_write):
        """If admin but netsh fails, return False."""
        result = firewall.ensure_firewall_rule()
        self.assertFalse(result)
        mock_write.assert_not_called()


class TestRemoveFirewallRule(unittest.TestCase):
    """Test firewall rule removal."""

    @mock.patch("firewall.firewall_rule_exists", return_value=False)
    def test_nothing_to_remove(self, mock_exists):
        result = firewall.remove_firewall_rule()
        self.assertTrue(result)

    @mock.patch("firewall.is_admin", return_value=False)
    @mock.patch("firewall.firewall_rule_exists", return_value=True)
    def test_not_admin_cannot_remove(self, mock_exists, mock_admin):
        result = firewall.remove_firewall_rule()
        self.assertFalse(result)

    @mock.patch("subprocess.run")
    @mock.patch("firewall.is_admin", return_value=True)
    @mock.patch("firewall.firewall_rule_exists", return_value=True)
    def test_successful_removal(self, mock_exists, mock_admin, mock_run):
        mock_run.return_value = mock.Mock(returncode=0, stdout="Ok.\n", stderr="")
        result = firewall.remove_firewall_rule()
        self.assertTrue(result)


class TestRequestUACElevation(unittest.TestCase):
    """Test UAC elevation request."""

    @mock.patch("firewall._marker_exists", return_value=True)
    @mock.patch("ctypes.windll.shell32.ShellExecuteW", return_value=42)
    def test_successful_elevation(self, mock_shell, mock_marker):
        """ShellExecuteW returns >32 on success, then marker appears."""
        result = firewall._request_uac_elevation()
        self.assertTrue(result)
        # Verify ShellExecuteW was called with "runas"
        mock_shell.assert_called_once()
        call_args = mock_shell.call_args[0]
        self.assertEqual(call_args[1], "runas")  # verb = "runas"

    @mock.patch("ctypes.windll.shell32.ShellExecuteW", return_value=5)
    def test_user_declined_uac(self, mock_shell):
        """Return code 5 = SE_ERR_ACCESSDENIED (user clicked No)."""
        result = firewall._request_uac_elevation()
        self.assertFalse(result)

    @mock.patch("ctypes.windll.shell32.ShellExecuteW", side_effect=OSError("fail"))
    def test_shell_execute_error(self, mock_shell):
        result = firewall._request_uac_elevation()
        self.assertFalse(result)


class TestConstants(unittest.TestCase):
    """Verify key constants are set correctly."""

    def test_udp_port(self):
        self.assertEqual(firewall.UDP_PORT, 8554)

    def test_rule_name_descriptive(self):
        self.assertIn("8554", firewall.RULE_NAME)
        self.assertIn("GoPro", firewall.RULE_NAME)

    def test_appdata_dir(self):
        self.assertIn("GoProBridge", firewall.APPDATA_DIR)


class TestFirewallRulePersistence(unittest.TestCase):
    """Verify the firewall rule is persistent (uses netsh advfirewall, not temporary rules)."""

    @mock.patch("subprocess.run")
    def test_rule_uses_profile_any(self, mock_run):
        """The rule should apply to all network profiles (domain, private, public)."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="Ok.\n", stderr="")
        firewall._create_firewall_rule_direct()
        args = mock_run.call_args[0][0]
        self.assertIn("profile=any", args)

    @mock.patch("subprocess.run")
    def test_rule_is_inbound_allow(self, mock_run):
        """The rule must be inbound and allow traffic."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="Ok.\n", stderr="")
        firewall._create_firewall_rule_direct()
        args = mock_run.call_args[0][0]
        self.assertIn("dir=in", args)
        self.assertIn("action=allow", args)

    @mock.patch("subprocess.run")
    def test_rule_is_enabled(self, mock_run):
        """The rule should be explicitly enabled."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="Ok.\n", stderr="")
        firewall._create_firewall_rule_direct()
        args = mock_run.call_args[0][0]
        self.assertIn("enable=yes", args)

    @mock.patch("subprocess.run")
    def test_rule_has_description(self, mock_run):
        """The rule should have a human-readable description."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="Ok.\n", stderr="")
        firewall._create_firewall_rule_direct()
        args = mock_run.call_args[0][0]
        # Find the description argument
        desc_args = [a for a in args if a.startswith("description=")]
        self.assertEqual(len(desc_args), 1)
        self.assertIn("8554", desc_args[0])

    @mock.patch("subprocess.run")
    def test_netsh_uses_advfirewall_add_rule(self, mock_run):
        """Must use 'netsh advfirewall firewall add rule' for persistent rules."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="Ok.\n", stderr="")
        firewall._create_firewall_rule_direct()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[:4], ["netsh", "advfirewall", "firewall", "add"])


class TestEnsureFirewallRuleIdempotent(unittest.TestCase):
    """Verify ensure_firewall_rule is idempotent — safe to call multiple times."""

    @mock.patch("firewall.firewall_rule_exists", return_value=True)
    @mock.patch("firewall._marker_exists", return_value=True)
    def test_no_action_when_rule_exists(self, mock_marker, mock_exists):
        """Calling ensure_firewall_rule when rule already exists does nothing."""
        with mock.patch("firewall._create_firewall_rule_direct") as mock_create, \
             mock.patch("firewall._request_uac_elevation") as mock_uac:
            result = firewall.ensure_firewall_rule()
            self.assertTrue(result)
            mock_create.assert_not_called()
            mock_uac.assert_not_called()


class TestHelperScript(unittest.TestCase):
    """Verify the UAC helper script is generated correctly."""

    @mock.patch("firewall._marker_exists", return_value=True)
    @mock.patch("ctypes.windll.shell32.ShellExecuteW", return_value=42)
    def test_helper_script_written_to_temp(self, mock_shell, mock_marker):
        """The helper script should be written to a temp directory."""
        firewall._request_uac_elevation()
        # Verify ShellExecuteW was called with the helper script path
        call_args = mock_shell.call_args[0]
        params = call_args[3]  # arguments parameter
        self.assertIn("gopro_bridge_firewall_setup.py", params)

    @mock.patch("firewall._marker_exists", return_value=True)
    @mock.patch("ctypes.windll.shell32.ShellExecuteW", return_value=42)
    def test_elevation_uses_runas_verb(self, mock_shell, mock_marker):
        """UAC elevation must use the 'runas' verb."""
        firewall._request_uac_elevation()
        call_args = mock_shell.call_args[0]
        self.assertEqual(call_args[1], "runas")

    @mock.patch("firewall._marker_exists", return_value=True)
    @mock.patch("ctypes.windll.shell32.ShellExecuteW", return_value=42)
    def test_elevation_hides_console(self, mock_shell, mock_marker):
        """UAC helper should run hidden (SW_HIDE = 0)."""
        firewall._request_uac_elevation()
        call_args = mock_shell.call_args[0]
        self.assertEqual(call_args[5], 0)  # SW_HIDE


class TestStartupFirewallIntegration(unittest.TestCase):
    """Test that the firewall check is called during app startup."""

    def test_prerequisites_calls_ensure_firewall_rule(self):
        """Verify _check_prerequisites calls ensure_firewall_rule."""
        from config import Config
        from app_controller import AppController

        config = Config()
        config._config_path = ""

        # Mock USBEventListener to avoid Win32 API crashes in tests
        with mock.patch('usb_event_listener.USBEventListener'):
            controller = AppController(config)

        with mock.patch('shutil.which', return_value='C:\\ffmpeg\\ffmpeg.exe'), \
             mock.patch('firewall.ensure_firewall_rule', return_value=True) as mock_fw:
            result = controller._check_prerequisites()
            self.assertTrue(result)
            mock_fw.assert_called_once()

    def test_prerequisites_warns_on_firewall_failure(self):
        """Verify startup continues with warning if firewall rule fails."""
        from config import Config
        from app_controller import AppController

        config = Config()
        config._config_path = ""

        statuses = []

        with mock.patch('usb_event_listener.USBEventListener'):
            controller = AppController(config)
        controller.on_status = lambda msg, lvl: statuses.append((msg, lvl))

        with mock.patch('shutil.which', return_value='C:\\ffmpeg\\ffmpeg.exe'), \
             mock.patch('firewall.ensure_firewall_rule', return_value=False):
            result = controller._check_prerequisites()
            # Should still return True (non-fatal)
            self.assertTrue(result)
            # Should have emitted a warning
            warning_msgs = [m for m, lvl in statuses if lvl == "warning"]
            self.assertTrue(
                any("firewall" in m.lower() for m in warning_msgs),
                f"Expected firewall warning, got: {statuses}"
            )

    def test_prerequisites_fails_if_ffmpeg_missing_before_firewall(self):
        """ffmpeg check should fail BEFORE the firewall check runs."""
        from config import Config
        from app_controller import AppController
        config = Config()
        config._config_path = ""

        with mock.patch('usb_event_listener.USBEventListener'):
            controller = AppController(config)
        controller.on_status = lambda msg, lvl: None

        with mock.patch('shutil.which', return_value=None), \
             mock.patch('firewall.ensure_firewall_rule') as mock_fw:
            result = controller._check_prerequisites()
            self.assertFalse(result)
            # Firewall check should NOT have been called
            mock_fw.assert_not_called()


if __name__ == "__main__":
    unittest.main()
