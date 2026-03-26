"""
test_config_defaults.py — Verifies that GoMaxWebcam config has sensible
defaults that work out of the box (AC 7).

Tests:
  1. Default config values are sensible for a first-run user
  2. Config loads cleanly when no file exists (creates default)
  3. Corrupted config file falls back gracefully to defaults
  4. Validation passes for fresh defaults
  5. Invalid enum values are caught with friendly messages
  6. Invalid numeric ranges are caught with friendly messages
  7. Cross-field validation (backoff_base <= backoff_cap)
  8. reset_to_defaults() restores settings while preserving camera identity
  9. is_valid property reflects validation state
 10. Resolution/FOV conversion helpers return correct codes
 11. Transport priority_for_manager() converts names correctly
 12. Bool fields accept true/false as expected
 13. Empty transport priority list is caught
 14. Unknown transport entries are filtered out with a warning

All tests are purely mocked unit tests — no filesystem writes to real
user config dirs, no hardware, no GoPro connections.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gomaxwebcam.config import (
    AdvancedSection,
    CameraSection,
    Config,
    ConfigValidationIssue,
    DashboardSection,
    LoggingSection,
    TransportSection,
    VideoSection,
    WifiApSection,
)

pytestmark = pytest.mark.no_gopro_needed


# ===========================================================================
# Helpers
# ===========================================================================


def fresh_config(tmp_path: Path) -> Config:
    """Return a brand-new Config backed by a temporary directory."""
    return Config.load(config_dir=tmp_path)


# ===========================================================================
# Default values
# ===========================================================================


class TestSensibleDefaults:
    """Verify every section default is appropriate for a first-time user."""

    def test_fresh_config_loads_without_error(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg is not None

    def test_transport_priority_defaults_to_usb_first(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.transport.priority[0] == "usb"

    def test_transport_priority_includes_all_three(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert set(cfg.transport.priority) == {"usb", "cohn", "wifi_ap"}

    def test_ble_wake_mode_default_is_always_on(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.transport.ble_wake_mode == "always_on"

    def test_video_resolution_defaults_to_1080p(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.video.resolution == "1080p"

    def test_video_fov_defaults_to_wide(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.video.fov == "wide"

    def test_dashboard_auto_start_defaults_to_false(self, tmp_path):
        """Users don't want the app auto-starting without explicit opt-in."""
        cfg = fresh_config(tmp_path)
        assert cfg.dashboard.auto_start is False

    def test_dashboard_open_browser_on_start_defaults_to_true(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.dashboard.open_browser_on_start is True

    def test_advanced_udp_port_default(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.advanced.udp_port == 8554

    def test_advanced_frame_timeout_positive(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.advanced.frame_timeout_seconds > 0

    def test_advanced_keepalive_interval_positive(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.advanced.keepalive_interval > 0

    def test_advanced_max_consecutive_failures_positive(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.advanced.max_consecutive_failures > 0

    def test_advanced_backoff_base_less_than_cap(self, tmp_path):
        """Backoff base must never exceed the cap — default must satisfy this."""
        cfg = fresh_config(tmp_path)
        assert cfg.advanced.backoff_base_seconds < cfg.advanced.backoff_cap_seconds

    def test_advanced_health_check_interval_positive(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.advanced.health_check_interval > 0

    def test_advanced_usb_poll_interval_positive(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.advanced.usb_poll_interval > 0

    def test_logging_debug_defaults_to_false(self, tmp_path):
        """Debug logging should be off by default — it produces a lot of output."""
        cfg = fresh_config(tmp_path)
        assert cfg.logging.debug is False

    def test_logging_retention_days_positive(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.logging.log_retention_days > 0

    def test_logging_max_size_mb_positive(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.logging.log_max_size_mb > 0

    def test_camera_fields_empty_by_default(self, tmp_path):
        """Camera identity is blank until the camera is discovered."""
        cfg = fresh_config(tmp_path)
        assert cfg.camera.ble_address == ""
        assert cfg.camera.camera_name == ""
        assert cfg.camera.camera_serial == ""
        assert cfg.camera.last_known_ip == ""
        assert cfg.camera.model == ""

    def test_wifi_ap_fields_empty_by_default(self, tmp_path):
        """WiFi AP credentials are blank until the user sets them up."""
        cfg = fresh_config(tmp_path)
        assert cfg.wifi_ap.ap_name == ""
        assert cfg.wifi_ap.ap_password == ""


# ===========================================================================
# Default file creation
# ===========================================================================


class TestDefaultFileCreation:
    """Verify the default config file is created on first run."""

    def test_config_file_created_on_first_load(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.config_path.exists()

    def test_default_config_file_is_valid_toml(self, tmp_path):
        """The written default must be parseable TOML."""
        import tomllib
        fresh_config(tmp_path)
        content = (tmp_path / "config.toml").read_bytes()
        parsed = tomllib.loads(content.decode("utf-8"))
        assert "transport" in parsed
        assert "video" in parsed
        assert "advanced" in parsed

    def test_default_config_contains_help_comments(self, tmp_path):
        """Comments in the default file guide first-time users."""
        fresh_config(tmp_path)
        text = (tmp_path / "config.toml").read_text(encoding="utf-8")
        assert "resolution" in text
        assert "priority" in text


# ===========================================================================
# Corrupted / unreadable config falls back to defaults
# ===========================================================================


class TestCorruptedConfigFallback:
    """Verify graceful degradation when the config file is broken."""

    def test_invalid_toml_falls_back_to_defaults(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("this is [[[not valid toml", encoding="utf-8")
        cfg = Config.load(config_dir=tmp_path)
        # Should still have usable defaults
        assert cfg.video.resolution == "1080p"
        assert cfg.transport.priority == ["usb", "cohn", "wifi_ap"]

    def test_empty_config_file_falls_back_to_defaults(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("", encoding="utf-8")
        cfg = Config.load(config_dir=tmp_path)
        assert cfg.video.resolution == "1080p"


# ===========================================================================
# Validation passes for fresh defaults
# ===========================================================================


class TestValidationOnDefaults:
    """Fresh defaults must produce zero validation issues."""

    def test_default_config_is_valid(self, tmp_path):
        cfg = fresh_config(tmp_path)
        issues = cfg.validate()
        assert issues == [], f"Fresh defaults produced issues: {issues}"

    def test_is_valid_true_for_defaults(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.is_valid is True


# ===========================================================================
# Validation catches invalid enum values
# ===========================================================================


class TestValidationEnums:
    """Enum-type fields produce friendly messages for unknown values."""

    def test_invalid_resolution_caught(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.video.resolution = "4K"
        issues = cfg.validate()
        fields = [i.field for i in issues]
        assert "video.resolution" in fields
        msg = next(i.message for i in issues if i.field == "video.resolution")
        assert "4K" in msg

    def test_invalid_fov_caught(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.video.fov = "fisheye"
        issues = cfg.validate()
        fields = [i.field for i in issues]
        assert "video.fov" in fields

    def test_invalid_ble_wake_mode_caught(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.transport.ble_wake_mode = "ultra_saver"
        issues = cfg.validate()
        assert any(i.field == "transport.ble_wake_mode" for i in issues)

    def test_invalid_fov_hint_mentions_valid_options(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.video.fov = "turbo"
        issues = cfg.validate()
        issue = next(i for i in issues if i.field == "video.fov")
        assert "wide" in issue.hint or "linear" in issue.hint

    def test_invalid_ble_wake_mode_hint_is_actionable(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.transport.ble_wake_mode = "eco"
        issues = cfg.validate()
        issue = next(i for i in issues if i.field == "transport.ble_wake_mode")
        assert issue.hint  # hint must be non-empty


# ===========================================================================
# Validation catches invalid numeric ranges
# ===========================================================================


class TestValidationNumericRanges:
    """Out-of-range numeric values produce friendly messages."""

    def test_udp_port_below_range_caught(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.advanced.udp_port = 80  # privileged port
        issues = cfg.validate()
        assert any(i.field == "advanced.udp_port" for i in issues)

    def test_udp_port_above_range_caught(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.advanced.udp_port = 99999
        issues = cfg.validate()
        assert any(i.field == "advanced.udp_port" for i in issues)

    def test_udp_port_in_range_is_valid(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.advanced.udp_port = 9000
        issues = cfg.validate()
        assert not any(i.field == "advanced.udp_port" for i in issues)

    def test_frame_timeout_zero_caught(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.advanced.frame_timeout_seconds = 0
        issues = cfg.validate()
        assert any(i.field == "advanced.frame_timeout_seconds" for i in issues)

    def test_keepalive_negative_caught(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.advanced.keepalive_interval = -1.0
        issues = cfg.validate()
        assert any(i.field == "advanced.keepalive_interval" for i in issues)

    def test_max_consecutive_failures_zero_caught(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.advanced.max_consecutive_failures = 0
        issues = cfg.validate()
        assert any(i.field == "advanced.max_consecutive_failures" for i in issues)

    def test_backoff_base_zero_caught(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.advanced.backoff_base_seconds = 0
        issues = cfg.validate()
        assert any(i.field == "advanced.backoff_base_seconds" for i in issues)

    def test_backoff_cap_zero_caught(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.advanced.backoff_cap_seconds = 0
        issues = cfg.validate()
        assert any(i.field == "advanced.backoff_cap_seconds" for i in issues)

    def test_log_retention_zero_caught(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.logging.log_retention_days = 0
        issues = cfg.validate()
        assert any(i.field == "logging.log_retention_days" for i in issues)

    def test_log_max_size_zero_caught(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.logging.log_max_size_mb = 0
        issues = cfg.validate()
        assert any(i.field == "logging.log_max_size_mb" for i in issues)

    def test_numeric_issue_has_non_empty_hint(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.advanced.udp_port = 80
        issues = cfg.validate()
        issue = next(i for i in issues if i.field == "advanced.udp_port")
        assert issue.hint  # hint must be non-empty


# ===========================================================================
# Cross-field validation
# ===========================================================================


class TestCrossFieldValidation:
    """Checks that depend on relationships between fields."""

    def test_backoff_base_greater_than_cap_caught(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.advanced.backoff_base_seconds = 120
        cfg.advanced.backoff_cap_seconds = 60
        issues = cfg.validate()
        cross_field_issues = [
            i for i in issues if "backoff" in i.field and "base" in i.message.lower()
        ]
        assert cross_field_issues, "Expected cross-field backoff issue"

    def test_backoff_base_equal_to_cap_is_valid(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.advanced.backoff_base_seconds = 60
        cfg.advanced.backoff_cap_seconds = 60
        issues = cfg.validate()
        # Equal values: no cross-field issue
        cross_field_msgs = [
            i for i in issues
            if "backoff" in i.field and "larger than" in i.message
        ]
        assert not cross_field_msgs

    def test_empty_transport_priority_caught(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.transport.priority = []
        issues = cfg.validate()
        assert any(i.field == "transport.priority" for i in issues)
        msg = next(i.message for i in issues if i.field == "transport.priority")
        assert "camera" in msg.lower() or "connect" in msg.lower()


# ===========================================================================
# reset_to_defaults()
# ===========================================================================


class TestResetToDefaults:
    """Verify reset_to_defaults() works correctly."""

    def test_reset_restores_video_resolution(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.video.resolution = "480p"
        cfg.reset_to_defaults()
        assert cfg.video.resolution == "1080p"

    def test_reset_restores_transport_priority(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.transport.priority = ["wifi_ap"]
        cfg.reset_to_defaults()
        assert "usb" in cfg.transport.priority

    def test_reset_preserves_camera_identity(self, tmp_path):
        """Camera pairing info must survive a settings reset."""
        cfg = fresh_config(tmp_path)
        cfg.camera.ble_address = "AA:BB:CC:DD:EE:FF"
        cfg.camera.camera_name = "GoPro Hero 13"
        cfg.camera.camera_serial = "C3501234567890"
        cfg.reset_to_defaults()
        assert cfg.camera.ble_address == "AA:BB:CC:DD:EE:FF"
        assert cfg.camera.camera_name == "GoPro Hero 13"
        assert cfg.camera.camera_serial == "C3501234567890"

    def test_reset_writes_config_file(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.reset_to_defaults()
        assert cfg.config_path.exists()

    def test_reset_config_is_valid_after_reset(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.video.resolution = "invalid_res"
        cfg.advanced.udp_port = 99999
        cfg.reset_to_defaults()
        assert cfg.is_valid


# ===========================================================================
# is_valid property
# ===========================================================================


class TestIsValidProperty:
    """is_valid returns correct boolean state."""

    def test_is_valid_true_for_clean_config(self, tmp_path):
        cfg = fresh_config(tmp_path)
        assert cfg.is_valid is True

    def test_is_valid_false_after_corruption(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.video.resolution = "8K"
        assert cfg.is_valid is False

    def test_is_valid_restored_after_fix(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.video.resolution = "8K"
        assert cfg.is_valid is False
        cfg.video.resolution = "1080p"
        assert cfg.is_valid is True


# ===========================================================================
# Resolution / FOV code helpers
# ===========================================================================


class TestVideoHelpers:
    """Verify resolution_code() and fov_code() return correct API codes."""

    def test_1080p_code(self):
        v = VideoSection(resolution="1080p")
        assert v.resolution_code() == 12

    def test_720p_code(self):
        v = VideoSection(resolution="720p")
        assert v.resolution_code() == 7

    def test_480p_code(self):
        v = VideoSection(resolution="480p")
        assert v.resolution_code() == 4

    def test_unknown_resolution_falls_back_to_1080p_code(self):
        v = VideoSection(resolution="4K")
        assert v.resolution_code() == 12

    def test_wide_fov_code(self):
        v = VideoSection(fov="wide")
        assert v.fov_code() == 0

    def test_narrow_fov_code(self):
        v = VideoSection(fov="narrow")
        assert v.fov_code() == 2

    def test_superview_fov_code(self):
        v = VideoSection(fov="superview")
        assert v.fov_code() == 3

    def test_linear_fov_code(self):
        v = VideoSection(fov="linear")
        assert v.fov_code() == 4

    def test_unknown_fov_falls_back_to_wide_code(self):
        v = VideoSection(fov="fisheye")
        assert v.fov_code() == 0


# ===========================================================================
# Transport priority_for_manager()
# ===========================================================================


class TestTransportPriorityForManager:
    """Verify priority_for_manager() converts TOML names to manager names."""

    def test_default_priority_converts_correctly(self):
        t = TransportSection()
        result = t.priority_for_manager()
        assert result == ["USB", "COHN", "WIFI_AP"]

    def test_single_usb_converts(self):
        t = TransportSection(priority=["usb"])
        assert t.priority_for_manager() == ["USB"]

    def test_single_cohn_converts(self):
        t = TransportSection(priority=["cohn"])
        assert t.priority_for_manager() == ["COHN"]

    def test_single_wifi_ap_converts(self):
        t = TransportSection(priority=["wifi_ap"])
        assert t.priority_for_manager() == ["WIFI_AP"]

    def test_unknown_entry_converts_to_uppercase(self):
        t = TransportSection(priority=["usb", "custom_transport"])
        result = t.priority_for_manager()
        assert result[1] == "CUSTOM_TRANSPORT"


# ===========================================================================
# Config file round-trip (save → load)
# ===========================================================================


class TestConfigRoundTrip:
    """Modifications are persisted and reloaded correctly."""

    def test_save_and_reload_preserves_resolution(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.video.resolution = "720p"
        cfg.save()
        reloaded = Config.load(config_dir=tmp_path)
        assert reloaded.video.resolution == "720p"

    def test_save_and_reload_preserves_transport_priority(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.transport.priority = ["cohn", "usb"]
        cfg.save()
        reloaded = Config.load(config_dir=tmp_path)
        assert reloaded.transport.priority == ["cohn", "usb"]

    def test_save_and_reload_preserves_camera_name(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.camera.camera_name = "My Hero 13"
        cfg.save()
        reloaded = Config.load(config_dir=tmp_path)
        assert reloaded.camera.camera_name == "My Hero 13"

    def test_save_and_reload_preserves_advanced_settings(self, tmp_path):
        cfg = fresh_config(tmp_path)
        cfg.advanced.udp_port = 9000
        cfg.advanced.frame_timeout_seconds = 8
        cfg.save()
        reloaded = Config.load(config_dir=tmp_path)
        assert reloaded.advanced.udp_port == 9000
        assert reloaded.advanced.frame_timeout_seconds == 8


# ===========================================================================
# _validate_field: invalid values in loaded TOML are discarded
# ===========================================================================


class TestFieldValidationOnLoad:
    """Invalid values in the TOML file are silently replaced with defaults."""

    def test_invalid_resolution_in_toml_uses_default(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[video]\nresolution = "8K"\nfov = "wide"\n',
            encoding="utf-8",
        )
        cfg = Config.load(config_dir=tmp_path)
        assert cfg.video.resolution == "1080p"

    def test_invalid_fov_in_toml_uses_default(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[video]\nresolution = "1080p"\nfov = "fisheye"\n',
            encoding="utf-8",
        )
        cfg = Config.load(config_dir=tmp_path)
        assert cfg.video.fov == "wide"

    def test_out_of_range_port_in_toml_uses_default(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[advanced]\nudp_port = 99999\n",
            encoding="utf-8",
        )
        cfg = Config.load(config_dir=tmp_path)
        # Default is 8554 (the out-of-range value is rejected)
        assert cfg.advanced.udp_port == 8554

    def test_invalid_ble_wake_mode_in_toml_uses_default(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[transport]\nble_wake_mode = "deep_sleep"\n',
            encoding="utf-8",
        )
        cfg = Config.load(config_dir=tmp_path)
        assert cfg.transport.ble_wake_mode == "always_on"

    def test_non_list_priority_in_toml_uses_default(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[transport]\npriority = "usb"\n',
            encoding="utf-8",
        )
        cfg = Config.load(config_dir=tmp_path)
        # Non-list value is rejected, default applied
        assert isinstance(cfg.transport.priority, list)
        assert "usb" in cfg.transport.priority

    def test_unknown_transport_entries_filtered_out(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[transport]\npriority = ["usb", "magic_transport"]\n',
            encoding="utf-8",
        )
        cfg = Config.load(config_dir=tmp_path)
        assert "magic_transport" not in cfg.transport.priority
        assert "usb" in cfg.transport.priority

    def test_negative_frame_timeout_in_toml_uses_default(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[advanced]\nframe_timeout_seconds = -1\n",
            encoding="utf-8",
        )
        cfg = Config.load(config_dir=tmp_path)
        assert cfg.advanced.frame_timeout_seconds == 5  # default

    def test_negative_log_retention_in_toml_uses_default(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[logging]\nlog_retention_days = 0\n",
            encoding="utf-8",
        )
        cfg = Config.load(config_dir=tmp_path)
        assert cfg.logging.log_retention_days == 7  # default


# ===========================================================================
# ConfigValidationIssue named tuple
# ===========================================================================


class TestConfigValidationIssue:
    """Ensure ConfigValidationIssue behaves as expected."""

    def test_basic_construction(self):
        issue = ConfigValidationIssue(
            field="video.resolution",
            message="Resolution is not valid.",
            hint="Use 480p, 720p, or 1080p.",
        )
        assert issue.field == "video.resolution"
        assert "not valid" in issue.message
        assert "480p" in issue.hint

    def test_default_hint_is_empty_string(self):
        issue = ConfigValidationIssue(field="foo.bar", message="Something wrong.")
        assert issue.hint == ""

    def test_issue_is_a_named_tuple(self):
        issue = ConfigValidationIssue(field="x", message="y")
        # Named tuples support indexing
        assert issue[0] == "x"
        assert issue[1] == "y"
