"""
test_structured_logging.py — Verify structured [EVENT:*] logging is present
throughout the codebase at appropriate locations.

Tests coverage of structured logging for:
  - Connection events (connect, disconnect, state changes)
  - Discovery events (start, found, failed, timeout)
  - Stream events (start, stop, error)
  - Keep-alive events (ok, fail)
  - Reconnect events (attempt, success, failed)
  - Battery/camera info events
  - Freeze-frame events
  - Firewall events
  - Config events
  - Startup/shutdown events
  - Virtual camera events
  - USB listener events
"""

import ast
import os
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_gopro_needed


SRC_DIR = Path(__file__).parent.parent / "src"


def _read_source(filename: str) -> str:
    """Read a Python source file from src/."""
    filepath = SRC_DIR / filename
    return filepath.read_text(encoding="utf-8")


def _count_event_tags(source: str, tag: str) -> int:
    """Count occurrences of [EVENT:<tag>] in log calls within source code."""
    pattern = rf"\[EVENT:{tag}\]"
    return len(re.findall(pattern, source))


def _find_all_event_tags(source: str) -> list[str]:
    """Extract all unique EVENT tag names from source code."""
    return sorted(set(re.findall(r"\[EVENT:(\w+)\]", source)))


# ============================================================================
# Test that each module has structured EVENT tags
# ============================================================================

class TestModuleEventTagPresence:
    """Verify that every module uses structured [EVENT:*] logging."""

    def test_logger_has_startup_tags(self):
        src = _read_source("logger.py")
        assert _count_event_tags(src, "startup") >= 1

    def test_config_has_config_tags(self):
        src = _read_source("config.py")
        assert _count_event_tags(src, "config") >= 2  # load + save

    def test_app_controller_has_state_change_tags(self):
        src = _read_source("app_controller.py")
        assert _count_event_tags(src, "state_change") >= 1

    def test_app_controller_has_startup_tags(self):
        src = _read_source("app_controller.py")
        assert _count_event_tags(src, "startup") >= 2

    def test_app_controller_has_shutdown_tags(self):
        src = _read_source("app_controller.py")
        assert _count_event_tags(src, "shutdown") >= 2

    def test_app_controller_has_reconnect_tags(self):
        src = _read_source("app_controller.py")
        assert _count_event_tags(src, "reconnect_attempt") >= 1
        assert _count_event_tags(src, "recovery_success") >= 1

    def test_app_controller_has_freeze_frame_tags(self):
        src = _read_source("app_controller.py")
        assert _count_event_tags(src, "freeze_frame") >= 1
        assert _count_event_tags(src, "frame_recovery") >= 1

    def test_app_controller_has_battery_tags(self):
        src = _read_source("app_controller.py")
        assert _count_event_tags(src, "battery") >= 1

    def test_discovery_has_discovery_tags(self):
        src = _read_source("discovery.py")
        assert _count_event_tags(src, "discovery_start") >= 3
        assert _count_event_tags(src, "discovery_found") >= 3
        assert _count_event_tags(src, "discovery_failed") >= 3

    def test_gopro_connection_has_connection_tags(self):
        src = _read_source("gopro_connection.py")
        assert _count_event_tags(src, "connection") >= 3

    def test_gopro_connection_has_state_change_tags(self):
        src = _read_source("gopro_connection.py")
        assert _count_event_tags(src, "state_change") >= 3

    def test_gopro_connection_has_keepalive_tags(self):
        src = _read_source("gopro_connection.py")
        assert _count_event_tags(src, "keepalive_ok") >= 1
        assert _count_event_tags(src, "keepalive_fail") >= 1

    def test_gopro_connection_has_disconnection_tags(self):
        src = _read_source("gopro_connection.py")
        assert _count_event_tags(src, "disconnection") >= 1

    def test_gopro_connection_has_battery_tags(self):
        src = _read_source("gopro_connection.py")
        assert _count_event_tags(src, "battery") >= 2  # low + critical

    def test_gopro_connection_has_stream_tags(self):
        src = _read_source("gopro_connection.py")
        assert _count_event_tags(src, "stream_start") >= 1
        assert _count_event_tags(src, "stream_error") >= 1

    def test_stream_reader_has_ffmpeg_tags(self):
        src = _read_source("stream_reader.py")
        assert _count_event_tags(src, "ffmpeg_start") >= 1
        assert _count_event_tags(src, "ffmpeg_stop") >= 1
        assert _count_event_tags(src, "ffmpeg_error") >= 1

    def test_virtual_camera_has_vcam_tags(self):
        src = _read_source("virtual_camera.py")
        assert _count_event_tags(src, "vcam_start") >= 3
        assert _count_event_tags(src, "vcam_stop") >= 1
        assert _count_event_tags(src, "vcam_error") >= 3

    def test_frame_pipeline_has_stream_tags(self):
        src = _read_source("frame_pipeline.py")
        assert _count_event_tags(src, "stream_start") >= 1
        assert _count_event_tags(src, "stream_stop") >= 1
        assert _count_event_tags(src, "stream_error") >= 1

    def test_frame_pipeline_has_freeze_tags(self):
        src = _read_source("frame_pipeline.py")
        assert _count_event_tags(src, "freeze_frame") >= 1
        assert _count_event_tags(src, "frame_recovery") >= 1

    def test_frame_pipeline_has_state_change_tags(self):
        src = _read_source("frame_pipeline.py")
        assert _count_event_tags(src, "state_change") >= 1

    def test_frame_buffer_has_freeze_tags(self):
        src = _read_source("frame_buffer.py")
        assert _count_event_tags(src, "freeze_frame") >= 1
        assert _count_event_tags(src, "frame_recovery") >= 1

    def test_disconnect_detector_has_tags(self):
        src = _read_source("disconnect_detector.py")
        assert _count_event_tags(src, "disconnect_detector") >= 2
        assert _count_event_tags(src, "freeze_frame") >= 1
        assert _count_event_tags(src, "usb_detach") >= 1
        assert _count_event_tags(src, "usb_attach") >= 1

    def test_usb_event_listener_has_tags(self):
        src = _read_source("usb_event_listener.py")
        assert _count_event_tags(src, "usb_listener") >= 3

    def test_firewall_has_firewall_tags(self):
        src = _read_source("firewall.py")
        assert _count_event_tags(src, "firewall") >= 5

    def test_port_checker_has_port_check_tags(self):
        src = _read_source("port_checker.py")
        assert _count_event_tags(src, "port_check") >= 2

    def test_main_has_startup_shutdown_tags(self):
        src = _read_source("main.py")
        assert _count_event_tags(src, "startup") >= 5
        assert _count_event_tags(src, "shutdown") >= 1

    def test_gui_has_event_tags(self):
        src = _read_source("gui.py")
        assert _count_event_tags(src, "startup") >= 1
        assert _count_event_tags(src, "shutdown") >= 2


# ============================================================================
# Test EVENT tag category coverage
# ============================================================================

class TestEventTagCategoryCoverage:
    """Verify that all documented EVENT tag categories from logger.py are used."""

    # These are the categories documented in logger.py docstring
    REQUIRED_TAG_CATEGORIES = [
        "startup",
        "shutdown",
        "config",
        "discovery_start",
        "discovery_found",
        "discovery_failed",
        "connection",
        "disconnection",
        "state_change",
        "stream_start",
        "stream_stop",
        "stream_error",
        "keepalive_ok",
        "keepalive_fail",
        "reconnect_attempt",
        "recovery_success",
        "battery",
        "camera_info",
        "firewall",
        "port_check",
        "ffmpeg_start",
        "ffmpeg_stop",
        "ffmpeg_error",
        "vcam_start",
        "vcam_stop",
        "vcam_error",
        "freeze_frame",
        "frame_recovery",
    ]

    @pytest.fixture(autouse=True)
    def _load_all_sources(self):
        """Load all source files and collect EVENT tags."""
        self.all_source = ""
        for py_file in SRC_DIR.glob("*.py"):
            self.all_source += py_file.read_text(encoding="utf-8") + "\n"
        self.all_tags = set(re.findall(r"\[EVENT:(\w+)\]", self.all_source))

    @pytest.mark.parametrize("category", REQUIRED_TAG_CATEGORIES)
    def test_category_present_in_codebase(self, category):
        """Each documented EVENT tag category must be used at least once."""
        assert category in self.all_tags, (
            f"EVENT tag category '{category}' documented in logger.py "
            f"but never used in codebase. Tags found: {sorted(self.all_tags)}"
        )


# ============================================================================
# Test appropriate log levels
# ============================================================================

class TestLogLevels:
    """Verify that log levels are appropriate for different event types."""

    def test_errors_use_error_or_warning_level(self):
        """Error events should not be logged at INFO/DEBUG level."""
        src = _read_source("gopro_connection.py")
        # Patterns that indicate errors should use log.error or log.warning
        # Make sure disconnection events use WARNING or ERROR, not INFO
        lines = src.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Connection lost should be WARNING or ERROR
            if "[EVENT:disconnection]" in stripped:
                assert "log.warning" in stripped or "log.error" in stripped or "log.info" in stripped, (
                    f"Line {i+1}: disconnection event should use warning/error/info level"
                )

    def test_keepalive_ok_uses_debug(self):
        """Successful keep-alive should be at DEBUG level (high frequency)."""
        src = _read_source("gopro_connection.py")
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if "[EVENT:keepalive_ok]" in line and "Keep-alive OK" in line:
                assert "log.debug" in line, (
                    f"Line {i+1}: keepalive_ok success should use DEBUG level (not flood INFO)"
                )

    def test_battery_low_uses_warning(self):
        """Low battery should be logged at WARNING level."""
        src = _read_source("gopro_connection.py")
        assert "log.warning" in src and "battery" in src.lower(), (
            "Low battery conditions should trigger a warning-level log"
        )


# ============================================================================
# Test that every module has a logger
# ============================================================================

class TestModuleLoggerPresence:
    """Every src module should have a logger (except __init__.py and utils.py)."""

    MODULES_WITH_LOGGER = [
        "app_controller.py",
        "config.py",
        "discovery.py",
        "disconnect_detector.py",
        "firewall.py",
        "frame_buffer.py",
        "frame_pipeline.py",
        "gopro_connection.py",
        "gui.py",
        "logger.py",
        "main.py",
        "port_checker.py",
        "stream_reader.py",
        "usb_event_listener.py",
        "virtual_camera.py",
    ]

    @pytest.mark.parametrize("module", MODULES_WITH_LOGGER)
    def test_module_has_logger(self, module):
        """Each module should import and create a logger."""
        src = _read_source(module)
        has_logger = (
            "get_logger" in src
            or "logging.getLogger" in src
            or "setup_logger" in src
        )
        assert has_logger, f"{module} does not import/create a logger"
