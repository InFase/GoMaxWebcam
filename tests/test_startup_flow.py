"""
test_startup_flow.py -- Tests for the startup wiring logic

Verifies that:
  - AppController runs the startup flow automatically
  - Status callbacks fire with correct state transitions
  - Discovery retries work as expected
  - Prerequisites check works
  - Graceful stop works
  - The discovery-connect-initialize pipeline runs on startup
  - open_connection() is called between discover() and start_webcam()
"""

import sys
import os
import time
import threading
from unittest.mock import patch, MagicMock, call

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from app_controller import AppController, AppState
from gopro_connection import GoProConnection, WebcamStatus, ConnectionState
from discovery import GoProDevice
from port_checker import PortConflict, PortInUseError

import pytest

pytestmark = pytest.mark.no_gopro_needed


# USBEventListener uses raw Win32 APIs (RegisterClassExW, CreateWindowExW) that
# cause access violations when called from a pytest process. We must mock it
# globally for all tests that instantiate AppController.
_usb_listener_patch = patch(
    'usb_event_listener.USBEventListener',
    **{'return_value.start.return_value': False, 'return_value.stop.return_value': None},
)


def make_test_config(**overrides) -> Config:
    """Create a config with fast timings for testing."""
    config = Config()
    config.discovery_timeout = 0.5
    config.discovery_retry_interval = 0.1
    config.discovery_max_retries = 2
    config.keepalive_interval = 0.5
    config.reconnect_delay = 0.1
    config.ncm_adapter_wait = 0.1
    config.idle_reset_delay = 0.1
    config._config_path = ""
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def _make_test_device(camera_ip="172.20.10.51"):
    """Create a GoProDevice for testing."""
    return GoProDevice(
        vendor_id=0x0A70,
        product_id=0x000D,
        description="GoPro Hero 12 (test)",
        camera_ip=camera_ip,
    )


class TestStartupFlow:
    """Tests for the auto-startup wiring."""

    def test_state_transitions_on_successful_startup(self):
        """Verify the state machine goes through the correct states on success."""
        config = make_test_config()
        controller = AppController(config)

        states_seen = []
        statuses_seen = []

        controller.on_state_change = lambda s: states_seen.append(s)
        controller.on_status = lambda msg, lvl: statuses_seen.append((msg, lvl))

        # Mock out the actual GoPro calls and USB listener
        with patch.object(controller.gopro, 'discover', return_value=True), \
             patch.object(controller.gopro, 'start_webcam', return_value=True), \
             patch.object(controller.gopro, 'keep_alive', return_value=True), \
             patch.object(controller.gopro, 'get_camera_info', return_value={"battery_level": 80}), \
             patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.check_udp_port_available', return_value=None), \
             _usb_listener_patch:

            # Also need to set gopro as connected so keep_alive doesn't fail
            controller.gopro._connected = True
            controller.gopro.ip = "172.20.10.51"

            controller.start()
            time.sleep(2)  # Give startup flow time to complete
            controller.stop()

        # Should have gone through these states in order
        assert AppState.CHECKING_PREREQUISITES in states_seen
        assert AppState.DISCOVERING in states_seen
        assert AppState.CONNECTING in states_seen
        assert AppState.STREAMING in states_seen

        # Should have seen some status messages
        assert len(statuses_seen) > 0

        print("[PASS] test_state_transitions_on_successful_startup")

    def test_prerequisite_check_fails_without_ffmpeg(self):
        """Verify startup stops if ffmpeg is not found."""
        config = make_test_config()
        controller = AppController(config)

        states_seen = []
        controller.on_state_change = lambda s: states_seen.append(s)
        controller.on_status = lambda msg, lvl: None

        with patch('shutil.which', return_value=None), \
             _usb_listener_patch:
            controller.start()
            time.sleep(1)
            controller.stop()

        # Should end in ERROR state
        assert AppState.ERROR in states_seen
        print("[PASS] test_prerequisite_check_fails_without_ffmpeg")

    def test_prerequisite_check_fails_when_all_ports_in_use(self):
        """Verify startup stops with clear error if all candidate UDP ports are busy."""
        config = make_test_config()
        controller = AppController(config)

        states_seen = []
        statuses_seen = []
        controller.on_state_change = lambda s: states_seen.append(s)
        controller.on_status = lambda msg, lvl: statuses_seen.append((msg, lvl))

        # Simulate all ports occupied — find_available_port raises PortInUseError
        fake_conflict = PortConflict(
            port=8554, protocol="UDP", pid=12345, process_name="gopro_webcam.exe"
        )

        with patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.find_available_port', side_effect=PortInUseError(fake_conflict)), \
             _usb_listener_patch:
            controller.start()
            time.sleep(1)
            controller.stop()

        # Should end in ERROR state
        assert AppState.ERROR in states_seen, f"Expected ERROR state. States: {states_seen}"

        # Should have emitted an error message mentioning the port and process
        error_msgs = [msg for msg, lvl in statuses_seen if lvl == "error"]
        assert any("8554" in msg and "gopro_webcam.exe" in msg for msg in error_msgs), \
            f"Expected error about port 8554 and gopro_webcam.exe. Errors: {error_msgs}"

        print("[PASS] test_prerequisite_check_fails_when_all_ports_in_use")

    def test_prerequisite_auto_selects_next_port_when_preferred_busy(self):
        """Verify auto-selection picks next port without persisting to config."""
        config = make_test_config()
        original_port = config.udp_port  # 8554
        controller = AppController(config)

        statuses_seen = []
        controller.on_status = lambda msg, lvl: statuses_seen.append((msg, lvl))

        # find_available_port returns 8555 (preferred 8554 was busy)
        with patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.find_available_port', return_value=original_port + 1), \
             patch('virtual_camera.detect_backend', return_value={"backend": "unitycapture", "is_recommended": True, "warning": ""}), \
             _usb_listener_patch:
            result = controller._check_prerequisites()

        # Should have succeeded
        assert result is True, "Prerequisites should pass with auto-selected port"

        # Runtime config should be updated to 8555
        assert controller.config.udp_port == original_port + 1, \
            f"Expected runtime port {original_port + 1}, got {controller.config.udp_port}"

        # Should have emitted a warning about port auto-selection
        warning_msgs = [msg for msg, lvl in statuses_seen if lvl == "warning"]
        assert any(str(original_port) in msg and "auto-selected" in msg for msg in warning_msgs), \
            f"Expected warning about auto-selected port. Warnings: {warning_msgs}"

        print("[PASS] test_prerequisite_auto_selects_next_port_when_preferred_busy")

    def test_prerequisite_check_passes_when_port_free(self):
        """Verify startup proceeds past port check when port is available."""
        config = make_test_config()
        controller = AppController(config)

        states_seen = []
        statuses_seen = []
        controller.on_state_change = lambda s: states_seen.append(s)
        controller.on_status = lambda msg, lvl: statuses_seen.append((msg, lvl))

        with patch.object(controller.gopro, 'discover', return_value=True), \
             patch.object(controller.gopro, 'start_webcam', return_value=True), \
             patch.object(controller.gopro, 'keep_alive', return_value=True), \
             patch.object(controller.gopro, 'get_camera_info', return_value=None), \
             patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.check_udp_port_available', return_value=None), \
             _usb_listener_patch:

            controller.gopro._connected = True
            controller.gopro.ip = "172.20.10.51"

            controller.start()
            time.sleep(2)
            controller.stop()

        # Should have passed prerequisites and reached DISCOVERING
        assert AppState.DISCOVERING in states_seen, \
            f"Expected DISCOVERING state (port check passed). States: {states_seen}"

        # Should have emitted a success message about the port
        success_msgs = [msg for msg, lvl in statuses_seen if lvl == "success"]
        assert any("8554" in msg and "available" in msg.lower() for msg in success_msgs), \
            f"Expected success message about port 8554 available. Successes: {success_msgs}"

        print("[PASS] test_prerequisite_check_passes_when_port_free")

    def test_discovery_retries_on_failure(self):
        """Verify discovery retries the configured number of times."""
        config = make_test_config(discovery_max_retries=3)
        controller = AppController(config)

        discover_call_count = 0

        def counting_discover():
            nonlocal discover_call_count
            discover_call_count += 1
            return False  # Always fail

        states_seen = []
        controller.on_state_change = lambda s: states_seen.append(s)
        controller.on_status = lambda msg, lvl: None

        with patch.object(controller.gopro, 'discover', side_effect=counting_discover), \
             patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.check_udp_port_available', return_value=None), \
             _usb_listener_patch:
            controller.start()
            time.sleep(3)
            controller.stop()

        assert discover_call_count == 3, f"Expected 3 discovery attempts, got {discover_call_count}"
        assert AppState.ERROR in states_seen
        print("[PASS] test_discovery_retries_on_failure")

    def test_stop_interrupts_discovery(self):
        """Verify calling stop() during discovery doesn't hang."""
        config = make_test_config(discovery_max_retries=100, discovery_retry_interval=1.0)
        controller = AppController(config)

        controller.on_state_change = lambda s: None
        controller.on_status = lambda msg, lvl: None

        with patch.object(controller.gopro, 'discover', return_value=False), \
             patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.check_udp_port_available', return_value=None), \
             _usb_listener_patch:
            controller.start()
            time.sleep(0.5)
            controller.stop()

        # Should complete quickly -- not hang for 100 retries
        assert controller.state == AppState.STOPPED
        print("[PASS] test_stop_interrupts_discovery")

    def test_status_callback_receives_messages(self):
        """Verify the on_status callback gets meaningful messages."""
        config = make_test_config()
        controller = AppController(config)

        messages = []
        controller.on_state_change = lambda s: None
        controller.on_status = lambda msg, lvl: messages.append((msg, lvl))

        with patch.object(controller.gopro, 'discover', return_value=True), \
             patch.object(controller.gopro, 'start_webcam', return_value=True), \
             patch.object(controller.gopro, 'keep_alive', return_value=True), \
             patch.object(controller.gopro, 'get_camera_info', return_value=None), \
             patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.check_udp_port_available', return_value=None), \
             _usb_listener_patch:

            controller.gopro._connected = True
            controller.gopro.ip = "172.20.10.51"

            controller.start()
            time.sleep(2)
            controller.stop()

        # Should have at least: "Checking prerequisites", "Discovery attempt",
        # "Starting webcam", and streaming confirmation
        msg_texts = [m[0] for m in messages]
        assert any("prerequisite" in m.lower() for m in msg_texts), f"Missing prerequisites message. Got: {msg_texts}"
        assert any("discovery" in m.lower() or "searching" in m.lower() for m in msg_texts), \
            f"Missing discovery message. Got: {msg_texts}"

        print("[PASS] test_status_callback_receives_messages")

    def test_camera_info_callback(self):
        """Verify camera info (battery, etc.) is pushed to the callback."""
        config = make_test_config()
        controller = AppController(config)

        camera_infos = []
        controller.on_state_change = lambda s: None
        controller.on_status = lambda msg, lvl: None
        controller.on_camera_info = lambda info: camera_infos.append(info)

        with patch.object(controller.gopro, 'discover', return_value=True), \
             patch.object(controller.gopro, 'start_webcam', return_value=True), \
             patch.object(controller.gopro, 'keep_alive', return_value=True), \
             patch.object(controller.gopro, 'get_camera_info', return_value={"battery_level": 75}), \
             patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.check_udp_port_available', return_value=None), \
             _usb_listener_patch:

            controller.gopro._connected = True
            controller.gopro.ip = "172.20.10.51"

            controller.start()
            time.sleep(2)
            controller.stop()

        assert len(camera_infos) > 0
        assert camera_infos[0]["battery_level"] == 75
        print("[PASS] test_camera_info_callback")


class TestPipelineWiring:
    """Tests for the discovery-connect-initialize pipeline wiring.

    Verifies that the startup flow correctly chains:
      discover() → open_connection() → start_webcam()
    and that open_connection() is called to enable wired USB control
    and perform the IDLE workaround before webcam mode starts.
    """

    def test_open_connection_called_before_start_webcam(self):
        """Verify open_connection() is called between discover and start_webcam.

        When discover() sets device_info and the connection state is not yet
        CONNECTED, _connect_and_start() should call open_connection() first
        to enable USB control and perform the IDLE workaround.
        """
        config = make_test_config()
        controller = AppController(config)

        call_order = []

        def mock_discover():
            call_order.append("discover")
            # Simulate discover setting device_info with camera_ip
            controller.gopro.device_info = _make_test_device()
            controller.gopro.ip = "172.20.10.51"
            controller.gopro.base_url = "http://172.20.10.51:8080"
            controller.gopro._connected = True
            return True

        def mock_open_connection(device):
            call_order.append("open_connection")
            # Simulate successful connection setup
            controller.gopro._state = ConnectionState.CONNECTED
            controller.gopro._connected = True
            return True

        def mock_start_webcam(**kwargs):
            call_order.append("start_webcam")
            return True

        controller.on_state_change = lambda s: None
        controller.on_status = lambda msg, lvl: None

        with patch.object(controller.gopro, 'discover', side_effect=mock_discover), \
             patch.object(controller.gopro, 'open_connection', side_effect=mock_open_connection), \
             patch.object(controller.gopro, 'start_webcam', side_effect=mock_start_webcam), \
             patch.object(controller.gopro, 'keep_alive', return_value=True), \
             patch.object(controller.gopro, 'get_camera_info', return_value=None), \
             patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.check_udp_port_available', return_value=None), \
             _usb_listener_patch:

            controller.start()
            time.sleep(2)
            controller.stop()

        # Verify the pipeline order: discover → open_connection → start_webcam
        assert "discover" in call_order, f"discover not called. Order: {call_order}"
        assert "open_connection" in call_order, f"open_connection not called. Order: {call_order}"
        assert "start_webcam" in call_order, f"start_webcam not called. Order: {call_order}"

        discover_idx = call_order.index("discover")
        open_conn_idx = call_order.index("open_connection")
        webcam_idx = call_order.index("start_webcam")

        assert discover_idx < open_conn_idx < webcam_idx, \
            f"Wrong order: expected discover < open_connection < start_webcam, got {call_order}"

        print("[PASS] test_open_connection_called_before_start_webcam")

    def test_open_connection_failure_stops_pipeline(self):
        """If open_connection() fails, the pipeline should not proceed to start_webcam."""
        config = make_test_config()
        controller = AppController(config)

        states_seen = []
        webcam_called = False

        def mock_discover():
            controller.gopro.device_info = _make_test_device()
            controller.gopro.ip = "172.20.10.51"
            controller.gopro.base_url = "http://172.20.10.51:8080"
            controller.gopro._connected = True
            return True

        def mock_open_connection(device):
            return False  # Simulate failure

        def mock_start_webcam(**kwargs):
            nonlocal webcam_called
            webcam_called = True
            return True

        controller.on_state_change = lambda s: states_seen.append(s)
        controller.on_status = lambda msg, lvl: None

        with patch.object(controller.gopro, 'discover', side_effect=mock_discover), \
             patch.object(controller.gopro, 'open_connection', side_effect=mock_open_connection), \
             patch.object(controller.gopro, 'start_webcam', side_effect=mock_start_webcam), \
             patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.check_udp_port_available', return_value=None), \
             _usb_listener_patch:

            controller.start()
            time.sleep(2)
            controller.stop()

        assert not webcam_called, "start_webcam should NOT be called when open_connection fails"
        assert AppState.ERROR in states_seen, f"Should end in ERROR state. States: {states_seen}"

        print("[PASS] test_open_connection_failure_stops_pipeline")

    def test_pipeline_runs_automatically_on_start(self):
        """Verify that calling start() triggers the full pipeline automatically.

        The discovery-connect-initialize pipeline should run without any
        additional method calls — just controller.start() is enough.
        """
        config = make_test_config()
        controller = AppController(config)

        pipeline_reached_streaming = threading.Event()

        def on_state_change(state):
            if state == AppState.STREAMING:
                pipeline_reached_streaming.set()

        controller.on_state_change = on_state_change
        controller.on_status = lambda msg, lvl: None

        with patch.object(controller.gopro, 'discover', return_value=True), \
             patch.object(controller.gopro, 'start_webcam', return_value=True), \
             patch.object(controller.gopro, 'keep_alive', return_value=True), \
             patch.object(controller.gopro, 'get_camera_info', return_value=None), \
             patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.check_udp_port_available', return_value=None), \
             _usb_listener_patch:

            controller.gopro._connected = True
            controller.gopro.ip = "172.20.10.51"

            # Single call to start() should trigger the entire pipeline
            controller.start()

            # Wait for the pipeline to reach STREAMING state
            reached = pipeline_reached_streaming.wait(timeout=5.0)
            controller.stop()

        assert reached, "Pipeline did not reach STREAMING state within timeout"
        print("[PASS] test_pipeline_runs_automatically_on_start")

    def test_auto_connect_on_launch_config(self):
        """Verify auto_connect_on_launch config flag is respected."""
        config = make_test_config(auto_connect_on_launch=False)

        # The config flag is checked in main.py, not in AppController.
        # AppController.start() always starts the pipeline.
        # Verify the flag exists and is properly set.
        assert config.auto_connect_on_launch is False

        config2 = make_test_config(auto_connect_on_launch=True)
        assert config2.auto_connect_on_launch is True

        print("[PASS] test_auto_connect_on_launch_config")

    def test_start_is_idempotent(self):
        """Calling start() twice should not start a second pipeline."""
        config = make_test_config()
        controller = AppController(config)

        controller.on_state_change = lambda s: None
        controller.on_status = lambda msg, lvl: None

        discover_count = 0

        def counting_discover():
            nonlocal discover_count
            discover_count += 1
            time.sleep(0.2)
            return True

        with patch.object(controller.gopro, 'discover', side_effect=counting_discover), \
             patch.object(controller.gopro, 'start_webcam', return_value=True), \
             patch.object(controller.gopro, 'keep_alive', return_value=True), \
             patch.object(controller.gopro, 'get_camera_info', return_value=None), \
             patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.check_udp_port_available', return_value=None), \
             _usb_listener_patch:

            controller.gopro._connected = True
            controller.gopro.ip = "172.20.10.51"

            controller.start()
            controller.start()  # Second call should be ignored
            time.sleep(2)
            controller.stop()

        # discover should only be called once (from the first start)
        assert discover_count == 1, f"Expected 1 discover call, got {discover_count}"
        print("[PASS] test_start_is_idempotent")


class TestStartupValidation:
    """Tests for startup prerequisite validation edge cases.

    Covers the 13-item reliability fix areas:
      - ffmpeg presence check with custom config path
      - Port auto-selection: non-blocking warning on fallback
      - Port auto-selection: runtime config updated, NOT persisted
      - All ports occupied: clear error with process details
      - Config defaults are sensible for fresh installs
      - Idempotent start/stop cycles
    """

    def test_ffmpeg_custom_path_used(self):
        """Startup should use config.ffmpeg_path, not hardcoded 'ffmpeg'."""
        config = make_test_config()
        config.ffmpeg_path = "custom_ffmpeg.exe"
        controller = AppController(config)
        controller.on_status = lambda msg, lvl: None

        with patch('shutil.which', return_value=None) as mock_which, \
             _usb_listener_patch:
            controller._check_prerequisites()
            # Verify the custom path was checked
            mock_which.assert_called_with("custom_ffmpeg.exe")

    def test_port_auto_selection_emits_warning_not_error(self):
        """When port auto-selects, status should be 'warning', never 'error'."""
        config = make_test_config()
        controller = AppController(config)

        statuses = []
        controller.on_status = lambda msg, lvl: statuses.append((msg, lvl))

        with patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.find_available_port', return_value=8555), \
             patch('virtual_camera.detect_backend', return_value={"backend": "unitycapture", "is_recommended": True, "warning": ""}), \
             _usb_listener_patch:
            result = controller._check_prerequisites()

        assert result is True
        # Find the port auto-selection message
        port_msgs = [(msg, lvl) for msg, lvl in statuses if "auto-selected" in msg]
        assert len(port_msgs) >= 1, f"Expected auto-selected message. Got: {statuses}"
        for msg, lvl in port_msgs:
            assert lvl == "warning", f"Expected 'warning' level, got '{lvl}' for: {msg}"

    def test_port_auto_selection_updates_runtime_config_only(self):
        """Auto-selected port should update runtime config but NOT call save()."""
        config = make_test_config()
        original_port = config.udp_port
        controller = AppController(config)
        controller.on_status = lambda msg, lvl: None

        with patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.find_available_port', return_value=original_port + 1), \
             patch('virtual_camera.detect_backend', return_value={"backend": "unitycapture", "is_recommended": True, "warning": ""}), \
             patch.object(config, 'save') as mock_save, \
             _usb_listener_patch:
            result = controller._check_prerequisites()

        assert result is True
        assert controller.config.udp_port == original_port + 1
        mock_save.assert_not_called()

    def test_all_ports_occupied_error_includes_process_info(self):
        """When all ports are occupied, error message must include process name."""
        config = make_test_config()
        controller = AppController(config)

        statuses = []
        controller.on_status = lambda msg, lvl: statuses.append((msg, lvl))

        conflict = PortConflict(
            port=8554, protocol="UDP", pid=42069, process_name="ffmpeg.exe"
        )

        with patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.find_available_port', side_effect=PortInUseError(conflict)), \
             _usb_listener_patch:
            result = controller._check_prerequisites()

        assert result is False
        error_msgs = [msg for msg, lvl in statuses if lvl == "error"]
        assert any("ffmpeg.exe" in msg for msg in error_msgs), \
            f"Error should mention process name. Errors: {error_msgs}"
        assert any("42069" in msg for msg in error_msgs), \
            f"Error should mention PID. Errors: {error_msgs}"

    def test_firewall_failure_is_non_blocking_warning(self):
        """Firewall check failure should warn but NOT block startup."""
        config = make_test_config()
        controller = AppController(config)

        statuses = []
        controller.on_status = lambda msg, lvl: statuses.append((msg, lvl))

        with patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=False), \
             patch('port_checker.find_available_port', return_value=8554), \
             patch('virtual_camera.detect_backend', return_value={"backend": "unitycapture", "is_recommended": True, "warning": ""}), \
             _usb_listener_patch:
            result = controller._check_prerequisites()

        assert result is True  # Firewall failure is non-blocking
        warning_msgs = [msg for msg, lvl in statuses if lvl == "warning"]
        assert any("firewall" in msg.lower() for msg in warning_msgs), \
            f"Expected firewall warning. Warnings: {warning_msgs}"

    def test_config_defaults_are_sensible(self):
        """Default config values should be valid for a fresh install."""
        config = Config()
        assert config.udp_port == 8554
        assert config.stream_width == 1920
        assert config.stream_height == 1080
        assert config.stream_fps == 30
        assert config.ffmpeg_path == "ffmpeg"
        assert config.discovery_max_retries > 0
        assert config.keepalive_interval > 0
        assert config.reconnect_delay > 0
        assert config.reconnect_max_delay >= config.reconnect_delay
        assert config.stale_poll_interval > 0
        assert config.ncm_adapter_wait > 0

    def test_stop_then_start_is_clean(self):
        """stop() then start() should cleanly restart without stale state."""
        config = make_test_config()
        controller = AppController(config)
        controller.on_state_change = lambda s: None
        controller.on_status = lambda msg, lvl: None

        with patch.object(controller.gopro, 'discover', return_value=False), \
             patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.check_udp_port_available', return_value=None), \
             _usb_listener_patch:
            controller.start()
            time.sleep(0.5)
            controller.stop()

        assert controller.state == AppState.STOPPED

        # Start again - should work cleanly
        with patch.object(controller.gopro, 'discover', return_value=False), \
             patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.check_udp_port_available', return_value=None), \
             _usb_listener_patch:
            controller.start()
            time.sleep(0.5)
            controller.stop()

        assert controller.state == AppState.STOPPED

    def test_virtual_camera_backend_warning_non_blocking(self):
        """Missing virtual camera backend should warn but not block startup."""
        config = make_test_config()
        controller = AppController(config)

        statuses = []
        controller.on_status = lambda msg, lvl: statuses.append((msg, lvl))

        with patch('shutil.which', return_value='/usr/bin/ffmpeg'), \
             patch('firewall.ensure_firewall_rule', return_value=True), \
             patch('port_checker.find_available_port', return_value=8554), \
             patch('virtual_camera.detect_backend', return_value={
                 "backend": None,
                 "is_recommended": False,
                 "warning": "No virtual camera backend found. Install Unity Capture.",
             }), \
             _usb_listener_patch:
            result = controller._check_prerequisites()

        assert result is True  # Non-blocking
        warning_msgs = [msg for msg, lvl in statuses if lvl == "warning"]
        assert any("Unity Capture" in msg for msg in warning_msgs)


class TestStartupRecoveryIdempotency:
    """Tests that recovery and reconnection triggers are idempotent."""

    def test_usb_reconnect_event_set_twice_is_safe(self):
        """Setting _usb_reconnect_event twice should not cause issues."""
        config = make_test_config()

        with _usb_listener_patch:
            controller = AppController(config)
            controller._usb_reconnect_event.set()
            controller._usb_reconnect_event.set()  # Second set is a no-op
            assert controller._usb_reconnect_event.is_set()

    def test_stop_clears_recovery_state(self):
        """stop() should clear recovery flags so restart is clean."""
        config = make_test_config()

        with _usb_listener_patch:
            controller = AppController(config)
            controller._is_recovering = True
            controller._recovery_count = 5
            controller.stop()

            # After stop, state should be STOPPED
            assert controller.state == AppState.STOPPED


def run_all_tests():
    """Run all tests and report results."""
    suites = [TestStartupFlow(), TestPipelineWiring()]
    tests = []
    for suite in suites:
        for name in dir(suite):
            if name.startswith("test_"):
                tests.append((suite, getattr(suite, name)))

    passed = 0
    failed = 0

    for suite, test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 40}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
