"""
test_webcam_init.py — Tests for Sub-AC 3: GoPro webcam initialization command sequence

Verifies that the GoPro-specific command sequence over USB correctly
initializes webcam/streaming mode without user interaction. Tests cover:

  1. Normal startup from OFF state
  2. IDLE workaround (start/stop reset) when camera reports stale IDLE
  3. Restart from already-STREAMING state (stop first, then start)
  4. Error recovery during initialization
  5. Status polling until READY/STREAMING confirmed
  6. Correct resolution and FOV parameters sent
"""

import sys
import os
import time
from unittest.mock import patch, MagicMock, call

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from gopro_connection import GoProConnection, WebcamStatus

import pytest

pytestmark = pytest.mark.no_gopro_needed


def make_test_config(**overrides) -> Config:
    """Create a config with fast timings for testing."""
    config = Config()
    config.discovery_timeout = 0.5
    config.discovery_retry_interval = 0.1
    config.discovery_max_retries = 2
    config.keepalive_interval = 0.5
    config.reconnect_delay = 0.1
    config.ncm_adapter_wait = 0.1
    config.idle_reset_delay = 0.05  # Very fast for tests
    config._config_path = ""
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def make_connected_gopro(**config_overrides) -> GoProConnection:
    """Create a GoProConnection that's already 'discovered' (has IP set)."""
    config = make_test_config(**config_overrides)
    gopro = GoProConnection(config)
    gopro.ip = "172.20.123.51"
    gopro.base_url = "http://172.20.123.51:8080"
    gopro._connected = True
    return gopro


class TestWebcamInitSequence:
    """Tests for start_webcam() — the full command sequence."""

    def test_normal_startup_from_off(self):
        """When camera is OFF, should just send start and wait for STREAMING."""
        gopro = make_connected_gopro()

        # Simulate: status=OFF, then start succeeds, then status=STREAMING
        status_sequence = [
            {"error": 0, "status": WebcamStatus.OFF},      # Initial check
            {"error": 0},                                     # start response
            {"error": 0, "status": WebcamStatus.READY},     # first poll
            {"error": 0, "status": WebcamStatus.STREAMING}, # second poll
        ]
        call_idx = [0]

        def mock_api_get(endpoint, timeout=5.0):
            idx = call_idx[0]
            call_idx[0] += 1
            if idx < len(status_sequence):
                return status_sequence[idx]
            return {"error": 0, "status": WebcamStatus.STREAMING}

        with patch.object(gopro, '_api_get', side_effect=mock_api_get):
            result = gopro.start_webcam(resolution=4, fov=4)

        assert result is True, "start_webcam should succeed when camera goes OFF → STREAMING"
        print("[PASS] test_normal_startup_from_off")

    def test_idle_workaround_triggered(self):
        """When camera reports IDLE, should do start→stop reset before real start."""
        gopro = make_connected_gopro()

        api_calls = []

        # Track all API calls to verify the sequence
        # reset_webcam_state now checks status first, so the sequence is:
        #   1. start_webcam checks status -> IDLE (triggers workaround)
        #   2. reset_webcam_state checks status -> IDLE (confirms workaround needed)
        #   3. workaround: start, stop
        #   4. reset_webcam_state verifies -> OFF
        #   5. start_webcam re-checks status -> OFF
        #   6. actual start command
        #   7. polling -> STREAMING
        status_responses = {
            "/gopro/webcam/status": iter([
                {"error": 0, "status": WebcamStatus.IDLE},      # start_webcam initial check
                {"error": 0, "status": WebcamStatus.IDLE},      # reset_webcam_state initial check
                {"error": 0, "status": WebcamStatus.OFF},       # reset_webcam_state final verify
                {"error": 0, "status": WebcamStatus.OFF},       # start_webcam re-check after reset
                {"error": 0, "status": WebcamStatus.STREAMING}, # Polling after real start
            ]),
        }

        def mock_api_get(endpoint, timeout=5.0):
            api_calls.append(endpoint)

            # Route status requests
            if "/gopro/webcam/status" in endpoint:
                try:
                    return next(status_responses["/gopro/webcam/status"])
                except StopIteration:
                    return {"error": 0, "status": WebcamStatus.STREAMING}

            # Start and stop always succeed
            if "/gopro/webcam/start" in endpoint:
                return {"error": 0}
            if "/gopro/webcam/stop" in endpoint:
                return {"error": 0}

            return {"error": 0}

        with patch.object(gopro, '_api_get', side_effect=mock_api_get):
            result = gopro.start_webcam()

        assert result is True, "start_webcam should succeed after IDLE workaround"

        # Verify the IDLE workaround sequence happened:
        # 1. status check (sees IDLE)
        # 2. start (workaround)
        # 3. stop (workaround)
        # 4. status check (verifies reset)
        # 5. real start
        # 6. status polling
        start_calls = [c for c in api_calls if "/gopro/webcam/start" in c]
        stop_calls = [c for c in api_calls if "/gopro/webcam/stop" in c]

        assert len(start_calls) >= 2, f"Expected at least 2 start calls (workaround + real), got {len(start_calls)}"
        assert len(stop_calls) >= 1, f"Expected at least 1 stop call (workaround), got {len(stop_calls)}"
        print("[PASS] test_idle_workaround_triggered")

    def test_already_streaming_stops_first(self):
        """When camera is already STREAMING, should stop before restarting."""
        gopro = make_connected_gopro()

        api_calls = []

        status_iter = iter([
            {"error": 0, "status": WebcamStatus.STREAMING},  # Initial check → already streaming
            {"error": 0, "status": WebcamStatus.STREAMING},  # After restart, polling
        ])

        def mock_api_get(endpoint, timeout=5.0):
            api_calls.append(endpoint)
            if "/gopro/webcam/status" in endpoint:
                try:
                    return next(status_iter)
                except StopIteration:
                    return {"error": 0, "status": WebcamStatus.STREAMING}
            return {"error": 0}

        with patch.object(gopro, '_api_get', side_effect=mock_api_get):
            result = gopro.start_webcam(resolution=12, fov=0)

        assert result is True

        # Should have called stop BEFORE the new start
        stop_idx = None
        start_idx = None
        for i, call_ep in enumerate(api_calls):
            if "/gopro/webcam/stop" in call_ep and stop_idx is None:
                stop_idx = i
            if "/gopro/webcam/start" in call_ep and stop_idx is not None:
                start_idx = i
                break

        assert stop_idx is not None, "Should have called stop when already STREAMING"
        assert start_idx is not None and start_idx > stop_idx, \
            "Start should come after stop when restarting"
        print("[PASS] test_already_streaming_stops_first")

    def test_resolution_and_fov_params_sent(self):
        """Verify resolution and FOV are sent as query params to webcam/start."""
        gopro = make_connected_gopro()

        start_endpoints = []

        def mock_api_get(endpoint, timeout=5.0):
            if "/gopro/webcam/start" in endpoint:
                start_endpoints.append(endpoint)
            if "/gopro/webcam/status" in endpoint:
                return {"error": 0, "status": WebcamStatus.STREAMING}
            return {"error": 0}

        with patch.object(gopro, '_api_get', side_effect=mock_api_get):
            gopro.start_webcam(resolution=12, fov=0)

        assert len(start_endpoints) > 0, "Should have sent at least one start command"
        # The last start call should have res=12&fov=0
        last_start = start_endpoints[-1]
        assert "res=12" in last_start, f"Expected res=12 in '{last_start}'"
        assert "fov=0" in last_start, f"Expected fov=0 in '{last_start}'"
        print("[PASS] test_resolution_and_fov_params_sent")

    def test_uses_config_defaults_when_no_params(self):
        """When no resolution/fov passed, should use config defaults."""
        gopro = make_connected_gopro(resolution=7, fov=2)

        start_endpoints = []

        def mock_api_get(endpoint, timeout=5.0):
            if "/gopro/webcam/start" in endpoint:
                start_endpoints.append(endpoint)
            if "/gopro/webcam/status" in endpoint:
                return {"error": 0, "status": WebcamStatus.STREAMING}
            return {"error": 0}

        with patch.object(gopro, '_api_get', side_effect=mock_api_get):
            gopro.start_webcam()  # No args — should use config

        last_start = start_endpoints[-1]
        assert "res=7" in last_start, f"Expected res=7 (from config) in '{last_start}'"
        assert "fov=2" in last_start, f"Expected fov=2 (from config) in '{last_start}'"
        print("[PASS] test_uses_config_defaults_when_no_params")

    def test_returns_false_when_camera_unreachable(self):
        """start_webcam returns False when the camera doesn't respond at all."""
        gopro = make_connected_gopro()

        def mock_api_get(endpoint, timeout=5.0):
            return None  # All calls fail

        with patch.object(gopro, '_api_get', side_effect=mock_api_get):
            result = gopro.start_webcam()

        assert result is False, "Should return False when camera is unreachable"
        print("[PASS] test_returns_false_when_camera_unreachable")

    def test_start_error_code_triggers_retry(self):
        """If webcam/start returns an error code, should try IDLE workaround."""
        gopro = make_connected_gopro()

        call_count = [0]

        def mock_api_get(endpoint, timeout=5.0):
            call_count[0] += 1
            if "/gopro/webcam/status" in endpoint:
                if call_count[0] <= 2:
                    return {"error": 0, "status": WebcamStatus.OFF}
                return {"error": 0, "status": WebcamStatus.STREAMING}
            if "/gopro/webcam/start" in endpoint:
                # First start fails, retries succeed
                if call_count[0] <= 3:
                    return {"error": 1}
                return {"error": 0}
            if "/gopro/webcam/stop" in endpoint:
                return {"error": 0}
            return {"error": 0}

        with patch.object(gopro, '_api_get', side_effect=mock_api_get):
            result = gopro.start_webcam()

        # Should have attempted recovery via IDLE workaround
        assert call_count[0] > 3, "Should have made multiple API calls during recovery"
        print("[PASS] test_start_error_code_triggers_retry")

    def test_status_callback_fires_during_init(self):
        """Verify the on_status_change callback gets notifications during startup."""
        gopro = make_connected_gopro()
        status_messages = []
        gopro.on_status_change = lambda msg, lvl: status_messages.append((msg, lvl))

        def mock_api_get(endpoint, timeout=5.0):
            if "/gopro/webcam/status" in endpoint:
                return {"error": 0, "status": WebcamStatus.STREAMING}
            return {"error": 0}

        with patch.object(gopro, '_api_get', side_effect=mock_api_get):
            gopro.start_webcam()

        assert len(status_messages) > 0, "Should have received status notifications"
        # Should have a success message
        levels = [m[1] for m in status_messages]
        assert "success" in levels, f"Expected a 'success' notification, got levels: {levels}"
        print("[PASS] test_status_callback_fires_during_init")

    def test_ready_state_accepted_after_timeout(self):
        """READY state should be accepted if STREAMING doesn't arrive quickly."""
        gopro = make_connected_gopro()

        def mock_api_get(endpoint, timeout=5.0):
            if "/gopro/webcam/status" in endpoint:
                # Camera stays READY (stream starts when consumer reads UDP)
                return {"error": 0, "status": WebcamStatus.READY}
            return {"error": 0}

        with patch.object(gopro, '_api_get', side_effect=mock_api_get):
            result = gopro.start_webcam()

        assert result is True, "READY should be accepted as success"
        print("[PASS] test_ready_state_accepted_after_timeout")

    def test_unavailable_returns_false_immediately(self):
        """When camera reports UNAVAILABLE, start_webcam should bail immediately."""
        gopro = make_connected_gopro()

        api_calls = []

        def mock_api_get(endpoint, timeout=5.0):
            api_calls.append(endpoint)
            if "/gopro/webcam/status" in endpoint:
                return {"error": 0, "status": WebcamStatus.UNAVAILABLE}
            return {"error": 0}

        with patch.object(gopro, '_api_get', side_effect=mock_api_get):
            result = gopro.start_webcam()

        assert result is False, "Should return False when camera is UNAVAILABLE"
        # Should NOT have sent webcam/start
        start_calls = [c for c in api_calls if "/gopro/webcam/start" in c]
        assert len(start_calls) == 0, (
            f"Should not send start when UNAVAILABLE, but got: {start_calls}"
        )
        print("[PASS] test_unavailable_returns_false_immediately")

    def test_unavailable_fires_error_notification(self):
        """UNAVAILABLE status should fire an error notification."""
        gopro = make_connected_gopro()
        messages = []
        gopro.on_status_change = lambda msg, lvl: messages.append((msg, lvl))

        def mock_api_get(endpoint, timeout=5.0):
            if "/gopro/webcam/status" in endpoint:
                return {"error": 0, "status": WebcamStatus.UNAVAILABLE}
            return {"error": 0}

        with patch.object(gopro, '_api_get', side_effect=mock_api_get):
            gopro.start_webcam()

        levels = [m[1] for m in messages]
        assert "error" in levels, f"Expected 'error' notification, got: {levels}"
        # Check that the message mentions unavailable
        error_msgs = [m[0] for m in messages if m[1] == "error"]
        assert any("unavailable" in m.lower() for m in error_msgs), (
            f"Expected 'unavailable' in error messages: {error_msgs}"
        )
        print("[PASS] test_unavailable_fires_error_notification")


class TestFullCommandSequenceIntegration:
    """End-to-end integration test for the full GoPro command sequence.

    Validates that open_connection() followed by start_webcam() sends the
    complete command sequence in the correct order without any user interaction:
      1. Verify HTTP API (/gopro/webcam/status)
      2. Enable wired USB control (/gopro/camera/control/wired_usb?p=1)
      3. IDLE workaround if needed (start → stop cycle)
      4. Start webcam (/gopro/webcam/start?res=...&fov=...)
      5. Poll until READY or STREAMING
      6. Keep-alive thread running
    """

    def test_full_sequence_from_discovery_to_streaming(self):
        """Complete automated sequence: open_connection → start_webcam."""
        from gopro_connection import ConnectionState, GOPRO_API_PORT
        from discovery import GoProDevice

        config = make_test_config()
        gopro = GoProConnection(config)
        device = GoProDevice(
            vendor_id=0x0A70,
            product_id=0x000D,
            description="GoPro Hero 12",
            camera_ip="172.20.145.51",
        )

        api_log = []

        def tracking_api_get(endpoint, timeout=5.0):
            api_log.append(endpoint)
            if "/gopro/webcam/status" in endpoint:
                # First few calls return OFF (during open_connection IDLE check),
                # then STREAMING (after start_webcam)
                if any("/gopro/webcam/start?res=" in c for c in api_log):
                    return {"error": 0, "status": WebcamStatus.STREAMING}
                return {"error": 0, "status": WebcamStatus.OFF}
            return {"error": 0}

        with patch.object(gopro, '_api_get', side_effect=tracking_api_get):
            # Step 1: Open connection (API verify → USB control → IDLE workaround → keep-alive)
            conn_result = gopro.open_connection(device)
            assert conn_result is True, "open_connection should succeed"
            assert gopro.state == ConnectionState.CONNECTED

            # Step 2: Start webcam mode
            webcam_result = gopro.start_webcam(resolution=4, fov=4)
            assert webcam_result is True, "start_webcam should succeed"
            assert gopro.state == ConnectionState.STREAMING

        # Verify the command sequence order
        # 1. API verification (webcam/status)
        assert any("/gopro/webcam/status" in c for c in api_log), \
            "Should verify API via webcam/status"

        # 2. Wired USB control enabled
        usb_control_calls = [c for c in api_log if "wired_usb?p=1" in c]
        assert len(usb_control_calls) >= 1, \
            f"Should enable wired USB control, got: {api_log}"

        # 3. USB control comes before webcam start
        usb_idx = next(i for i, c in enumerate(api_log) if "wired_usb?p=1" in c)
        start_calls = [i for i, c in enumerate(api_log) if "webcam/start?res=" in c]
        assert len(start_calls) >= 1, "Should have sent webcam/start with params"
        assert usb_idx < start_calls[0], \
            "USB control must come before webcam start"

        # 4. Start sent with correct params
        start_with_params = [c for c in api_log if "webcam/start?res=4&fov=4" in c]
        assert len(start_with_params) >= 1, \
            f"Expected start with res=4&fov=4, got starts: {[c for c in api_log if 'start' in c]}"

        # Clean up keep-alive
        gopro._stop_keepalive()
        print("[PASS] test_full_sequence_from_discovery_to_streaming")

    def test_full_sequence_with_idle_workaround(self):
        """Complete sequence where IDLE workaround is triggered during open_connection."""
        from gopro_connection import ConnectionState, GOPRO_API_PORT
        from discovery import GoProDevice

        config = make_test_config()
        gopro = GoProConnection(config)
        device = GoProDevice(
            vendor_id=0x0A70,
            product_id=0x000D,
            description="GoPro Hero 12",
            camera_ip="172.20.145.51",
        )

        api_log = []
        # Use a state machine to simulate realistic camera behavior
        camera_state = {"webcam": WebcamStatus.IDLE}

        def tracking_api_get(endpoint, timeout=5.0):
            api_log.append(endpoint)

            if "/gopro/webcam/status" in endpoint:
                return {"error": 0, "status": camera_state["webcam"]}

            if "/gopro/webcam/stop" in endpoint:
                camera_state["webcam"] = WebcamStatus.OFF
                return {"error": 0}

            if "/gopro/webcam/start" in endpoint:
                # If this is the real start (with params), go to STREAMING
                if "res=" in endpoint:
                    camera_state["webcam"] = WebcamStatus.STREAMING
                else:
                    # Workaround start — camera goes to READY briefly
                    camera_state["webcam"] = WebcamStatus.READY
                return {"error": 0}

            return {"error": 0}

        with patch.object(gopro, '_api_get', side_effect=tracking_api_get):
            conn_result = gopro.open_connection(device)
            assert conn_result is True

            webcam_result = gopro.start_webcam(resolution=4, fov=4)
            assert webcam_result is True
            assert gopro.state == ConnectionState.STREAMING

        # Verify IDLE workaround happened (start+stop before the real start)
        start_calls = [i for i, c in enumerate(api_log) if "/gopro/webcam/start" in c]
        stop_calls = [i for i, c in enumerate(api_log) if "/gopro/webcam/stop" in c]
        assert len(start_calls) >= 2, \
            f"Expected at least 2 starts (workaround + real), got {len(start_calls)}"
        assert len(stop_calls) >= 1, \
            f"Expected at least 1 stop (workaround), got {len(stop_calls)}"

        gopro._stop_keepalive()
        print("[PASS] test_full_sequence_with_idle_workaround")

    def test_no_user_interaction_required(self):
        """Verify the entire sequence completes without any user prompts or waits.

        The sequence must be fully automated — no input() calls, no manual
        mode-switch requirements, no GUI confirmations needed.
        """
        from gopro_connection import ConnectionState
        from discovery import GoProDevice

        config = make_test_config()
        gopro = GoProConnection(config)
        device = GoProDevice(
            vendor_id=0x0A70,
            product_id=0x000D,
            description="GoPro Hero 12",
            camera_ip="172.20.145.51",
        )

        # Track state transitions to verify no intermediate "waiting for user" states
        states = []
        gopro.on_connection_state = lambda s: states.append(s)

        def mock_api_get(endpoint, timeout=5.0):
            if "/gopro/webcam/status" in endpoint:
                return {"error": 0, "status": WebcamStatus.OFF}
            return {"error": 0}

        with patch.object(gopro, '_api_get', side_effect=mock_api_get):
            gopro.open_connection(device)

        # The sequence should go directly from CONNECTING → CONNECTED
        # with no manual steps needed
        assert ConnectionState.CONNECTING in states, "Should have entered CONNECTING"
        assert ConnectionState.CONNECTED in states, "Should have reached CONNECTED"

        gopro._stop_keepalive()
        print("[PASS] test_no_user_interaction_required")


class TestIdleWorkaround:
    """Tests specifically for reset_webcam_state() — the IDLE fix."""

    def test_workaround_sends_start_then_stop(self):
        """The IDLE workaround must send start, wait, stop, wait when not OFF."""
        gopro = make_connected_gopro()
        api_calls = []

        # First status check returns IDLE (triggers workaround),
        # final verification returns OFF
        status_iter = iter([
            {"error": 0, "status": WebcamStatus.IDLE},  # initial check -> IDLE
            {"error": 0, "status": WebcamStatus.OFF},    # after stop verification
        ])

        def mock_api_get(endpoint, timeout=5.0):
            api_calls.append(endpoint)
            if "/gopro/webcam/status" in endpoint:
                try:
                    return next(status_iter)
                except StopIteration:
                    return {"error": 0, "status": WebcamStatus.OFF}
            return {"error": 0}

        with patch.object(gopro, '_api_get', side_effect=mock_api_get):
            result = gopro.reset_webcam_state()

        assert result is True

        # Verify: start was called, then stop, in that order
        start_idx = next(i for i, c in enumerate(api_calls) if "start" in c)
        stop_idx = next(i for i, c in enumerate(api_calls) if "stop" in c)
        assert start_idx < stop_idx, "Start must come before stop in IDLE workaround"
        print("[PASS] test_workaround_sends_start_then_stop")

    def test_workaround_tolerates_start_error(self):
        """IDLE workaround should succeed even if the start command errors.

        During the workaround, the start is just to poke the state machine —
        an error response is fine as long as we can still send stop.
        """
        gopro = make_connected_gopro()

        # First status returns IDLE to trigger workaround, then OFF after
        status_iter = iter([
            {"error": 0, "status": WebcamStatus.IDLE},
            {"error": 0, "status": WebcamStatus.OFF},
        ])

        def mock_api_get(endpoint, timeout=5.0):
            if "/gopro/webcam/start" in endpoint:
                return None  # Start fails completely
            if "/gopro/webcam/stop" in endpoint:
                return {"error": 0}  # Stop succeeds
            if "/gopro/webcam/status" in endpoint:
                try:
                    return next(status_iter)
                except StopIteration:
                    return {"error": 0, "status": WebcamStatus.OFF}
            return {"error": 0}

        with patch.object(gopro, '_api_get', side_effect=mock_api_get):
            result = gopro.reset_webcam_state()

        # Should still succeed because stop worked
        assert result is True, "Workaround should tolerate start errors"
        print("[PASS] test_workaround_tolerates_start_error")

    def test_workaround_fails_when_camera_gone(self):
        """IDLE workaround should return False when camera is completely gone."""
        gopro = make_connected_gopro()

        def mock_api_get(endpoint, timeout=5.0):
            return None  # Everything fails

        with patch.object(gopro, '_api_get', side_effect=mock_api_get):
            result = gopro.reset_webcam_state()

        assert result is False, "Workaround should fail when camera is unreachable"
        print("[PASS] test_workaround_fails_when_camera_gone")


class TestKeepAlive:
    """Tests for the keep_alive ping."""

    def test_keep_alive_returns_true_on_success(self):
        """keep_alive should return True when the camera responds."""
        gopro = make_connected_gopro()

        with patch.object(gopro, '_api_get', return_value={"error": 0, "status": 3}):
            assert gopro.keep_alive() is True

        print("[PASS] test_keep_alive_returns_true_on_success")

    def test_keep_alive_returns_false_on_failure(self):
        """keep_alive should return False when the camera doesn't respond."""
        gopro = make_connected_gopro()

        with patch.object(gopro, '_api_get', return_value=None):
            assert gopro.keep_alive() is False

        print("[PASS] test_keep_alive_returns_false_on_failure")


class TestApiGet:
    """Tests for the _api_get HTTP helper."""

    def test_successful_request(self):
        """Successful API call should return parsed JSON and mark as connected."""
        gopro = make_connected_gopro()
        gopro._connected = False  # Start disconnected

        mock_response = MagicMock()
        mock_response.json.return_value = {"error": 0, "status": 3}
        mock_response.raise_for_status = MagicMock()

        with patch('requests.get', return_value=mock_response):
            result = gopro._api_get("/gopro/webcam/status")

        assert result == {"error": 0, "status": 3}
        assert gopro._connected is True, "Should mark as connected after success"
        print("[PASS] test_successful_request")

    def test_connection_error_marks_disconnected(self):
        """ConnectionError should set _connected = False."""
        gopro = make_connected_gopro()

        import requests as req
        with patch('requests.get', side_effect=req.ConnectionError("gone")):
            result = gopro._api_get("/gopro/webcam/status")

        assert result is None
        assert gopro._connected is False
        print("[PASS] test_connection_error_marks_disconnected")

    def test_timeout_does_not_mark_disconnected(self):
        """Timeout should NOT mark as disconnected (could be transient)."""
        gopro = make_connected_gopro()
        gopro._connected = True

        import requests as req
        with patch('requests.get', side_effect=req.Timeout("slow")):
            result = gopro._api_get("/gopro/webcam/status")

        assert result is None
        assert gopro._connected is True, "Timeout should not mark as disconnected"
        print("[PASS] test_timeout_does_not_mark_disconnected")


def run_all_tests():
    """Run all test classes and methods."""
    test_classes = [
        TestWebcamInitSequence(),
        TestFullCommandSequenceIntegration(),
        TestIdleWorkaround(),
        TestKeepAlive(),
        TestApiGet(),
    ]

    passed = 0
    failed = 0

    for suite in test_classes:
        # Run every method that starts with 'test_'
        for name in sorted(dir(suite)):
            if not name.startswith("test_"):
                continue
            method = getattr(suite, name)
            try:
                method()
                passed += 1
            except Exception as e:
                print(f"[FAIL] {suite.__class__.__name__}.{name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    print(f"\n{'=' * 50}")
    print(f"Webcam Init Tests: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
