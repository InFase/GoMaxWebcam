"""
test_stream_reinit.py -- Tests for stream re-initialization logic upon reconnection

Tests verify that:
  1. _create_and_swap_stream_reader() stops the old ffmpeg, creates a new one,
     waits for port release, verifies first frame, and swaps into the pipeline
  2. _wait_for_first_frame() correctly validates stream stabilization
  3. Port release delay is respected before creating new ffmpeg
  4. Recovery succeeds even when first-frame verification times out (non-fatal)
  5. Pipeline fallback: if pipeline is dead, full restart is attempted
  6. Disconnect detector is updated with the new reader reference
  7. _try_reconnect() integrates discover + connect + webcam + stream reinit
  8. Config parameters (stream_startup_timeout, ffmpeg_port_release_delay) are used

IMPORTANT: USBEventListener uses raw Win32 APIs that crash in pytest.
All tests that instantiate AppController MUST mock usb_event_listener.USBEventListener.
"""

import sys
import os
import time
import threading
from unittest.mock import MagicMock, patch, PropertyMock, call

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from app_controller import AppController, AppState

pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_config(**overrides) -> Config:
    """Create a config with fast timings for testing."""
    config = Config()
    config.stream_width = 320
    config.stream_height = 240
    config.stream_fps = 30
    config.udp_port = 8554
    config.ffmpeg_path = "ffmpeg"
    config.keepalive_interval = 0.5
    config.discovery_timeout = 0.5
    config.discovery_retry_interval = 0.1
    config.discovery_max_retries = 2
    config.reconnect_delay = 0.1
    config.ncm_adapter_wait = 0.1
    config.idle_reset_delay = 0.1
    config.stream_startup_timeout = 1.0  # Short timeout for tests
    config.ffmpeg_port_release_delay = 0.0  # No delay for tests
    config._config_path = ""
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def make_test_frame(width=320, height=240, color=(128, 64, 32)):
    """Create a test BGR24 frame as numpy array."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _make_controller(**config_overrides):
    """Create an AppController with test config, mocking USBEventListener."""
    config = make_test_config(**config_overrides)
    with patch(
        'usb_event_listener.USBEventListener',
        **{'return_value.start.return_value': False,
           'return_value.stop.return_value': None,
           'return_value.is_running': False},
    ):
        controller = AppController(config)
    return controller


# ---------------------------------------------------------------------------
# Tests: _create_and_swap_stream_reader
# ---------------------------------------------------------------------------

class TestCreateAndSwapStreamReader:
    """Tests for the core stream re-initialization method."""

    def test_stops_old_reader_before_creating_new(self):
        """Old stream reader should be stopped before new one is created."""
        ctrl = _make_controller()

        old_reader = MagicMock()
        ctrl._stream_reader = old_reader

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        new_reader = MagicMock()
        new_reader.start.return_value = True
        new_reader._process = MagicMock()
        new_reader._process.pid = 99999
        new_reader.is_running = True
        new_reader.read_frame.return_value = make_test_frame()

        with patch('stream_reader.StreamReader', return_value=new_reader):
            result = ctrl._create_and_swap_stream_reader()

        assert result is True
        old_reader.stop.assert_called_once()

    def test_new_reader_started_and_swapped(self):
        """New StreamReader should be created, started, and swapped into pipeline."""
        ctrl = _make_controller()

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        new_reader = MagicMock()
        new_reader.start.return_value = True
        new_reader._process = MagicMock()
        new_reader._process.pid = 11111
        new_reader.is_running = True
        new_reader.read_frame.return_value = make_test_frame()

        with patch('stream_reader.StreamReader', return_value=new_reader):
            result = ctrl._create_and_swap_stream_reader()

        assert result is True
        new_reader.start.assert_called_once()
        mock_pipeline.swap_reader.assert_called_once_with(new_reader)

    def test_returns_false_when_new_reader_fails_to_start(self):
        """Should return False if the new StreamReader fails to start."""
        ctrl = _make_controller()

        new_reader = MagicMock()
        new_reader.start.return_value = False

        with patch('stream_reader.StreamReader', return_value=new_reader):
            result = ctrl._create_and_swap_stream_reader()

        assert result is False

    def test_updates_disconnect_detector(self):
        """Disconnect detector should be updated with new reader reference."""
        ctrl = _make_controller()

        mock_detector = MagicMock()
        ctrl._disconnect_detector = mock_detector

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        new_reader = MagicMock()
        new_reader.start.return_value = True
        new_reader._process = MagicMock()
        new_reader._process.pid = 22222
        new_reader.is_running = True
        new_reader.read_frame.return_value = make_test_frame()

        with patch('stream_reader.StreamReader', return_value=new_reader):
            ctrl._create_and_swap_stream_reader()

        mock_detector.update_stream_reader.assert_called_once_with(new_reader)
        mock_detector.mark_connected.assert_called_once()

    def test_succeeds_even_when_first_frame_times_out(self):
        """Should still swap reader even if first-frame verification times out."""
        ctrl = _make_controller(stream_startup_timeout=0.1)

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        new_reader = MagicMock()
        new_reader.start.return_value = True
        new_reader._process = MagicMock()
        new_reader._process.pid = 33333
        new_reader.is_running = True
        # read_frame always returns None (no frames arriving yet)
        new_reader.read_frame.return_value = None

        with patch('stream_reader.StreamReader', return_value=new_reader):
            result = ctrl._create_and_swap_stream_reader()

        # Should still succeed -- timeout is non-fatal
        assert result is True
        mock_pipeline.swap_reader.assert_called_once()

    def test_pipeline_dead_triggers_full_restart(self):
        """If pipeline is not running, should attempt full pipeline restart."""
        ctrl = _make_controller(stream_startup_timeout=0.1)

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = False
        ctrl._frame_pipeline = mock_pipeline

        new_reader = MagicMock()
        new_reader.start.return_value = True
        new_reader._process = MagicMock()
        new_reader._process.pid = 44444
        new_reader.is_running = True
        new_reader.read_frame.return_value = None

        with patch('stream_reader.StreamReader', return_value=new_reader), \
             patch.object(ctrl, '_start_streaming_pipeline', return_value=True) as mock_restart:
            result = ctrl._create_and_swap_stream_reader()

        assert result is True
        mock_restart.assert_called_once()

    def test_port_release_delay_respected(self):
        """Port release delay should cause a wait between old stop and new start."""
        ctrl = _make_controller(ffmpeg_port_release_delay=0.2)

        old_reader = MagicMock()
        ctrl._stream_reader = old_reader

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        new_reader = MagicMock()
        new_reader.start.return_value = True
        new_reader._process = MagicMock()
        new_reader._process.pid = 55555
        new_reader.is_running = True
        new_reader.read_frame.return_value = make_test_frame()

        start_time = time.monotonic()
        with patch('stream_reader.StreamReader', return_value=new_reader):
            ctrl._create_and_swap_stream_reader()
        elapsed = time.monotonic() - start_time

        # Should have waited at least the port release delay
        assert elapsed >= 0.15  # Allow some tolerance

    def test_stop_event_during_port_release_aborts(self):
        """If stop event fires during port release wait, should abort."""
        ctrl = _make_controller(ffmpeg_port_release_delay=5.0)

        old_reader = MagicMock()
        ctrl._stream_reader = old_reader

        # Set stop event immediately
        ctrl._stop_event.set()

        new_reader = MagicMock()
        with patch('stream_reader.StreamReader', return_value=new_reader):
            result = ctrl._create_and_swap_stream_reader()

        assert result is False
        new_reader.start.assert_not_called()

    def test_old_reader_stop_exception_handled(self):
        """Exception from stopping old reader should be caught gracefully."""
        ctrl = _make_controller()

        old_reader = MagicMock()
        old_reader.stop.side_effect = RuntimeError("Already dead")
        ctrl._stream_reader = old_reader

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        new_reader = MagicMock()
        new_reader.start.return_value = True
        new_reader._process = MagicMock()
        new_reader._process.pid = 66666
        new_reader.is_running = True
        new_reader.read_frame.return_value = make_test_frame()

        with patch('stream_reader.StreamReader', return_value=new_reader):
            result = ctrl._create_and_swap_stream_reader()

        # Should succeed despite old reader throwing
        assert result is True


# ---------------------------------------------------------------------------
# Tests: _wait_for_first_frame
# ---------------------------------------------------------------------------

class TestWaitForFirstFrame:
    """Tests for the stream stabilization verification."""

    def test_returns_true_when_frame_received(self):
        """Should return True when a frame is received within timeout."""
        ctrl = _make_controller(stream_startup_timeout=2.0)

        reader = MagicMock()
        reader.is_running = True
        reader.read_frame.return_value = make_test_frame()

        result = ctrl._wait_for_first_frame(reader)
        assert result is True

    def test_returns_false_on_timeout(self):
        """Should return False when no frame is received within timeout."""
        ctrl = _make_controller(stream_startup_timeout=0.3)

        reader = MagicMock()
        reader.is_running = True
        reader.read_frame.return_value = None

        start = time.monotonic()
        result = ctrl._wait_for_first_frame(reader)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed >= 0.25  # Should have waited near the timeout

    def test_returns_false_when_reader_dies(self):
        """Should return False immediately if reader dies during wait."""
        ctrl = _make_controller(stream_startup_timeout=5.0)

        reader = MagicMock()
        reader.is_running = False  # Already dead
        reader.exit_code = 1
        reader.last_error = "Connection refused"

        result = ctrl._wait_for_first_frame(reader)
        assert result is False

    def test_returns_true_after_some_none_frames(self):
        """Should succeed even if first few reads return None."""
        ctrl = _make_controller(stream_startup_timeout=5.0)

        frame = make_test_frame()
        reader = MagicMock()
        reader.is_running = True
        # Return None twice, then a frame
        reader.read_frame.side_effect = [None, None, frame]

        result = ctrl._wait_for_first_frame(reader)
        assert result is True

    def test_stop_event_aborts_wait(self):
        """Should return False if stop event fires during wait."""
        ctrl = _make_controller(stream_startup_timeout=10.0)
        ctrl._stop_event.set()

        reader = MagicMock()
        reader.is_running = True
        reader.read_frame.return_value = None

        result = ctrl._wait_for_first_frame(reader)
        assert result is False


# ---------------------------------------------------------------------------
# Tests: Full _try_reconnect flow
# ---------------------------------------------------------------------------

class TestTryReconnect:
    """Tests for the complete reconnection flow: discover -> connect -> stream."""

    def test_successful_reconnection_flow(self):
        """Full reconnect: discover -> connect -> webcam -> swap reader."""
        ctrl = _make_controller()

        # Mock gopro connection
        ctrl.gopro = MagicMock()
        ctrl.gopro.discover.return_value = True
        ctrl.gopro.device_info = MagicMock()
        ctrl.gopro.open_connection.return_value = True
        ctrl.gopro.start_webcam.return_value = True

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        # Mock new stream reader
        new_reader = MagicMock()
        new_reader.start.return_value = True
        new_reader._process = MagicMock()
        new_reader._process.pid = 77777
        new_reader.is_running = True
        new_reader.read_frame.return_value = make_test_frame()

        with patch('stream_reader.StreamReader', return_value=new_reader):
            result = ctrl._try_reconnect()

        assert result is True
        ctrl.gopro.discover.assert_called_once()
        ctrl.gopro.open_connection.assert_called_once()
        ctrl.gopro.start_webcam.assert_called_once()
        mock_pipeline.swap_reader.assert_called_once()

    def test_reconnect_fails_on_discover(self):
        """Reconnect should fail if discovery fails."""
        ctrl = _make_controller()
        ctrl.gopro = MagicMock()
        ctrl.gopro.discover.return_value = False

        result = ctrl._try_reconnect()
        assert result is False

    def test_reconnect_fails_on_open_connection(self):
        """Reconnect should fail if control connection fails."""
        ctrl = _make_controller()
        ctrl.gopro = MagicMock()
        ctrl.gopro.discover.return_value = True
        ctrl.gopro.device_info = MagicMock()
        ctrl.gopro.open_connection.return_value = False

        result = ctrl._try_reconnect()
        assert result is False

    def test_reconnect_fails_on_webcam_start(self):
        """Reconnect should fail if webcam mode fails to start."""
        ctrl = _make_controller()
        ctrl.gopro = MagicMock()
        ctrl.gopro.discover.return_value = True
        ctrl.gopro.device_info = MagicMock()
        ctrl.gopro.open_connection.return_value = True
        ctrl.gopro.start_webcam.return_value = False

        result = ctrl._try_reconnect()
        assert result is False


# ---------------------------------------------------------------------------
# Tests: Config parameters
# ---------------------------------------------------------------------------

class TestStreamReinitConfig:
    """Tests for stream re-initialization config parameters."""

    def test_stream_startup_timeout_default(self):
        """Default stream_startup_timeout should be 10 seconds."""
        config = Config()
        assert config.stream_startup_timeout == 10.0

    def test_ffmpeg_port_release_delay_default(self):
        """Default ffmpeg_port_release_delay should be 0.5 seconds."""
        config = Config()
        assert config.ffmpeg_port_release_delay == 0.5

    def test_config_overrides_work(self):
        """Config overrides should be respected."""
        config = make_test_config(
            stream_startup_timeout=5.0,
            ffmpeg_port_release_delay=1.0,
        )
        assert config.stream_startup_timeout == 5.0
        assert config.ffmpeg_port_release_delay == 1.0


# ---------------------------------------------------------------------------
# Tests: Recovery loop integration
# ---------------------------------------------------------------------------

class TestRecoveryLoopIntegration:
    """Tests for the recovery loop's integration with stream re-init."""

    def test_auto_recover_enters_freeze_frame(self):
        """_auto_recover should enter freeze-frame on the pipeline."""
        ctrl = _make_controller()
        ctrl._running = True

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        # Prevent the recovery loop from actually running (it would block)
        with patch.object(ctrl, '_recovery_loop'):
            ctrl._auto_recover()

        mock_pipeline.enter_freeze_frame.assert_called_once()
        assert ctrl.state == AppState.RECONNECTING

    def test_auto_recover_stops_old_stream_reader(self):
        """_auto_recover should stop the old stream reader."""
        ctrl = _make_controller()
        ctrl._running = True

        old_reader = MagicMock()
        ctrl._stream_reader = old_reader

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        with patch.object(ctrl, '_recovery_loop'):
            ctrl._auto_recover()

        old_reader.stop.assert_called_once()

    def test_auto_recover_calls_reset_for_recovery(self):
        """_auto_recover should use gopro.reset_for_recovery() instead of private fields."""
        ctrl = _make_controller()
        ctrl._running = True

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        with patch.object(ctrl, '_recovery_loop'), \
             patch.object(ctrl.gopro, 'reset_for_recovery') as mock_reset:
            ctrl._auto_recover()

        mock_reset.assert_called_once()

    def test_recovery_loop_success_restores_streaming(self):
        """Successful recovery should set state back to STREAMING."""
        ctrl = _make_controller()
        ctrl._running = True

        # Mock _try_reconnect to succeed immediately
        with patch.object(ctrl, '_try_reconnect', return_value=True), \
             patch.object(ctrl, '_start_keepalive'), \
             patch.object(ctrl, '_fetch_camera_info'):
            ctrl._is_recovering = True
            ctrl._recovery_loop()

        assert ctrl.state == AppState.STREAMING
        assert ctrl._recovery_count == 1

    def test_recovery_increments_count(self):
        """Each successful recovery should increment recovery_count."""
        ctrl = _make_controller()
        ctrl._running = True
        ctrl._recovery_count = 3  # Already had 3 recoveries

        with patch.object(ctrl, '_try_reconnect', return_value=True), \
             patch.object(ctrl, '_start_keepalive'), \
             patch.object(ctrl, '_fetch_camera_info'):
            ctrl._is_recovering = True
            ctrl._recovery_loop()

        assert ctrl._recovery_count == 4

    def test_recovery_skips_if_already_recovering(self):
        """_auto_recover should skip if already recovering."""
        ctrl = _make_controller()
        ctrl._running = True
        ctrl._is_recovering = True

        # Mock _recovery_loop on threading.Thread to prevent it from running
        with patch('threading.Thread') as mock_thread:
            ctrl._auto_recover()

        # Thread should NOT be created since already recovering
        mock_thread.assert_not_called()

    def test_usb_detected_uses_polling_not_blind_sleep(self):
        """When USB reconnection is event-driven, recovery should poll
        wait_for_network_interface instead of doing a blind sleep."""
        ctrl = _make_controller()
        ctrl._running = True

        # Pre-set the USB reconnect event so the recovery loop sees it immediately
        ctrl._usb_reconnect_event.set()

        # Mock wait_for_network_interface to return an IP (network ready)
        ctrl.gopro.wait_for_network_interface = MagicMock(return_value="172.27.187.52")

        # Mock _try_reconnect to succeed
        with patch.object(ctrl, '_try_reconnect', return_value=True), \
             patch.object(ctrl, '_start_keepalive'), \
             patch.object(ctrl, '_fetch_camera_info'):
            ctrl._is_recovering = True
            ctrl._recovery_loop()

        # Verify wait_for_network_interface was called with the ncm_adapter_wait timeout
        ctrl.gopro.wait_for_network_interface.assert_called_once_with(
            timeout=ctrl.config.ncm_adapter_wait
        )
        assert ctrl.state == AppState.STREAMING

    def test_usb_detected_proceeds_even_if_network_poll_times_out(self):
        """If wait_for_network_interface returns None (timeout), recovery
        should still attempt _try_reconnect (discovery handles its own polling)."""
        ctrl = _make_controller()
        ctrl._running = True

        # Pre-set the USB reconnect event
        ctrl._usb_reconnect_event.set()

        # Mock wait_for_network_interface to return None (timed out)
        ctrl.gopro.wait_for_network_interface = MagicMock(return_value=None)

        # Mock _try_reconnect to succeed anyway (discovery has its own retry)
        with patch.object(ctrl, '_try_reconnect', return_value=True), \
             patch.object(ctrl, '_start_keepalive'), \
             patch.object(ctrl, '_fetch_camera_info'):
            ctrl._is_recovering = True
            ctrl._recovery_loop()

        # Still called wait_for_network_interface (polling, not blind sleep)
        ctrl.gopro.wait_for_network_interface.assert_called_once()
        assert ctrl.state == AppState.STREAMING

    def test_exponential_backoff_doubles_delay_up_to_max(self):
        """Poll-based reconnects should use exponential backoff: 2s→4s→8s→max 30s.

        When _try_reconnect fails repeatedly, the wait timeout between attempts
        should double each time, capped at reconnect_max_delay.
        """
        ctrl = _make_controller(
            reconnect_delay=2.0,
            reconnect_max_delay=30.0,
            reconnect_max_retries=6,  # Stop after 6 attempts
        )
        ctrl._running = True
        ctrl._is_recovering = True

        # Track the timeout values passed to _usb_reconnect_event.wait()
        wait_timeouts = []
        original_wait = ctrl._usb_reconnect_event.wait

        def tracking_wait(timeout=None):
            wait_timeouts.append(timeout)
            return False  # Simulate no USB event (poll-based fallback)

        ctrl._usb_reconnect_event.wait = tracking_wait

        # _try_reconnect always fails so we exercise all backoff steps
        with patch.object(ctrl, '_try_reconnect', return_value=False):
            ctrl._recovery_loop()

        # Should have reached max retries and entered ERROR state
        assert ctrl.state == AppState.ERROR

        # Verify exponential backoff sequence: 2, 4, 8, 16, 30, 30
        # First attempt uses initial delay, then doubles, capped at max
        assert wait_timeouts == [2.0, 4.0, 8.0, 16.0, 30.0, 30.0]

    def test_exponential_backoff_resets_on_new_recovery(self):
        """Each new recovery loop should start with the base delay, not the
        previous backoff value."""
        ctrl = _make_controller(
            reconnect_delay=2.0,
            reconnect_max_delay=30.0,
        )
        ctrl._running = True

        wait_timeouts = []

        def tracking_wait(timeout=None):
            wait_timeouts.append(timeout)
            return False  # Poll-based fallback

        # Run recovery loop twice — first fails twice then succeeds,
        # second should start fresh at base delay
        for run in range(2):
            ctrl._is_recovering = True
            wait_timeouts.clear()

            attempt_count = [0]
            def try_reconnect_side_effect():
                attempt_count[0] += 1
                return attempt_count[0] >= 3  # Succeed on 3rd attempt

            ctrl._usb_reconnect_event.wait = tracking_wait
            attempt_count[0] = 0

            with patch.object(ctrl, '_try_reconnect', side_effect=try_reconnect_side_effect), \
                 patch.object(ctrl, '_start_keepalive'), \
                 patch.object(ctrl, '_fetch_camera_info'):
                ctrl._recovery_loop()

            # Each run: attempts at delay 2.0, then 4.0, then succeeds at 8.0
            assert wait_timeouts == [2.0, 4.0, 8.0], \
                f"Run {run+1}: expected [2.0, 4.0, 8.0], got {wait_timeouts}"

    def test_usb_event_driven_reconnect_skips_backoff_wait(self):
        """When USB reconnect event fires, the wait returns immediately
        without waiting for the full backoff delay."""
        ctrl = _make_controller(
            reconnect_delay=2.0,
            reconnect_max_delay=30.0,
        )
        ctrl._running = True
        ctrl._is_recovering = True

        # Pre-set USB event so wait returns True immediately
        ctrl._usb_reconnect_event.set()

        # Mock wait_for_network_interface (required for USB-detected path)
        ctrl.gopro.wait_for_network_interface = MagicMock(return_value="172.27.187.52")

        start = time.monotonic()
        with patch.object(ctrl, '_try_reconnect', return_value=True), \
             patch.object(ctrl, '_start_keepalive'), \
             patch.object(ctrl, '_fetch_camera_info'):
            ctrl._recovery_loop()
        elapsed = time.monotonic() - start

        # Should complete nearly instantly (well under the 2s base delay)
        assert elapsed < 1.0
        assert ctrl.state == AppState.STREAMING


# ---------------------------------------------------------------------------
# Tests: Health monitor pause/resume during stream reader swap
# ---------------------------------------------------------------------------

class TestHealthMonitorPauseDuringSwap:
    """Tests that health monitor is paused for the entire _create_and_swap_stream_reader()."""

    def test_health_monitor_paused_and_resumed_on_success(self):
        """Health monitor should be paused before swap and resumed after success."""
        ctrl = _make_controller()

        mock_detector = MagicMock()
        ctrl._disconnect_detector = mock_detector

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        new_reader = MagicMock()
        new_reader.start.return_value = True
        new_reader._process = MagicMock()
        new_reader._process.pid = 88888
        new_reader.is_running = True
        new_reader.read_frame.return_value = make_test_frame()

        with patch('stream_reader.StreamReader', return_value=new_reader):
            result = ctrl._create_and_swap_stream_reader()

        assert result is True
        mock_detector.pause_health_monitor.assert_called_once()
        mock_detector.resume_health_monitor.assert_called_once()

    def test_health_monitor_resumed_on_failure(self):
        """Health monitor should be resumed even if new reader fails to start."""
        ctrl = _make_controller()

        mock_detector = MagicMock()
        ctrl._disconnect_detector = mock_detector

        new_reader = MagicMock()
        new_reader.start.return_value = False

        with patch('stream_reader.StreamReader', return_value=new_reader):
            result = ctrl._create_and_swap_stream_reader()

        assert result is False
        mock_detector.pause_health_monitor.assert_called_once()
        mock_detector.resume_health_monitor.assert_called_once()

    def test_health_monitor_resumed_on_exception(self):
        """Health monitor should be resumed even if an exception occurs during swap."""
        ctrl = _make_controller()

        mock_detector = MagicMock()
        ctrl._disconnect_detector = mock_detector

        # StreamReader constructor raises
        with patch('stream_reader.StreamReader', side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                ctrl._create_and_swap_stream_reader()

        mock_detector.pause_health_monitor.assert_called_once()
        mock_detector.resume_health_monitor.assert_called_once()

    def test_health_monitor_resumed_on_stop_event_abort(self):
        """Health monitor should be resumed even if stop event aborts the swap."""
        ctrl = _make_controller(ffmpeg_port_release_delay=5.0)

        mock_detector = MagicMock()
        ctrl._disconnect_detector = mock_detector

        old_reader = MagicMock()
        ctrl._stream_reader = old_reader

        # Set stop event to abort during port release wait
        ctrl._stop_event.set()

        new_reader = MagicMock()
        with patch('stream_reader.StreamReader', return_value=new_reader):
            result = ctrl._create_and_swap_stream_reader()

        assert result is False
        mock_detector.pause_health_monitor.assert_called_once()
        mock_detector.resume_health_monitor.assert_called_once()

    def test_pause_order_before_old_reader_stop(self):
        """Health monitor must be paused BEFORE the old reader is stopped."""
        ctrl = _make_controller()

        call_order = []

        mock_detector = MagicMock()
        mock_detector.pause_health_monitor.side_effect = lambda: call_order.append('pause')
        mock_detector.resume_health_monitor.side_effect = lambda: call_order.append('resume')
        ctrl._disconnect_detector = mock_detector

        old_reader = MagicMock()
        old_reader.stop.side_effect = lambda: call_order.append('old_stop')
        ctrl._stream_reader = old_reader

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        new_reader = MagicMock()
        new_reader.start.return_value = True
        new_reader._process = MagicMock()
        new_reader._process.pid = 99999
        new_reader.is_running = True
        new_reader.read_frame.return_value = make_test_frame()

        with patch('stream_reader.StreamReader', return_value=new_reader):
            ctrl._create_and_swap_stream_reader()

        # pause must come before old reader stop, resume must come last
        assert call_order[0] == 'pause'
        assert call_order[1] == 'old_stop'
        assert call_order[-1] == 'resume'

    def test_no_detector_skips_pause_resume(self):
        """When no disconnect detector is set, swap should still work without pause/resume."""
        ctrl = _make_controller()
        ctrl._disconnect_detector = None

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        new_reader = MagicMock()
        new_reader.start.return_value = True
        new_reader._process = MagicMock()
        new_reader._process.pid = 11111
        new_reader.is_running = True
        new_reader.read_frame.return_value = make_test_frame()

        with patch('stream_reader.StreamReader', return_value=new_reader):
            result = ctrl._create_and_swap_stream_reader()

        assert result is True


# ---------------------------------------------------------------------------
# Tests: Infinite retries (never give up)
# ---------------------------------------------------------------------------

class TestInfiniteRetries:
    """Tests that the recovery loop never gives up when reconnect_max_retries=0 (default)."""

    def test_infinite_retries_never_enters_error_state(self):
        """With reconnect_max_retries=0 (default), the recovery loop should keep
        retrying indefinitely and never enter ERROR state.

        We simulate 20 consecutive failures then a success to prove the loop
        doesn't cap out at any fixed number of attempts.
        """
        ctrl = _make_controller(
            reconnect_delay=0.01,
            reconnect_max_delay=0.02,
            reconnect_max_retries=0,  # 0 = infinite (the default)
        )
        ctrl._running = True
        ctrl._is_recovering = True

        attempt_count = [0]
        max_failures = 20  # Fail 20 times then succeed

        def try_reconnect_side_effect():
            attempt_count[0] += 1
            return attempt_count[0] > max_failures  # Succeed on attempt 21

        with patch.object(ctrl, '_try_reconnect', side_effect=try_reconnect_side_effect), \
             patch.object(ctrl, '_start_keepalive'), \
             patch.object(ctrl, '_fetch_camera_info'):
            ctrl._recovery_loop()

        # Should have succeeded, NOT entered ERROR
        assert ctrl.state == AppState.STREAMING
        assert attempt_count[0] == max_failures + 1  # 21 attempts total

    def test_infinite_retries_does_not_emit_max_retries_error(self):
        """With reconnect_max_retries=0, the 'Could not recover after N attempts'
        error message should never be emitted, regardless of how many failures occur."""
        ctrl = _make_controller(
            reconnect_delay=0.01,
            reconnect_max_delay=0.02,
            reconnect_max_retries=0,
        )
        ctrl._running = True
        ctrl._is_recovering = True

        attempt_count = [0]

        def try_reconnect_side_effect():
            attempt_count[0] += 1
            return attempt_count[0] >= 15  # Succeed on attempt 15

        status_messages = []

        def capture_status(msg, level="info"):
            status_messages.append((msg, level))

        ctrl._emit_status = capture_status

        with patch.object(ctrl, '_try_reconnect', side_effect=try_reconnect_side_effect), \
             patch.object(ctrl, '_start_keepalive'), \
             patch.object(ctrl, '_fetch_camera_info'):
            ctrl._recovery_loop()

        # No "Could not recover" error should appear
        error_msgs = [msg for msg, level in status_messages if "Could not recover" in msg]
        assert error_msgs == [], f"Unexpected error messages: {error_msgs}"
        assert attempt_count[0] == 15

    def test_finite_retries_enters_error_after_max(self):
        """With reconnect_max_retries > 0, the loop should give up after that many attempts."""
        ctrl = _make_controller(
            reconnect_delay=0.01,
            reconnect_max_delay=0.02,
            reconnect_max_retries=3,
        )
        ctrl._running = True
        ctrl._is_recovering = True

        with patch.object(ctrl, '_try_reconnect', return_value=False):
            ctrl._recovery_loop()

        assert ctrl.state == AppState.ERROR

    def test_infinite_retries_respects_stop_event(self):
        """Even with infinite retries, the loop must exit when stop_event is set."""
        ctrl = _make_controller(
            reconnect_delay=0.01,
            reconnect_max_delay=0.02,
            reconnect_max_retries=0,
        )
        ctrl._running = True
        ctrl._is_recovering = True

        attempt_count = [0]

        def try_reconnect_side_effect():
            attempt_count[0] += 1
            if attempt_count[0] >= 5:
                ctrl._stop_event.set()
            return False

        with patch.object(ctrl, '_try_reconnect', side_effect=try_reconnect_side_effect):
            ctrl._recovery_loop()

        # Should have stopped, not entered ERROR
        assert ctrl.state != AppState.ERROR
        assert attempt_count[0] >= 5

    def test_default_config_has_infinite_retries(self):
        """The default config should have reconnect_max_retries=0 (infinite)."""
        config = Config()
        assert config.reconnect_max_retries == 0
