"""
test_ffmpeg_crash_recovery.py — Tests for AC 6: ffmpeg crash triggers same
freeze-frame recovery as USB disconnect

Verifies that:
  1. ffmpeg process death triggers freeze-frame on the pipeline (same as USB disconnect)
  2. DisconnectDetector distinguishes ffmpeg crash from USB disconnect
  3. ffmpeg crash fires on_ffmpeg_crash callback (not on_disconnect)
  4. AppController._on_ffmpeg_crash triggers lightweight recovery
  5. Lightweight recovery restarts ffmpeg without full USB reconnect dance
  6. If GoPro API is unreachable after ffmpeg crash, falls back to full recovery
  7. Pipeline stays in freeze-frame during ffmpeg restart (no frame gaps)
  8. Virtual camera never disappears during ffmpeg crash recovery
  9. Multiple consecutive ffmpeg crashes are handled correctly
  10. Recovery is skipped if already in progress

IMPORTANT: USBEventListener uses raw Win32 APIs that crash in pytest.
All tests that touch DisconnectDetector.start() or AppController MUST
mock usb_event_listener.USBEventListener.
"""

import collections
import sys
import os
import time
import threading
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from disconnect_detector import DisconnectDetector
from frame_pipeline import FramePipeline, PipelineState
from frame_buffer import FrameBuffer
from stream_reader import StreamReader



from tests.test_utils import FrameLog

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
    config._config_path = ""
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def make_test_frame(width=320, height=240, color=(128, 64, 32)):
    """Create a test RGB24 frame as numpy array."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


class MockStreamReader:
    """Mock StreamReader that can simulate ffmpeg crash."""

    def __init__(self, frames=None, is_running=True, crash_after=None):
        self._frames = frames or []
        self._index = 0
        self._is_running = is_running
        self._crash_after = crash_after
        self.width = 320
        self.height = 240
        self.fps = 30
        self._read_count = 0

    @property
    def is_running(self):
        return self._is_running

    @is_running.setter
    def is_running(self, val):
        self._is_running = val

    def read_frame(self):
        self._read_count += 1
        if self._crash_after is not None and self._read_count > self._crash_after:
            self._is_running = False
            return None
        if not self._frames:
            if not self._is_running:
                return None
            return make_test_frame()
        frame = self._frames[self._index % len(self._frames)]
        self._index += 1
        return frame

    def start(self):
        self._is_running = True
        return True

    def stop(self):
        self._is_running = False


class MockVirtualCamera:
    """Mock VirtualCamera that records frames."""

    def __init__(self):
        self.frames_sent = FrameLog()
        self.frame_count = 0
        self._is_running = True
        self._last_frame = None
        self.width = 320
        self.height = 240
        self.fps = 30
        self._lock = threading.Lock()

    @property
    def is_running(self):
        return self._is_running

    def start(self):
        self._is_running = True
        return True

    def stop(self):
        self._is_running = False

    def send_frame(self, frame):
        with self._lock:
            self.frame_count += 1
            self.frames_sent.append(frame.copy())
            self._last_frame = frame
        return True

    def send_last_frame(self):
        with self._lock:
            if self._last_frame is not None:
                self.frame_count += 1
                self.frames_sent.append(self._last_frame.copy())
                return True
        return False

    def sleep_until_next_frame(self):
        time.sleep(0.001)


# Mock USBEventListener globally for all tests
_usb_listener_patch = patch(
    'usb_event_listener.USBEventListener',
    **{'return_value.start.return_value': False, 'return_value.stop.return_value': None,
       'return_value.is_running': False},
)


# ---------------------------------------------------------------------------
# Test 1: ffmpeg crash triggers freeze-frame on pipeline (same as USB disconnect)
# ---------------------------------------------------------------------------

class TestFFmpegCrashTriggersFreezeFrame:
    """Verify that ffmpeg process death triggers freeze-frame on the pipeline,
    identical to USB disconnect behavior."""

    def test_pipeline_enters_freeze_on_ffmpeg_crash(self):
        """When StreamReader stops producing frames (ffmpeg crash),
        pipeline enters freeze-frame mode."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        # Reader produces 3 frames then "crashes"
        live_frame = make_test_frame(color=(200, 100, 50))
        reader = MockStreamReader(
            frames=[live_frame, live_frame, live_frame],
            crash_after=3,
        )
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.5)

        # Pipeline should be in freeze-frame mode
        assert pipeline.state == PipelineState.FREEZE_FRAME
        assert pipeline.is_frozen is True

        # Virtual camera should still be running
        assert vcam.is_running is True

        pipeline.stop()

    def test_freeze_frames_pixel_identical_after_ffmpeg_crash(self):
        """Freeze frames after ffmpeg crash must be pixel-identical
        to the last live frame (same guarantee as USB disconnect)."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        last_live = make_test_frame(color=(42, 84, 168))
        reader = MockStreamReader(frames=[last_live], crash_after=1)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.5)
        pipeline.stop()

        # All frames after the first should match the last live frame
        assert vcam.frame_count > 1
        for i, frame in enumerate(list(vcam.frames_sent)[1:], start=1):
            np.testing.assert_array_equal(
                frame, last_live,
                err_msg=f"Freeze frame #{i} after ffmpeg crash does not match last live frame"
            )

    def test_no_black_frames_during_ffmpeg_crash(self):
        """No black frames should appear during ffmpeg crash → freeze transition."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        frames = [make_test_frame(color=(128, 128, 128)) for _ in range(3)]
        reader = MockStreamReader(frames=frames, crash_after=3)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.5)
        pipeline.stop()

        for i, frame in enumerate(vcam.frames_sent):
            is_black = np.all(frame == 0)
            assert not is_black, f"Frame #{i} is black during ffmpeg crash recovery"

    def test_vcam_never_disappears_during_ffmpeg_crash(self):
        """Virtual camera device must remain registered during ffmpeg crash."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        reader = MockStreamReader(
            frames=[make_test_frame() for _ in range(3)],
            crash_after=3,
        )
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.5)

        # Check at various points
        assert vcam.is_running is True
        assert pipeline.is_running  # Pipeline thread still alive in freeze mode

        pipeline.stop()


# ---------------------------------------------------------------------------
# Test 2: DisconnectDetector distinguishes ffmpeg crash from USB disconnect
# ---------------------------------------------------------------------------

class TestDetectorDistinguishesFFmpegCrash:
    """Verify DisconnectDetector fires on_ffmpeg_crash (not on_disconnect)
    when ffmpeg dies without USB detach."""

    @_usb_listener_patch
    def test_ffmpeg_crash_fires_on_ffmpeg_crash_callback(self, mock_usb):
        """When stream reader dies without prior USB detach, on_ffmpeg_crash fires."""
        pipeline = MagicMock()
        reader = MockStreamReader(is_running=True)

        ffmpeg_crash_events = []
        disconnect_events = []

        detector = DisconnectDetector(
            pipeline=pipeline,
            stream_reader=reader,
        )
        detector._health_check_interval = 0.05
        detector.on_ffmpeg_crash = lambda: ffmpeg_crash_events.append(True)
        detector.on_disconnect = lambda: disconnect_events.append(True)

        detector.start()
        time.sleep(0.1)

        # Simulate ffmpeg crash (no USB detach)
        reader._is_running = False
        time.sleep(0.3)

        detector.stop()

        assert len(ffmpeg_crash_events) >= 1, "on_ffmpeg_crash should have fired"
        assert len(disconnect_events) == 0, "on_disconnect should NOT fire for ffmpeg crash"

    @_usb_listener_patch
    def test_usb_disconnect_then_ffmpeg_death_fires_on_disconnect(self, mock_usb):
        """When USB detach precedes stream death, on_disconnect fires (not on_ffmpeg_crash)."""
        pipeline = MagicMock()
        reader = MockStreamReader(is_running=True)

        ffmpeg_crash_events = []
        disconnect_events = []

        detector = DisconnectDetector(
            pipeline=pipeline,
            stream_reader=reader,
        )
        detector._health_check_interval = 0.05
        detector.on_ffmpeg_crash = lambda: ffmpeg_crash_events.append(True)
        detector.on_disconnect = lambda: disconnect_events.append(True)

        detector.start()
        time.sleep(0.1)

        # USB detach first (sets _disconnected = True)
        detector._on_usb_detach("USB\\VID_2672&PID_0052\\test")
        # Then ffmpeg dies
        reader._is_running = False
        time.sleep(0.3)

        detector.stop()

        # on_disconnect should have fired (from USB detach and/or health monitor backup)
        assert len(disconnect_events) >= 1, "on_disconnect should fire for USB disconnect"
        # on_ffmpeg_crash should NOT fire (USB detach already set disconnected flag)
        assert len(ffmpeg_crash_events) == 0, "on_ffmpeg_crash should NOT fire when USB was disconnected"

    def test_ffmpeg_crash_triggers_freeze_same_as_usb_detach(self):
        """Both ffmpeg crash and USB detach call pipeline.enter_freeze_frame()."""
        pipeline = MagicMock()

        # Test USB detach path
        detector1 = DisconnectDetector(pipeline=pipeline)
        detector1._on_usb_detach("USB\\VID_2672&PID_0052\\test")
        pipeline.enter_freeze_frame.assert_called()

        # Reset
        pipeline.reset_mock()

        # Test ffmpeg crash path (via _trigger_freeze_frame directly)
        detector2 = DisconnectDetector(pipeline=pipeline)
        detector2._trigger_freeze_frame("ffmpeg process exited")
        pipeline.enter_freeze_frame.assert_called()


# ---------------------------------------------------------------------------
# Test 3: AppController._on_ffmpeg_crash triggers lightweight recovery
# ---------------------------------------------------------------------------

# Global mock for all AppController tests
_app_usb_patch = patch(
    'usb_event_listener.USBEventListener',
    **{'return_value.start.return_value': False, 'return_value.stop.return_value': None,
       'return_value.is_running': False},
)


class TestAppControllerFFmpegCrashRecovery:
    """Tests that AppController handles ffmpeg crash with fast recovery."""

    @_app_usb_patch
    @patch('app_controller.GoProConnection')
    def test_on_ffmpeg_crash_exists(self, mock_gopro_cls, mock_usb):
        """AppController should have _on_ffmpeg_crash method."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)
        assert hasattr(ctrl, '_on_ffmpeg_crash')
        assert callable(ctrl._on_ffmpeg_crash)

    @_app_usb_patch
    @patch('app_controller.GoProConnection')
    def test_on_ffmpeg_crash_skips_when_already_recovering(self, mock_gopro_cls, mock_usb):
        """_on_ffmpeg_crash should skip if recovery is already in progress."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)
        ctrl._is_recovering = True
        ctrl._recover_ffmpeg_crash = MagicMock()

        ctrl._on_ffmpeg_crash()

        # Should not start a new recovery thread
        ctrl._recover_ffmpeg_crash.assert_not_called()

    @_app_usb_patch
    @patch('app_controller.GoProConnection')
    def test_recover_ffmpeg_crash_fast_path_api_reachable(self, mock_gopro_cls, mock_usb):
        """When GoPro API is still reachable, recovery just restarts ffmpeg."""
        from app_controller import AppController, AppState
        config = make_test_config()
        ctrl = AppController(config)
        ctrl._running = True

        # Mock pipeline in freeze mode
        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        # Mock GoPro API as reachable
        ctrl.gopro.keep_alive = MagicMock(return_value=True)

        # Mock _create_and_swap_stream_reader
        ctrl._create_and_swap_stream_reader = MagicMock(return_value=True)

        # Mock _auto_recover to ensure it's NOT called
        ctrl._auto_recover = MagicMock()

        ctrl._recover_ffmpeg_crash()

        # Should have called pipeline.enter_freeze_frame()
        mock_pipeline.enter_freeze_frame.assert_called_once()

        # Should have restarted ffmpeg via create_and_swap
        ctrl._create_and_swap_stream_reader.assert_called_once()

        # Should NOT have fallen back to full recovery
        ctrl._auto_recover.assert_not_called()

        # Recovery count should be incremented
        assert ctrl._recovery_count == 1

    @_app_usb_patch
    @patch('app_controller.GoProConnection')
    def test_recover_ffmpeg_crash_falls_back_when_api_unreachable(self, mock_gopro_cls, mock_usb):
        """When GoPro API is not reachable, falls back to full _auto_recover."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)
        ctrl._running = True

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        # GoPro API NOT reachable
        ctrl.gopro.keep_alive = MagicMock(return_value=False)

        # Track auto_recover calls
        ctrl._auto_recover = MagicMock()

        ctrl._recover_ffmpeg_crash()

        # Should fall back to full recovery
        ctrl._auto_recover.assert_called_once()

    @_app_usb_patch
    @patch('app_controller.GoProConnection')
    def test_recover_ffmpeg_crash_falls_back_on_swap_failure(self, mock_gopro_cls, mock_usb):
        """If restarting ffmpeg fails, falls back to full recovery."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)
        ctrl._running = True

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        ctrl.gopro.keep_alive = MagicMock(return_value=True)
        ctrl._create_and_swap_stream_reader = MagicMock(return_value=False)
        ctrl._auto_recover = MagicMock()

        ctrl._recover_ffmpeg_crash()

        ctrl._auto_recover.assert_called_once()

    @_app_usb_patch
    @patch('app_controller.GoProConnection')
    def test_disconnect_detector_wired_with_ffmpeg_crash_callback(self, mock_gopro_cls, mock_usb):
        """_start_disconnect_detector should wire on_ffmpeg_crash callback."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)

        # Set up required components
        ctrl._frame_pipeline = MagicMock()
        ctrl._stream_reader = MagicMock()

        with patch('disconnect_detector.DisconnectDetector') as mock_dd_cls:
            mock_detector = MagicMock()
            mock_detector.is_running = False
            mock_detector.start.return_value = True
            mock_dd_cls.return_value = mock_detector

            ctrl._start_disconnect_detector()

            # Verify on_ffmpeg_crash was wired
            assert mock_detector.on_ffmpeg_crash == ctrl._on_ffmpeg_crash


# ---------------------------------------------------------------------------
# Test 4: Pipeline integration — ffmpeg crash recovery flow
# ---------------------------------------------------------------------------

class TestFFmpegCrashPipelineIntegration:
    """Integration tests: ffmpeg crash → freeze → recovery → live."""

    def test_pipeline_survives_ffmpeg_crash_and_reader_swap(self):
        """Pipeline survives: live frames → ffmpeg crash → freeze → new reader → live."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        # Phase 1: Live streaming then crash
        live_color = (200, 100, 50)
        reader1 = MockStreamReader(
            frames=[make_test_frame(color=live_color) for _ in range(5)],
            crash_after=5,
        )

        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.4)

        assert pipeline.state == PipelineState.FREEZE_FRAME
        frames_before_swap = vcam.frame_count

        # Phase 2: Swap in new reader (simulating ffmpeg restart)
        new_color = (0, 255, 0)
        reader2 = MockStreamReader(
            frames=[make_test_frame(color=new_color)],
        )
        pipeline.swap_reader(reader2)
        time.sleep(0.3)

        pipeline.stop()

        # Should have more frames after swap
        assert vcam.frame_count > frames_before_swap

        # New live frames should be present somewhere in the frame log
        new_frames = [
            f for f in vcam.frames_sent
            if tuple(f[0, 0]) == new_color
        ]
        assert len(new_frames) > 0, "No new live frames after ffmpeg restart"

    def test_multiple_ffmpeg_crashes_handled(self):
        """Pipeline survives multiple consecutive ffmpeg crash/recovery cycles."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        # Cycle 1: crash after 3 frames
        reader1 = MockStreamReader(
            frames=[make_test_frame(color=(200, 0, 0)) for _ in range(3)],
            crash_after=3,
        )
        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.3)
        assert pipeline.is_running

        # Cycle 2: swap and crash again
        reader2 = MockStreamReader(
            frames=[make_test_frame(color=(0, 200, 0)) for _ in range(3)],
            crash_after=3,
        )
        pipeline.swap_reader(reader2)
        time.sleep(0.3)
        assert pipeline.is_running

        # Cycle 3: swap with working reader
        reader3 = MockStreamReader(
            frames=[make_test_frame(color=(0, 0, 200))],
        )
        pipeline.swap_reader(reader3)
        time.sleep(0.2)

        assert pipeline.is_running
        assert vcam.is_running

        pipeline.stop()

        # Verify frames were pushed in all cycles
        assert pipeline.frames_pushed > 10

    def test_ffmpeg_crash_continuous_frame_emission(self):
        """Frames must be pushed continuously through crash and recovery."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        reader = MockStreamReader(
            frames=[make_test_frame(color=(100, 100, 100)) for _ in range(3)],
            crash_after=3,
        )

        pipeline.start(reader, vcam, frame_buffer=buffer)

        # Sample frame count over time
        samples = []
        for _ in range(8):
            time.sleep(0.1)
            with vcam._lock:
                samples.append(vcam.frame_count)

        pipeline.stop()

        # Frame count should be monotonically non-decreasing
        for i in range(1, len(samples)):
            assert samples[i] >= samples[i - 1], (
                f"Frame count decreased: {samples[i-1]} → {samples[i]}"
            )

        # Should have pushed frames throughout (no long gaps)
        assert samples[-1] > samples[0], "No frames pushed over time"


# ---------------------------------------------------------------------------
# Test 5: Health monitor detects ffmpeg crash correctly
# ---------------------------------------------------------------------------

class TestHealthMonitorFFmpegCrash:
    """Tests that the health monitor correctly identifies ffmpeg crashes."""

    @_usb_listener_patch
    def test_health_monitor_triggers_freeze_on_ffmpeg_crash(self, mock_usb):
        """Health monitor should trigger freeze-frame when ffmpeg crashes."""
        pipeline = MagicMock()
        reader = MockStreamReader(is_running=True)

        detector = DisconnectDetector(
            pipeline=pipeline,
            stream_reader=reader,
        )
        detector._health_check_interval = 0.05

        detector.start()
        time.sleep(0.1)

        # Simulate ffmpeg crash
        reader._is_running = False
        time.sleep(0.3)

        detector.stop()

        # Freeze should have been triggered
        pipeline.enter_freeze_frame.assert_called()

    @_usb_listener_patch
    def test_health_monitor_sets_disconnected_on_ffmpeg_crash(self, mock_usb):
        """Health monitor should set disconnected flag on ffmpeg crash."""
        pipeline = MagicMock()
        reader = MockStreamReader(is_running=True)

        detector = DisconnectDetector(
            pipeline=pipeline,
            stream_reader=reader,
        )
        detector._health_check_interval = 0.05

        detector.start()
        time.sleep(0.1)

        reader._is_running = False
        time.sleep(0.3)

        detector.stop()

        assert detector.is_disconnected is True

    def test_no_on_ffmpeg_crash_falls_back_to_on_disconnect(self):
        """If on_ffmpeg_crash is not set, falls back to on_disconnect."""
        pipeline = MagicMock()
        reader = MockStreamReader(is_running=True)

        disconnect_events = []

        detector = DisconnectDetector(
            pipeline=pipeline,
            stream_reader=reader,
        )
        detector._health_check_interval = 0.05
        # No on_ffmpeg_crash set — only on_disconnect
        detector.on_disconnect = lambda: disconnect_events.append(True)
        detector._start_usb_listener = MagicMock(return_value=False)

        detector.start()
        time.sleep(0.1)

        reader._is_running = False
        time.sleep(0.3)

        detector.stop()

        assert len(disconnect_events) >= 1, "Should fall back to on_disconnect"


# ---------------------------------------------------------------------------
# Test 6: Edge cases
# ---------------------------------------------------------------------------

class TestFFmpegCrashEdgeCases:
    """Edge cases for ffmpeg crash recovery."""

    def test_detector_ffmpeg_crash_with_no_pipeline(self):
        """ffmpeg crash detection with no pipeline should not crash."""
        reader = MockStreamReader(is_running=True)
        detector = DisconnectDetector(
            pipeline=None,
            stream_reader=reader,
        )
        detector._health_check_interval = 0.05
        detector._start_usb_listener = MagicMock(return_value=False)

        detector.start()
        time.sleep(0.1)

        reader._is_running = False
        time.sleep(0.2)

        detector.stop()
        # Should not crash

    @_app_usb_patch
    @patch('app_controller.GoProConnection')
    def test_recover_ffmpeg_crash_exception_handling(self, mock_gopro_cls, mock_usb):
        """_recover_ffmpeg_crash should handle exceptions and fall back."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)
        ctrl._running = True

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        # Make keep_alive raise an exception
        ctrl.gopro.keep_alive = MagicMock(side_effect=RuntimeError("boom"))
        ctrl._auto_recover = MagicMock()

        ctrl._recover_ffmpeg_crash()

        # Should fall back to full recovery
        ctrl._auto_recover.assert_called()

    @_app_usb_patch
    @patch('app_controller.GoProConnection')
    def test_recover_ffmpeg_crash_sets_streaming_state(self, mock_gopro_cls, mock_usb):
        """Successful ffmpeg crash recovery should set state to STREAMING."""
        from app_controller import AppController, AppState
        config = make_test_config()
        ctrl = AppController(config)
        ctrl._running = True

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        ctrl.gopro.keep_alive = MagicMock(return_value=True)
        ctrl._create_and_swap_stream_reader = MagicMock(return_value=True)

        ctrl._recover_ffmpeg_crash()

        assert ctrl.state == AppState.STREAMING


# ---------------------------------------------------------------------------
# Test 7: StreamReader stderr buffer and get_stderr_lines() accessor
# ---------------------------------------------------------------------------

class TestStreamReaderStderrBuffer:
    """Tests for the stderr line buffer and get_stderr_lines() accessor."""

    def _make_reader(self):
        """Create a StreamReader with test config (no ffmpeg launched)."""
        config = make_test_config()
        return StreamReader(config)

    def test_get_stderr_lines_returns_empty_initially(self):
        """get_stderr_lines() should return an empty list before any stderr output."""
        reader = self._make_reader()
        assert reader.get_stderr_lines() == []

    def test_get_stderr_lines_returns_buffered_lines(self):
        """get_stderr_lines() should return lines added to the buffer."""
        reader = self._make_reader()
        with reader._stderr_lock:
            reader._stderr_buffer.append("line one")
            reader._stderr_buffer.append("line two")

        lines = reader.get_stderr_lines()
        assert lines == ["line one", "line two"]

    def test_get_stderr_lines_returns_list_copy(self):
        """get_stderr_lines() should return a new list (not the internal deque)."""
        reader = self._make_reader()
        with reader._stderr_lock:
            reader._stderr_buffer.append("test line")

        lines = reader.get_stderr_lines()
        assert isinstance(lines, list)
        assert not isinstance(lines, collections.deque)

        # Mutating the returned list should not affect the internal buffer
        lines.clear()
        assert reader.get_stderr_lines() == ["test line"]

    def test_stderr_buffer_caps_at_50_lines(self):
        """Buffer should keep only the 50 most recent lines."""
        reader = self._make_reader()
        with reader._stderr_lock:
            for i in range(70):
                reader._stderr_buffer.append(f"line {i}")

        lines = reader.get_stderr_lines()
        assert len(lines) == 50
        # Should contain the most recent 50 (lines 20-69)
        assert lines[0] == "line 20"
        assert lines[-1] == "line 69"

    def test_last_error_uses_lock(self):
        """last_error property should be thread-safe (uses lock)."""
        reader = self._make_reader()
        with reader._stderr_lock:
            reader._last_stderr = "fatal error occurred"

        assert reader.last_error == "fatal error occurred"

    def test_read_stderr_populates_buffer(self):
        """_read_stderr should populate the buffer from stderr pipe data."""
        reader = self._make_reader()

        # Create a mock process with stderr that returns data then EOF
        mock_stderr = MagicMock()
        mock_stderr.read.side_effect = [
            b"Opening stream...\r\nConnection established\r\n",
            b"",  # EOF
        ]
        mock_process = MagicMock()
        mock_process.stderr = mock_stderr
        reader._process = mock_process

        reader._read_stderr()

        lines = reader.get_stderr_lines()
        assert "Opening stream..." in lines
        assert "Connection established" in lines

    def test_read_stderr_logs_error_keywords(self):
        """_read_stderr should log warnings for lines containing error keywords."""
        reader = self._make_reader()

        mock_stderr = MagicMock()
        mock_stderr.read.side_effect = [
            b"normal output\r\nfatal: something broke\r\n",
            b"",
        ]
        mock_process = MagicMock()
        mock_process.stderr = mock_stderr
        reader._process = mock_process

        reader._read_stderr()

        lines = reader.get_stderr_lines()
        assert "normal output" in lines
        assert "fatal: something broke" in lines
        assert reader.last_error == "fatal: something broke"

    def test_read_stderr_handles_carriage_returns(self):
        """_read_stderr should split on \\r (ffmpeg progress lines) as well as \\n."""
        reader = self._make_reader()

        mock_stderr = MagicMock()
        # ffmpeg progress uses \r without \n
        mock_stderr.read.side_effect = [
            b"frame=  100 fps=30\rframe=  200 fps=30\r\n",
            b"",
        ]
        mock_process = MagicMock()
        mock_process.stderr = mock_stderr
        reader._process = mock_process

        reader._read_stderr()

        lines = reader.get_stderr_lines()
        assert len(lines) >= 2
        assert any("frame=" in line for line in lines)
