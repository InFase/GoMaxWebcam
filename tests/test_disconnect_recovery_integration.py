"""
test_disconnect_recovery_integration.py — Integration tests for the
disconnect/recovery pipeline and ffmpeg diagnostics

These tests verify end-to-end behavior across multiple components:
  1. USB detach → DisconnectDetector → freeze-frame → VirtualCamera keeps running
  2. ffmpeg crash → health monitor → freeze-frame → lightweight recovery path
  3. Full disconnect → freeze → USB reconnect → reader swap → live video restored
  4. Health monitor pause/resume prevents false triggers during reader swap
  5. ffmpeg stderr diagnostics accessible after crash
  6. Recovery idempotency — duplicate events produce no side effects
  7. Memory safety — all tests stay within 2GB Job Object RAM cap

IMPORTANT: USBEventListener uses raw Win32 APIs that crash in pytest.
All tests that touch DisconnectDetector.start() or AppController MUST
mock usb_event_listener.USBEventListener.
"""

import collections
import gc
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import psutil
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from disconnect_detector import DisconnectDetector
from frame_buffer import FrameBuffer
from frame_pipeline import FramePipeline, PipelineState
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
    """Create a test BGR24 frame as numpy array."""
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
        self._stderr_buffer = collections.deque(maxlen=50)
        self._stderr_lock = threading.Lock()
        self._last_stderr = ""

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

    def get_stderr_lines(self):
        with self._stderr_lock:
            return list(self._stderr_buffer)

    @property
    def last_error(self):
        with self._stderr_lock:
            return self._last_stderr


class MockVirtualCamera:
    """Mock VirtualCamera that records frames with bounded memory."""

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


# Mock USBEventListener globally for all tests that start the detector
_usb_listener_patch = patch(
    'usb_event_listener.USBEventListener',
    **{'return_value.start.return_value': False, 'return_value.stop.return_value': None,
       'return_value.is_running': False},
)

# Mock USBEventListener for AppController tests
_app_usb_patch = patch(
    'usb_event_listener.USBEventListener',
    **{'return_value.start.return_value': False, 'return_value.stop.return_value': None,
       'return_value.is_running': False},
)


# ---------------------------------------------------------------------------
# Integration Test 1: Full disconnect → freeze → recovery pipeline
# ---------------------------------------------------------------------------

class TestFullDisconnectRecoveryPipeline:
    """End-to-end: USB detach → freeze-frame → reader swap → live video restored.

    Verifies the complete disconnect/recovery flow across DisconnectDetector,
    FramePipeline, FrameBuffer, and VirtualCamera.
    """

    def test_detach_freeze_swap_recovery_flow(self):
        """Full flow: live → USB detach → freeze → swap reader → live again.

        This is the core integration scenario: GoPro is unplugged mid-stream,
        the pipeline freezes on the last good frame, then a new reader is
        swapped in simulating reconnection, and live video resumes.
        """
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()

        # Phase 1: Streaming live video
        live_color = (200, 100, 50)
        live_frames = [make_test_frame(color=live_color) for _ in range(20)]
        _idx = [0]

        def cycling_read():
            f = live_frames[_idx[0] % len(live_frames)]
            _idx[0] += 1
            return f

        reader1 = MagicMock()
        reader1.is_running = True
        reader1.width = 320
        reader1.height = 240
        reader1.read_frame = MagicMock(side_effect=cycling_read)
        vcam = MockVirtualCamera()

        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.1)
        assert pipeline.state == PipelineState.STREAMING

        # Phase 2: USB detach triggers freeze via DisconnectDetector
        detector = DisconnectDetector(pipeline=pipeline)
        detector._on_usb_detach("\\\\?\\USB#VID_0A70&PID_000D#serial#{guid}")

        assert pipeline.state == PipelineState.FREEZE_FRAME
        assert detector.is_disconnected is True
        frames_at_freeze = vcam.frame_count

        # Phase 3: Virtual camera continues receiving frames (frozen)
        time.sleep(0.2)
        assert vcam.frame_count > frames_at_freeze, "VCam should keep receiving freeze frames"
        assert vcam.is_running is True

        # Verify freeze frames match last live frame
        freeze_frames = list(vcam.frames_sent)[-5:]
        for ff in freeze_frames:
            np.testing.assert_array_equal(ff, live_frames[0])

        # Phase 4: Swap reader (simulating reconnection)
        recovery_color = (0, 255, 0)
        recovery_frames = [make_test_frame(color=recovery_color) for _ in range(10)]
        reader2 = MockStreamReader(frames=recovery_frames)
        pipeline.swap_reader(reader2)
        time.sleep(0.3)

        pipeline.stop()

        # Phase 5: Verify new live frames appeared
        new_live = [f for f in vcam.frames_sent if tuple(f[0, 0]) == recovery_color]
        assert len(new_live) > 0, "Recovery reader frames should appear in vcam output"

    def test_freeze_frame_pixel_perfect_through_recovery(self):
        """Freeze frames must be pixel-identical to last live frame throughout recovery."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()

        # Distinctive last frame
        last_live = make_test_frame(color=(42, 84, 168))
        reader = MockStreamReader(frames=[last_live], crash_after=1)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.5)

        # Should be frozen with pixel-perfect copies
        assert pipeline.state == PipelineState.FREEZE_FRAME

        # Check all freeze frames
        for i, frame in enumerate(list(vcam.frames_sent)[1:], start=1):
            np.testing.assert_array_equal(
                frame, last_live,
                err_msg=f"Freeze frame #{i} is not pixel-identical to last live frame"
            )

        pipeline.stop()

    def test_no_black_frames_in_disconnect_recovery_cycle(self):
        """No black frames should appear at any point: live → freeze → recovery."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()

        # Phase 1: live with non-black frames
        live_frame = make_test_frame(color=(128, 128, 128))
        reader1 = MockStreamReader(frames=[live_frame], crash_after=5)
        vcam = MockVirtualCamera()

        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.3)

        # Phase 2: crash → freeze
        assert pipeline.state == PipelineState.FREEZE_FRAME

        # Phase 3: swap (recovery)
        reader2 = MockStreamReader(frames=[make_test_frame(color=(64, 200, 100))])
        pipeline.swap_reader(reader2)
        time.sleep(0.2)

        pipeline.stop()

        # Verify no black frames anywhere
        for i, frame in enumerate(vcam.frames_sent):
            is_black = np.all(frame == 0)
            assert not is_black, f"Frame #{i} is black during disconnect/recovery cycle"

    def test_vcam_never_stops_during_disconnect_recovery(self):
        """VirtualCamera must remain running through entire disconnect/recovery."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()

        reader1 = MockStreamReader(
            frames=[make_test_frame(color=(200, 0, 0))],
            crash_after=5,
        )
        vcam = MockVirtualCamera()

        pipeline.start(reader1, vcam, frame_buffer=buffer)

        # Sample vcam.is_running every 50ms through the whole cycle
        running_checks = []
        for _ in range(10):
            time.sleep(0.05)
            running_checks.append(vcam.is_running)

        # Swap reader (recovery)
        reader2 = MockStreamReader(frames=[make_test_frame(color=(0, 200, 0))])
        pipeline.swap_reader(reader2)

        for _ in range(5):
            time.sleep(0.05)
            running_checks.append(vcam.is_running)

        pipeline.stop()

        # VCam should have been running the whole time
        assert all(running_checks), "VirtualCamera stopped during disconnect/recovery"


# ---------------------------------------------------------------------------
# Integration Test 2: ffmpeg crash detection and lightweight recovery
# ---------------------------------------------------------------------------

class TestFFmpegCrashIntegration:
    """Integration tests for ffmpeg crash → health monitor → freeze → recovery.

    Verifies the health monitor correctly detects ffmpeg process death,
    distinguishes it from USB disconnect, and triggers appropriate callbacks.
    """

    @_usb_listener_patch
    def test_ffmpeg_crash_detection_fires_correct_callback(self, mock_usb):
        """Health monitor fires on_ffmpeg_crash (not on_disconnect) for pure ffmpeg crash."""
        pipeline = MagicMock()
        reader = MockStreamReader(is_running=True)

        ffmpeg_events = []
        disconnect_events = []

        detector = DisconnectDetector(
            pipeline=pipeline,
            stream_reader=reader,
        )
        detector._health_check_interval = 0.05
        detector.on_ffmpeg_crash = lambda: ffmpeg_events.append(True)
        detector.on_disconnect = lambda: disconnect_events.append(True)

        detector.start()
        time.sleep(0.15)

        # Simulate pure ffmpeg crash (no USB detach)
        reader._is_running = False
        time.sleep(0.3)

        detector.stop()

        assert len(ffmpeg_events) >= 1, "on_ffmpeg_crash should fire"
        assert len(disconnect_events) == 0, "on_disconnect should NOT fire for ffmpeg crash"

    @_usb_listener_patch
    def test_usb_disconnect_before_ffmpeg_death_fires_on_disconnect(self, mock_usb):
        """When USB detach precedes ffmpeg death, on_disconnect fires correctly."""
        pipeline = MagicMock()
        reader = MockStreamReader(is_running=True)

        ffmpeg_events = []
        disconnect_events = []

        detector = DisconnectDetector(
            pipeline=pipeline,
            stream_reader=reader,
        )
        detector._health_check_interval = 0.05
        detector.on_ffmpeg_crash = lambda: ffmpeg_events.append(True)
        detector.on_disconnect = lambda: disconnect_events.append(True)

        detector.start()
        time.sleep(0.15)

        # USB detach first, then ffmpeg dies
        detector._on_usb_detach("USB\\VID_2672&PID_0052\\test")
        reader._is_running = False
        time.sleep(0.3)

        detector.stop()

        assert len(disconnect_events) >= 1, "on_disconnect should fire"
        assert len(ffmpeg_events) == 0, "on_ffmpeg_crash should NOT fire after USB detach"

    def test_ffmpeg_crash_pipeline_stays_frozen_during_recovery(self):
        """Pipeline should stay in freeze-frame during ffmpeg crash recovery window."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()

        reader = MockStreamReader(
            frames=[make_test_frame(color=(100, 100, 100))],
            crash_after=3,
        )
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.4)

        # Pipeline should be frozen after ffmpeg crash
        assert pipeline.state == PipelineState.FREEZE_FRAME
        assert pipeline.is_running is True  # Thread still alive

        # VCam should still be running
        assert vcam.is_running is True

        # Frame count should still be growing (freeze frames)
        count_before = vcam.frame_count
        time.sleep(0.2)
        assert vcam.frame_count > count_before, "Freeze frames should keep being pushed"

        pipeline.stop()

    def test_multiple_ffmpeg_crash_recovery_cycles(self):
        """Pipeline should survive multiple consecutive ffmpeg crash/recovery cycles."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()
        vcam = MockVirtualCamera()

        colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]

        # Cycle 1: crash after 3 frames
        reader1 = MockStreamReader(
            frames=[make_test_frame(color=colors[0])],
            crash_after=3,
        )
        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.3)
        assert pipeline.is_running

        # Cycle 2: swap and crash again
        reader2 = MockStreamReader(
            frames=[make_test_frame(color=colors[1])],
            crash_after=3,
        )
        pipeline.swap_reader(reader2)
        time.sleep(0.3)
        assert pipeline.is_running

        # Cycle 3: swap with stable reader
        reader3 = MockStreamReader(
            frames=[make_test_frame(color=colors[2])],
        )
        pipeline.swap_reader(reader3)
        time.sleep(0.2)

        assert pipeline.is_running
        assert vcam.is_running

        pipeline.stop()

        # Verify pipeline survived all cycles and pushed frames throughout
        assert pipeline.frames_pushed > 10, "Should have pushed frames across all cycles"
        # At minimum, first and last cycle colors should appear
        all_colors = set()
        for frame in vcam.frames_sent:
            all_colors.add(tuple(frame[0, 0]))
        assert colors[0] in all_colors, f"Missing frames from cycle 1 with color {colors[0]}"
        assert colors[2] in all_colors, f"Missing frames from cycle 3 with color {colors[2]}"


# ---------------------------------------------------------------------------
# Integration Test 3: Health monitor pause/resume during reader swap
# ---------------------------------------------------------------------------

class TestHealthMonitorReaderSwapIntegration:
    """Integration: health monitor pause/resume eliminates false triggers during swap."""

    @_usb_listener_patch
    def test_paused_monitor_no_false_trigger_during_swap(self, mock_usb):
        """Paused health monitor should not trigger freeze when old reader stops."""
        pipeline = MagicMock()
        reader = MockStreamReader(is_running=True)

        ffmpeg_events = []
        disconnect_events = []

        detector = DisconnectDetector(
            pipeline=pipeline,
            stream_reader=reader,
        )
        detector._health_check_interval = 0.05
        detector.on_ffmpeg_crash = lambda: ffmpeg_events.append(True)
        detector.on_disconnect = lambda: disconnect_events.append(True)

        detector.start()
        time.sleep(0.15)

        # Pause → stop old reader → swap → resume (simulates reader swap)
        detector.pause_health_monitor()
        reader._is_running = False
        time.sleep(0.15)

        # No triggers should fire while paused
        assert len(ffmpeg_events) == 0, "No ffmpeg crash should fire while paused"
        assert len(disconnect_events) == 0, "No disconnect should fire while paused"
        pipeline.enter_freeze_frame.assert_not_called()

        # Swap to new reader and resume
        new_reader = MockStreamReader(is_running=True)
        detector.update_stream_reader(new_reader)
        detector.resume_health_monitor()

        time.sleep(0.15)

        # Still no false triggers (new reader is healthy)
        assert len(ffmpeg_events) == 0
        assert len(disconnect_events) == 0

        detector.stop()

    @_usb_listener_patch
    def test_resume_detects_new_reader_death(self, mock_usb):
        """After resume, health monitor should detect if new reader also dies."""
        pipeline = MagicMock()
        reader1 = MockStreamReader(is_running=True)

        ffmpeg_events = []

        detector = DisconnectDetector(
            pipeline=pipeline,
            stream_reader=reader1,
        )
        detector._health_check_interval = 0.05
        detector.on_ffmpeg_crash = lambda: ffmpeg_events.append(True)

        detector.start()
        time.sleep(0.15)

        # Swap reader
        detector.pause_health_monitor()
        reader1._is_running = False

        reader2 = MockStreamReader(is_running=True)
        detector.update_stream_reader(reader2)
        detector.resume_health_monitor()

        # Let monitor see reader2 running
        time.sleep(0.15)

        # Now crash reader2
        reader2._is_running = False
        time.sleep(0.3)

        detector.stop()

        # Monitor should detect reader2's crash
        assert len(ffmpeg_events) >= 1, "Should detect new reader's crash after resume"


# ---------------------------------------------------------------------------
# Integration Test 4: ffmpeg stderr diagnostics
# ---------------------------------------------------------------------------

class TestFFmpegDiagnosticsIntegration:
    """Tests for ffmpeg stderr buffer accessibility after crash."""

    def test_stderr_buffer_available_after_mock_crash(self):
        """Stderr diagnostics should be accessible via get_stderr_lines() after crash."""
        reader = MockStreamReader(is_running=True)

        # Simulate stderr output before crash
        with reader._stderr_lock:
            reader._stderr_buffer.append("frame=  100 fps=30 time=00:00:03.33")
            reader._stderr_buffer.append("frame=  200 fps=30 time=00:00:06.67")
            reader._stderr_buffer.append("error: connection refused")
            reader._last_stderr = "error: connection refused"

        # Simulate crash
        reader._is_running = False

        # Diagnostics should still be accessible after crash
        lines = reader.get_stderr_lines()
        assert len(lines) == 3
        assert "error: connection refused" in lines
        assert reader.last_error == "error: connection refused"

    def test_real_stream_reader_stderr_buffer_initial_state(self):
        """Real StreamReader should have empty stderr buffer initially."""
        config = make_test_config()
        reader = StreamReader(config)

        assert reader.get_stderr_lines() == []
        assert reader.last_error == ""

    def test_real_stream_reader_stderr_buffer_populated(self):
        """Real StreamReader should populate stderr buffer from ffmpeg output."""
        config = make_test_config()
        reader = StreamReader(config)

        # Manually populate buffer (simulating _read_stderr behavior)
        with reader._stderr_lock:
            reader._stderr_buffer.append("Opening stream...")
            reader._stderr_buffer.append("Connection established")
            reader._stderr_buffer.append("fatal: something broke")
            reader._last_stderr = "fatal: something broke"

        lines = reader.get_stderr_lines()
        assert len(lines) == 3
        assert "Opening stream..." in lines
        assert "fatal: something broke" in lines
        assert reader.last_error == "fatal: something broke"

    def test_stderr_buffer_is_list_copy(self):
        """get_stderr_lines() should return a list copy, not the internal deque."""
        config = make_test_config()
        reader = StreamReader(config)

        with reader._stderr_lock:
            reader._stderr_buffer.append("test line")

        lines = reader.get_stderr_lines()
        assert isinstance(lines, list)
        assert not isinstance(lines, collections.deque)

        # Mutating returned list should not affect internal buffer
        lines.clear()
        assert reader.get_stderr_lines() == ["test line"]

    def test_stderr_buffer_caps_at_50_lines(self):
        """Stderr buffer should keep only the 50 most recent lines."""
        config = make_test_config()
        reader = StreamReader(config)

        with reader._stderr_lock:
            for i in range(70):
                reader._stderr_buffer.append(f"line {i}")

        lines = reader.get_stderr_lines()
        assert len(lines) == 50
        assert lines[0] == "line 20"
        assert lines[-1] == "line 69"


# ---------------------------------------------------------------------------
# Integration Test 5: Recovery idempotency
# ---------------------------------------------------------------------------

class TestRecoveryIdempotency:
    """Duplicate disconnect/freeze events should produce no side effects."""

    def test_duplicate_detach_events_safe(self):
        """Rapid duplicate USB detach events should be debounced."""
        pipeline = MagicMock()
        detector = DisconnectDetector(pipeline=pipeline)
        detector._debounce_interval = 1.0

        detector._on_usb_detach("device1")
        detector._on_usb_detach("device1")
        detector._on_usb_detach("device1")

        pipeline.enter_freeze_frame.assert_called_once()

    def test_duplicate_freeze_triggers_safe(self):
        """Rapid freeze triggers via _trigger_freeze_frame should be debounced."""
        pipeline = MagicMock()
        detector = DisconnectDetector(pipeline=pipeline)
        detector._debounce_interval = 1.0

        detector._trigger_freeze_frame("reason1")
        detector._trigger_freeze_frame("reason2")
        detector._trigger_freeze_frame("reason3")

        pipeline.enter_freeze_frame.assert_called_once()

    def test_pipeline_enter_freeze_frame_idempotent(self):
        """Calling enter_freeze_frame() when already frozen should be a no-op."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        pipeline._state = PipelineState.STREAMING

        callback_count = []
        pipeline.on_stream_lost = lambda: callback_count.append(1)

        pipeline.enter_freeze_frame()
        assert pipeline.state == PipelineState.FREEZE_FRAME
        assert len(callback_count) == 1

        freeze_time_1 = pipeline._freeze_start_time

        # Second call should be no-op
        pipeline.enter_freeze_frame()
        assert pipeline.state == PipelineState.FREEZE_FRAME
        assert len(callback_count) == 1  # NOT called again
        assert pipeline._freeze_start_time == freeze_time_1  # Timestamp unchanged

    def test_concurrent_detach_and_health_monitor_trigger(self):
        """USB detach + health monitor should not double-trigger freeze."""
        pipeline = MagicMock()
        detector = DisconnectDetector(pipeline=pipeline)
        detector._debounce_interval = 1.0

        # USB detach triggers freeze
        detector._on_usb_detach("device")

        # Health monitor also notices stream death — debounced
        detector._trigger_freeze_frame("ffmpeg process exited")

        # Only one freeze trigger
        pipeline.enter_freeze_frame.assert_called_once()


# ---------------------------------------------------------------------------
# Integration Test 6: AppController ffmpeg crash recovery wiring
# ---------------------------------------------------------------------------

class TestAppControllerFFmpegCrashWiring:
    """Tests that AppController correctly wires ffmpeg crash callback
    and routes it to lightweight recovery."""

    @_app_usb_patch
    @patch('app_controller.GoProConnection')
    def test_detector_wired_with_ffmpeg_crash_callback(self, mock_gopro_cls, mock_usb):
        """_start_disconnect_detector should wire on_ffmpeg_crash to _on_ffmpeg_crash."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)

        ctrl._frame_pipeline = MagicMock()
        ctrl._stream_reader = MagicMock()

        with patch('disconnect_detector.DisconnectDetector') as mock_dd_cls:
            mock_detector = MagicMock()
            mock_detector.is_running = False
            mock_detector.start.return_value = True
            mock_dd_cls.return_value = mock_detector

            ctrl._start_disconnect_detector()

            assert mock_detector.on_ffmpeg_crash == ctrl._on_ffmpeg_crash

    @_app_usb_patch
    @patch('app_controller.GoProConnection')
    def test_ffmpeg_crash_skips_when_already_recovering(self, mock_gopro_cls, mock_usb):
        """_on_ffmpeg_crash should be no-op when recovery is already in progress."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)
        ctrl._is_recovering = True
        ctrl._recover_ffmpeg_crash = MagicMock()

        ctrl._on_ffmpeg_crash()

        ctrl._recover_ffmpeg_crash.assert_not_called()

    @_app_usb_patch
    @patch('app_controller.GoProConnection')
    def test_lightweight_recovery_restarts_ffmpeg_when_api_reachable(self, mock_gopro_cls, mock_usb):
        """When GoPro API is still reachable, recovery restarts just ffmpeg."""
        from app_controller import AppController, AppState
        config = make_test_config()
        ctrl = AppController(config)
        ctrl._running = True

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline
        ctrl.gopro.keep_alive = MagicMock(return_value=True)
        ctrl._create_and_swap_stream_reader = MagicMock(return_value=True)
        ctrl._auto_recover = MagicMock()

        ctrl._recover_ffmpeg_crash()

        # Should restart ffmpeg, NOT do full recovery
        ctrl._create_and_swap_stream_reader.assert_called_once()
        ctrl._auto_recover.assert_not_called()
        assert ctrl._recovery_count == 1

    @_app_usb_patch
    @patch('app_controller.GoProConnection')
    def test_falls_back_to_full_recovery_when_api_unreachable(self, mock_gopro_cls, mock_usb):
        """When GoPro API is unreachable, falls back to full _auto_recover."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)
        ctrl._running = True

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline
        ctrl.gopro.keep_alive = MagicMock(return_value=False)
        ctrl._auto_recover = MagicMock()

        ctrl._recover_ffmpeg_crash()

        ctrl._auto_recover.assert_called_once()

    @_app_usb_patch
    @patch('app_controller.GoProConnection')
    def test_falls_back_on_swap_failure(self, mock_gopro_cls, mock_usb):
        """If ffmpeg restart fails, falls back to full recovery."""
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
    def test_exception_in_recovery_falls_back(self, mock_gopro_cls, mock_usb):
        """Exception during recovery should trigger fallback to full recovery."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)
        ctrl._running = True

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline
        ctrl.gopro.keep_alive = MagicMock(side_effect=RuntimeError("boom"))
        ctrl._auto_recover = MagicMock()

        ctrl._recover_ffmpeg_crash()

        ctrl._auto_recover.assert_called()


# ---------------------------------------------------------------------------
# Integration Test 7: Continuous frame emission through crash and recovery
# ---------------------------------------------------------------------------

class TestContinuousFrameEmission:
    """Frames must be pushed continuously through crash and recovery cycles."""

    def test_monotonic_frame_count_through_crash(self):
        """Frame count should be monotonically non-decreasing through crash."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()
        vcam = MockVirtualCamera()

        reader = MockStreamReader(
            frames=[make_test_frame(color=(100, 100, 100))],
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

        # Monotonically non-decreasing
        for i in range(1, len(samples)):
            assert samples[i] >= samples[i - 1], (
                f"Frame count decreased: {samples[i-1]} → {samples[i]}"
            )

        # Should have pushed frames throughout
        assert samples[-1] > samples[0], "No frames pushed over time"

    def test_no_frame_gap_during_crash_to_freeze_transition(self):
        """The transition from live streaming to freeze-frame should be seamless."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()
        vcam = MockVirtualCamera()

        live_frame = make_test_frame(color=(150, 75, 200))
        reader = MockStreamReader(frames=[live_frame], crash_after=5)

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.5)
        pipeline.stop()

        # Should have frames from both live and freeze phases
        assert vcam.frame_count > 5, "Should have both live and freeze frames"

        # No None frames (all should be valid numpy arrays)
        for i, frame in enumerate(vcam.frames_sent):
            assert frame is not None, f"Frame #{i} is None"
            assert frame.shape == (240, 320, 3), f"Frame #{i} has wrong shape"


# ---------------------------------------------------------------------------
# Integration Test 8: Memory safety verification
# ---------------------------------------------------------------------------

class TestMemorySafety:
    """Verify tests stay within 2GB Job Object RAM cap.

    These tests explicitly check that frame handling doesn't leak memory
    by using the FrameLog bounded collection and checking RSS.
    """

    def test_framelog_bounds_memory(self):
        """FrameLog should cap stored frames to prevent unbounded growth."""
        log = FrameLog(first_cap=10, recent_cap=90)

        # Push 1000 frames
        for i in range(1000):
            frame = make_test_frame(color=(i % 256, 0, 0))
            log.append(frame)

        # Should only store 100 frames (10 first + 90 recent)
        assert len(log) == 100
        assert log._count == 1000

    def test_pipeline_with_framelog_vcam_bounded_memory(self):
        """Pipeline with FrameLog-backed vcam should not grow memory unboundedly."""
        proc = psutil.Process(os.getpid())
        gc.collect()
        rss_before = proc.memory_info().rss

        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()
        vcam = MockVirtualCamera()

        # Run pipeline for a moderate time
        reader = MockStreamReader(frames=[make_test_frame()])
        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.5)
        pipeline.stop()

        gc.collect()
        rss_after = proc.memory_info().rss
        delta_mb = (rss_after - rss_before) / (1024 * 1024)

        # Memory growth should be reasonable (< 100MB for a short test)
        assert delta_mb < 100, f"Memory grew by {delta_mb:.1f} MB — possible leak"

    def test_freeze_frame_cycle_no_memory_leak(self):
        """Multiple freeze/recovery cycles should not leak memory."""
        proc = psutil.Process(os.getpid())
        gc.collect()
        rss_before = proc.memory_info().rss

        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()
        vcam = MockVirtualCamera()

        # Run 3 crash/recovery cycles
        for cycle in range(3):
            reader = MockStreamReader(
                frames=[make_test_frame(color=(cycle * 80, 0, 0))],
                crash_after=5,
            )
            if cycle == 0:
                pipeline.start(reader, vcam, frame_buffer=buffer)
            else:
                pipeline.swap_reader(reader)
            time.sleep(0.3)

        pipeline.stop()

        gc.collect()
        rss_after = proc.memory_info().rss
        delta_mb = (rss_after - rss_before) / (1024 * 1024)

        assert delta_mb < 200, f"Memory grew by {delta_mb:.1f} MB after 3 cycles — possible leak"


# ---------------------------------------------------------------------------
# Integration Test 9: Detector + Pipeline + Buffer end-to-end
# ---------------------------------------------------------------------------

class TestDetectorPipelineBufferE2E:
    """End-to-end: DisconnectDetector health monitor + FramePipeline + FrameBuffer."""

    @_usb_listener_patch
    def test_health_monitor_triggers_freeze_pipeline_serves_frozen_frame(self, mock_usb):
        """Health monitor detects stream death → pipeline freezes → buffer serves frame."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()

        live_frame = make_test_frame(color=(200, 100, 50))
        reader = MockStreamReader(is_running=True)

        # Override read to return live frames, then crash
        call_count = [0]
        def read_fn():
            call_count[0] += 1
            if call_count[0] <= 5:
                return live_frame
            reader._is_running = False
            return None
        reader.read_frame = read_fn

        vcam = MockVirtualCamera()
        pipeline.start(reader, vcam, frame_buffer=buffer)

        # Wire detector
        detector = DisconnectDetector(
            pipeline=pipeline,
            stream_reader=reader,
        )
        detector._health_check_interval = 0.05

        ffmpeg_events = []
        detector.on_ffmpeg_crash = lambda: ffmpeg_events.append(True)

        detector.start()
        time.sleep(0.8)

        detector.stop()
        pipeline.stop()

        # Pipeline should have entered freeze
        assert pipeline.state == PipelineState.STOPPED  # We stopped it
        # But freeze frames should have been pushed to vcam
        assert vcam.frame_count > 5, "Should have both live and freeze frames"

        # All frames should be non-black
        for frame in vcam.frames_sent:
            assert not np.all(frame == 0), "No black frames allowed"

    def test_detector_mark_connected_after_reader_swap(self):
        """After successful reader swap, detector should be marked connected."""
        pipeline = MagicMock()
        detector = DisconnectDetector(pipeline=pipeline)

        # Simulate disconnect
        detector._on_usb_detach("device")
        assert detector.is_disconnected is True

        # Simulate recovery: swap reader and mark connected
        new_reader = MockStreamReader(is_running=True)
        detector.update_stream_reader(new_reader)
        detector.mark_connected()

        assert detector.is_disconnected is False
        assert detector._stream_reader is new_reader
