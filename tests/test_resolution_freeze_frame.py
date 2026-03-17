"""
test_resolution_freeze_frame.py — Tests for Sub-AC 16.2: Freeze-frame during resolution transition

Verifies that:
  1. FrameBuffer.change_resolution() scales the last good frame to new dimensions
  2. FrameBuffer.change_resolution() uses placeholder when no live frame exists
  3. FramePipeline.handle_resolution_transition() enters freeze-frame and updates targets
  4. Pipeline continues emitting scaled freeze-frame at new resolution during transition
  5. swap_reader() after resolution transition exits freeze-frame and resumes live video
  6. No black frames or gaps appear during the resolution transition
  7. AppController.change_resolution() orchestrates the full transition
  8. Resolution change is a no-op when dimensions are unchanged
  9. Frame scaling preserves visual content (nearest-neighbor, not black)

IMPORTANT: USBEventListener uses raw Win32 APIs that crash in pytest.
All tests that instantiate AppController MUST mock usb_event_listener.USBEventListener.
"""

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
from frame_buffer import FrameBuffer
from frame_pipeline import FramePipeline, PipelineState



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
    """Mock StreamReader that yields frames then optionally dies."""

    def __init__(self, frames=None, max_frames=None, width=320, height=240):
        self._frames = frames or []
        self._index = 0
        self._max_frames = max_frames
        self._is_running = True
        self.width = width
        self.height = height
        self.fps = 30

    @property
    def is_running(self):
        return self._is_running

    @is_running.setter
    def is_running(self, val):
        self._is_running = val

    def read_frame(self):
        if self._max_frames is not None and self._index >= self._max_frames:
            self._is_running = False
            return None
        if not self._frames:
            self._is_running = False
            return None
        frame = self._frames[self._index % len(self._frames)]
        self._index += 1
        return frame

    def stop(self):
        self._is_running = False


class MockVirtualCamera:
    """Mock VirtualCamera that records every frame sent for verification."""

    def __init__(self, width=320, height=240):
        self.frames_sent = FrameLog()
        self.frame_count = 0
        self._is_running = True
        self._frame_count = 0
        self._last_frame = None
        self.width = width
        self.height = height
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
            self._frame_count += 1
        return True

    def send_last_frame(self):
        with self._lock:
            if self._last_frame is not None:
                self.frame_count += 1
                self.frames_sent.append(self._last_frame.copy())
                self._frame_count += 1
                return True
        return False

    def sleep_until_next_frame(self):
        time.sleep(0.001)


# ---------------------------------------------------------------------------
# Test: FrameBuffer.change_resolution() — scales last good frame
# ---------------------------------------------------------------------------

class TestFrameBufferChangeResolution:
    """Tests for FrameBuffer.change_resolution() with freeze-frame preservation."""

    def test_change_resolution_scales_last_live_frame(self):
        """change_resolution() should scale the last live frame to new dimensions."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        # Push a distinctive live frame
        live_frame = make_test_frame(width=320, height=240, color=(200, 100, 50))
        buf.update(live_frame)

        # Change resolution
        result = buf.change_resolution(640, 480)
        assert result is True
        assert buf.width == 640
        assert buf.height == 480

        # The frozen frame should be at the new dimensions
        frame = buf.get_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)

        # The scaled frame should have the same color (not black, not placeholder)
        assert tuple(frame[0, 0]) == (200, 100, 50)

    def test_change_resolution_uses_placeholder_when_no_live_frame(self):
        """change_resolution() should create placeholder at new size when no live frame exists."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        # No live frame pushed — change resolution
        result = buf.change_resolution(640, 480)
        assert result is True

        frame = buf.get_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)

        # Should be placeholder color (40, 40, 40), not black
        assert tuple(frame[0, 0]) == (40, 40, 40)

    def test_change_resolution_noop_when_unchanged(self):
        """change_resolution() should be a no-op when dimensions are the same."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        live_frame = make_test_frame(color=(200, 100, 50))
        buf.update(live_frame)

        # Same resolution — no change
        result = buf.change_resolution(320, 240)
        assert result is True
        assert buf.width == 320
        assert buf.height == 240

        # Frame should be unchanged
        frame = buf.get_frame()
        np.testing.assert_array_equal(frame, live_frame)

    def test_change_resolution_downscale(self):
        """change_resolution() should downscale correctly (1920x1080 → 1280x720)."""
        buf = FrameBuffer(width=1920, height=1080)
        buf.start()

        # Push a frame with a known color
        frame_1080p = make_test_frame(width=1920, height=1080, color=(100, 200, 50))
        buf.update(frame_1080p)

        # Downscale to 720p
        result = buf.change_resolution(1280, 720)
        assert result is True

        frame = buf.get_frame()
        assert frame.shape == (720, 1280, 3)
        # Color should be preserved
        assert tuple(frame[0, 0]) == (100, 200, 50)

    def test_change_resolution_upscale(self):
        """change_resolution() should upscale correctly (1280x720 → 1920x1080)."""
        buf = FrameBuffer(width=1280, height=720)
        buf.start()

        frame_720p = make_test_frame(width=1280, height=720, color=(50, 150, 250))
        buf.update(frame_720p)

        # Upscale to 1080p
        result = buf.change_resolution(1920, 1080)
        assert result is True

        frame = buf.get_frame()
        assert frame.shape == (1080, 1920, 3)
        assert tuple(frame[0, 0]) == (50, 150, 250)

    def test_change_resolution_resets_stats(self):
        """change_resolution() should reset update/read counts."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        # Accumulate some stats
        for _ in range(5):
            buf.update(make_test_frame())
        for _ in range(3):
            buf.get_frame()

        assert buf.total_updates == 5
        assert buf.total_reads == 3

        buf.change_resolution(640, 480)

        assert buf.total_updates == 0
        assert buf.total_reads == 0

    def test_change_resolution_accepts_new_size_frames(self):
        """After change_resolution(), update() should accept frames at the new size."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        buf.update(make_test_frame(width=320, height=240, color=(100, 0, 0)))

        buf.change_resolution(640, 480)

        # New frame at new resolution should be accepted
        new_frame = make_test_frame(width=640, height=480, color=(0, 200, 0))
        accepted = buf.update(new_frame)
        assert accepted is True

        frame = buf.get_frame()
        np.testing.assert_array_equal(frame, new_frame)

    def test_change_resolution_rejects_old_size_frames(self):
        """After change_resolution(), update() should reject frames at old size."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        buf.update(make_test_frame(width=320, height=240, color=(100, 0, 0)))

        buf.change_resolution(640, 480)

        # Old-resolution frame should be rejected
        old_frame = make_test_frame(width=320, height=240, color=(0, 0, 200))
        accepted = buf.update(old_frame)
        assert accepted is False


# ---------------------------------------------------------------------------
# Test: FrameBuffer._scale_frame() — nearest-neighbor scaling
# ---------------------------------------------------------------------------

class TestFrameBufferScaleFrame:
    """Tests for the internal frame scaling helper."""

    def test_scale_same_size_returns_copy(self):
        """Scaling to the same size should return a copy (not the original)."""
        frame = make_test_frame(width=320, height=240, color=(100, 200, 50))
        scaled = FrameBuffer._scale_frame(frame, 320, 240)

        np.testing.assert_array_equal(scaled, frame)
        assert scaled is not frame  # Should be a copy

    def test_scale_preserves_solid_color(self):
        """Scaling a solid-color frame should produce a frame of the same color."""
        frame = make_test_frame(width=100, height=100, color=(42, 84, 168))
        scaled = FrameBuffer._scale_frame(frame, 200, 200)

        assert scaled.shape == (200, 200, 3)
        # Every pixel should have the same color
        for y in range(200):
            for x in range(0, 200, 50):  # Sample pixels
                assert tuple(scaled[y, x]) == (42, 84, 168)

    def test_scale_result_is_contiguous(self):
        """Scaled frame should be a contiguous numpy array for pyvirtualcam."""
        frame = make_test_frame(width=160, height=120, color=(255, 0, 0))
        scaled = FrameBuffer._scale_frame(frame, 320, 240)
        assert scaled.flags['C_CONTIGUOUS']

    def test_scale_half_size(self):
        """Scaling to half size should work correctly."""
        frame = make_test_frame(width=640, height=480, color=(128, 128, 128))
        scaled = FrameBuffer._scale_frame(frame, 320, 240)
        assert scaled.shape == (240, 320, 3)
        assert tuple(scaled[0, 0]) == (128, 128, 128)


# ---------------------------------------------------------------------------
# Test: FramePipeline.handle_resolution_transition() — freeze + reconfigure
# ---------------------------------------------------------------------------

class TestPipelineResolutionTransition:
    """Tests for FramePipeline.handle_resolution_transition()."""

    def test_transition_enters_freeze_frame(self):
        """handle_resolution_transition() should enter freeze-frame mode."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        # Start pipeline with a cycling reader so it stays in STREAMING
        frames = [make_test_frame() for _ in range(20)]
        _idx = [0]
        reader = MagicMock()
        reader.is_running = True
        reader.width = 320
        reader.height = 240
        def _cycling_read():
            f = frames[_idx[0] % len(frames)]
            _idx[0] += 1
            return f
        reader.read_frame = MagicMock(side_effect=_cycling_read)
        vcam = MockVirtualCamera()
        buf = FrameBuffer(width=320, height=240)

        pipeline.start(reader, vcam, frame_buffer=buf)
        time.sleep(0.05)  # Let it start streaming
        assert pipeline.state == PipelineState.STREAMING

        # Trigger resolution transition
        result = pipeline.handle_resolution_transition(640, 480)
        assert result is True
        assert pipeline.state == PipelineState.FREEZE_FRAME

        # Check internal targets updated
        assert pipeline.target_width == 640
        assert pipeline.target_height == 480

        pipeline.stop()

    def test_transition_updates_frame_buffer(self):
        """handle_resolution_transition() should update the FrameBuffer's resolution."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        buf = FrameBuffer(width=320, height=240)
        buf.start()

        # Use the SAME distinctive color for all frames so the buffer
        # always has this color regardless of pipeline thread timing
        distinctive_color = (200, 100, 50)
        frames = [make_test_frame(color=distinctive_color) for _ in range(20)]
        frame_iter = iter(frames)
        reader = MagicMock()
        reader.is_running = True
        reader.width = 320
        reader.height = 240
        reader.read_frame = MagicMock(side_effect=lambda: next(frame_iter, None))
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buf)
        time.sleep(0.05)

        pipeline.handle_resolution_transition(640, 480)

        # Buffer should be at new resolution
        assert buf.width == 640
        assert buf.height == 480

        # Buffer's frame should be scaled to new dimensions
        frame = buf.get_frame()
        assert frame.shape == (480, 640, 3)
        assert tuple(frame[0, 0]) == distinctive_color

        pipeline.stop()

    def test_transition_noop_same_resolution(self):
        """handle_resolution_transition() with same resolution is a no-op."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        # Use a cycling reader so pipeline stays in STREAMING
        frames = [make_test_frame() for _ in range(20)]
        _idx = [0]
        reader = MagicMock()
        reader.is_running = True
        reader.width = 320
        reader.height = 240
        def _cycling_read():
            f = frames[_idx[0] % len(frames)]
            _idx[0] += 1
            return f
        reader.read_frame = MagicMock(side_effect=_cycling_read)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.05)
        assert pipeline.state == PipelineState.STREAMING

        # Same resolution — should stay STREAMING
        result = pipeline.handle_resolution_transition(320, 240)
        assert result is True
        assert pipeline.state == PipelineState.STREAMING

        pipeline.stop()

    def test_transition_updates_fps(self):
        """handle_resolution_transition() with new_fps should update timing."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        assert pipeline.target_fps == 30
        assert pipeline._frame_interval == pytest.approx(1.0 / 30, abs=0.001)

        # Apply transition with new FPS (pipeline doesn't need to be running)
        pipeline.handle_resolution_transition(640, 480, new_fps=60)

        assert pipeline.target_fps == 60
        assert pipeline._frame_interval == pytest.approx(1.0 / 60, abs=0.001)

    def test_transition_without_buffer(self):
        """handle_resolution_transition() should work even without a FrameBuffer."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        # No frame buffer wired in
        frames = [make_test_frame() for _ in range(20)]
        frame_iter = iter(frames)
        reader = MagicMock()
        reader.is_running = True
        reader.width = 320
        reader.height = 240
        reader.read_frame = MagicMock(side_effect=lambda: next(frame_iter, None))
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)  # No buffer
        time.sleep(0.05)

        # Should not crash even without buffer
        result = pipeline.handle_resolution_transition(640, 480)
        assert result is True
        assert pipeline.target_width == 640
        assert pipeline.target_height == 480

        pipeline.stop()


# ---------------------------------------------------------------------------
# Test: Full resolution transition — freeze → swap → live at new resolution
# ---------------------------------------------------------------------------

class TestFullResolutionTransition:
    """End-to-end tests for resolution transition with freeze-frame hold."""

    def test_freeze_hold_during_transition(self):
        """During resolution transition, vcam should receive scaled freeze-frames."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buf = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        # Start with live frames
        live_color = (200, 100, 50)
        live_frames = [make_test_frame(color=live_color) for _ in range(20)]
        frame_iter = iter(live_frames)
        reader1 = MagicMock()
        reader1.is_running = True
        reader1.width = 320
        reader1.height = 240
        reader1.read_frame = MagicMock(side_effect=lambda: next(frame_iter, None))

        pipeline.start(reader1, vcam, frame_buffer=buf)
        time.sleep(0.1)  # Let live frames flow

        frames_before_transition = vcam.frame_count
        assert frames_before_transition > 0

        # Trigger resolution transition (enters freeze, updates buffer)
        pipeline.handle_resolution_transition(640, 480)
        assert pipeline.state == PipelineState.FREEZE_FRAME

        time.sleep(0.2)  # Let freeze frames be pushed

        # Check that frames are still being pushed during freeze
        frames_during_freeze = vcam.frame_count - frames_before_transition
        assert frames_during_freeze > 0, "No frames pushed during resolution transition freeze"

        pipeline.stop()

    def test_no_black_frames_during_transition(self):
        """No black frames should appear during the resolution transition."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buf = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        live_frames = [make_test_frame(color=(180, 90, 45)) for _ in range(20)]
        frame_iter = iter(live_frames)
        reader = MagicMock()
        reader.is_running = True
        reader.width = 320
        reader.height = 240
        reader.read_frame = MagicMock(side_effect=lambda: next(frame_iter, None))

        pipeline.start(reader, vcam, frame_buffer=buf)
        time.sleep(0.1)

        pipeline.handle_resolution_transition(640, 480)
        time.sleep(0.2)

        pipeline.stop()

        # Check NO frames are black
        for i, frame in enumerate(vcam.frames_sent):
            assert not np.all(frame == 0), (
                f"Frame #{i} is black — signal loss during resolution transition!"
            )

    def test_swap_reader_after_transition_resumes_live(self):
        """After resolution transition, swapping reader should resume live video."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buf = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        # Phase 1: Live at 320x240
        old_frames = [make_test_frame(color=(200, 0, 0)) for _ in range(20)]
        frame_iter = iter(old_frames)
        reader1 = MagicMock()
        reader1.is_running = True
        reader1.width = 320
        reader1.height = 240
        reader1.read_frame = MagicMock(side_effect=lambda: next(frame_iter, None))

        pipeline.start(reader1, vcam, frame_buffer=buf)
        time.sleep(0.1)

        # Phase 2: Resolution transition
        pipeline.handle_resolution_transition(640, 480)
        assert pipeline.state == PipelineState.FREEZE_FRAME
        time.sleep(0.1)

        # Phase 3: Swap in new reader at 640x480
        new_color = (0, 200, 0)
        new_frames = [make_test_frame(width=640, height=480, color=new_color)]
        reader2 = MockStreamReader(frames=new_frames, width=640, height=480)
        pipeline.swap_reader(reader2)
        time.sleep(0.3)

        pipeline.stop()

        # Verify new green frames appear after the swap
        green_frames = [
            f for f in vcam.frames_sent
            if f.shape == (480, 640, 3) and tuple(f[0, 0]) == new_color
        ]
        assert len(green_frames) > 0, "No new live frames found after resolution transition swap"

    def test_vcam_never_stops_during_transition(self):
        """Virtual camera should remain running throughout the resolution transition."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buf = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()
        vcam.stop = MagicMock(side_effect=vcam.stop)

        frames = [make_test_frame() for _ in range(20)]
        frame_iter = iter(frames)
        reader = MagicMock()
        reader.is_running = True
        reader.width = 320
        reader.height = 240
        reader.read_frame = MagicMock(side_effect=lambda: next(frame_iter, None))

        pipeline.start(reader, vcam, frame_buffer=buf)
        time.sleep(0.05)

        pipeline.handle_resolution_transition(640, 480)
        time.sleep(0.1)

        # vcam.stop should NOT have been called by the pipeline
        vcam.stop.assert_not_called()
        assert vcam.is_running is True

        pipeline.stop()


# ---------------------------------------------------------------------------
# Test: AppController.change_resolution() integration
# ---------------------------------------------------------------------------

# Global mock for USBEventListener
_usb_listener_patch = patch(
    'usb_event_listener.USBEventListener',
    **{'return_value.start.return_value': False, 'return_value.stop.return_value': None,
       'return_value.is_running': False},
)


class TestAppControllerChangeResolution:
    """Tests for AppController.change_resolution() orchestration."""

    @_usb_listener_patch
    @patch('app_controller.GoProConnection')
    def test_change_resolution_enters_freeze_frame(self, mock_gopro_cls, mock_usb):
        """change_resolution() should enter freeze-frame on the pipeline."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)

        # Set up mocks for pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        # Mock the internal methods
        ctrl._apply_resolution_on_camera = MagicMock(return_value=True)
        ctrl._resume_stream_after_resolution_change = MagicMock(return_value=True)

        ctrl.change_resolution(7)  # 720p

        # Pipeline should have been told to enter freeze-frame
        mock_pipeline.enter_freeze_frame.assert_called_once()

    @_usb_listener_patch
    @patch('app_controller.GoProConnection')
    def test_change_resolution_updates_config_on_success(self, mock_gopro_cls, mock_usb):
        """Successful resolution change should update config dimensions via _resume_stream."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        # Mock apply and resume to succeed
        ctrl._apply_resolution_on_camera = MagicMock(return_value=True)
        ctrl._resume_stream_after_resolution_change = MagicMock(return_value=True)

        result = ctrl.change_resolution(7)  # 720p

        # _resume_stream_after_resolution_change should be called with 720p dims
        ctrl._resume_stream_after_resolution_change.assert_called_once_with(
            1280, 720, 7, config.fov
        )

    @_usb_listener_patch
    @patch('app_controller.GoProConnection')
    def test_change_resolution_emits_status(self, mock_gopro_cls, mock_usb):
        """change_resolution() should emit status messages."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)

        statuses = []
        ctrl.on_status = lambda msg, lvl: statuses.append((msg, lvl))

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True
        ctrl._frame_pipeline = mock_pipeline

        ctrl._apply_resolution_on_camera = MagicMock(return_value=True)
        ctrl._resume_stream_after_resolution_change = MagicMock(return_value=True)

        ctrl.change_resolution(7)  # 720p

        status_texts = [s[0] for s in statuses]
        assert any("1280x720" in t for t in status_texts)

    @_usb_listener_patch
    @patch('app_controller.GoProConnection')
    def test_change_resolution_invalid_code_rejected(self, mock_gopro_cls, mock_usb):
        """change_resolution() with invalid code should return False."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)

        result = ctrl.change_resolution(999)  # Invalid
        assert result is False

    @_usb_listener_patch
    @patch('app_controller.GoProConnection')
    def test_change_resolution_noop_same_settings(self, mock_gopro_cls, mock_usb):
        """change_resolution() with same resolution/fov should be a no-op."""
        from app_controller import AppController
        config = make_test_config()
        config.resolution = 4
        config.fov = 4
        ctrl = AppController(config)

        # Set the GoPro's current resolution to match so change_resolution sees no-op
        ctrl.gopro.current_resolution = 4
        ctrl.gopro.current_fov = 4

        # Same resolution and fov — should return True without doing anything
        result = ctrl.change_resolution(4, fov=4)
        assert result is True
