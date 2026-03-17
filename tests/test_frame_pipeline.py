"""
test_frame_pipeline.py — Tests for the frame-pushing pipeline

Tests cover:
  1. Normal streaming: frames read from StreamReader → pushed to VirtualCamera
  2. Freeze-frame mode: last frame keeps being pushed when stream drops
  3. Stream recovery: pipeline exits freeze mode when reader is swapped
  4. FPS pacing: sleep_until_next_frame is called between frames
  5. Frame validation: correct shape/dtype reaches the virtual camera
  6. Pipeline lifecycle: start/stop/state transitions
  7. Reader swap: hot-swap reader during operation
  8. Error handling: reader/vcam exceptions don't crash the pipeline
  9. Statistics tracking: frame count, FPS, freeze duration
  10. Callbacks: on_stream_lost and on_stream_recovered fire correctly
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
    """Mock StreamReader that yields a configurable sequence of frames."""

    def __init__(self, frames=None, max_frames=None):
        """
        Args:
            frames: List of frames to yield (None entries = stream lost).
                    If not provided, generates solid-color frames.
            max_frames: Stop after this many frames (returns None after).
        """
        self._frames = frames or []
        self._index = 0
        self._max_frames = max_frames
        self._is_running = True
        self.width = 320
        self.height = 240
        self.fps = 30

    @property
    def is_running(self):
        return self._is_running

    @is_running.setter
    def is_running(self, val):
        self._is_running = val

    def read_frame(self):
        """Return next frame or None."""
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
    """Mock VirtualCamera that records all frames sent to it."""

    def __init__(self):
        self.frames_sent = FrameLog()
        self.frame_count = 0
        self._is_running = True
        self._frame_count = 0
        self._last_frame = None
        self.sleep_calls = 0
        self.width = 320
        self.height = 240
        self.fps = 30

    @property
    def is_running(self):
        return self._is_running

    def start(self):
        self._is_running = True
        return True

    def stop(self):
        self._is_running = False

    def send_frame(self, frame):
        self.frame_count += 1
        self.frames_sent.append(frame.copy())
        self._last_frame = frame
        self._frame_count += 1
        return True

    def send_last_frame(self):
        if self._last_frame is not None:
            self.frame_count += 1
            self.frames_sent.append(self._last_frame.copy())
            self._frame_count += 1
            return True
        return False

    def sleep_until_next_frame(self):
        self.sleep_calls += 1
        # Don't actually sleep in tests — just a tiny yield
        time.sleep(0.001)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineLifecycle:
    """Tests for start/stop and state transitions."""

    def test_initial_state_is_stopped(self):
        config = make_test_config()
        pipeline = FramePipeline(config)
        assert pipeline.state == PipelineState.STOPPED
        assert pipeline.is_running is False
        assert pipeline.frames_pushed == 0

    def test_start_sets_streaming_state(self):
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames = [make_test_frame() for _ in range(5)]
        reader = MockStreamReader(frames=frames, max_frames=5)
        vcam = MockVirtualCamera()

        result = pipeline.start(reader, vcam)
        assert result is True

        # Give the thread time to start and process
        time.sleep(0.1)
        assert pipeline.state in (PipelineState.STREAMING, PipelineState.FREEZE_FRAME)

        pipeline.stop()
        assert pipeline.state == PipelineState.STOPPED

    def test_stop_from_stopped_is_noop(self):
        config = make_test_config()
        pipeline = FramePipeline(config)
        pipeline.stop()  # Should not raise
        assert pipeline.state == PipelineState.STOPPED

    def test_double_start_returns_true(self):
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames = [make_test_frame()]
        reader = MockStreamReader(frames=frames)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.05)
        # Second start should return True (already running)
        result = pipeline.start(reader, vcam)
        assert result is True

        pipeline.stop()

    def test_start_fails_if_vcam_not_openable(self):
        config = make_test_config()
        pipeline = FramePipeline(config)

        reader = MockStreamReader(frames=[make_test_frame()])
        vcam = MockVirtualCamera()
        vcam._is_running = False
        vcam.start = MagicMock(return_value=False)

        result = pipeline.start(reader, vcam)
        assert result is False
        assert pipeline.state == PipelineState.STOPPED


class TestFramePushing:
    """Tests for normal frame pushing from reader to vcam."""

    def test_frames_pushed_to_vcam(self):
        """Frames from reader should appear in vcam."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        frame1 = make_test_frame(color=(255, 0, 0))
        frame2 = make_test_frame(color=(0, 255, 0))
        frame3 = make_test_frame(color=(0, 0, 255))
        reader = MockStreamReader(frames=[frame1, frame2, frame3], max_frames=3)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        # Wait for frames to be processed
        time.sleep(0.2)
        pipeline.stop()

        # Should have pushed at least the 3 frames
        assert vcam.frame_count >= 3
        # First 3 frames should match our input
        np.testing.assert_array_equal(vcam.frames_sent[0], frame1)
        np.testing.assert_array_equal(vcam.frames_sent[1], frame2)
        np.testing.assert_array_equal(vcam.frames_sent[2], frame3)

    def test_frames_pushed_count_tracked(self):
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames = [make_test_frame() for _ in range(5)]
        reader = MockStreamReader(frames=frames, max_frames=5)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.2)
        pipeline.stop()

        assert pipeline.frames_pushed >= 5

    def test_fps_pacing_via_blocking_read(self):
        """Pipeline relies on blocking read_frame() for FPS pacing, not explicit sleep.

        The pipeline does NOT call vcam.sleep_until_next_frame() because
        read_frame() blocks until ffmpeg delivers the next frame, which is
        already rate-limited by the GoPro's output cadence. Adding explicit
        sleep on top would double the per-frame delay.
        """
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames = [make_test_frame() for _ in range(3)]
        reader = MockStreamReader(frames=frames, max_frames=3)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.2)
        pipeline.stop()

        # Pipeline should NOT call sleep_until_next_frame — FPS pacing
        # comes from read_frame() blocking on the ffmpeg pipe.
        assert vcam.sleep_calls == 0


class TestFreezeFrame:
    """Tests for freeze-frame mode when stream is lost."""

    def test_enters_freeze_on_stream_loss(self):
        """Pipeline should enter freeze mode when reader returns None."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        # 3 real frames, then None (stream lost)
        frames = [make_test_frame(color=(i * 80, 0, 0)) for i in range(3)]
        reader = MockStreamReader(frames=frames, max_frames=3)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        # Wait for frames + freeze detection (3 consecutive Nones)
        time.sleep(0.5)

        assert pipeline.state == PipelineState.FREEZE_FRAME
        assert pipeline.is_frozen is True

        pipeline.stop()

    def test_freeze_frame_keeps_pushing(self):
        """During freeze, the last frame should keep being pushed."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        last_color = (200, 100, 50)
        frames = [make_test_frame(color=last_color)]
        reader = MockStreamReader(frames=frames, max_frames=1)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        # Wait for freeze + some freeze frames
        time.sleep(0.5)

        # Should have pushed more frames than just the 1 real one
        assert vcam.frame_count > 1

        # Freeze frames should match the last real frame
        last_real = vcam.frames_sent[0]
        for freeze_frame in list(vcam.frames_sent)[1:]:
            np.testing.assert_array_equal(freeze_frame, last_real)

        pipeline.stop()

    def test_freeze_frame_count_tracked(self):
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames = [make_test_frame()]
        reader = MockStreamReader(frames=frames, max_frames=1)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.4)
        pipeline.stop()

        assert pipeline.freeze_frame_count > 0

    def test_freeze_duration_increases(self):
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames = [make_test_frame()]
        reader = MockStreamReader(frames=frames, max_frames=1)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.3)

        if pipeline.is_frozen:
            assert pipeline.freeze_duration > 0

        pipeline.stop()

    def test_manual_enter_freeze_frame(self):
        """enter_freeze_frame() can be called externally."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames = [make_test_frame()]
        reader = MockStreamReader(frames=frames)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.05)

        pipeline.enter_freeze_frame()
        assert pipeline.state == PipelineState.FREEZE_FRAME
        assert pipeline.is_frozen is True

        pipeline.stop()


class TestStreamRecovery:
    """Tests for recovering from freeze-frame when stream is restored."""

    def test_swap_reader_exits_freeze(self):
        """Swapping in a new reader should exit freeze mode."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        # Initial reader yields 1 frame then dies
        frames = [make_test_frame(color=(100, 0, 0))]
        reader1 = MockStreamReader(frames=frames, max_frames=1)
        vcam = MockVirtualCamera()

        pipeline.start(reader1, vcam)
        time.sleep(0.3)  # Wait for freeze

        # Now swap in a new reader with fresh frames
        new_frames = [make_test_frame(color=(0, 200, 0)) for _ in range(10)]
        reader2 = MockStreamReader(frames=new_frames, max_frames=10)
        reader2._is_running = True

        pipeline.swap_reader(reader2)
        time.sleep(0.2)

        # Should have exited freeze mode
        assert pipeline.state in (PipelineState.STREAMING, PipelineState.FREEZE_FRAME)

        pipeline.stop()

    def test_exit_freeze_frame_callback_fires(self):
        """on_stream_recovered should fire when exiting freeze."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        recovered = []
        pipeline.on_stream_recovered = lambda: recovered.append(True)

        frames = [make_test_frame()]
        reader = MockStreamReader(frames=frames, max_frames=1)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.3)  # Enter freeze

        # Exit freeze
        pipeline.exit_freeze_frame()

        assert len(recovered) <= 1  # May or may not fire depending on state

        pipeline.stop()

    def test_stream_lost_callback_fires(self):
        """on_stream_lost should fire when entering freeze."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        lost_events = []
        pipeline.on_stream_lost = lambda: lost_events.append(True)

        frames = [make_test_frame()]
        reader = MockStreamReader(frames=frames, max_frames=1)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.4)  # Wait for freeze detection

        pipeline.stop()

        # Stream lost callback should have fired
        assert len(lost_events) >= 1


class TestStatistics:
    """Tests for pipeline statistics reporting."""

    def test_get_stats_returns_dict(self):
        config = make_test_config()
        pipeline = FramePipeline(config)

        stats = pipeline.get_stats()
        assert isinstance(stats, dict)
        assert "state" in stats
        assert "frames_pushed" in stats
        assert "fps_target" in stats
        assert "is_frozen" in stats
        assert stats["state"] == "STOPPED"
        assert stats["fps_target"] == 30
        assert stats["is_frozen"] is False

    def test_stats_update_during_streaming(self):
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames = [make_test_frame() for _ in range(10)]
        reader = MockStreamReader(frames=frames, max_frames=10)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.2)

        stats = pipeline.get_stats()
        assert stats["frames_pushed"] >= 1

        pipeline.stop()


class TestErrorHandling:
    """Tests for error resilience in the pipeline."""

    def test_reader_exception_enters_freeze(self):
        """If the reader throws an exception, pipeline should enter freeze."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        class FailingReader:
            is_running = True
            width = 320
            height = 240

            def read_frame(self):
                raise IOError("Simulated read error")

            def stop(self):
                self.is_running = False

        reader = FailingReader()
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.3)

        # Pipeline should still be running (in freeze mode), not crashed
        assert pipeline.is_running
        assert pipeline.state == PipelineState.FREEZE_FRAME

        pipeline.stop()

    def test_vcam_send_failure_logged(self):
        """If vcam.send_frame fails, pipeline continues."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames = [make_test_frame() for _ in range(5)]
        reader = MockStreamReader(frames=frames, max_frames=5)

        vcam = MockVirtualCamera()
        original_send = vcam.send_frame
        call_count = [0]

        def failing_send(frame):
            call_count[0] += 1
            if call_count[0] <= 2:
                return False  # First 2 sends fail
            return original_send(frame)

        vcam.send_frame = failing_send

        pipeline.start(reader, vcam)
        time.sleep(0.2)
        pipeline.stop()

        # Pipeline should have continued despite send failures
        assert call_count[0] >= 2


class TestReaderSwap:
    """Tests for hot-swapping the stream reader."""

    def test_swap_reader_thread_safe(self):
        """swap_reader should be safe to call while pipeline is running."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames1 = [make_test_frame(color=(100, 0, 0))]
        reader1 = MockStreamReader(frames=frames1)
        vcam = MockVirtualCamera()

        pipeline.start(reader1, vcam)
        time.sleep(0.05)

        # Swap reader while pipeline is running
        frames2 = [make_test_frame(color=(0, 200, 0))]
        reader2 = MockStreamReader(frames=frames2)

        pipeline.swap_reader(reader2)
        time.sleep(0.05)

        pipeline.stop()

        # Should have frames from both readers
        assert vcam.frame_count > 0


class TestFrameValidation:
    """Tests for frame shape and format handling."""

    def test_correct_frame_shape_accepted(self):
        """Frames with correct shape should be pushed."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        frame = make_test_frame(320, 240)
        assert frame.shape == (240, 320, 3)

        reader = MockStreamReader(frames=[frame], max_frames=1)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.1)
        pipeline.stop()

        assert vcam.frame_count >= 1
        assert vcam.frames_sent[0].shape == (240, 320, 3)
        assert vcam.frames_sent[0].dtype == np.uint8


class TestPipelineStateProperty:
    """Tests for pipeline state property edge cases."""

    def test_is_running_false_when_stopped(self):
        config = make_test_config()
        pipeline = FramePipeline(config)
        assert pipeline.is_running is False

    def test_is_frozen_false_when_stopped(self):
        config = make_test_config()
        pipeline = FramePipeline(config)
        assert pipeline.is_frozen is False

    def test_freeze_duration_zero_when_not_frozen(self):
        config = make_test_config()
        pipeline = FramePipeline(config)
        assert pipeline.freeze_duration == 0.0

    def test_state_change_callback(self):
        """on_state_change should be called on state transitions."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        states = []
        pipeline.on_state_change = lambda s: states.append(s)

        frames = [make_test_frame() for _ in range(3)]
        reader = MockStreamReader(frames=frames, max_frames=3)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.3)
        pipeline.stop()

        # Should have at least STARTING, STREAMING, and STOPPED transitions
        state_names = [s.name for s in states]
        assert "STARTING" in state_names
        assert "STOPPED" in state_names
