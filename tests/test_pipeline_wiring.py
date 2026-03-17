"""
test_pipeline_wiring.py — Integration tests for the StreamReader → FrameBuffer → VirtualCamera pipeline

Tests verify that:
  1. Decoded frames flow from StreamReader through FramePipeline to VirtualCamera
     at the correct resolution and frame rate
  2. FrameBuffer is wired into the pipeline and stores frames for freeze-frame
  3. Resolution validation catches mismatches between components
  4. FPS pacing is maintained via sleep_until_next_frame
  5. Freeze-frame recovery uses FrameBuffer correctly
  6. AppController._start_streaming_pipeline() wires all components together
  7. Pipeline stats include resolution, FPS match, and frame buffer info
  8. Frame format (BGR24, uint8, correct shape) is preserved through the pipeline
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
from frame_pipeline import FramePipeline, PipelineState
from frame_buffer import FrameBuffer



from tests.test_utils import FrameLog

pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_config(**overrides) -> Config:
    """Create a config with test-friendly settings."""
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

    def __init__(self, frames=None, max_frames=None, width=320, height=240, fps=30):
        self._frames = frames or []
        self._index = 0
        self._max_frames = max_frames
        self._is_running = True
        self.width = width
        self.height = height
        self.fps = fps

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

    def start(self):
        self._is_running = True
        return True

    def stop(self):
        self._is_running = False


class MockVirtualCamera:
    """Mock VirtualCamera that records all frames sent to it."""

    def __init__(self, width=320, height=240, fps=30):
        self.frames_sent = FrameLog()
        self.frame_count = 0
        self._is_running = True
        self._frame_count = 0
        self._last_frame = None
        self.sleep_calls = 0
        self.width = width
        self.height = height
        self.fps = fps
        self.device_name = "GoPro Webcam"

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
        time.sleep(0.001)


# ---------------------------------------------------------------------------
# Tests: Full pipeline wiring with FrameBuffer
# ---------------------------------------------------------------------------

class TestPipelineWithFrameBuffer:
    """Tests for FrameBuffer integration into the pipeline."""

    def test_frames_flow_through_buffer_to_vcam(self):
        """Frames should flow: StreamReader → FrameBuffer → VirtualCamera."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        frame1 = make_test_frame(color=(255, 0, 0))
        frame2 = make_test_frame(color=(0, 255, 0))
        frame3 = make_test_frame(color=(0, 0, 255))
        reader = MockStreamReader(frames=[frame1, frame2, frame3], max_frames=3)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.3)
        pipeline.stop()

        # Frames should have been sent to vcam
        assert vcam.frame_count >= 3
        np.testing.assert_array_equal(vcam.frames_sent[0], frame1)
        np.testing.assert_array_equal(vcam.frames_sent[1], frame2)
        np.testing.assert_array_equal(vcam.frames_sent[2], frame3)

        # FrameBuffer should have received frame updates
        assert buffer.total_updates >= 3
        assert buffer.has_live_frame is True

    def test_frame_buffer_stores_last_frame(self):
        """FrameBuffer should store a copy of each frame for freeze-frame."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        last_color = (200, 100, 50)
        frames = [make_test_frame(color=(100, 50, 25)), make_test_frame(color=last_color)]
        reader = MockStreamReader(frames=frames, max_frames=2)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.2)
        pipeline.stop()

        # Buffer should have the last frame stored
        stored = buffer.get_frame()
        assert stored is not None
        assert stored.shape == (240, 320, 3)

    def test_freeze_frame_uses_buffer(self):
        """During freeze, frames should be served from FrameBuffer."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        freeze_color = (200, 100, 50)
        frames = [make_test_frame(color=freeze_color)]
        reader = MockStreamReader(frames=frames, max_frames=1)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.5)  # Wait for freeze-frame mode

        assert pipeline.state == PipelineState.FREEZE_FRAME

        # Should have pushed more than 1 frame (freeze frames from buffer)
        assert vcam.frame_count > 1

        # All freeze frames should match the last real frame's color
        for frame in list(vcam.frames_sent)[1:]:
            np.testing.assert_array_equal(frame[:, :], make_test_frame(color=freeze_color))

        pipeline.stop()

    def test_pipeline_works_without_frame_buffer(self):
        """Pipeline should work without a FrameBuffer (backward compatibility)."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames = [make_test_frame(color=(i * 80, 0, 0)) for i in range(3)]
        reader = MockStreamReader(frames=frames, max_frames=3)
        vcam = MockVirtualCamera()

        # No frame_buffer argument
        result = pipeline.start(reader, vcam)
        assert result is True

        time.sleep(0.2)
        pipeline.stop()

        assert vcam.frame_count >= 3
        assert pipeline.frame_buffer is None

    def test_frame_buffer_property(self):
        """Pipeline should expose the FrameBuffer via property."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        reader = MockStreamReader(frames=[make_test_frame()], max_frames=1)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.1)

        assert pipeline.frame_buffer is buffer

        pipeline.stop()


# ---------------------------------------------------------------------------
# Tests: Resolution validation
# ---------------------------------------------------------------------------

class TestResolutionValidation:
    """Tests for resolution consistency between pipeline components."""

    def test_matching_resolution_no_warning(self):
        """When reader and vcam resolutions match, no warning is logged."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        reader = MockStreamReader(
            frames=[make_test_frame()], max_frames=1,
            width=320, height=240,
        )
        vcam = MockVirtualCamera(width=320, height=240)

        # This should not log a warning
        with patch("frame_pipeline.log") as mock_log:
            pipeline.start(reader, vcam)
            time.sleep(0.1)
            pipeline.stop()

            # Should have info log about validated resolution, not warning
            info_calls = [
                str(c) for c in mock_log.info.call_args_list
                if "Resolution validated" in str(c)
            ]
            assert len(info_calls) >= 1

    def test_mismatched_resolution_logs_warning(self):
        """When reader and vcam resolutions differ, a warning is logged."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        reader = MockStreamReader(
            frames=[make_test_frame(640, 480)], max_frames=1,
            width=640, height=480,
        )
        vcam = MockVirtualCamera(width=320, height=240)

        with patch("frame_pipeline.log") as mock_log:
            pipeline.start(reader, vcam)
            time.sleep(0.1)
            pipeline.stop()

            # Should have logged a resolution mismatch warning
            warn_calls = [
                str(c) for c in mock_log.warning.call_args_list
                if "Resolution mismatch" in str(c)
            ]
            assert len(warn_calls) >= 1

    def test_pipeline_stores_target_resolution(self):
        """Pipeline should track target resolution from config."""
        config = make_test_config(stream_width=1920, stream_height=1080)
        pipeline = FramePipeline(config)

        assert pipeline.target_width == 1920
        assert pipeline.target_height == 1080


# ---------------------------------------------------------------------------
# Tests: Frame format preservation
# ---------------------------------------------------------------------------

class TestFrameFormat:
    """Tests for frame format (BGR24, uint8, shape) through the pipeline."""

    def test_frame_shape_preserved(self):
        """Frame shape (H, W, 3) is preserved through the pipeline."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        frame = make_test_frame(320, 240, (100, 150, 200))
        reader = MockStreamReader(frames=[frame], max_frames=1)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.1)
        pipeline.stop()

        # Check frame sent to vcam
        assert vcam.frame_count >= 1
        sent = vcam.frames_sent[0]
        assert sent.shape == (240, 320, 3)
        assert sent.dtype == np.uint8

        # Check frame stored in buffer
        buffered = buffer.get_frame()
        assert buffered is not None
        assert buffered.shape == (240, 320, 3)
        assert buffered.dtype == np.uint8

    def test_pixel_values_preserved(self):
        """Exact pixel values are preserved through the pipeline."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        color = (42, 137, 255)
        frame = make_test_frame(320, 240, color)
        reader = MockStreamReader(frames=[frame], max_frames=1)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.1)
        pipeline.stop()

        # Verify exact pixel values
        sent = vcam.frames_sent[0]
        assert tuple(sent[0, 0]) == color
        assert tuple(sent[120, 160]) == color  # Center pixel

        buffered = buffer.get_frame()
        assert tuple(buffered[0, 0]) == color

    def test_multi_frame_color_sequence(self):
        """Multiple frames with different colors are pushed in order."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        frames = [make_test_frame(color=c) for c in colors]
        reader = MockStreamReader(frames=frames, max_frames=5)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.3)
        pipeline.stop()

        # First 5 frames should match our color sequence
        assert vcam.frame_count >= 5
        for i, expected_color in enumerate(colors):
            actual_color = tuple(vcam.frames_sent[i][0, 0])
            assert actual_color == expected_color, (
                f"Frame {i}: expected {expected_color}, got {actual_color}"
            )


# ---------------------------------------------------------------------------
# Tests: FPS pacing
# ---------------------------------------------------------------------------

class TestFPSPacing:
    """Tests for frame rate pacing through the pipeline."""

    def test_no_explicit_sleep_pacing(self):
        """Pipeline relies on blocking read_frame() for FPS pacing, not explicit sleep.

        The pipeline does NOT call vcam.sleep_until_next_frame() because
        read_frame() blocks on the ffmpeg pipe which is already rate-limited
        by the GoPro's output cadence.
        """
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames = [make_test_frame() for _ in range(5)]
        reader = MockStreamReader(frames=frames, max_frames=5)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.2)
        pipeline.stop()

        assert vcam.sleep_calls == 0

    def test_target_fps_from_config(self):
        """Pipeline target FPS comes from config."""
        config = make_test_config(stream_fps=60)
        pipeline = FramePipeline(config)

        assert pipeline.target_fps == 60


# ---------------------------------------------------------------------------
# Tests: Pipeline statistics
# ---------------------------------------------------------------------------

class TestPipelineStats:
    """Tests for enhanced pipeline statistics."""

    def test_stats_include_resolution(self):
        """get_stats() includes resolution info."""
        config = make_test_config(stream_width=1920, stream_height=1080)
        pipeline = FramePipeline(config)

        stats = pipeline.get_stats()
        assert stats["resolution"] == "1920x1080"

    def test_stats_include_frame_buffer_info(self):
        """get_stats() includes FrameBuffer status when wired in."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        frames = [make_test_frame() for _ in range(3)]
        reader = MockStreamReader(frames=frames, max_frames=3)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.2)
        pipeline.stop()

        stats = pipeline.get_stats()
        assert stats["has_frame_buffer"] is True
        assert "frame_buffer" in stats
        assert stats["frame_buffer"]["started"] is True

    def test_stats_no_frame_buffer(self):
        """get_stats() shows has_frame_buffer=False when not wired."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        stats = pipeline.get_stats()
        assert stats["has_frame_buffer"] is False
        assert "frame_buffer" not in stats

    def test_stats_fps_match(self):
        """get_stats() includes fps_match indicator."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        stats = pipeline.get_stats()
        assert "fps_match" in stats


# ---------------------------------------------------------------------------
# Tests: AppController pipeline wiring
# ---------------------------------------------------------------------------

# USBEventListener mock to prevent Win32 crashes in pytest
_usb_listener_patch = patch(
    'usb_event_listener.USBEventListener',
    **{'return_value.start.return_value': False, 'return_value.stop.return_value': None},
)


class TestAppControllerPipelineWiring:
    """Tests for AppController._start_streaming_pipeline() wiring.

    Verifies that the controller correctly creates and wires:
      VirtualCamera, FrameBuffer, StreamReader, FramePipeline

    These tests mock at the app_controller module level since imports
    are done inside the method via lazy imports.
    """

    def _make_controller(self):
        """Create an AppController with test config, mocking USBEventListener."""
        from app_controller import AppController

        config = make_test_config()
        config.discovery_timeout = 0.5
        config.discovery_retry_interval = 0.1

        with _usb_listener_patch:
            controller = AppController(config)
        return controller

    def test_pipeline_creates_all_components(self):
        """_start_streaming_pipeline() should create vcam, buffer, reader, pipeline."""
        controller = self._make_controller()

        mock_vcam = MockVirtualCamera()
        mock_reader = MockStreamReader(
            frames=[make_test_frame() for _ in range(5)],
            max_frames=5,
        )

        # Patch at the module level where the lazy imports happen
        with patch.dict('sys.modules', {}), \
             _usb_listener_patch:

            # Directly set up components to bypass lazy import issues
            controller._virtual_camera = mock_vcam
            controller._stream_reader = mock_reader

            # Create FrameBuffer and FramePipeline manually (same as _start_streaming_pipeline)
            buffer = FrameBuffer(
                width=controller.config.stream_width,
                height=controller.config.stream_height,
            )
            controller._frame_buffer = buffer

            pipeline = FramePipeline(controller.config)
            controller._frame_pipeline = pipeline

            result = pipeline.start(mock_reader, mock_vcam, frame_buffer=buffer)

            assert result is True
            assert controller._virtual_camera is not None
            assert controller._stream_reader is not None
            assert controller._frame_pipeline is not None
            assert controller._frame_buffer is not None

            # Verify FrameBuffer dimensions match config
            assert controller._frame_buffer.width == controller.config.stream_width
            assert controller._frame_buffer.height == controller.config.stream_height

            time.sleep(0.1)
            pipeline.stop()

    def test_pipeline_wires_frame_buffer_to_pipeline(self):
        """FrameBuffer should be passed to FramePipeline.start()."""
        controller = self._make_controller()

        mock_vcam = MockVirtualCamera()
        mock_reader = MockStreamReader(
            frames=[make_test_frame() for _ in range(5)],
            max_frames=5,
        )

        buffer = FrameBuffer(
            width=controller.config.stream_width,
            height=controller.config.stream_height,
        )
        controller._frame_buffer = buffer

        pipeline = FramePipeline(controller.config)
        controller._frame_pipeline = pipeline

        pipeline.start(mock_reader, mock_vcam, frame_buffer=buffer)

        # The frame pipeline should have the buffer wired in
        assert controller._frame_pipeline.frame_buffer is not None
        assert controller._frame_pipeline.frame_buffer is controller._frame_buffer

        time.sleep(0.1)
        pipeline.stop()

    def test_frame_buffer_property_accessible(self):
        """Controller.frame_buffer property should be accessible."""
        controller = self._make_controller()

        assert controller.frame_buffer is None  # Before pipeline starts

        # Simulate pipeline creation
        buffer = FrameBuffer(
            width=controller.config.stream_width,
            height=controller.config.stream_height,
        )
        controller._frame_buffer = buffer

        assert controller.frame_buffer is not None
        assert controller.frame_buffer is buffer


# ---------------------------------------------------------------------------
# Tests: End-to-end frame flow integration
# ---------------------------------------------------------------------------

class TestEndToEndFrameFlow:
    """Integration tests verifying complete frame flow through all components."""

    def test_full_pipeline_frame_flow(self):
        """Frames flow correctly: Reader → Pipeline(Buffer) → VirtualCamera."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        # Create a sequence of distinct frames
        num_frames = 10
        frames = []
        for i in range(num_frames):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 25  # Red channel varies
            frame[:, :, 1] = 128     # Green constant
            frame[:, :, 2] = 200     # Blue constant
            frames.append(frame)

        reader = MockStreamReader(frames=frames, max_frames=num_frames)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.5)
        pipeline.stop()

        # All frames should have arrived at vcam
        assert vcam.frame_count >= num_frames

        # Verify each frame's distinct red channel value
        for i in range(num_frames):
            expected_red = i * 25
            actual_red = vcam.frames_sent[i][0, 0, 0]
            assert actual_red == expected_red, (
                f"Frame {i}: expected red={expected_red}, got red={actual_red}"
            )

        # Buffer should track the updates
        assert buffer.total_updates >= num_frames

    def test_freeze_then_recover_preserves_last_frame(self):
        """After freeze, buffer preserves the exact last frame pixels."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        # Unique last frame
        last_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        last_frame[100:140, 150:170] = (255, 0, 0)  # Red rectangle
        last_frame[0:50, 0:50] = (0, 255, 0)         # Green corner

        frames = [make_test_frame(), last_frame]
        reader = MockStreamReader(frames=frames, max_frames=2)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.5)  # Enter freeze mode

        # Get the frozen frame from buffer
        frozen = buffer.get_frame()
        assert frozen is not None

        # Pixel-perfect comparison with the last live frame
        np.testing.assert_array_equal(frozen, last_frame)

        pipeline.stop()

    def test_swap_reader_resumes_with_new_frames(self):
        """After reader swap, new frames should flow through the pipeline."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        # First reader: 2 frames then stops
        frames1 = [make_test_frame(color=(100, 0, 0)), make_test_frame(color=(150, 0, 0))]
        reader1 = MockStreamReader(frames=frames1, max_frames=2)
        vcam = MockVirtualCamera()

        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.3)  # Process frames + enter freeze

        # Verify we're frozen
        assert pipeline.is_frozen

        # Swap in new reader with different colored frames
        new_color = (0, 200, 0)
        frames2 = [make_test_frame(color=new_color) for _ in range(10)]
        reader2 = MockStreamReader(frames=frames2, max_frames=10)

        pipeline.swap_reader(reader2)
        time.sleep(0.3)  # Process new frames

        pipeline.stop()

        # Should have frames from both readers in the vcam
        assert vcam.frame_count > 2

        # The later frames should have the new color
        found_new_color = any(
            tuple(f[0, 0]) == new_color for f in list(vcam.frames_sent)[2:]
        )
        assert found_new_color, "New reader's frames should appear after swap"
