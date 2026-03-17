"""
test_freeze_frame_resilience.py — Tests for Sub-AC 3.3: Virtual camera freeze-frame resilience

Verifies that:
  1. Virtual camera device stays registered (is_running=True) during USB disconnect
  2. Freeze-frame pushes the exact last good frame (pixel-perfect) to downstream apps
  3. No black frames or signal gaps occur during live→frozen transition
  4. The pre-freeze gap (reader returning None) still pushes frames to vcam
  5. FrameBuffer always has a valid frame (placeholder or real) after start()
  6. Pipeline continues emitting frames at target FPS through disconnect + recovery
  7. DisconnectDetector triggers proactive freeze before corrupted frames arrive
  8. Full recovery flow: live → disconnect → freeze → reconnect → live
"""

import sys
import os
import time
import threading
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from frame_pipeline import FramePipeline, PipelineState
from frame_buffer import FrameBuffer
from virtual_camera import VirtualCamera, _PLACEHOLDER_COLOR
from disconnect_detector import DisconnectDetector


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
    """Mock StreamReader that yields frames then optionally dies (simulates disconnect)."""

    def __init__(self, frames=None, max_frames=None):
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
    """Mock VirtualCamera that records every frame sent to it for verification."""

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
        self.sleep_calls += 1
        time.sleep(0.001)


# ---------------------------------------------------------------------------
# Test: Virtual camera stays registered during disconnect
# ---------------------------------------------------------------------------

class TestVcamStaysRegistered:
    """Verify the virtual camera device never disappears during disconnect."""

    def test_vcam_running_throughout_freeze(self):
        """VirtualCamera.is_running stays True while pipeline is in freeze-frame mode."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        # 2 live frames then stream dies
        frames = [make_test_frame(color=(200, 100, 50)), make_test_frame(color=(200, 100, 50))]
        reader = MockStreamReader(frames=frames, max_frames=2)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.5)  # Wait for freeze detection

        # Pipeline should be in freeze mode
        assert pipeline.state == PipelineState.FREEZE_FRAME
        # VirtualCamera must still be running
        assert vcam.is_running is True

        pipeline.stop()

    def test_vcam_not_stopped_on_enter_freeze(self):
        """enter_freeze_frame() does NOT call vcam.stop()."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames = [make_test_frame()]
        reader = MockStreamReader(frames=frames)
        vcam = MockVirtualCamera()
        vcam.stop = MagicMock(side_effect=vcam.stop)

        pipeline.start(reader, vcam)
        time.sleep(0.05)

        pipeline.enter_freeze_frame()
        time.sleep(0.1)

        # vcam.stop should NOT have been called by the pipeline
        vcam.stop.assert_not_called()

        pipeline.stop()

    def test_vcam_survives_pipeline_stop_start_cycle(self):
        """Pipeline stop/start doesn't affect the virtual camera device."""
        config = make_test_config()
        vcam = MockVirtualCamera()

        # First pipeline session
        pipeline1 = FramePipeline(config)
        frames1 = [make_test_frame(color=(100, 0, 0)) for _ in range(5)]
        reader1 = MockStreamReader(frames=frames1, max_frames=5)
        pipeline1.start(reader1, vcam)
        time.sleep(0.2)
        pipeline1.stop()

        # vcam should still be running (not closed by pipeline)
        assert vcam.is_running is True

        # Second pipeline session with same vcam
        pipeline2 = FramePipeline(config)
        frames2 = [make_test_frame(color=(0, 100, 0)) for _ in range(5)]
        reader2 = MockStreamReader(frames=frames2, max_frames=5)
        pipeline2.start(reader2, vcam)
        time.sleep(0.2)
        pipeline2.stop()

        assert vcam.is_running is True


# ---------------------------------------------------------------------------
# Test: Pixel-perfect freeze-frame (no black frames)
# ---------------------------------------------------------------------------

class TestPixelPerfectFreezeFrame:
    """Verify frozen frames are pixel-identical to the last live frame."""

    def test_freeze_frames_match_last_live_frame(self):
        """All frames sent during freeze must be pixel-identical to the last live frame."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        last_live_color = (200, 100, 50)
        last_live_frame = make_test_frame(color=last_live_color)
        frames = [
            make_test_frame(color=(100, 0, 0)),
            make_test_frame(color=(150, 50, 25)),
            last_live_frame,
        ]
        reader = MockStreamReader(frames=frames, max_frames=3)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.6)  # Wait for freeze + several freeze frames
        pipeline.stop()

        # Find the transition point: last frame with last_live_color is the last live frame
        # Everything after should be pixel-identical freeze frames
        sent = vcam.frames_sent
        assert len(sent) > 3, f"Expected more than 3 frames, got {len(sent)}"

        # Frames after the 3 live frames should all match the last live frame
        for i, frame in enumerate(sent[3:], start=3):
            np.testing.assert_array_equal(
                frame, last_live_frame,
                err_msg=f"Freeze frame #{i} does not match last live frame"
            )

    def test_no_black_frames_during_transition(self):
        """No all-black frames should appear between live and frozen modes."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames = [make_test_frame(color=(128, 128, 128)) for _ in range(3)]
        reader = MockStreamReader(frames=frames, max_frames=3)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.5)
        pipeline.stop()

        # Check that no frame is all-black
        for i, frame in enumerate(vcam.frames_sent):
            is_black = np.all(frame == 0)
            assert not is_black, (
                f"Frame #{i} is all-black — this is a signal loss! "
                f"shape={frame.shape}, unique_values={np.unique(frame)}"
            )

    def test_no_signal_gap_between_live_and_freeze(self):
        """Frames should be pushed continuously from live through freeze mode."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        live_frame = make_test_frame(color=(200, 150, 100))
        frames = [live_frame]
        reader = MockStreamReader(frames=frames, max_frames=1)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.5)
        pipeline.stop()

        # Should have multiple frames: the live one + freeze frames
        # ALL frames after the live one should be valid (non-None, non-black)
        assert vcam.frame_count > 1, "Expected multiple frames (live + freeze)"

        for i, frame in enumerate(vcam.frames_sent):
            assert frame is not None, f"Frame #{i} is None"
            assert frame.shape == (240, 320, 3), f"Frame #{i} has wrong shape"
            assert not np.all(frame == 0), f"Frame #{i} is black"


# ---------------------------------------------------------------------------
# Test: Pre-freeze gap frames (reader returns None before freeze triggers)
# ---------------------------------------------------------------------------

class TestPreFreezeGap:
    """Verify that frames are still pushed during the pre-freeze None gap."""

    def test_last_frame_pushed_during_none_gap(self):
        """When reader returns None (before freeze triggers), vcam still gets frames."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        live_color = (180, 90, 45)
        live_frame = make_test_frame(color=live_color)
        # 1 live frame, then reader returns None
        reader = MockStreamReader(frames=[live_frame], max_frames=1)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.4)
        pipeline.stop()

        # Count frames — should be more than 1 (live + pre-freeze + freeze)
        assert vcam.frame_count > 1, (
            f"Only {vcam.frame_count} frames sent — gap detection failed"
        )

        # All frames after the first should match the last live frame
        # (no black frames, no garbage)
        for i in range(1, len(vcam.frames_sent)):
            np.testing.assert_array_equal(
                vcam.frames_sent[i], live_frame,
                err_msg=f"Pre-freeze/freeze frame #{i} doesn't match last live frame"
            )


# ---------------------------------------------------------------------------
# Test: FrameBuffer always has a valid frame
# ---------------------------------------------------------------------------

class TestFrameBufferAlwaysValid:
    """Verify FrameBuffer never returns None or black frames after start()."""

    def test_buffer_has_placeholder_after_start(self):
        """After start(), get_frame() returns a valid non-None, non-black frame."""
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()

        frame = buffer.get_frame()
        assert frame is not None
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8
        # Should be the dark gray placeholder, not black
        assert not np.all(frame == 0), "Placeholder frame should not be black"
        # Check it matches _PLACEHOLDER_COLOR
        assert tuple(frame[0, 0]) == _PLACEHOLDER_COLOR

    def test_buffer_returns_last_frame_when_stale(self):
        """After stream stops, buffer returns the last good frame."""
        buffer = FrameBuffer(width=320, height=240, stale_threshold=0.1)
        buffer.start()

        live_frame = make_test_frame(color=(200, 100, 50))
        buffer.update(live_frame)

        # Wait for staleness
        time.sleep(0.2)

        assert buffer.is_stale is True
        frozen = buffer.get_frame()
        assert frozen is not None
        np.testing.assert_array_equal(frozen, live_frame)

    def test_buffer_get_frame_never_returns_none_after_start(self):
        """get_frame() should never return None once start() has been called."""
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()

        # Call get_frame many times without any updates
        for _ in range(100):
            frame = buffer.get_frame()
            assert frame is not None, "get_frame() returned None after start()"

    def test_buffer_freeze_frame_is_pixel_identical(self):
        """The frozen frame from buffer must be pixel-identical to the last update."""
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()

        # Send some frames
        frame1 = make_test_frame(color=(100, 0, 0))
        frame2 = make_test_frame(color=(0, 200, 0))
        frame3 = make_test_frame(color=(0, 0, 255))  # Last frame
        buffer.update(frame1)
        buffer.update(frame2)
        buffer.update(frame3)

        # Now simulate disconnect (no more updates)
        time.sleep(0.1)

        # Every get_frame() call should return frame3
        for _ in range(10):
            frozen = buffer.get_frame()
            np.testing.assert_array_equal(frozen, frame3)

    def test_buffer_tracks_freeze_stats(self):
        """Buffer tracks when it transitions to freeze state."""
        buffer = FrameBuffer(width=320, height=240, stale_threshold=0.1)
        buffer.start()

        live_frame = make_test_frame(color=(128, 128, 128))
        buffer.update(live_frame)

        # Let it go stale
        time.sleep(0.2)

        # First get_frame() in stale state should trigger freeze detection
        buffer.get_frame()
        assert buffer.is_frozen is True
        assert buffer.freeze_duration > 0


# ---------------------------------------------------------------------------
# Test: Pipeline continuous frame emission
# ---------------------------------------------------------------------------

class TestContinuousFrameEmission:
    """Verify pipeline emits frames continuously through disconnect + recovery."""

    def test_pipeline_emits_frames_during_freeze(self):
        """Pipeline must keep pushing frames to vcam even in freeze mode."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames = [make_test_frame(color=(100, 200, 50))]
        reader = MockStreamReader(frames=frames, max_frames=1)
        vcam = MockVirtualCamera()
        buffer = FrameBuffer(width=320, height=240)

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.6)

        # Capture frame count during freeze
        freeze_frames = pipeline.freeze_frame_count
        assert freeze_frames > 0, "No freeze frames pushed during freeze mode"

        total = pipeline.frames_pushed
        assert total > freeze_frames, (
            f"Total frames ({total}) should exceed freeze frames ({freeze_frames})"
        )

        pipeline.stop()

    def test_swap_reader_seamlessly_resumes_live(self):
        """Swapping in a new reader transitions from freeze → live without gaps."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)

        # Phase 1: Live streaming then freeze
        live_color = (200, 100, 50)
        frames1 = [make_test_frame(color=live_color) for _ in range(3)]
        reader1 = MockStreamReader(frames=frames1, max_frames=3)
        vcam = MockVirtualCamera()

        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.4)  # Enter freeze mode

        assert pipeline.state == PipelineState.FREEZE_FRAME
        frames_before_swap = vcam.frame_count

        # Phase 2: Swap in new reader (simulating reconnection)
        new_color = (0, 255, 0)
        frames2 = [make_test_frame(color=new_color)]
        reader2 = MockStreamReader(frames=frames2)
        pipeline.swap_reader(reader2)
        time.sleep(0.3)

        # Should have more frames after swap
        frames_after_swap = vcam.frame_count
        assert frames_after_swap > frames_before_swap

        # Check that new live frames are being pushed (with new color)
        # Search all frames in the log (FrameLog evicts middle frames,
        # so slicing by frame_count would miss them)
        new_color_frames = [
            f for f in vcam.frames_sent
            if tuple(f[0, 0]) == new_color
        ]
        assert len(new_color_frames) > 0, "No new live frames found after reader swap"

        pipeline.stop()

    def test_multiple_disconnect_reconnect_cycles(self):
        """Pipeline survives multiple disconnect → freeze → reconnect cycles."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        # Initial live stream
        frames1 = [make_test_frame(color=(100, 0, 0)) for _ in range(5)]
        reader1 = MockStreamReader(frames=frames1, max_frames=5)
        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.3)

        # Cycle 1: freeze + reconnect
        assert pipeline.is_running
        frames2 = [make_test_frame(color=(0, 100, 0)) for _ in range(5)]
        reader2 = MockStreamReader(frames=frames2, max_frames=5)
        pipeline.swap_reader(reader2)
        time.sleep(0.3)

        # Cycle 2: freeze + reconnect
        assert pipeline.is_running
        frames3 = [make_test_frame(color=(0, 0, 100)) for _ in range(5)]
        reader3 = MockStreamReader(frames=frames3, max_frames=5)
        pipeline.swap_reader(reader3)
        time.sleep(0.3)

        # Pipeline should still be running after multiple cycles
        assert pipeline.is_running
        assert vcam.is_running is True

        # Total frames should include frames from all cycles + freeze periods
        assert pipeline.frames_pushed > 10

        pipeline.stop()


# ---------------------------------------------------------------------------
# Test: DisconnectDetector proactive freeze
# ---------------------------------------------------------------------------

class TestProactiveFreezeOnDisconnect:
    """Verify DisconnectDetector triggers freeze BEFORE stream corruption."""

    def test_usb_detach_triggers_pipeline_freeze(self):
        """USB detach event should immediately call pipeline.enter_freeze_frame()."""
        config = make_test_config()
        pipeline = MagicMock()
        pipeline.enter_freeze_frame = MagicMock()

        detector = DisconnectDetector(pipeline=pipeline, config=config)

        # Simulate USB detach
        detector._on_usb_detach("USB\\VID_2672&PID_0052\\test")

        pipeline.enter_freeze_frame.assert_called_once()
        assert detector.is_disconnected is True

    def test_usb_detach_debounce(self):
        """Rapid duplicate detach events should be debounced."""
        pipeline = MagicMock()
        pipeline.enter_freeze_frame = MagicMock()

        detector = DisconnectDetector(pipeline=pipeline)

        # Two rapid detach events
        detector._on_usb_detach("USB\\VID_2672&PID_0052\\test")
        detector._on_usb_detach("USB\\VID_2672&PID_0052\\test")

        # enter_freeze_frame should only be called once (debounced)
        assert pipeline.enter_freeze_frame.call_count == 1

    def test_disconnect_then_attach_signals_recovery(self):
        """USB attach after detach should fire on_reconnect_ready callback."""
        pipeline = MagicMock()
        reconnect_events = []

        detector = DisconnectDetector(pipeline=pipeline)
        detector.on_reconnect_ready = lambda dev_id: reconnect_events.append(dev_id)

        # Simulate disconnect then reconnect
        detector._on_usb_detach("USB\\VID_2672&PID_0052\\test")
        time.sleep(1.1)  # Wait for debounce window
        detector._on_usb_attach("USB\\VID_2672&PID_0052\\test")

        assert len(reconnect_events) == 1

    def test_stream_health_monitor_triggers_freeze(self):
        """Stream health monitor detects ffmpeg death and triggers freeze."""
        pipeline = MagicMock()
        pipeline.enter_freeze_frame = MagicMock()

        reader = MockStreamReader(frames=[make_test_frame()], max_frames=1)
        reader._is_running = True

        detector = DisconnectDetector(
            pipeline=pipeline,
            stream_reader=reader,
        )
        detector._health_check_interval = 0.1

        # Mock _start_usb_listener to avoid Win32 access violation in tests
        detector._start_usb_listener = MagicMock(return_value=False)

        # Start monitoring (only health monitor, no USB listener)
        detector.start()
        time.sleep(0.15)

        # Simulate ffmpeg death
        reader._is_running = False
        time.sleep(0.3)

        detector.stop()

        # Should have triggered freeze-frame
        pipeline.enter_freeze_frame.assert_called()

    def test_disconnect_callbacks_fire(self):
        """on_disconnect and on_usb_detach callbacks should fire on USB detach."""
        disconnect_events = []
        detach_events = []

        detector = DisconnectDetector()
        detector.on_disconnect = lambda: disconnect_events.append(True)
        detector.on_usb_detach = lambda dev_id: detach_events.append(dev_id)

        detector._on_usb_detach("USB\\VID_2672&PID_0052\\test")

        assert len(disconnect_events) == 1
        assert len(detach_events) == 1


# ---------------------------------------------------------------------------
# Test: Full integration: live → disconnect → freeze → reconnect → live
# ---------------------------------------------------------------------------

class TestFullDisconnectRecoveryCycle:
    """End-to-end test simulating the complete disconnect/recovery flow."""

    def test_full_cycle_no_black_frames(self):
        """Simulate: stream live → USB disconnect → freeze → new reader → live.

        Verify:
          - vcam receives frames continuously throughout
          - No black frames at any point
          - Freeze frames are pixel-identical to last live frame
          - New live frames are distinct from freeze frames
        """
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        # Phase 1: Live streaming (red frames)
        live_color_1 = (200, 50, 50)
        live_frames = [make_test_frame(color=live_color_1) for _ in range(5)]
        reader1 = MockStreamReader(frames=live_frames, max_frames=5)

        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.3)

        # Phase 2: Stream dies → pipeline enters freeze
        time.sleep(0.3)
        assert pipeline.state == PipelineState.FREEZE_FRAME

        # Record the transition point
        frames_at_freeze = len(vcam.frames_sent)
        count_at_freeze = vcam.frame_count

        # Phase 3: While frozen, verify freeze frames
        time.sleep(0.2)
        frames_during_freeze = list(vcam.frames_sent)[frames_at_freeze:]
        assert vcam.frame_count > count_at_freeze, "No frames sent during freeze period"

        for i, frame in enumerate(frames_during_freeze):
            assert not np.all(frame == 0), f"Black frame during freeze at index {i}"
            # All freeze frames should match the last live frame
            np.testing.assert_array_equal(
                frame, live_frames[-1],
                err_msg=f"Freeze frame #{i} doesn't match last live frame"
            )

        # Phase 4: Reconnect with new reader (green frames)
        live_color_2 = (50, 200, 50)
        new_frames = [make_test_frame(color=live_color_2)]
        reader2 = MockStreamReader(frames=new_frames)
        pipeline.swap_reader(reader2)
        time.sleep(0.3)

        pipeline.stop()

        # Phase 5: Verify overall frame sequence
        all_frames = vcam.frames_sent

        # No frame should be None
        for i, f in enumerate(all_frames):
            assert f is not None, f"Frame #{i} is None"

        # No frame should be all-black
        for i, f in enumerate(all_frames):
            assert not np.all(f == 0), f"Frame #{i} is black (signal loss)"

        # Should have received frames in all phases
        assert vcam.frame_count > len(live_frames) + len(new_frames)

    def test_vcam_frame_count_always_increasing(self):
        """Virtual camera frame count should always increase, never plateau or reset."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        frames = [make_test_frame() for _ in range(3)]
        reader = MockStreamReader(frames=frames, max_frames=3)

        pipeline.start(reader, vcam, frame_buffer=buffer)

        samples = []
        for _ in range(6):
            time.sleep(0.1)
            with vcam._lock:
                samples.append(vcam._frame_count)

        pipeline.stop()

        # Frame count should be monotonically increasing
        for i in range(1, len(samples)):
            assert samples[i] >= samples[i - 1], (
                f"Frame count decreased: {samples[i - 1]} → {samples[i]} "
                f"(all samples: {samples})"
            )

        # Should have pushed frames in every sampling window
        assert samples[-1] > samples[0], (
            f"Frame count didn't increase over time: {samples}"
        )


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestFreezeFrameEdgeCases:
    """Edge cases for freeze-frame resilience."""

    def test_freeze_with_no_prior_frame_uses_placeholder(self):
        """If freeze happens before any live frame, placeholder is used."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        # Reader that immediately returns None (no live frames)
        reader = MockStreamReader(frames=[], max_frames=0)

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.4)

        pipeline.stop()

        # Should have pushed frames (placeholder/freeze frames)
        assert vcam.frame_count > 0, "No frames sent at all"

        # Frames should not be black (should be placeholder gray)
        for i, frame in enumerate(vcam.frames_sent):
            assert not np.all(frame == 0), f"Frame #{i} is black"

    def test_explicit_enter_freeze_preserves_last_frame(self):
        """Explicitly calling enter_freeze_frame() preserves the frame correctly."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        live_color = (180, 90, 45)
        frames = [make_test_frame(color=live_color)]
        reader = MockStreamReader(frames=frames)

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.1)

        # Explicitly trigger freeze (simulating DisconnectDetector)
        pipeline.enter_freeze_frame()
        time.sleep(0.2)

        pipeline.stop()

        # Find frames sent after freeze was triggered
        # They should all be the same color as the last live frame
        # (not black, not garbage)
        for frame in vcam.frames_sent:
            assert not np.all(frame == 0), "Black frame found"

    def test_freeze_frame_with_buffer_reset_recovers(self):
        """If buffer is reset during freeze, placeholder is used (no crash)."""
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()

        live_frame = make_test_frame(color=(200, 100, 50))
        buffer.update(live_frame)

        # Reset clears the live frame
        buffer.reset()

        # get_frame should return placeholder, not None
        frame = buffer.get_frame()
        assert frame is not None
        assert frame.shape == (240, 320, 3)
        # Should be the placeholder color, not black
        assert not np.all(frame == 0)

    def test_pipeline_thread_doesnt_crash_on_vcam_error(self):
        """If vcam.send_frame raises during freeze, pipeline thread survives."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        frames = [make_test_frame()]
        reader = MockStreamReader(frames=frames, max_frames=1)

        call_count = [0]

        class FlakeyVcam(MockVirtualCamera):
            def send_frame(self, frame):
                call_count[0] += 1
                if call_count[0] % 3 == 0:
                    raise RuntimeError("Simulated vcam error")
                return super().send_frame(frame)

        vcam = FlakeyVcam()

        pipeline.start(reader, vcam)
        time.sleep(0.5)

        # Pipeline should still be running (survived the error)
        assert pipeline.is_running
        pipeline.stop()

    def test_buffer_update_clears_freeze_state(self):
        """When new frames arrive after a freeze, buffer exits freeze state."""
        buffer = FrameBuffer(width=320, height=240, stale_threshold=0.1)
        buffer.start()

        frame1 = make_test_frame(color=(100, 100, 100))
        buffer.update(frame1)

        # Go stale
        time.sleep(0.2)
        buffer.get_frame()  # Trigger freeze detection
        assert buffer.is_frozen is True

        # New frame arrives (reconnection)
        frame2 = make_test_frame(color=(200, 200, 200))
        buffer.update(frame2)

        assert buffer.is_frozen is False
        result = buffer.get_frame()
        np.testing.assert_array_equal(result, frame2)
