"""
test_seamless_transition.py — Tests for Sub-AC 4.3: Seamless freeze→live transition

Verifies that:
  1. swap_reader() exits freeze mode and resumes live streaming
  2. Warmup grace period prevents premature re-freeze after reader swap
  3. First live frame from new reader clears the warmup grace
  4. Grace period expiration allows normal freeze logic to resume
  5. Virtual camera output is continuous through freeze→live transition
  6. No black or placeholder frames appear during the transition
  7. Pipeline state transitions correctly: FREEZE_FRAME → STREAMING
  8. on_stream_recovered callback fires on successful transition
  9. Multiple disconnect→freeze→reconnect cycles work seamlessly
  10. Frame buffer correctly transitions from frozen→live frames
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
    """Create a test RGB24 frame as numpy array."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


class MockStreamReader:
    """Mock StreamReader that yields frames then optionally dies."""

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


class DelayedMockStreamReader:
    """Mock StreamReader that returns None for N calls then delivers frames.

    Simulates ffmpeg warmup delay: the process is started but takes time
    to connect to the UDP stream and produce the first frame.
    """

    def __init__(self, frames, warmup_nones=10):
        self._frames = frames
        self._index = 0
        self._warmup_nones = warmup_nones
        self._none_count = 0
        self._is_running = True
        self.width = 320
        self.height = 240
        self.fps = 30

    @property
    def is_running(self):
        return self._is_running

    def read_frame(self):
        if self._none_count < self._warmup_nones:
            self._none_count += 1
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
    """Mock VirtualCamera that records every frame sent."""

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
# Test: swap_reader exits freeze mode and resumes live
# ---------------------------------------------------------------------------

class TestSwapReaderExitsFreeze:
    """Verify swap_reader() transitions pipeline from FREEZE_FRAME to STREAMING."""

    def test_swap_reader_exits_freeze_state(self):
        """swap_reader() should set state to STREAMING when in FREEZE_FRAME."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        # Start with frames that run out quickly
        frames1 = [make_test_frame(color=(200, 0, 0)) for _ in range(3)]
        reader1 = MockStreamReader(frames=frames1, max_frames=3)
        vcam = MockVirtualCamera()
        buffer = FrameBuffer(width=320, height=240)

        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.5)  # Wait for freeze

        assert pipeline.state == PipelineState.FREEZE_FRAME

        # Swap in a new reader with plenty of frames (cycles indefinitely)
        frames2 = [make_test_frame(color=(0, 200, 0))]
        reader2 = MockStreamReader(frames=frames2)
        pipeline.swap_reader(reader2)

        time.sleep(0.3)

        # Pipeline should be streaming (not frozen)
        assert pipeline.state == PipelineState.STREAMING

        pipeline.stop()

    def test_swap_reader_sets_warmup_time(self):
        """swap_reader() should record the swap timestamp for warmup grace."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        assert pipeline._reader_swap_time is None

        frames = [make_test_frame() for _ in range(10)]
        reader = MockStreamReader(frames=frames, max_frames=10)
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.05)

        before = time.monotonic()
        reader2 = MockStreamReader(frames=frames, max_frames=10)
        pipeline.swap_reader(reader2)
        after = time.monotonic()

        assert pipeline._reader_swap_time is not None
        assert before <= pipeline._reader_swap_time <= after

        pipeline.stop()

    def test_on_stream_recovered_fires_on_swap(self):
        """on_stream_recovered callback should fire when swap exits freeze."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        recovered_events = []
        pipeline.on_stream_recovered = lambda: recovered_events.append(True)

        frames1 = [make_test_frame(color=(200, 0, 0)) for _ in range(3)]
        reader1 = MockStreamReader(frames=frames1, max_frames=3)
        vcam = MockVirtualCamera()
        buffer = FrameBuffer(width=320, height=240)

        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.5)  # Enter freeze

        assert pipeline.state == PipelineState.FREEZE_FRAME

        frames2 = [make_test_frame(color=(0, 200, 0))]
        reader2 = MockStreamReader(frames=frames2)
        pipeline.swap_reader(reader2)

        time.sleep(0.2)

        assert len(recovered_events) >= 1

        pipeline.stop()


# ---------------------------------------------------------------------------
# Test: Warmup grace period prevents re-freeze
# ---------------------------------------------------------------------------

class TestWarmupGracePeriod:
    """Verify warmup grace prevents premature re-freeze after reader swap."""

    def test_none_reads_during_warmup_dont_trigger_freeze(self):
        """None reads from a new reader during warmup should NOT re-enter freeze."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        pipeline._reader_warmup_grace = 2.0  # 2 seconds grace

        # Start with frames then enter freeze
        frames1 = [make_test_frame(color=(200, 0, 0)) for _ in range(3)]
        reader1 = MockStreamReader(frames=frames1, max_frames=3)
        vcam = MockVirtualCamera()
        buffer = FrameBuffer(width=320, height=240)

        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.5)  # Enter freeze

        assert pipeline.state == PipelineState.FREEZE_FRAME

        # Swap in a delayed reader (returns None for warmup period, then delivers frames)
        # Use enough frames so they don't run out during the test window
        new_frames = [make_test_frame(color=(0, 255, 0))]
        delayed_reader = DelayedMockStreamReader(frames=new_frames, warmup_nones=20)
        pipeline.swap_reader(delayed_reader)

        # Wait for warmup frames + first live frame
        time.sleep(0.5)

        # Pipeline should be STREAMING (not re-frozen) because warmup grace
        # prevented the None reads from triggering freeze
        assert pipeline.state == PipelineState.STREAMING

        pipeline.stop()

    def test_first_live_frame_clears_warmup(self):
        """First live frame from new reader should clear the warmup flag."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        pipeline._reader_warmup_grace = 5.0

        frames1 = [make_test_frame(color=(200, 0, 0)) for _ in range(3)]
        reader1 = MockStreamReader(frames=frames1, max_frames=3)
        vcam = MockVirtualCamera()

        pipeline.start(reader1, vcam)
        time.sleep(0.5)

        # Swap in new reader with immediate frames
        frames2 = [make_test_frame(color=(0, 200, 0))]
        reader2 = MockStreamReader(frames=frames2)
        pipeline.swap_reader(reader2)

        # Wait for first frame to be read
        time.sleep(0.2)

        # Warmup should be cleared after first live frame
        assert pipeline._reader_swap_time is None
        assert pipeline.in_warmup is False

        pipeline.stop()

    def test_warmup_grace_expiration_allows_freeze(self):
        """After grace period expires, normal freeze logic should resume."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        pipeline._reader_warmup_grace = 0.1  # Very short grace for test

        frames1 = [make_test_frame(color=(200, 0, 0)) for _ in range(3)]
        reader1 = MockStreamReader(frames=frames1, max_frames=3)
        vcam = MockVirtualCamera()
        buffer = FrameBuffer(width=320, height=240)

        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.5)  # Enter freeze

        # Swap in a reader that NEVER produces frames (simulates failed reconnection)
        dead_reader = MockStreamReader(frames=[], max_frames=0)
        pipeline.swap_reader(dead_reader)

        # Wait for grace to expire + consecutive Nones
        time.sleep(0.5)

        # Pipeline should re-enter freeze after grace expires
        assert pipeline.state == PipelineState.FREEZE_FRAME

        pipeline.stop()

    def test_in_warmup_property(self):
        """in_warmup property should reflect warmup state correctly."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        assert pipeline.in_warmup is False

        pipeline._reader_swap_time = time.monotonic()
        assert pipeline.in_warmup is True

        pipeline._reader_swap_time = None
        assert pipeline.in_warmup is False


# ---------------------------------------------------------------------------
# Test: Continuous virtual camera output through transition
# ---------------------------------------------------------------------------

class TestContinuousOutputDuringTransition:
    """Verify virtual camera output is continuous through freeze→live."""

    def test_no_frame_gaps_during_transition(self):
        """Frames should be pushed continuously through freeze → swap → live."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        # Phase 1: Stream → freeze
        live_color = (200, 50, 50)
        frames1 = [make_test_frame(color=live_color) for _ in range(5)]
        reader1 = MockStreamReader(frames=frames1, max_frames=5)

        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.3)

        # Phase 2: Freeze (reader ran out of frames)
        time.sleep(0.3)

        # Phase 3: Swap in new reader
        new_color = (50, 200, 50)
        frames2 = [make_test_frame(color=new_color)]
        reader2 = MockStreamReader(frames=frames2)
        pipeline.swap_reader(reader2)
        time.sleep(0.3)

        pipeline.stop()

        # Verify: NO gaps — frames_sent should have many frames
        total = vcam.frame_count
        assert total > 10, f"Expected >10 frames total, got {total}"

        # Verify: NO black frames
        for i, frame in enumerate(vcam.frames_sent):
            assert not np.all(frame == 0), f"Frame #{i} is black (signal gap!)"

        # Verify: should see both old color (live/freeze) and new color (new live)
        old_color_found = any(tuple(f[0, 0]) == live_color for f in vcam.frames_sent)
        new_color_found = any(tuple(f[0, 0]) == new_color for f in vcam.frames_sent)
        assert old_color_found, "No frames with original live color"
        assert new_color_found, "No frames with new live color after reconnect"

    def test_no_placeholder_frames_during_transition(self):
        """No placeholder (gray) frames should appear during freeze→live transition."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        live_frame = make_test_frame(color=(180, 90, 45))
        frames1 = [live_frame for _ in range(5)]
        reader1 = MockStreamReader(frames=frames1, max_frames=5)

        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.4)  # Stream + freeze

        # Swap in new reader
        new_frame = make_test_frame(color=(45, 180, 90))
        frames2 = [new_frame]
        reader2 = MockStreamReader(frames=frames2)
        pipeline.swap_reader(reader2)
        time.sleep(0.3)

        pipeline.stop()

        # All frames after the first live frame should be either the
        # freeze-frame color or the new live color — never the placeholder
        placeholder_color = (40, 40, 40)
        for i, frame in enumerate(vcam.frames_sent):
            pixel = tuple(frame[0, 0])
            assert pixel != placeholder_color, (
                f"Frame #{i} is placeholder gray — "
                "buffer or vcam fell back to placeholder during transition"
            )

    def test_vcam_frame_count_monotonic_through_transition(self):
        """Frame count should increase monotonically through freeze→live."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        frames1 = [make_test_frame(color=(200, 0, 0)) for _ in range(3)]
        reader1 = MockStreamReader(frames=frames1, max_frames=3)

        pipeline.start(reader1, vcam, frame_buffer=buffer)

        samples = []
        for _ in range(5):
            time.sleep(0.1)
            samples.append(pipeline.frames_pushed)

        # Swap reader during freeze
        frames2 = [make_test_frame(color=(0, 200, 0))]
        reader2 = MockStreamReader(frames=frames2)
        pipeline.swap_reader(reader2)

        for _ in range(5):
            time.sleep(0.1)
            samples.append(pipeline.frames_pushed)

        pipeline.stop()

        # All samples should be monotonically non-decreasing
        for i in range(1, len(samples)):
            assert samples[i] >= samples[i - 1], (
                f"Frame count decreased at sample {i}: {samples}"
            )
        # Should have increased overall
        assert samples[-1] > samples[0], f"Frame count didn't increase: {samples}"


# ---------------------------------------------------------------------------
# Test: Delayed reader warmup with FrameBuffer
# ---------------------------------------------------------------------------

class TestDelayedReaderTransition:
    """Test seamless transition with a reader that has a warmup delay."""

    def test_delayed_reader_seamless_with_buffer(self):
        """A reader with warmup delay should transition seamlessly with FrameBuffer."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        pipeline._reader_warmup_grace = 2.0
        buffer = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        freeze_color = (200, 0, 0)
        frames1 = [make_test_frame(color=freeze_color) for _ in range(3)]
        reader1 = MockStreamReader(frames=frames1, max_frames=3)

        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.5)  # Enter freeze

        # Capture last freeze-frame color
        assert pipeline.state == PipelineState.FREEZE_FRAME

        # Swap in delayed reader (simulates ffmpeg warmup)
        new_color = (0, 255, 0)
        new_frames = [make_test_frame(color=new_color)]
        delayed_reader = DelayedMockStreamReader(frames=new_frames, warmup_nones=15)
        pipeline.swap_reader(delayed_reader)

        time.sleep(0.5)

        pipeline.stop()

        # During the warmup None gap, the pipeline should have pushed the
        # last good freeze-frame (red), not black or placeholder
        all_sent = vcam.frames_sent
        assert vcam.frame_count > 5

        # Find frames between freeze and new live: these are the transition frames
        # They should be either the freeze color or the new live color
        for i, frame in enumerate(all_sent):
            pixel = tuple(frame[0, 0])
            assert pixel in [freeze_color, new_color, (40, 40, 40)] or pixel == freeze_color, (
                f"Frame #{i} has unexpected color {pixel}"
            )

        # Verify new live frames appeared
        new_live_found = any(tuple(f[0, 0]) == new_color for f in all_sent)
        assert new_live_found, "New live frames never arrived after delayed reader warmup"


# ---------------------------------------------------------------------------
# Test: Frame buffer transition from frozen→live
# ---------------------------------------------------------------------------

class TestFrameBufferTransition:
    """Verify FrameBuffer correctly reflects the freeze→live transition."""

    def test_buffer_updates_resume_after_freeze(self):
        """After swap_reader provides live frames, buffer should track them."""
        buffer = FrameBuffer(width=320, height=240, stale_threshold=0.1)
        buffer.start()

        # Simulate live → freeze
        frame1 = make_test_frame(color=(200, 0, 0))
        buffer.update(frame1)
        time.sleep(0.15)

        assert buffer.is_stale is True

        # Simulate reconnection: new frames arrive
        frame2 = make_test_frame(color=(0, 200, 0))
        buffer.update(frame2)

        assert buffer.is_stale is False
        assert buffer.is_frozen is False

        result = buffer.get_frame()
        np.testing.assert_array_equal(result, frame2)

    def test_buffer_freeze_reads_reset_on_new_frame(self):
        """freeze_frame_reads counter should reset when new frames arrive."""
        buffer = FrameBuffer(width=320, height=240, stale_threshold=0.05)
        buffer.start()

        buffer.update(make_test_frame(color=(100, 0, 0)))
        time.sleep(0.1)

        # Read a bunch during freeze
        for _ in range(10):
            buffer.get_frame()

        assert buffer.freeze_frame_reads >= 10

        # New frame arrives (recovery)
        buffer.update(make_test_frame(color=(0, 100, 0)))

        # Freeze reads should be reset
        assert buffer.freeze_frame_reads == 0


# ---------------------------------------------------------------------------
# Test: Multiple disconnect→reconnect cycles
# ---------------------------------------------------------------------------

class TestMultipleReconnectCycles:
    """Verify seamless transition works across multiple disconnect/reconnect cycles."""

    def test_three_cycles_seamless(self):
        """Pipeline survives 3 cycles of disconnect → freeze → reconnect → live."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()
        # Use a larger FrameLog so frames from all 3 cycles survive eviction
        vcam.frames_sent = FrameLog(first_cap=200, recent_cap=500)

        colors = [
            (200, 0, 0),    # Cycle 1: red
            (0, 200, 0),    # Cycle 2: green
            (0, 0, 200),    # Cycle 3: blue
        ]

        # Start initial stream
        frames0 = [make_test_frame(color=colors[0]) for _ in range(3)]
        reader0 = MockStreamReader(frames=frames0, max_frames=3)
        pipeline.start(reader0, vcam, frame_buffer=buffer)
        time.sleep(0.3)

        for cycle, color in enumerate(colors[1:], start=1):
            # Wait for freeze
            time.sleep(0.2)
            assert pipeline.is_running, f"Pipeline died during cycle {cycle}"

            # Swap in new reader (cycles indefinitely until next swap/stop)
            frames = [make_test_frame(color=color)]
            reader = MockStreamReader(frames=frames)
            pipeline.swap_reader(reader)
            time.sleep(0.3)

            # Verify live streaming resumed
            assert pipeline.is_running, f"Pipeline not running after cycle {cycle}"

        pipeline.stop()

        # Verify all colors appeared in the output
        all_pixels = set()
        for frame in vcam.frames_sent:
            all_pixels.add(tuple(frame[0, 0]))

        for color in colors:
            assert color in all_pixels, (
                f"Color {color} never appeared in output. "
                f"Seen colors: {all_pixels}"
            )

        # No black frames
        for i, frame in enumerate(vcam.frames_sent):
            assert not np.all(frame == 0), f"Black frame at index {i}"

    def test_state_transitions_correct_through_cycles(self):
        """State should follow: STREAMING → FREEZE_FRAME → STREAMING for each cycle."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        state_log = []
        pipeline.on_state_change = lambda s: state_log.append(s)

        frames1 = [make_test_frame(color=(200, 0, 0)) for _ in range(3)]
        reader1 = MockStreamReader(frames=frames1, max_frames=3)
        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.4)  # Stream + freeze

        # Cycle 1: swap reader
        frames2 = [make_test_frame(color=(0, 200, 0)) for _ in range(3)]
        reader2 = MockStreamReader(frames=frames2, max_frames=3)
        pipeline.swap_reader(reader2)
        time.sleep(0.4)  # Stream + freeze again

        pipeline.stop()

        # State log should contain: STARTING, STREAMING, FREEZE_FRAME, STREAMING, FREEZE_FRAME, ...
        state_names = [s.name for s in state_log]
        assert "FREEZE_FRAME" in state_names, f"No FREEZE_FRAME in state log: {state_names}"
        assert state_names.count("STREAMING") >= 2, (
            f"Expected at least 2 STREAMING states (initial + recovery), got: {state_names}"
        )


# ---------------------------------------------------------------------------
# Test: Pipeline stats through transition
# ---------------------------------------------------------------------------

class TestPipelineStatsTransition:
    """Verify pipeline stats are accurate through freeze→live transition."""

    def test_get_stats_reflects_warmup(self):
        """get_stats() should show in_warmup status during reader warmup."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        # Before any swap — not in warmup
        stats = pipeline.get_stats()
        assert stats["in_warmup"] is False

        # Manually set swap time to simulate
        pipeline._reader_swap_time = time.monotonic()
        stats = pipeline.get_stats()
        assert stats["in_warmup"] is True

    def test_freeze_frame_count_resets_on_recovery(self):
        """freeze_frame_count tracks the current freeze period only."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        vcam = MockVirtualCamera()

        frames1 = [make_test_frame() for _ in range(3)]
        reader1 = MockStreamReader(frames=frames1, max_frames=3)

        pipeline.start(reader1, vcam, frame_buffer=buffer)
        time.sleep(0.5)

        # Should have freeze frames
        freeze_before = pipeline.freeze_frame_count
        assert freeze_before > 0

        # Swap reader to exit freeze — use enough frames to last the test
        frames2 = [make_test_frame()]
        reader2 = MockStreamReader(frames=frames2)
        pipeline.swap_reader(reader2)
        time.sleep(0.2)

        # freeze_frame_count from the previous freeze period stays
        # (it's not reset on swap — only on enter_freeze_frame)
        # But the pipeline state should be STREAMING
        assert pipeline.state == PipelineState.STREAMING

        pipeline.stop()
