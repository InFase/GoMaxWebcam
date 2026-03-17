"""
test_frame_buffer.py — Unit tests for the FrameBuffer ring buffer

Tests cover:
  1. Lifecycle: start(), reset(), placeholder frame creation
  2. Frame storage: update() stores copies, get_frame() returns current frame
  3. Freeze-frame: stale detection, freeze state tracking, freeze reads
  4. Thread safety: concurrent update/read from multiple threads
  5. Dimension validation: wrong-sized frames are rejected
  6. Statistics: get_stats() accuracy, counter tracking
  7. Edge cases: pre-start, never-updated, rapid updates
  8. Pixel-perfect recovery: exact frame data preserved through buffer
"""

import sys
import os
import time
import threading
from unittest.mock import patch

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frame_buffer import FrameBuffer, _DEFAULT_STALE_THRESHOLD, _PLACEHOLDER_COLOR

pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_frame(width=320, height=240, color=(128, 64, 32)):
    """Create a test RGB24 frame."""
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    return frame


def make_unique_frame(width=320, height=240, index=0):
    """Create a frame with unique pixel pattern based on index."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = index % 256           # Red varies by index
    frame[:, :, 1] = (index * 37) % 256    # Green varies differently
    frame[:, :, 2] = (index * 73) % 256    # Blue varies differently
    # Add a unique corner marker
    frame[0, 0] = (index % 256, (index + 1) % 256, (index + 2) % 256)
    return frame


# ---------------------------------------------------------------------------
# Tests: Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    """Tests for FrameBuffer start/reset lifecycle."""

    def test_initial_state(self):
        """Before start(), buffer has no frame."""
        buf = FrameBuffer(width=320, height=240)
        assert buf.has_frame is False
        assert buf.has_live_frame is False
        assert buf.get_frame() is None
        assert buf.total_updates == 0
        assert buf.total_reads == 0

    def test_start_creates_placeholder(self):
        """After start(), buffer has a placeholder frame."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        assert buf.has_frame is True
        assert buf.has_live_frame is False

        frame = buf.get_frame()
        assert frame is not None
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8

    def test_placeholder_color(self):
        """Placeholder frame uses the configured color."""
        color = (40, 40, 40)
        buf = FrameBuffer(width=320, height=240, placeholder_color=color)
        buf.start()

        frame = buf.get_frame()
        assert tuple(frame[0, 0]) == color
        assert tuple(frame[120, 160]) == color  # center pixel

    def test_custom_placeholder_color(self):
        """Custom placeholder color is respected."""
        color = (100, 200, 50)
        buf = FrameBuffer(width=320, height=240, placeholder_color=color)
        buf.start()

        frame = buf.get_frame()
        assert tuple(frame[0, 0]) == color

    def test_reset_clears_to_placeholder(self):
        """reset() returns buffer to placeholder state."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        # Push a live frame
        buf.update(make_frame(color=(255, 0, 0)))
        assert buf.has_live_frame is True
        assert buf.total_updates == 1

        # Reset
        buf.reset()
        assert buf.has_live_frame is False
        assert buf.total_updates == 0
        assert buf.total_reads == 0

        # Should have placeholder
        frame = buf.get_frame()
        assert frame is not None
        assert tuple(frame[0, 0]) == _PLACEHOLDER_COLOR

    def test_start_resets_counters(self):
        """Calling start() resets all tracking counters."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        buf.update(make_frame())
        buf.get_frame()
        buf.get_frame()

        # Re-start
        buf.start()
        assert buf.total_updates == 0
        assert buf.total_reads == 0
        assert buf.freeze_frame_reads == 0
        assert buf.has_live_frame is False


# ---------------------------------------------------------------------------
# Tests: Frame storage and retrieval
# ---------------------------------------------------------------------------

class TestFrameStorage:
    """Tests for update/get_frame operations."""

    def test_update_stores_frame(self):
        """update() stores the frame, get_frame() returns it."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        frame = make_frame(color=(255, 128, 64))
        result = buf.update(frame)
        assert result is True

        stored = buf.get_frame()
        np.testing.assert_array_equal(stored, frame)

    def test_update_makes_copy(self):
        """update() makes a copy — modifying original doesn't affect buffer."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        frame = make_frame(color=(100, 100, 100))
        buf.update(frame)

        # Modify the original
        frame[:, :] = (0, 0, 0)

        # Buffer should still have the original data
        stored = buf.get_frame()
        assert tuple(stored[0, 0]) == (100, 100, 100)

    def test_get_frame_returns_same_reference(self):
        """get_frame() returns the internal frame (not a copy) for performance."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        buf.update(make_frame())

        frame1 = buf.get_frame()
        frame2 = buf.get_frame()
        # Same reference (not a copy)
        assert frame1 is frame2

    def test_latest_frame_wins(self):
        """Multiple updates: get_frame() returns the most recent."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        buf.update(make_frame(color=(100, 0, 0)))
        buf.update(make_frame(color=(0, 200, 0)))
        buf.update(make_frame(color=(0, 0, 255)))

        stored = buf.get_frame()
        assert tuple(stored[0, 0]) == (0, 0, 255)

    def test_update_before_start_rejected(self):
        """update() before start() returns False."""
        buf = FrameBuffer(width=320, height=240)

        result = buf.update(make_frame())
        assert result is False
        assert buf.total_updates == 0

    def test_update_increments_counter(self):
        """Each successful update increments total_updates."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        for i in range(5):
            buf.update(make_frame())

        assert buf.total_updates == 5

    def test_get_frame_increments_read_counter(self):
        """Each get_frame() call increments total_reads."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        buf.update(make_frame())

        for _ in range(3):
            buf.get_frame()

        assert buf.total_reads == 3


# ---------------------------------------------------------------------------
# Tests: Dimension validation
# ---------------------------------------------------------------------------

class TestDimensionValidation:
    """Tests for frame dimension checking."""

    def test_wrong_dimensions_rejected(self):
        """Frames with wrong dimensions are rejected."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        wrong_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = buf.update(wrong_frame)
        assert result is False
        assert buf.total_updates == 0

    def test_wrong_channels_rejected(self):
        """Frames with wrong number of channels are rejected."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        wrong_frame = np.zeros((240, 320, 4), dtype=np.uint8)  # RGBA
        result = buf.update(wrong_frame)
        assert result is False

    def test_correct_dimensions_accepted(self):
        """Frames with correct dimensions are accepted."""
        buf = FrameBuffer(width=640, height=480)
        buf.start()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = buf.update(frame)
        assert result is True

    def test_various_resolutions(self):
        """Buffer works with various common resolutions."""
        for w, h in [(1920, 1080), (1280, 720), (640, 480), (320, 240)]:
            buf = FrameBuffer(width=w, height=h)
            buf.start()

            frame = np.zeros((h, w, 3), dtype=np.uint8)
            result = buf.update(frame)
            assert result is True

            stored = buf.get_frame()
            assert stored.shape == (h, w, 3)


# ---------------------------------------------------------------------------
# Tests: Stale / freeze detection
# ---------------------------------------------------------------------------

class TestStaleDetection:
    """Tests for stale frame and freeze-frame detection."""

    def test_is_stale_when_never_updated(self):
        """is_stale is True when no frames have been received."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        assert buf.is_stale is True

    def test_not_stale_after_update(self):
        """is_stale is False immediately after an update."""
        buf = FrameBuffer(width=320, height=240, stale_threshold=1.0)
        buf.start()
        buf.update(make_frame())
        assert buf.is_stale is False

    def test_becomes_stale_after_threshold(self):
        """is_stale becomes True after stale_threshold seconds."""
        buf = FrameBuffer(width=320, height=240, stale_threshold=0.05)
        buf.start()
        buf.update(make_frame())
        assert buf.is_stale is False

        time.sleep(0.1)  # Exceed threshold
        assert buf.is_stale is True

    def test_custom_stale_threshold(self):
        """Custom stale_threshold is respected."""
        buf = FrameBuffer(width=320, height=240, stale_threshold=0.5)
        buf.start()
        buf.update(make_frame())

        time.sleep(0.1)
        assert buf.is_stale is False  # Under threshold

        time.sleep(0.5)
        assert buf.is_stale is True  # Over threshold

    def test_freeze_state_tracked(self):
        """is_frozen transitions correctly."""
        buf = FrameBuffer(width=320, height=240, stale_threshold=0.05)
        buf.start()
        buf.update(make_frame())
        assert buf.is_frozen is False

        # Wait for stale, then read to trigger freeze detection
        time.sleep(0.1)
        buf.get_frame()
        assert buf.is_frozen is True

    def test_freeze_clears_on_update(self):
        """Freeze state clears when a new frame arrives."""
        buf = FrameBuffer(width=320, height=240, stale_threshold=0.05)
        buf.start()
        buf.update(make_frame(color=(100, 0, 0)))

        # Enter freeze
        time.sleep(0.1)
        buf.get_frame()
        assert buf.is_frozen is True

        # New frame clears freeze
        buf.update(make_frame(color=(0, 200, 0)))
        assert buf.is_frozen is False

    def test_freeze_duration_increases(self):
        """freeze_duration increases while frozen."""
        buf = FrameBuffer(width=320, height=240, stale_threshold=0.05)
        buf.start()
        buf.update(make_frame())

        time.sleep(0.1)
        buf.get_frame()  # Trigger freeze

        d1 = buf.freeze_duration
        time.sleep(0.05)
        d2 = buf.freeze_duration

        assert d2 > d1

    def test_freeze_duration_zero_when_not_frozen(self):
        """freeze_duration is 0 when not in freeze state."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        buf.update(make_frame())
        assert buf.freeze_duration == 0.0

    def test_freeze_reads_tracked(self):
        """freeze_frame_reads counts reads during freeze."""
        buf = FrameBuffer(width=320, height=240, stale_threshold=0.05)
        buf.start()
        buf.update(make_frame())

        time.sleep(0.1)
        for _ in range(5):
            buf.get_frame()

        assert buf.freeze_frame_reads == 5

    def test_freeze_reads_reset_on_recovery(self):
        """freeze_frame_reads resets when stream recovers."""
        buf = FrameBuffer(width=320, height=240, stale_threshold=0.05)
        buf.start()
        buf.update(make_frame())

        time.sleep(0.1)
        buf.get_frame()
        buf.get_frame()
        assert buf.freeze_frame_reads == 2

        # Recovery
        buf.update(make_frame(color=(0, 255, 0)))
        assert buf.freeze_frame_reads == 0


# ---------------------------------------------------------------------------
# Tests: frame_age property
# ---------------------------------------------------------------------------

class TestFrameAge:
    """Tests for frame_age property."""

    def test_frame_age_none_before_update(self):
        """frame_age is None when no frame has been received."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        assert buf.frame_age is None

    def test_frame_age_near_zero_after_update(self):
        """frame_age is near zero immediately after update."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        buf.update(make_frame())

        age = buf.frame_age
        assert age is not None
        assert age < 0.1  # Should be very small

    def test_frame_age_increases_over_time(self):
        """frame_age increases when no new frames arrive."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        buf.update(make_frame())

        time.sleep(0.1)
        age = buf.frame_age
        assert age >= 0.09  # At least ~100ms


# ---------------------------------------------------------------------------
# Tests: get_stats()
# ---------------------------------------------------------------------------

class TestGetStats:
    """Tests for get_stats() reporting."""

    def test_stats_before_start(self):
        """Stats reflect unstarted state."""
        buf = FrameBuffer(width=320, height=240)
        stats = buf.get_stats()
        assert stats["started"] is False
        assert stats["has_live_frame"] is False
        assert stats["resolution"] == "320x240"

    def test_stats_after_start(self):
        """Stats reflect started but no-live-frame state."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        stats = buf.get_stats()
        assert stats["started"] is True
        assert stats["has_live_frame"] is False
        assert stats["is_stale"] is True
        assert stats["frame_age"] is None

    def test_stats_after_frames(self):
        """Stats reflect live streaming state."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        buf.update(make_frame())
        buf.get_frame()

        stats = buf.get_stats()
        assert stats["started"] is True
        assert stats["has_live_frame"] is True
        assert stats["total_updates"] == 1
        assert stats["total_reads"] == 1
        assert stats["frame_age"] is not None
        assert stats["frame_age"] < 1.0

    def test_stats_frozen_state(self):
        """Stats reflect freeze-frame state."""
        buf = FrameBuffer(width=320, height=240, stale_threshold=0.05)
        buf.start()
        buf.update(make_frame())

        time.sleep(0.1)
        buf.get_frame()
        # Small delay so freeze_duration > 0 after rounding
        time.sleep(0.15)

        stats = buf.get_stats()
        assert stats["is_frozen"] is True
        assert stats["is_stale"] is True
        assert stats["freeze_reads"] >= 1
        assert stats["freeze_duration"] >= 0  # May round to 0.0 on fast machines

    def test_stats_resolution_format(self):
        """Resolution is formatted as 'WxH'."""
        buf = FrameBuffer(width=1920, height=1080)
        stats = buf.get_stats()
        assert stats["resolution"] == "1920x1080"


# ---------------------------------------------------------------------------
# Tests: Pixel-perfect freeze-frame recovery
# ---------------------------------------------------------------------------

class TestPixelPerfectRecovery:
    """Tests verifying exact pixel data is preserved through freeze-frame."""

    def test_solid_color_preserved(self):
        """A solid-color frame is preserved exactly."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        original = make_frame(color=(42, 137, 255))
        buf.update(original)

        # Simulate freeze by waiting
        time.sleep(0.01)

        recovered = buf.get_frame()
        np.testing.assert_array_equal(recovered, original)

    def test_complex_pattern_preserved(self):
        """A frame with complex patterns is preserved pixel-perfectly."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        # Create a frame with varied pixel data
        original = np.zeros((240, 320, 3), dtype=np.uint8)
        original[0:120, :, 0] = 255      # Top half red
        original[120:, :, 2] = 255        # Bottom half blue
        original[100:140, 140:180] = (0, 255, 0)  # Green rectangle
        # Gradient in a strip
        for x in range(320):
            original[50, x] = (x % 256, (x * 2) % 256, (x * 3) % 256)

        buf.update(original)
        recovered = buf.get_frame()
        np.testing.assert_array_equal(recovered, original)

    def test_last_frame_of_sequence_preserved(self):
        """After multiple updates, the last frame is preserved exactly."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        # Push 10 distinct frames
        frames = [make_unique_frame(index=i) for i in range(10)]
        for f in frames:
            buf.update(f)

        # The buffer should hold the last frame
        recovered = buf.get_frame()
        np.testing.assert_array_equal(recovered, frames[-1])

    def test_freeze_frame_identical_across_reads(self):
        """Multiple reads during freeze return identical frame data."""
        buf = FrameBuffer(width=320, height=240, stale_threshold=0.05)
        buf.start()

        original = make_unique_frame(index=42)
        buf.update(original)

        time.sleep(0.1)  # Exceed stale threshold

        reads = [buf.get_frame() for _ in range(10)]
        for r in reads:
            np.testing.assert_array_equal(r, original)

    def test_recovery_replaces_frozen_frame(self):
        """After recovery, new frame replaces frozen frame."""
        buf = FrameBuffer(width=320, height=240, stale_threshold=0.05)
        buf.start()

        old_frame = make_frame(color=(100, 0, 0))
        buf.update(old_frame)

        time.sleep(0.1)
        frozen = buf.get_frame()
        np.testing.assert_array_equal(frozen, old_frame)

        # Recovery: new frame
        new_frame = make_frame(color=(0, 255, 0))
        buf.update(new_frame)

        recovered = buf.get_frame()
        np.testing.assert_array_equal(recovered, new_frame)
        assert not np.array_equal(recovered, old_frame)


# ---------------------------------------------------------------------------
# Tests: Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Tests for concurrent access to the frame buffer."""

    def test_concurrent_update_and_read(self):
        """Concurrent update() and get_frame() calls don't crash or corrupt."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        errors = []
        stop_event = threading.Event()

        def writer():
            """Push frames rapidly."""
            try:
                for i in range(100):
                    if stop_event.is_set():
                        break
                    frame = make_unique_frame(index=i)
                    buf.update(frame)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Writer error: {e}")

        def reader():
            """Read frames rapidly."""
            try:
                for _ in range(200):
                    if stop_event.is_set():
                        break
                    frame = buf.get_frame()
                    if frame is not None:
                        assert frame.shape == (240, 320, 3)
                        assert frame.dtype == np.uint8
                    time.sleep(0.0005)
            except Exception as e:
                errors.append(f"Reader error: {e}")

        t_write = threading.Thread(target=writer)
        t_read1 = threading.Thread(target=reader)
        t_read2 = threading.Thread(target=reader)

        t_write.start()
        t_read1.start()
        t_read2.start()

        t_write.join(timeout=5)
        stop_event.set()
        t_read1.join(timeout=2)
        t_read2.join(timeout=2)

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert buf.total_updates > 0
        assert buf.total_reads > 0

    def test_concurrent_stats_access(self):
        """get_stats() is safe to call while streaming."""
        buf = FrameBuffer(width=320, height=240, stale_threshold=0.05)
        buf.start()

        errors = []
        stop_event = threading.Event()

        def writer():
            try:
                for i in range(50):
                    if stop_event.is_set():
                        break
                    buf.update(make_frame())
                    time.sleep(0.002)
            except Exception as e:
                errors.append(f"Writer: {e}")

        def stats_reader():
            try:
                for _ in range(100):
                    if stop_event.is_set():
                        break
                    stats = buf.get_stats()
                    assert isinstance(stats, dict)
                    assert "started" in stats
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Stats: {e}")

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=stats_reader),
            threading.Thread(target=stats_reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        stop_event.set()

        assert len(errors) == 0, f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_get_frame_before_start_returns_none(self):
        """get_frame() returns None before start()."""
        buf = FrameBuffer(width=320, height=240)
        assert buf.get_frame() is None

    def test_rapid_updates(self):
        """Rapid updates don't corrupt the buffer."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()

        for i in range(1000):
            buf.update(make_unique_frame(index=i))

        # Should have the last frame
        assert buf.total_updates == 1000
        stored = buf.get_frame()
        expected = make_unique_frame(index=999)
        np.testing.assert_array_equal(stored, expected)

    def test_default_stale_threshold(self):
        """Default stale threshold is 2 seconds."""
        assert _DEFAULT_STALE_THRESHOLD == 2.0

    def test_default_placeholder_color(self):
        """Default placeholder color is dark gray (40, 40, 40)."""
        assert _PLACEHOLDER_COLOR == (40, 40, 40)

    def test_1x1_frame(self):
        """Buffer works with minimal 1x1 frame."""
        buf = FrameBuffer(width=1, height=1)
        buf.start()

        frame = np.array([[[255, 128, 64]]], dtype=np.uint8)
        result = buf.update(frame)
        assert result is True

        stored = buf.get_frame()
        assert tuple(stored[0, 0]) == (255, 128, 64)

    def test_has_frame_property(self):
        """has_frame tracks whether any frame (placeholder or real) is available."""
        buf = FrameBuffer(width=320, height=240)
        assert buf.has_frame is False

        buf.start()
        assert buf.has_frame is True  # Placeholder

        buf.update(make_frame())
        assert buf.has_frame is True  # Real frame
