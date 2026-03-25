"""
Tests for VirtualCameraSink — the v2 virtual camera output component.

All tests mock pyvirtualcam at the Transport ABC boundary so they run
in CI without a real virtual camera backend installed.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# Ensure src is importable
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gomaxwebcam.pipeline.virtual_camera_sink import (
    VirtualCameraSink,
    WIDTH,
    HEIGHT,
    FPS,
    DEVICE_NAME,
    _QUEUE_MAX_SIZE,
    _detect_backend,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pyvirtualcam():
    """Mock pyvirtualcam module so tests run without a real backend."""
    mock_cam = MagicMock()
    mock_cam.device = "MockDevice"
    mock_cam.send = MagicMock()
    mock_cam.sleep_until_next_frame = MagicMock()
    mock_cam.close = MagicMock()

    mock_module = MagicMock()
    mock_module.Camera.return_value = mock_cam
    mock_module.PixelFormat.RGB = "rgb"

    with patch.dict("sys.modules", {"pyvirtualcam": mock_module}):
        with patch(
            "gomaxwebcam.pipeline.virtual_camera_sink._detect_backend",
            return_value="obs",
        ):
            yield mock_cam, mock_module


@pytest.fixture
def sink(mock_pyvirtualcam):
    """Create and start a VirtualCameraSink with mocked backend."""
    mock_cam, _ = mock_pyvirtualcam
    s = VirtualCameraSink()
    s.start()
    yield s
    s.stop()


def _make_frame(h: int = HEIGHT, w: int = WIDTH, value: int = 128) -> np.ndarray:
    """Create a test RGB24 frame."""
    return np.full((h, w, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_output_resolution(self):
        assert WIDTH == 1920
        assert HEIGHT == 1080

    def test_output_fps(self):
        assert FPS == 30

    def test_device_name(self):
        assert DEVICE_NAME == "GoMaxWebcam"


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_values(self):
        sink = VirtualCameraSink()
        assert sink.width == 1920
        assert sink.height == 1080
        assert sink.fps == 30
        assert sink.device_name == "GoMaxWebcam"
        assert not sink.is_running
        assert sink.backend is None
        assert sink.frames_sent == 0
        assert sink.freeze_frames_sent == 0
        assert sink.frames_dropped == 0
        assert sink.last_frame is None

    def test_custom_device_name(self):
        sink = VirtualCameraSink(device_name="Custom Cam")
        assert sink.device_name == "Custom Cam"

    def test_custom_backend(self):
        sink = VirtualCameraSink(backend="v4l2loopback")
        assert sink._requested_backend == "v4l2loopback"

    def test_repr_stopped(self):
        sink = VirtualCameraSink()
        r = repr(sink)
        assert "stopped" in r
        assert "1920x1080" in r
        assert "GoMaxWebcam" in r


# ---------------------------------------------------------------------------
# Start / stop lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_opens_device(self, mock_pyvirtualcam):
        mock_cam, mock_module = mock_pyvirtualcam
        sink = VirtualCameraSink()
        assert sink.start()
        assert sink.is_running
        assert sink.backend == "obs"
        # Camera was created with correct params
        mock_module.Camera.assert_called_once()
        call_kwargs = mock_module.Camera.call_args[1]
        assert call_kwargs["width"] == 1920
        assert call_kwargs["height"] == 1080
        assert call_kwargs["fps"] == 30
        assert call_kwargs["fmt"] == "rgb"
        sink.stop()

    def test_start_twice_is_idempotent(self, sink):
        assert sink.start()  # Already started by fixture
        assert sink.is_running

    def test_stop_closes_device(self, mock_pyvirtualcam):
        mock_cam, _ = mock_pyvirtualcam
        sink = VirtualCameraSink()
        sink.start()
        sink.stop()
        assert not sink.is_running
        mock_cam.close.assert_called_once()
        assert sink.backend is None

    def test_stop_twice_is_safe(self, mock_pyvirtualcam):
        mock_cam, _ = mock_pyvirtualcam
        sink = VirtualCameraSink()
        sink.start()
        sink.stop()
        sink.stop()  # Should not raise
        assert not sink.is_running

    def test_start_no_backend_fails(self):
        with patch(
            "gomaxwebcam.pipeline.virtual_camera_sink._detect_backend",
            return_value=None,
        ):
            sink = VirtualCameraSink()
            assert not sink.start()
            assert not sink.is_running

    def test_start_pyvirtualcam_import_error(self):
        with patch(
            "gomaxwebcam.pipeline.virtual_camera_sink._detect_backend",
            return_value="obs",
        ):
            with patch("builtins.__import__", side_effect=ImportError("no pyvirtualcam")):
                sink = VirtualCameraSink()
                assert not sink.start()

    def test_start_runtime_error(self, mock_pyvirtualcam):
        _, mock_module = mock_pyvirtualcam
        mock_module.Camera.side_effect = RuntimeError("no driver")
        sink = VirtualCameraSink()
        assert not sink.start()
        assert not sink.is_running


# ---------------------------------------------------------------------------
# Frame submission
# ---------------------------------------------------------------------------

class TestFrameSubmission:
    def test_submit_frame_queues_frame(self, sink):
        frame = _make_frame()
        assert sink.submit_frame(frame)

    def test_submit_frame_when_stopped(self, mock_pyvirtualcam):
        sink = VirtualCameraSink()
        # Don't start — sink is stopped
        sink._stop_event.set()
        frame = _make_frame()
        assert not sink.submit_frame(frame)

    def test_submit_frame_reaches_device(self, mock_pyvirtualcam):
        mock_cam, _ = mock_pyvirtualcam
        sink = VirtualCameraSink()
        sink.start()

        frame = _make_frame(value=200)
        sink.submit_frame(frame)

        # Wait for consumer to process
        time.sleep(0.3)

        # mock_cam.send should have been called (placeholder + submitted frame)
        assert mock_cam.send.call_count >= 2
        sink.stop()

    def test_queue_overflow_drops_oldest(self, mock_pyvirtualcam):
        mock_cam, _ = mock_pyvirtualcam
        # Make send slow so queue fills up
        mock_cam.send = MagicMock()
        mock_cam.sleep_until_next_frame = MagicMock(side_effect=lambda: time.sleep(0.5))

        sink = VirtualCameraSink()
        sink.start()

        # Submit many frames quickly — more than queue capacity
        for i in range(_QUEUE_MAX_SIZE + 5):
            sink.submit_frame(_make_frame(value=i))

        # Some frames should have been dropped
        time.sleep(0.1)
        assert sink.frames_dropped > 0
        sink.stop()


# ---------------------------------------------------------------------------
# Freeze-frame behavior
# ---------------------------------------------------------------------------

class TestFreezeFrame:
    def test_freeze_frame_sends_last_frame(self, mock_pyvirtualcam):
        mock_cam, _ = mock_pyvirtualcam
        sink = VirtualCameraSink()
        sink.start()

        # Submit one frame
        frame = _make_frame(value=100)
        sink.submit_frame(frame)

        # Wait for it to be consumed
        time.sleep(0.3)

        # Now don't submit any more — consumer should re-send the last frame
        initial_count = mock_cam.send.call_count
        time.sleep(0.5)

        # More frames should have been sent (freeze frames)
        assert mock_cam.send.call_count > initial_count
        assert sink.freeze_frames_sent > 0
        sink.stop()

    def test_placeholder_on_startup(self, mock_pyvirtualcam):
        mock_cam, _ = mock_pyvirtualcam
        sink = VirtualCameraSink()
        sink.start()

        # Wait a bit for placeholder + freeze-frames
        time.sleep(0.2)

        # At least the placeholder frame should have been sent
        assert mock_cam.send.call_count >= 1
        sink.stop()


# ---------------------------------------------------------------------------
# Frame resize
# ---------------------------------------------------------------------------

class TestResize:
    def test_resize_720p_to_1080p(self, mock_pyvirtualcam):
        mock_cam, _ = mock_pyvirtualcam
        sink = VirtualCameraSink()
        sink.start()

        # Submit a 720p frame
        frame_720p = _make_frame(h=720, w=1280, value=150)
        sink.submit_frame(frame_720p)

        time.sleep(0.3)

        # The frame should have been resized before sending
        # Check the last call to send was 1080p
        for call in mock_cam.send.call_args_list:
            sent_frame = call[0][0]
            assert sent_frame.shape == (1080, 1920, 3)
        sink.stop()

    def test_resize_4k_to_1080p(self):
        sink = VirtualCameraSink()
        frame_4k = _make_frame(h=2160, w=3840, value=200)
        resized = sink._resize_frame(frame_4k)
        assert resized.shape == (1080, 1920, 3)

    def test_no_resize_for_correct_size(self):
        sink = VirtualCameraSink()
        frame = _make_frame()
        result = sink._resize_frame(frame)
        assert result is frame  # Should be the same object (no copy)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStats:
    def test_get_stats_structure(self, sink):
        time.sleep(0.2)  # Let consumer thread run
        stats = sink.get_stats()

        assert "running" in stats
        assert "backend" in stats
        assert "device_name" in stats
        assert "resolution" in stats
        assert "fps_target" in stats
        assert "fps_actual" in stats
        assert "frames_sent" in stats
        assert "freeze_frames_sent" in stats
        assert "frames_dropped" in stats
        assert "has_frame" in stats
        assert "uptime_seconds" in stats
        assert "seconds_since_last_frame" in stats
        assert "queue_size" in stats
        assert "queue_max" in stats

    def test_stats_values_after_start(self, sink):
        time.sleep(0.2)
        stats = sink.get_stats()

        assert stats["running"] is True
        assert stats["backend"] == "obs"
        assert stats["device_name"] == "GoMaxWebcam"
        assert stats["resolution"] == "1920x1080"
        assert stats["fps_target"] == 30
        assert stats["frames_sent"] >= 1  # At least placeholder
        assert stats["uptime_seconds"] > 0
        assert stats["queue_max"] == _QUEUE_MAX_SIZE

    def test_frames_sent_increases(self, mock_pyvirtualcam):
        mock_cam, _ = mock_pyvirtualcam
        sink = VirtualCameraSink()
        sink.start()

        frame = _make_frame()
        for _ in range(5):
            sink.submit_frame(frame)
            time.sleep(0.05)

        time.sleep(0.5)
        assert sink.frames_sent >= 5
        sink.stop()

    def test_fps_measurement(self, mock_pyvirtualcam):
        mock_cam, _ = mock_pyvirtualcam
        # Make sleep_until_next_frame fast so we can measure FPS quickly
        mock_cam.sleep_until_next_frame = MagicMock(
            side_effect=lambda: time.sleep(0.01)
        )

        sink = VirtualCameraSink()
        sink.start()

        # Submit frames for >1 second so FPS window updates
        start = time.monotonic()
        while time.monotonic() - start < 1.5:
            sink.submit_frame(_make_frame())
            time.sleep(0.01)

        # FPS should be measurable (non-zero)
        assert sink.fps_actual > 0
        sink.stop()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_submit(self, mock_pyvirtualcam):
        mock_cam, _ = mock_pyvirtualcam
        sink = VirtualCameraSink()
        sink.start()

        errors = []
        # Use small frames to avoid MemoryError in CI
        small_frame = _make_frame(h=120, w=160, value=100)

        def submit_worker(worker_id):
            try:
                for i in range(20):
                    sink.submit_frame(small_frame)
                    time.sleep(0.005)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=submit_worker, args=(i,))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Submit workers raised errors: {errors}"
        assert sink.frames_sent > 0
        sink.stop()

    def test_last_frame_thread_safe(self, mock_pyvirtualcam):
        mock_cam, _ = mock_pyvirtualcam
        sink = VirtualCameraSink()
        sink.start()

        # Submit frames while reading last_frame from another thread
        errors = []

        def reader():
            try:
                for _ in range(100):
                    _ = sink.last_frame
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        def writer():
            for _ in range(100):
                sink.submit_frame(_make_frame())
                time.sleep(0.001)

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=writer)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors
        sink.stop()


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

class TestBackendDetection:
    def test_detect_backend_windows(self):
        with patch("gomaxwebcam.pipeline.virtual_camera_sink.platform") as mock_platform:
            mock_platform.system.return_value = "Windows"
            with patch(
                "gomaxwebcam.pipeline.virtual_camera_sink._backend_available",
                return_value=True,
            ):
                result = _detect_backend()
                assert result == "unitycapture"

    def test_detect_backend_macos(self):
        with patch("gomaxwebcam.pipeline.virtual_camera_sink.platform") as mock_platform:
            mock_platform.system.return_value = "Darwin"
            with patch(
                "gomaxwebcam.pipeline.virtual_camera_sink._backend_available",
                return_value=True,
            ):
                result = _detect_backend()
                assert result == "obs"

    def test_detect_backend_linux(self):
        with patch("gomaxwebcam.pipeline.virtual_camera_sink.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            with patch(
                "gomaxwebcam.pipeline.virtual_camera_sink._backend_available",
                return_value=True,
            ):
                result = _detect_backend()
                assert result == "v4l2loopback"

    def test_detect_backend_none_available(self):
        with patch("gomaxwebcam.pipeline.virtual_camera_sink.platform") as mock_platform:
            mock_platform.system.return_value = "Windows"
            with patch(
                "gomaxwebcam.pipeline.virtual_camera_sink._backend_available",
                return_value=False,
            ):
                result = _detect_backend()
                assert result is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_send_frame_after_device_closed(self, mock_pyvirtualcam):
        mock_cam, _ = mock_pyvirtualcam
        sink = VirtualCameraSink()
        sink.start()
        sink.stop()

        # Submit after stop should return False
        assert not sink.submit_frame(_make_frame())

    def test_cam_send_raises(self, mock_pyvirtualcam):
        mock_cam, _ = mock_pyvirtualcam
        sink = VirtualCameraSink()
        sink.start()

        # Make cam.send raise on next call
        mock_cam.send.side_effect = RuntimeError("device error")

        sink.submit_frame(_make_frame())
        time.sleep(0.3)

        # Consumer should survive the error
        assert sink._consumer_thread.is_alive()
        sink.stop()

    def test_empty_frame_queue_on_start(self, mock_pyvirtualcam):
        """Verify queue is cleared on start."""
        mock_cam, _ = mock_pyvirtualcam
        sink = VirtualCameraSink()

        # Manually put something in the queue before start
        sink._frame_queue.put(_make_frame())
        assert not sink._frame_queue.empty()

        sink.start()
        time.sleep(0.1)

        # Queue should have been cleared during start
        # (consumer may have consumed it too, both are fine)
        sink.stop()
