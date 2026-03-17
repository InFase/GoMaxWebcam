"""
test_virtual_camera.py — Tests for virtual camera initialization and operation

Tests cover:
  1. VirtualCamera initialization with config and defaults
  2. Backend selection logic (Unity Capture preferred, OBS fallback)
  3. Device opening/closing lifecycle
  4. Frame sending (correct dimensions, wrong dimensions, placeholder)
  5. Freeze-frame (send_last_frame) for disconnect resilience
  6. Stats/diagnostics reporting
  7. Thread safety of send_frame
  8. Error handling (missing backend, import failures)
  9. Context manager support

All tests mock pyvirtualcam so they work without a virtual camera driver installed.
"""

import sys
import os
import time
import threading
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from virtual_camera import (
    VirtualCamera,
    check_backend_available,
    detect_backend,
    select_best_backend,
    _PLACEHOLDER_COLOR,
)
from config import Config

pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def make_test_config(**overrides) -> Config:
    """Create a Config with test-friendly defaults."""
    config = Config()
    config.stream_width = 1920
    config.stream_height = 1080
    config.stream_fps = 30
    config.virtual_camera_name = "GoPro Webcam"
    config._config_path = ""
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def make_mock_camera():
    """Create a mock pyvirtualcam.Camera instance."""
    cam = MagicMock()
    cam.device = "GoPro Webcam"
    cam.close = MagicMock()
    cam.send = MagicMock()
    cam.sleep_until_next_frame = MagicMock()
    return cam


def make_rgb_frame(height=1080, width=1920, color=(128, 64, 32)):
    """Create a test RGB frame with a solid color."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

class TestVirtualCameraInit:
    """Tests for VirtualCamera.__init__()."""

    def test_init_with_config(self):
        """VirtualCamera picks up settings from config."""
        config = make_test_config(
            stream_width=1280,
            stream_height=720,
            stream_fps=60,
            virtual_camera_name="Test Cam",
        )
        vcam = VirtualCamera(config)

        assert vcam.width == 1280
        assert vcam.height == 720
        assert vcam.fps == 60
        assert vcam.device_name == "Test Cam"
        assert vcam.is_running is False
        assert vcam.backend is None
        assert vcam.frame_count == 0

    def test_init_without_config(self):
        """VirtualCamera uses sensible defaults when no config given."""
        vcam = VirtualCamera()

        assert vcam.width == 1920
        assert vcam.height == 1080
        assert vcam.fps == 30
        assert vcam.device_name == "GoPro Webcam"

    def test_init_preserves_default_name(self):
        """Default camera name is 'GoPro Webcam' per requirements."""
        config = make_test_config()
        vcam = VirtualCamera(config)
        assert vcam.device_name == "GoPro Webcam"


# ---------------------------------------------------------------------------
# Test: Backend selection
# ---------------------------------------------------------------------------

class TestBackendSelection:
    """Tests for check_backend_available() and select_best_backend()."""

    def test_unity_capture_preferred(self):
        """Unity Capture is checked first (supports custom device names)."""
        with patch("virtual_camera.check_backend_available") as mock_check:
            mock_check.side_effect = lambda b: b == "unitycapture"
            result = select_best_backend()
        assert result == "unitycapture"

    def test_obs_fallback(self):
        """OBS VirtualCam is used when Unity Capture unavailable."""
        with patch("virtual_camera.check_backend_available") as mock_check:
            mock_check.side_effect = lambda b: b == "obs"
            result = select_best_backend()
        assert result == "obs"

    def test_no_backend_returns_none(self):
        """Returns None when no backends are available."""
        with patch("virtual_camera.check_backend_available", return_value=False):
            result = select_best_backend()
        assert result is None

    def test_check_backend_unity_import_success(self):
        """check_backend_available returns True when unity native module exists."""
        mock_pyvirtualcam = MagicMock()
        mock_native = MagicMock()
        with patch.dict("sys.modules", {
            "pyvirtualcam": mock_pyvirtualcam,
            "pyvirtualcam._native_windows_unity": mock_native,
        }):
            result = check_backend_available("unitycapture")
        assert result is True

    def test_check_backend_obs_import_success(self):
        """check_backend_available returns True when OBS native module exists."""
        mock_pyvirtualcam = MagicMock()
        mock_native = MagicMock()
        with patch.dict("sys.modules", {
            "pyvirtualcam": mock_pyvirtualcam,
            "pyvirtualcam._native_windows_obs": mock_native,
        }):
            result = check_backend_available("obs")
        assert result is True

    def test_check_backend_unknown_returns_false(self):
        """check_backend_available returns False for unknown backend names."""
        result = check_backend_available("nonexistent_backend")
        assert result is False


# ---------------------------------------------------------------------------
# Test: Start / Stop lifecycle
# ---------------------------------------------------------------------------

class TestStartStop:
    """Tests for VirtualCamera.start() and stop()."""

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_start_opens_camera(self, mock_fmt, mock_cam_cls, mock_backend):
        """start() opens a pyvirtualcam.Camera and returns True."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        result = vcam.start()

        assert result is True
        assert vcam.is_running is True
        assert vcam.backend == "unitycapture"
        # Camera was opened with correct params
        mock_cam_cls.assert_called_once()
        call_kwargs = mock_cam_cls.call_args[1]
        assert call_kwargs["width"] == 1920
        assert call_kwargs["height"] == 1080
        assert call_kwargs["fps"] == 30
        assert call_kwargs["device"] == "GoPro Webcam"

        vcam.stop()

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_stop_closes_camera(self, mock_fmt, mock_cam_cls, mock_backend):
        """stop() closes the camera and clears state."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()
        vcam.stop()

        assert vcam.is_running is False
        assert vcam.backend is None
        mock_cam.close.assert_called_once()

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_start_idempotent(self, mock_fmt, mock_cam_cls, mock_backend):
        """Calling start() twice doesn't open a second camera."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()
        result = vcam.start()  # Second call

        assert result is True
        assert mock_cam_cls.call_count == 1  # Only opened once

        vcam.stop()

    def test_start_fails_without_pyvirtualcam(self):
        """start() returns False if pyvirtualcam is not installed."""
        vcam = VirtualCamera(make_test_config())

        with patch.dict("sys.modules", {"pyvirtualcam": None}):
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                result = vcam.start()

        assert result is False
        assert vcam.is_running is False

    @patch("virtual_camera.select_best_backend", return_value=None)
    def test_start_fails_without_backend(self, mock_backend):
        """start() returns False when no backend is available."""
        vcam = VirtualCamera(make_test_config())

        # Need to mock the pyvirtualcam import to succeed
        mock_pyvirtualcam = MagicMock()
        with patch.dict("sys.modules", {"pyvirtualcam": mock_pyvirtualcam}):
            result = vcam.start()

        assert result is False

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.PixelFormat")
    def test_start_handles_runtime_error(self, mock_fmt, mock_backend):
        """start() returns False if the driver raises RuntimeError."""
        mock_fmt.RGB = "RGB"

        with patch("pyvirtualcam.Camera", side_effect=RuntimeError("driver not found")):
            vcam = VirtualCamera(make_test_config())
            result = vcam.start()

        assert result is False
        assert vcam.is_running is False

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_stop_without_start(self, mock_fmt, mock_cam_cls, mock_backend):
        """stop() on an un-started camera is a no-op (no crash)."""
        vcam = VirtualCamera(make_test_config())
        vcam.stop()  # Should not raise
        assert vcam.is_running is False

    @patch("virtual_camera.select_best_backend", return_value="obs")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_obs_backend_no_device_name(self, mock_fmt, mock_cam_cls, mock_backend):
        """OBS backend doesn't receive 'device' kwarg (not supported)."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()

        call_kwargs = mock_cam_cls.call_args[1]
        assert "device" not in call_kwargs  # OBS doesn't support custom names

        vcam.stop()

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_start_with_forced_backend(self, mock_fmt, mock_cam_cls, mock_backend):
        """start(preferred_backend=...) overrides auto-selection."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start(preferred_backend="obs")

        # Should use "obs" even though select_best_backend would pick unitycapture
        call_kwargs = mock_cam_cls.call_args[1]
        assert call_kwargs["backend"] == "obs"
        assert "device" not in call_kwargs  # OBS backend

        vcam.stop()


# ---------------------------------------------------------------------------
# Test: Frame sending
# ---------------------------------------------------------------------------

class TestFrameSending:
    """Tests for send_frame(), send_last_frame(), and placeholder."""

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_send_frame_correct_dimensions(self, mock_fmt, mock_cam_cls, mock_backend):
        """send_frame() with correct dimensions sends to camera and stores copy."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()

        frame = make_rgb_frame(1080, 1920, (255, 0, 0))
        result = vcam.send_frame(frame)

        assert result is True
        # frame_count includes the placeholder sent on start + this frame
        assert vcam.frame_count >= 2
        # Last frame should be a copy of what we sent
        assert vcam.last_frame is not None
        assert np.array_equal(vcam.last_frame, frame)

        vcam.stop()

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_send_frame_wrong_dimensions_resized(self, mock_fmt, mock_cam_cls, mock_backend):
        """send_frame() with wrong dimensions attempts resize."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())  # 1920x1080
        vcam.start()

        # Send a 720p frame to a 1080p camera
        frame = make_rgb_frame(720, 1280, (0, 255, 0))
        result = vcam.send_frame(frame)

        assert result is True
        # The sent frame should have been resized
        sent_frame = mock_cam.send.call_args_list[-1][0][0]
        assert sent_frame.shape == (1080, 1920, 3)

        vcam.stop()

    def test_send_frame_when_not_running(self):
        """send_frame() returns False when camera is not started."""
        vcam = VirtualCamera(make_test_config())
        frame = make_rgb_frame()
        assert vcam.send_frame(frame) is False

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_send_last_frame_freezes(self, mock_fmt, mock_cam_cls, mock_backend):
        """send_last_frame() re-sends the stored frame (freeze-frame)."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()

        # Send a real frame
        frame = make_rgb_frame(1080, 1920, (100, 200, 50))
        vcam.send_frame(frame)
        count_before = vcam.frame_count

        # Now freeze-frame
        result = vcam.send_last_frame()
        assert result is True
        assert vcam.frame_count == count_before + 1

        # Verify the same frame was sent again
        last_sent = mock_cam.send.call_args_list[-1][0][0]
        assert np.array_equal(last_sent, frame)

        vcam.stop()

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_send_last_frame_no_prior_frame(self, mock_fmt, mock_cam_cls, mock_backend):
        """send_last_frame() sends placeholder when no real frame received yet."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()

        # Clear the last_frame that was set by placeholder
        with vcam._lock:
            vcam._last_frame = None

        result = vcam.send_last_frame()
        assert result is True

        vcam.stop()

    def test_send_last_frame_when_not_running(self):
        """send_last_frame() returns False when camera not started."""
        vcam = VirtualCamera(make_test_config())
        assert vcam.send_last_frame() is False

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_placeholder_frame_sent_on_start(self, mock_fmt, mock_cam_cls, mock_backend):
        """start() sends a placeholder frame immediately."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()

        # Camera.send should have been called once (placeholder)
        assert mock_cam.send.call_count == 1
        placeholder = mock_cam.send.call_args[0][0]
        assert placeholder.shape == (1080, 1920, 3)
        # Check it's the dark gray placeholder color
        assert tuple(placeholder[0, 0]) == _PLACEHOLDER_COLOR

        vcam.stop()


# ---------------------------------------------------------------------------
# Test: Statistics
# ---------------------------------------------------------------------------

class TestStats:
    """Tests for get_stats() diagnostic output."""

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_stats_while_running(self, mock_fmt, mock_cam_cls, mock_backend):
        """get_stats() returns correct info when running."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()

        stats = vcam.get_stats()

        assert stats["running"] is True
        assert stats["backend"] == "unitycapture"
        assert stats["device_name"] == "GoPro Webcam"
        assert stats["resolution"] == "1920x1080"
        assert stats["fps"] == 30
        assert stats["frame_count"] >= 1  # At least the placeholder
        assert stats["uptime_seconds"] >= 0.0

        vcam.stop()

    def test_stats_when_stopped(self):
        """get_stats() returns sensible defaults when not running."""
        vcam = VirtualCamera(make_test_config())
        stats = vcam.get_stats()

        assert stats["running"] is False
        assert stats["backend"] is None
        assert stats["frame_count"] == 0
        assert stats["uptime_seconds"] == 0.0
        assert stats["seconds_since_last_frame"] is None


# ---------------------------------------------------------------------------
# Test: Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Tests that send_frame() is safe to call from multiple threads."""

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_concurrent_send_frame(self, mock_fmt, mock_cam_cls, mock_backend):
        """Multiple threads can call send_frame() without crashing."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()

        errors = []

        def send_frames(n):
            try:
                for _ in range(n):
                    frame = make_rgb_frame()
                    vcam.send_frame(frame)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=send_frames, args=(10,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Thread errors: {errors}"
        # 4 threads x 10 frames + 1 placeholder = 41
        assert vcam.frame_count >= 40

        vcam.stop()


# ---------------------------------------------------------------------------
# Test: Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    """Tests for with-statement support."""

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_context_manager(self, mock_fmt, mock_cam_cls, mock_backend):
        """VirtualCamera works as a context manager."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        config = make_test_config()

        with VirtualCamera(config) as vcam:
            assert vcam.is_running is True
            vcam.send_frame(make_rgb_frame())

        # After exiting the context, camera should be closed
        mock_cam.close.assert_called_once()


# ---------------------------------------------------------------------------
# Test: sleep_until_next_frame
# ---------------------------------------------------------------------------

class TestFrameTiming:
    """Tests for sleep_until_next_frame()."""

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_sleep_delegates_to_cam(self, mock_fmt, mock_cam_cls, mock_backend):
        """sleep_until_next_frame() delegates to pyvirtualcam."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()
        vcam.sleep_until_next_frame()

        mock_cam.sleep_until_next_frame.assert_called_once()
        vcam.stop()

    def test_sleep_when_not_running(self):
        """sleep_until_next_frame() is a no-op when camera isn't started."""
        vcam = VirtualCamera(make_test_config())
        vcam.sleep_until_next_frame()  # Should not raise


# ---------------------------------------------------------------------------
# Test: Resize
# ---------------------------------------------------------------------------

class TestFrameResize:
    """Tests for the internal _resize_frame helper."""

    def test_resize_upscale(self):
        """Resizing a smaller frame to larger dimensions."""
        vcam = VirtualCamera(make_test_config())  # 1920x1080
        small = make_rgb_frame(720, 1280, (255, 128, 0))

        resized = vcam._resize_frame(small)

        assert resized.shape == (1080, 1920, 3)
        assert resized.dtype == np.uint8

    def test_resize_downscale(self):
        """Resizing a larger frame to smaller dimensions."""
        config = make_test_config(stream_width=640, stream_height=480)
        vcam = VirtualCamera(config)
        big = make_rgb_frame(1080, 1920, (0, 0, 255))

        resized = vcam._resize_frame(big)

        assert resized.shape == (480, 640, 3)

    def test_resize_same_size_passthrough(self):
        """Same-size frame is returned as-is."""
        vcam = VirtualCamera(make_test_config())
        frame = make_rgb_frame(1080, 1920)

        resized = vcam._resize_frame(frame)

        assert resized.shape == (1080, 1920, 3)
        assert np.array_equal(resized, frame)


# ---------------------------------------------------------------------------
# Test: detect_backend()
# ---------------------------------------------------------------------------

class TestDetectBackend:
    """Tests for detect_backend() startup warning logic."""

    def test_unity_capture_detected_returns_success(self):
        """When Unity Capture is available, returns success with no warning."""
        with patch("virtual_camera.check_backend_available") as mock_check:
            mock_check.side_effect = lambda b: b == "unitycapture"
            with patch.dict("sys.modules", {"pyvirtualcam": MagicMock()}):
                result = detect_backend()

        assert result["backend"] == "unitycapture"
        assert result["is_recommended"] is True
        assert result["warning"] is None
        assert result["level"] == "success"

    def test_obs_fallback_returns_warning(self):
        """When only OBS is available, returns warning recommending Unity Capture."""
        with patch("virtual_camera.check_backend_available") as mock_check:
            mock_check.side_effect = lambda b: b == "obs"
            with patch.dict("sys.modules", {"pyvirtualcam": MagicMock()}):
                result = detect_backend()

        assert result["backend"] == "obs"
        assert result["is_recommended"] is False
        assert result["warning"] is not None
        assert "Unity Capture" in result["warning"]
        assert result["level"] == "warning"

    def test_no_backend_returns_error(self):
        """When no backend is available, returns error with install instructions."""
        with patch("virtual_camera.check_backend_available", return_value=False):
            with patch.dict("sys.modules", {"pyvirtualcam": MagicMock()}):
                result = detect_backend()

        assert result["backend"] is None
        assert result["is_recommended"] is False
        assert result["warning"] is not None
        assert "Unity Capture" in result["warning"]
        assert result["level"] == "error"

    def test_pyvirtualcam_not_installed_returns_error(self):
        """When pyvirtualcam is missing, returns error."""
        with patch.dict("sys.modules", {"pyvirtualcam": None}):
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                result = detect_backend()

        assert result["backend"] is None
        assert result["warning"] is not None
        assert "pyvirtualcam" in result["warning"].lower()
        assert result["level"] == "error"

    def test_recommended_field_always_unitycapture(self):
        """The 'recommended' field always says 'unitycapture'."""
        with patch("virtual_camera.check_backend_available", return_value=False):
            with patch.dict("sys.modules", {"pyvirtualcam": MagicMock()}):
                result = detect_backend()
        assert result["recommended"] == "unitycapture"

    def test_non_blocking_on_obs_fallback(self):
        """detect_backend() with OBS returns a warning, not an error — non-blocking."""
        with patch("virtual_camera.check_backend_available") as mock_check:
            mock_check.side_effect = lambda b: b == "obs"
            with patch.dict("sys.modules", {"pyvirtualcam": MagicMock()}):
                result = detect_backend()

        # Level should be 'warning', not 'error' — startup should proceed
        assert result["level"] == "warning"
        assert result["backend"] == "obs"
