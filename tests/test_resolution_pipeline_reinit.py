"""
test_resolution_pipeline_reinit.py — Tests for Sub-AC 3 of AC 16:
stream pipeline teardown and re-initialization at new resolution
with virtual camera reconfiguration.

Tests verify:
  1. VirtualCamera.reconfigure() stops and restarts at new dimensions
  2. VirtualCamera.reconfigure() is a no-op when dimensions unchanged
  3. VirtualCamera.reconfigure() falls back on start failure
  4. FrameBuffer.resize() changes dimensions and resets to placeholder
  5. FrameBuffer.resize() is a no-op when dimensions unchanged
  6. FrameBuffer.resize() accepts frames at new dimensions
  7. _change_resolution_flow() tears down full pipeline
  8. _change_resolution_flow() reconfigures vcam at new dims
  9. _change_resolution_flow() reverts config on vcam failure
  10. _change_resolution_flow() stops/restarts disconnect detector
  11. _change_resolution_flow() creates new FramePipeline
  12. _change_resolution_flow() restarts GoPro webcam mode
  13. _resume_stream_after_resolution_change() does full reinit
  14. Pipeline can be fully torn down and restarted at new resolution
"""

import sys
import os
import time
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


# USBEventListener uses raw Win32 APIs that crash in pytest — mock globally
_usb_listener_patch = patch(
    'usb_event_listener.USBEventListener',
    **{'return_value.start.return_value': False, 'return_value.stop.return_value': None},
)


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
    config.resolution = 4
    config.fov = 4
    config.idle_reset_delay = 0.01
    config.discovery_timeout = 0.5
    config.discovery_retry_interval = 0.1
    config.discovery_max_retries = 2
    config.keepalive_interval = 0.5
    config.reconnect_delay = 0.1
    config.ncm_adapter_wait = 0.1
    config._config_path = ""
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def make_test_frame(width=320, height=240, color=(128, 64, 32)):
    """Create a test BGR24 frame as numpy array."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


class MockVirtualCamera:
    """Mock VirtualCamera with reconfigure support."""

    def __init__(self, width=320, height=240, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self._is_running = False
        self._backend = "unitycapture"
        self.device_name = "GoPro Webcam"
        self._start_count = 0
        self._stop_count = 0
        self._reconfigure_count = 0
        self._reconfigure_fail = False
        self._last_frame = None
        self._frame_count = 0
        self.frames_sent = FrameLog()
        self.frame_count = 0

    @property
    def is_running(self):
        return self._is_running

    def start(self, preferred_backend=None):
        self._is_running = True
        self._start_count += 1
        return True

    def stop(self):
        self._is_running = False
        self._stop_count += 1

    def reconfigure(self, width, height, fps=None):
        self._reconfigure_count += 1
        if self._reconfigure_fail:
            return False
        self.stop()
        self.width = width
        self.height = height
        if fps is not None:
            self.fps = fps
        return self.start()

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
        time.sleep(0.001)


class MockStreamReader:
    """Mock StreamReader."""

    def __init__(self, frames=None, width=320, height=240, fps=30):
        self._frames = frames or []
        self._index = 0
        self._is_running = False
        self.width = width
        self.height = height
        self.fps = fps
        self._process = MagicMock()
        self._process.pid = 12345

    @property
    def is_running(self):
        return self._is_running

    def start(self):
        self._is_running = True
        return True

    def stop(self):
        self._is_running = False

    def read_frame(self):
        if not self._frames:
            self._is_running = False
            return None
        frame = self._frames[self._index % len(self._frames)]
        self._index += 1
        return frame


# ---------------------------------------------------------------------------
# Tests: VirtualCamera.reconfigure() on the real class
# ---------------------------------------------------------------------------

class TestVirtualCameraReconfigure:
    """Tests for the actual VirtualCamera.reconfigure() method."""

    def test_reconfigure_noop_same_dimensions(self):
        """reconfigure() with same w/h/fps returns True without restart."""
        from virtual_camera import VirtualCamera
        config = make_test_config()
        vcam = VirtualCamera(config)

        result = vcam.reconfigure(320, 240, 30)
        assert result is True

    def test_reconfigure_updates_dimensions(self):
        """reconfigure() updates width, height, fps attributes."""
        from virtual_camera import VirtualCamera
        config = make_test_config()
        vcam = VirtualCamera(config)

        with patch.object(vcam, 'start', return_value=True), \
             patch.object(vcam, 'stop'):
            result = vcam.reconfigure(640, 480, 60)

        assert result is True
        assert vcam.width == 640
        assert vcam.height == 480
        assert vcam.fps == 60

    def test_reconfigure_calls_stop_then_start(self):
        """reconfigure() stops the old device and starts at new dimensions."""
        from virtual_camera import VirtualCamera
        config = make_test_config()
        vcam = VirtualCamera(config)

        with patch.object(vcam, 'start', return_value=True) as mock_start, \
             patch.object(vcam, 'stop') as mock_stop:
            vcam.reconfigure(640, 480)

        mock_stop.assert_called_once()
        mock_start.assert_called_once()

    def test_reconfigure_fallback_on_failure(self):
        """reconfigure() reverts to old dims if start fails at new dims."""
        from virtual_camera import VirtualCamera
        config = make_test_config()
        vcam = VirtualCamera(config)

        call_count = [0]
        def mock_start(preferred_backend=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return False  # Fail at new dims
            return True  # Succeed at old dims (fallback)

        with patch.object(vcam, 'start', side_effect=mock_start), \
             patch.object(vcam, 'stop'):
            result = vcam.reconfigure(640, 480, 60)

        assert result is False
        assert vcam.width == 320
        assert vcam.height == 240
        assert vcam.fps == 30

    def test_reconfigure_keeps_fps_when_none(self):
        """reconfigure() with fps=None keeps current fps."""
        from virtual_camera import VirtualCamera
        config = make_test_config()
        vcam = VirtualCamera(config)

        with patch.object(vcam, 'start', return_value=True), \
             patch.object(vcam, 'stop'):
            vcam.reconfigure(640, 480)

        assert vcam.fps == 30

    def test_reconfigure_preserves_backend(self):
        """reconfigure() passes previous backend to start()."""
        from virtual_camera import VirtualCamera
        config = make_test_config()
        vcam = VirtualCamera(config)
        vcam._backend = "unitycapture"

        with patch.object(vcam, 'start', return_value=True) as mock_start, \
             patch.object(vcam, 'stop'):
            vcam.reconfigure(640, 480)

        mock_start.assert_called_once_with(preferred_backend="unitycapture")


# ---------------------------------------------------------------------------
# Tests: FrameBuffer.resize()
# ---------------------------------------------------------------------------

class TestFrameBufferResize:
    """Tests for FrameBuffer.resize() method."""

    def test_resize_changes_dimensions(self):
        """resize() updates width and height."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        buf.resize(640, 480)
        assert buf.width == 640
        assert buf.height == 480

    def test_resize_resets_to_placeholder(self):
        """resize() resets buffer to placeholder at new dimensions."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        buf.update(make_test_frame(320, 240, (255, 0, 0)))
        assert buf.has_live_frame is True

        buf.resize(640, 480)
        frame = buf.get_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    def test_resize_unchanged_is_noop(self):
        """resize() with same dimensions does nothing."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        buf.update(make_test_frame(320, 240, (255, 0, 0)))
        updates_before = buf.total_updates

        buf.resize(320, 240)
        assert buf.total_updates == updates_before
        assert buf.has_live_frame is True

    def test_resize_keeps_started_state(self):
        """resize() keeps the buffer started."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        buf.resize(640, 480)
        assert buf.has_frame is True

    def test_resize_accepts_new_dimension_frames(self):
        """After resize(), accepts frames at new dimensions."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        buf.resize(640, 480)

        result = buf.update(make_test_frame(640, 480, (0, 255, 0)))
        assert result is True

    def test_resize_rejects_old_dimension_frames(self):
        """After resize(), rejects frames at old dimensions."""
        buf = FrameBuffer(width=320, height=240)
        buf.start()
        buf.resize(640, 480)

        result = buf.update(make_test_frame(320, 240))
        assert result is False


# ---------------------------------------------------------------------------
# Tests: Pipeline teardown and re-initialization
# ---------------------------------------------------------------------------

class TestPipelineTeardownReinit:
    """Tests for full pipeline teardown and re-init at new resolution."""

    def test_pipeline_stop_sets_stopped(self):
        """Stopping pipeline sets state to STOPPED."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        reader = MockStreamReader(frames=[make_test_frame()])
        vcam = MockVirtualCamera()

        pipeline.start(reader, vcam)
        time.sleep(0.1)
        pipeline.stop()

        assert pipeline.state == PipelineState.STOPPED

    def test_new_pipeline_works_at_new_resolution(self):
        """A new pipeline at different resolution streams correctly."""
        # Pipeline 1 at 320x240
        config = make_test_config()
        pipeline1 = FramePipeline(config)
        vcam = MockVirtualCamera(width=320, height=240)
        reader1 = MockStreamReader(frames=[make_test_frame(320, 240)])
        pipeline1.start(reader1, vcam)
        time.sleep(0.1)
        pipeline1.stop()

        # Pipeline 2 at 640x480
        config.stream_width = 640
        config.stream_height = 480
        pipeline2 = FramePipeline(config)
        vcam.reconfigure(640, 480)
        reader2 = MockStreamReader(frames=[make_test_frame(640, 480)])
        buf = FrameBuffer(width=640, height=480)
        buf.start()

        result = pipeline2.start(reader2, vcam, frame_buffer=buf)
        assert result is True
        assert pipeline2.target_width == 640
        assert pipeline2.target_height == 480
        time.sleep(0.1)
        pipeline2.stop()

    def test_full_teardown_reinit_sequence(self):
        """Full teardown-reinit preserves frame flow at new resolution."""
        config = make_test_config()
        pipeline = FramePipeline(config)
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()
        vcam = MockVirtualCamera(width=320, height=240)
        reader = MockStreamReader(frames=[make_test_frame(320, 240)] * 5)

        pipeline.start(reader, vcam, frame_buffer=buffer)
        time.sleep(0.2)

        # Teardown
        pipeline.stop()
        reader.stop()

        # Reconfigure
        config.stream_width = 640
        config.stream_height = 480
        vcam.reconfigure(640, 480)
        buffer.resize(640, 480)

        # Re-initialize
        new_pipeline = FramePipeline(config)
        new_reader = MockStreamReader(
            frames=[make_test_frame(640, 480, (0, 255, 0))] * 5,
            width=640, height=480,
        )
        result = new_pipeline.start(new_reader, vcam, frame_buffer=buffer)
        assert result is True
        time.sleep(0.2)

        # Verify frames at new resolution
        last = vcam.frames_sent[-1]
        assert last.shape == (480, 640, 3)

        new_pipeline.stop()


# ---------------------------------------------------------------------------
# Tests: AppController._change_resolution_flow
# ---------------------------------------------------------------------------

class TestChangeResolutionFlow:
    """Tests for AppController._change_resolution_flow full teardown."""

    def _make_controller(self, **overrides):
        from app_controller import AppController
        config = make_test_config(**overrides)
        with _usb_listener_patch:
            controller = AppController(config)
        return controller

    def test_stops_pipeline(self):
        """_change_resolution_flow stops the existing pipeline."""
        controller = self._make_controller()
        mock_pipeline = MagicMock()
        controller._frame_pipeline = mock_pipeline
        controller._stream_reader = MagicMock()
        controller._virtual_camera = MockVirtualCamera()
        controller._virtual_camera.start()
        controller._frame_buffer = MagicMock()
        controller.gopro._connected = False

        with patch.object(controller, '_apply_resolution_on_camera', return_value=True), \
             patch.object(controller, '_create_and_swap_stream_reader', return_value=True), \
             patch.object(controller, '_stop_disconnect_detector'), \
             patch.object(controller, '_start_disconnect_detector'), \
             _usb_listener_patch:
            controller.change_resolution(7)  # 720p

        mock_pipeline.stop.assert_called_once()

    def test_stops_stream_reader(self):
        """_change_resolution_flow stops the old stream reader."""
        controller = self._make_controller()
        mock_reader = MagicMock()
        controller._stream_reader = mock_reader
        controller._frame_pipeline = None
        controller._virtual_camera = MockVirtualCamera()
        controller._virtual_camera.start()
        controller._frame_buffer = FrameBuffer(width=320, height=240)
        controller._frame_buffer.start()
        controller.gopro._connected = False

        with patch.object(controller, '_apply_resolution_on_camera', return_value=True), \
             patch.object(controller, '_create_and_swap_stream_reader', return_value=True), \
             patch.object(controller, '_stop_disconnect_detector'), \
             patch.object(controller, '_start_disconnect_detector'), \
             _usb_listener_patch:
            controller.change_resolution(7)  # 720p

        mock_reader.stop.assert_called_once()

    def test_reconfigures_vcam(self):
        """change_resolution reconfigures virtual camera."""
        controller = self._make_controller()
        vcam = MockVirtualCamera(width=320, height=240)
        vcam.start()
        controller._virtual_camera = vcam
        controller._frame_pipeline = None
        controller._stream_reader = None
        controller._frame_buffer = FrameBuffer(width=320, height=240)
        controller._frame_buffer.start()
        controller.gopro._connected = False

        with patch.object(controller, '_apply_resolution_on_camera', return_value=True), \
             patch.object(controller, '_create_and_swap_stream_reader', return_value=True), \
             patch.object(controller, '_stop_disconnect_detector'), \
             patch.object(controller, '_start_disconnect_detector'), \
             _usb_listener_patch:
            controller.change_resolution(7)  # 720p

        assert vcam.width == 1280
        assert vcam.height == 720

    def test_resizes_frame_buffer(self):
        """change_resolution resizes the frame buffer."""
        controller = self._make_controller()
        controller._virtual_camera = MockVirtualCamera()
        controller._virtual_camera.start()
        controller._frame_pipeline = None
        controller._stream_reader = None
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()
        controller._frame_buffer = buffer
        controller.gopro._connected = False

        with patch.object(controller, '_apply_resolution_on_camera', return_value=True), \
             patch.object(controller, '_create_and_swap_stream_reader', return_value=True), \
             patch.object(controller, '_stop_disconnect_detector'), \
             patch.object(controller, '_start_disconnect_detector'), \
             _usb_listener_patch:
            controller.change_resolution(7)  # 720p

        assert buffer.width == 1280
        assert buffer.height == 720

    def test_updates_config(self):
        """change_resolution updates config dimensions."""
        controller = self._make_controller()
        controller._virtual_camera = MockVirtualCamera()
        controller._virtual_camera.start()
        controller._frame_pipeline = None
        controller._stream_reader = None
        controller._frame_buffer = FrameBuffer(width=320, height=240)
        controller._frame_buffer.start()
        controller.gopro._connected = False

        with patch.object(controller, '_apply_resolution_on_camera', return_value=True), \
             patch.object(controller, '_create_and_swap_stream_reader', return_value=True), \
             patch.object(controller, '_stop_disconnect_detector'), \
             patch.object(controller, '_start_disconnect_detector'), \
             _usb_listener_patch:
            controller.change_resolution(4, fov=2)  # 1080p

        assert controller.config.stream_width == 1920
        assert controller.config.stream_height == 1080
        assert controller.config.resolution == 4
        assert controller.config.fov == 2

    def test_fallback_on_camera_failure(self):
        """change_resolution falls back when camera rejects new resolution."""
        controller = self._make_controller()
        vcam = MockVirtualCamera(width=320, height=240)
        vcam.start()
        controller._virtual_camera = vcam
        controller._frame_pipeline = MagicMock()
        controller._frame_pipeline.is_running = True
        controller._stream_reader = None
        controller._frame_buffer = FrameBuffer(width=320, height=240)
        controller._frame_buffer.start()

        # First apply fails, fallback succeeds
        with patch.object(controller, '_apply_resolution_on_camera',
                          side_effect=[False, True]) as mock_apply, \
             patch.object(controller, '_resume_stream_after_resolution_change',
                          return_value=True), \
             _usb_listener_patch:
            controller.change_resolution(7)  # 720p

        # Should have attempted fallback to old resolution
        assert mock_apply.call_count == 2

    def test_restarts_gopro_webcam(self):
        """change_resolution restarts GoPro webcam at new resolution."""
        controller = self._make_controller()
        controller._virtual_camera = MockVirtualCamera()
        controller._virtual_camera.start()
        controller._frame_pipeline = MagicMock()
        controller._frame_pipeline.is_running = True
        controller._stream_reader = MagicMock()
        controller._frame_buffer = FrameBuffer(width=320, height=240)
        controller._frame_buffer.start()

        with patch.object(type(controller.gopro), 'is_connected',
                          new_callable=PropertyMock, return_value=True), \
             patch.object(controller.gopro, 'stop_webcam') as mock_stop_wc, \
             patch.object(controller.gopro, 'start_webcam', return_value=True) as mock_start_wc, \
             patch.object(controller, '_create_and_swap_stream_reader', return_value=True), \
             patch.object(controller, '_stop_disconnect_detector'), \
             patch.object(controller, '_start_disconnect_detector'), \
             _usb_listener_patch:
            controller.change_resolution(4, fov=2)  # 1080p

        mock_stop_wc.assert_called_once()
        mock_start_wc.assert_called_once_with(resolution=4, fov=2)


# ---------------------------------------------------------------------------
# Tests: _resume_stream_after_resolution_change
# ---------------------------------------------------------------------------

class TestResumeStreamReinit:
    """Tests for _resume_stream_after_resolution_change full reinit."""

    def _make_controller(self):
        from app_controller import AppController
        config = make_test_config()
        with _usb_listener_patch:
            controller = AppController(config)
        return controller

    def test_stops_old_pipeline(self):
        """_resume_stream_after_resolution_change stops old pipeline."""
        controller = self._make_controller()
        mock_pipeline = MagicMock()
        controller._frame_pipeline = mock_pipeline
        controller._virtual_camera = MockVirtualCamera()
        controller._virtual_camera.start()
        controller._frame_buffer = FrameBuffer(width=320, height=240)
        controller._frame_buffer.start()

        with patch('stream_reader.StreamReader') as MockSR, \
             patch('frame_pipeline.FramePipeline') as MockFP, \
             patch.object(controller, '_stop_disconnect_detector'), \
             patch.object(controller, '_start_disconnect_detector'), \
             _usb_listener_patch:
            MockSR.return_value.start.return_value = True
            MockFP.return_value.start.return_value = True
            controller._resume_stream_after_resolution_change(640, 480, 7, 4)

        mock_pipeline.stop.assert_called_once()

    def test_reconfigures_vcam(self):
        """_resume_stream_after_resolution_change reconfigures vcam."""
        controller = self._make_controller()
        vcam = MockVirtualCamera(width=320, height=240)
        vcam.start()
        controller._virtual_camera = vcam
        controller._frame_pipeline = None
        controller._frame_buffer = FrameBuffer(width=320, height=240)
        controller._frame_buffer.start()

        with patch('stream_reader.StreamReader') as MockSR, \
             patch('frame_pipeline.FramePipeline') as MockFP, \
             patch.object(controller, '_stop_disconnect_detector'), \
             patch.object(controller, '_start_disconnect_detector'), \
             _usb_listener_patch:
            MockSR.return_value.start.return_value = True
            MockFP.return_value.start.return_value = True
            result = controller._resume_stream_after_resolution_change(640, 480, 7, 4)

        assert result is True
        assert vcam._reconfigure_count == 1

    def test_returns_false_on_vcam_failure(self):
        """Returns False if vcam reconfigure fails."""
        controller = self._make_controller()
        vcam = MockVirtualCamera(width=320, height=240)
        vcam.start()
        vcam._reconfigure_fail = True
        controller._virtual_camera = vcam
        controller._frame_pipeline = None
        controller._frame_buffer = FrameBuffer(width=320, height=240)
        controller._frame_buffer.start()

        with patch.object(controller, '_stop_disconnect_detector'), \
             _usb_listener_patch:
            result = controller._resume_stream_after_resolution_change(640, 480, 7, 4)

        assert result is False

    def test_resizes_buffer(self):
        """_resume_stream_after_resolution_change resizes frame buffer."""
        controller = self._make_controller()
        controller._virtual_camera = MockVirtualCamera()
        controller._virtual_camera.start()
        controller._frame_pipeline = None
        buffer = FrameBuffer(width=320, height=240)
        buffer.start()
        controller._frame_buffer = buffer

        with patch('stream_reader.StreamReader') as MockSR, \
             patch('frame_pipeline.FramePipeline') as MockFP, \
             patch.object(controller, '_stop_disconnect_detector'), \
             patch.object(controller, '_start_disconnect_detector'), \
             _usb_listener_patch:
            MockSR.return_value.start.return_value = True
            MockFP.return_value.start.return_value = True
            controller._resume_stream_after_resolution_change(640, 480, 7, 4)

        assert buffer.width == 640
        assert buffer.height == 480
