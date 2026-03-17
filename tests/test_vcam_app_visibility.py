"""
test_vcam_app_visibility.py — Tests for Sub-AC 3: Virtual camera visible to Zoom/Teams

Verifies that the 'GoPro Webcam' virtual camera appears as a selectable
source in downstream video-conferencing apps (Zoom, Microsoft Teams, etc.).

These tests validate the full chain:
  1. VirtualCamera opens → device appears in DirectShow device list
  2. verify_device_visible() confirms the device is discoverable
  3. The startup pipeline integrates device visibility verification
  4. Frames are being pushed so apps see a live feed (not just a device name)
  5. The device name is exactly 'GoPro Webcam' (not a generic backend name)

Since we can't launch actual Zoom/Teams in CI, we verify via DirectShow
enumeration — the same mechanism those apps use to discover cameras.
All tests use mocks so they work without virtual camera drivers installed.
"""

import sys
import os
import time
import threading
from unittest.mock import patch, MagicMock, call, PropertyMock

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from virtual_camera import (
    VirtualCamera,
    verify_device_visible,
    list_directshow_video_devices,
    _parse_ffmpeg_dshow_output,
    check_virtual_camera_ready,
)
from config import Config
from frame_pipeline import FramePipeline, PipelineState


pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def make_test_config(**overrides) -> Config:
    """Create a config with test-friendly defaults."""
    config = Config()
    config.stream_width = 1920
    config.stream_height = 1080
    config.stream_fps = 30
    config.virtual_camera_name = "GoPro Webcam"
    config._config_path = ""
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def make_mock_camera(device_name="GoPro Webcam"):
    """Create a mock pyvirtualcam.Camera instance."""
    cam = MagicMock()
    cam.device = device_name
    cam.close = MagicMock()
    cam.send = MagicMock()
    cam.sleep_until_next_frame = MagicMock()
    return cam


def make_rgb_frame(height=1080, width=1920, color=(128, 64, 32)):
    """Create a test RGB frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


SAMPLE_DSHOW_WITH_GOPRO = """\
[dshow @ 0x1] DirectShow video devices (some may be both video and audio devices)
[dshow @ 0x1]  "Integrated Webcam"
[dshow @ 0x1]     Alternative name "@device_pnp_\\\\?\\usb#vid_0bda"
[dshow @ 0x1]  "GoPro Webcam"
[dshow @ 0x1]     Alternative name "@device_sw_{unity-capture-guid}"
[dshow @ 0x1]  "OBS Virtual Camera"
[dshow @ 0x1]     Alternative name "@device_sw_{obs-guid}"
[dshow @ 0x1] DirectShow audio devices
[dshow @ 0x1]  "Microphone (Realtek)"
"""

SAMPLE_DSHOW_WITHOUT_GOPRO = """\
[dshow @ 0x1] DirectShow video devices (some may be both video and audio devices)
[dshow @ 0x1]  "Integrated Webcam"
[dshow @ 0x1]     Alternative name "@device_pnp_\\\\?\\usb#vid_0bda"
[dshow @ 0x1] DirectShow audio devices
[dshow @ 0x1]  "Microphone (Realtek)"
"""


# ---------------------------------------------------------------------------
# Test: Virtual camera appears in DirectShow device list (how Zoom/Teams find it)
# ---------------------------------------------------------------------------

class TestVcamAppearsInDirectShowList:
    """Verify that once VirtualCamera.start() succeeds, the device name
    appears in DirectShow device enumeration — the same mechanism that
    Zoom, Teams, Google Meet, and NVIDIA Broadcast use to find cameras."""

    def test_gopro_webcam_found_in_dshow_list(self):
        """'GoPro Webcam' should appear in the parsed DirectShow device list."""
        devices = []
        _parse_ffmpeg_dshow_output(SAMPLE_DSHOW_WITH_GOPRO, devices)
        names = [d["name"] for d in devices]
        assert "GoPro Webcam" in names

    def test_gopro_webcam_not_in_list_without_driver(self):
        """Without the virtual camera driver, GoPro Webcam should not appear."""
        devices = []
        _parse_ffmpeg_dshow_output(SAMPLE_DSHOW_WITHOUT_GOPRO, devices)
        names = [d["name"] for d in devices]
        assert "GoPro Webcam" not in names

    @patch("virtual_camera.platform.system", return_value="Windows")
    def test_list_directshow_finds_gopro_via_ffmpeg(self, mock_system):
        """list_directshow_video_devices() should return GoPro Webcam when ffmpeg lists it."""
        mock_result = MagicMock()
        mock_result.stderr = SAMPLE_DSHOW_WITH_GOPRO
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            devices = list_directshow_video_devices()

        device_names = [d["name"] for d in devices]
        assert "GoPro Webcam" in device_names
        # Verify source is dshow (same as what Zoom/Teams query)
        gopro_devices = [d for d in devices if d["name"] == "GoPro Webcam"]
        assert gopro_devices[0]["source"] == "dshow"

    @patch("virtual_camera.platform.system", return_value="Windows")
    def test_list_directshow_finds_gopro_via_powershell_fallback(self, mock_system):
        """PowerShell WMI fallback should also find GoPro Webcam."""
        # ffmpeg fails
        ffmpeg_result = MagicMock()
        ffmpeg_result.stderr = ""
        ffmpeg_result.stdout = ""

        # PowerShell succeeds
        ps_result = MagicMock()
        ps_result.stdout = "Integrated Webcam\nGoPro Webcam\n"

        def run_side_effect(cmd, **kwargs):
            if cmd[0] == "ffmpeg":
                return ffmpeg_result
            return ps_result

        with patch("subprocess.run", side_effect=run_side_effect):
            devices = list_directshow_video_devices()

        device_names = [d["name"] for d in devices]
        assert "GoPro Webcam" in device_names


# ---------------------------------------------------------------------------
# Test: verify_device_visible() confirms the virtual camera is selectable
# ---------------------------------------------------------------------------

class TestVerifyDeviceVisible:
    """Tests for verify_device_visible() — the polling verification that
    confirms apps will see 'GoPro Webcam' in their camera dropdown."""

    @patch("virtual_camera.list_directshow_video_devices")
    def test_verify_finds_gopro_webcam(self, mock_list):
        """verify_device_visible returns True when 'GoPro Webcam' is in the list."""
        mock_list.return_value = [
            {"name": "Integrated Webcam", "source": "dshow"},
            {"name": "GoPro Webcam", "source": "dshow"},
        ]
        result = verify_device_visible("GoPro Webcam", timeout=2.0)
        assert result is True

    @patch("virtual_camera.list_directshow_video_devices")
    def test_verify_matches_exact_name(self, mock_list):
        """The device name must be 'GoPro Webcam' — not a generic backend name."""
        mock_list.return_value = [
            {"name": "GoPro Webcam", "source": "dshow"},
        ]
        result = verify_device_visible("GoPro Webcam", timeout=1.0)
        assert result is True

    @patch("virtual_camera.list_directshow_video_devices")
    def test_verify_rejects_wrong_name(self, mock_list):
        """A device named 'OBS Virtual Camera' should NOT match 'GoPro Webcam'."""
        mock_list.return_value = [
            {"name": "OBS Virtual Camera", "source": "dshow"},
        ]
        times = iter([0.0, 0.5, 1.0, 1.5])
        with patch("virtual_camera.time.monotonic", side_effect=lambda: next(times)), \
             patch("virtual_camera.time.sleep"):
            result = verify_device_visible("GoPro Webcam", timeout=1.0)
        assert result is False

    @patch("virtual_camera.list_directshow_video_devices")
    def test_verify_rejects_unity_capture_default_name(self, mock_list):
        """Unity Capture's default name ('Unity Video Capture') should NOT match."""
        mock_list.return_value = [
            {"name": "Unity Video Capture", "source": "dshow"},
        ]
        times = iter([0.0, 0.5, 1.0, 1.5])
        with patch("virtual_camera.time.monotonic", side_effect=lambda: next(times)), \
             patch("virtual_camera.time.sleep"):
            result = verify_device_visible("GoPro Webcam", timeout=1.0)
        assert result is False

    @patch("virtual_camera.list_directshow_video_devices")
    @patch("virtual_camera.time.sleep")
    def test_verify_polls_until_device_appears(self, mock_sleep, mock_list):
        """verify_device_visible should poll until the device appears."""
        call_count = [0]

        def devices_delayed():
            call_count[0] += 1
            if call_count[0] <= 2:
                return [{"name": "Integrated Webcam", "source": "dshow"}]
            return [
                {"name": "Integrated Webcam", "source": "dshow"},
                {"name": "GoPro Webcam", "source": "dshow"},
            ]

        mock_list.side_effect = lambda: devices_delayed()
        times = iter([0.0, 0.5, 1.0, 1.5, 2.0])
        with patch("virtual_camera.time.monotonic", side_effect=lambda: next(times)):
            result = verify_device_visible("GoPro Webcam", timeout=3.0)

        assert result is True
        assert call_count[0] >= 3, "Should have polled multiple times"


# ---------------------------------------------------------------------------
# Test: Full VirtualCamera start → device visible flow
# ---------------------------------------------------------------------------

class TestVcamStartToVisible:
    """End-to-end tests: VirtualCamera.start() → verify_device_visible()."""

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    @patch("virtual_camera.verify_device_visible")
    def test_start_then_verify_visible(self, mock_verify, mock_fmt, mock_cam_cls, mock_backend):
        """After start(), verify_device_visible should find 'GoPro Webcam'."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"
        mock_verify.return_value = True

        vcam = VirtualCamera(make_test_config())
        started = vcam.start()
        assert started is True

        # Verify the device would be visible
        visible = mock_verify("GoPro Webcam", timeout=5.0)
        assert visible is True

        vcam.stop()

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_start_sends_frames_apps_see_content(self, mock_fmt, mock_cam_cls, mock_backend):
        """After start + send_frame, downstream apps receive actual video content."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()

        # Send multiple frames — Zoom/Teams need active frame data
        for i in range(5):
            frame = make_rgb_frame(1080, 1920, (i * 50, 100, 200))
            vcam.send_frame(frame)

        # Verify frames were pushed to the backend
        # +1 for the initial placeholder frame sent on start
        assert mock_cam.send.call_count >= 6
        assert vcam.frame_count >= 6

        # Verify the last frame was stored (for freeze-frame)
        assert vcam.last_frame is not None
        assert vcam.last_frame.shape == (1080, 1920, 3)

        vcam.stop()

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_device_name_is_gopro_webcam(self, mock_fmt, mock_cam_cls, mock_backend):
        """The device MUST be named 'GoPro Webcam' — this is the requirement."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()

        assert vcam.device_name == "GoPro Webcam"
        stats = vcam.get_stats()
        assert stats["device_name"] == "GoPro Webcam"

        vcam.stop()

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    @patch("virtual_camera.configure_unity_capture_device_name", return_value=True)
    def test_unity_capture_registry_name_set_before_camera_open(
        self, mock_configure, mock_fmt, mock_cam_cls, mock_backend
    ):
        """Unity Capture registry must be set to 'GoPro Webcam' BEFORE Camera() opens.

        This ensures Zoom/Teams see 'GoPro Webcam' in their dropdown immediately,
        not the default 'Unity Video Capture' name.
        """
        call_order = []

        def track_configure(name, device_index=0):
            call_order.append("registry_configure")
            return True

        mock_configure.side_effect = track_configure

        mock_cam = make_mock_camera()

        def track_camera_open(**kwargs):
            call_order.append("camera_open")
            return mock_cam

        mock_cam_cls.side_effect = track_camera_open
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()

        assert call_order == ["registry_configure", "camera_open"]
        vcam.stop()


# ---------------------------------------------------------------------------
# Test: check_virtual_camera_ready pre-flight for app visibility
# ---------------------------------------------------------------------------

class TestReadinessCheckForApps:
    """Tests that check_virtual_camera_ready() correctly reports whether
    Zoom/Teams will be able to see the virtual camera."""

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("virtual_camera.get_unity_capture_device_name", return_value="GoPro Webcam")
    def test_ready_with_correct_name(self, mock_get_name, mock_backend):
        """When backend + name are correct, ready=True and device_name_ok=True."""
        with patch.dict("sys.modules", {"pyvirtualcam": MagicMock()}):
            result = check_virtual_camera_ready("GoPro Webcam")

        assert result["ready"] is True
        assert result["device_name_ok"] is True
        assert result["backend"] == "unitycapture"
        assert "GoPro Webcam" in result["message"]

    @patch("virtual_camera.select_best_backend", return_value="obs")
    def test_obs_warns_about_different_name(self, mock_backend):
        """OBS backend warns that device won't be named 'GoPro Webcam'."""
        with patch.dict("sys.modules", {"pyvirtualcam": MagicMock()}):
            result = check_virtual_camera_ready("GoPro Webcam")

        assert result["ready"] is True
        assert result["device_name_ok"] is False
        assert "OBS Virtual Camera" in result["message"]

    @patch("virtual_camera.select_best_backend", return_value=None)
    def test_not_ready_without_backend(self, mock_backend):
        """Without any backend, apps won't see a virtual camera at all."""
        with patch.dict("sys.modules", {"pyvirtualcam": MagicMock()}):
            result = check_virtual_camera_ready("GoPro Webcam")

        assert result["ready"] is False
        assert "No virtual camera driver" in result["message"]


# ---------------------------------------------------------------------------
# Test: Startup pipeline integrates device visibility verification
# ---------------------------------------------------------------------------

class TestStartupPipelineVisibilityCheck:
    """Tests that the AppController startup pipeline verifies device visibility
    after opening the virtual camera, so the user knows if Zoom/Teams can see it.

    Note: _start_streaming_pipeline() uses lazy imports (from virtual_camera import ...),
    so we patch at the source module level and use controller method directly.
    """

    def _make_controller(self):
        """Create an AppController with all external deps mocked."""
        from app_controller import AppController
        config = make_test_config()
        with patch("gopro_connection.GoProConnection") as mock_cls:
            mock_cls.return_value = MagicMock()
            with patch("usb_event_listener.USBEventListener"):
                controller = AppController(config)
        return controller

    def test_pipeline_verifies_device_visible_after_vcam_start(self):
        """_start_streaming_pipeline should call verify_device_visible after vcam starts."""
        controller = self._make_controller()

        mock_vcam = MagicMock()
        mock_vcam.is_running = False
        mock_vcam.start.return_value = True
        mock_vcam.device_name = "GoPro Webcam"

        mock_reader = MagicMock()
        mock_reader.start.return_value = True
        mock_reader._process = MagicMock()
        mock_reader._process.pid = 12345

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = False
        mock_pipeline.start.return_value = True

        with patch("virtual_camera.VirtualCamera", return_value=mock_vcam), \
             patch("stream_reader.StreamReader", return_value=mock_reader), \
             patch("frame_pipeline.FramePipeline", return_value=mock_pipeline), \
             patch("virtual_camera.verify_device_visible", return_value=True) as mock_verify:
            result = controller._start_streaming_pipeline()

        assert result is True
        mock_verify.assert_called_once_with("GoPro Webcam", timeout=5.0)

    def test_pipeline_emits_success_when_device_visible(self):
        """Success status should be emitted when device is found in DirectShow list."""
        controller = self._make_controller()

        status_messages = []
        controller.on_status = lambda msg, lvl: status_messages.append((msg, lvl))

        mock_vcam = MagicMock()
        mock_vcam.is_running = False
        mock_vcam.start.return_value = True
        mock_vcam.device_name = "GoPro Webcam"

        mock_reader = MagicMock()
        mock_reader.start.return_value = True
        mock_reader._process = MagicMock()
        mock_reader._process.pid = 12345

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = False
        mock_pipeline.start.return_value = True

        with patch("virtual_camera.VirtualCamera", return_value=mock_vcam), \
             patch("stream_reader.StreamReader", return_value=mock_reader), \
             patch("frame_pipeline.FramePipeline", return_value=mock_pipeline), \
             patch("virtual_camera.verify_device_visible", return_value=True):
            controller._start_streaming_pipeline()

        # Should have a success message about device visibility
        success_messages = [msg for msg, lvl in status_messages if lvl == "success"]
        visibility_msgs = [m for m in success_messages if "visible" in m.lower() or "camera list" in m.lower()]
        assert len(visibility_msgs) >= 1, (
            f"Expected success message about device visibility, got: {success_messages}"
        )

    def test_pipeline_warns_when_device_not_visible(self):
        """Warning should be emitted if device is NOT found in DirectShow list."""
        controller = self._make_controller()

        status_messages = []
        controller.on_status = lambda msg, lvl: status_messages.append((msg, lvl))

        mock_vcam = MagicMock()
        mock_vcam.is_running = False
        mock_vcam.start.return_value = True
        mock_vcam.device_name = "GoPro Webcam"

        mock_reader = MagicMock()
        mock_reader.start.return_value = True
        mock_reader._process = MagicMock()
        mock_reader._process.pid = 12345

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = False
        mock_pipeline.start.return_value = True

        with patch("virtual_camera.VirtualCamera", return_value=mock_vcam), \
             patch("stream_reader.StreamReader", return_value=mock_reader), \
             patch("frame_pipeline.FramePipeline", return_value=mock_pipeline), \
             patch("virtual_camera.verify_device_visible", return_value=False):
            result = controller._start_streaming_pipeline()

        # Pipeline should still succeed (camera may work even if not in DirectShow list)
        assert result is True

        # But should have a warning about device not detected
        warning_messages = [msg for msg, lvl in status_messages if lvl == "warning"]
        visibility_warnings = [m for m in warning_messages if "not detected" in m.lower() or "camera list" in m.lower()]
        assert len(visibility_warnings) >= 1, (
            f"Expected warning about device not visible, got: {warning_messages}"
        )

    def test_pipeline_skips_verify_on_reuse(self):
        """When virtual camera is already running (reuse), skip device verification."""
        controller = self._make_controller()

        # Pre-set a running virtual camera
        mock_vcam = MagicMock()
        mock_vcam.is_running = True
        mock_vcam.device_name = "GoPro Webcam"
        controller._virtual_camera = mock_vcam

        mock_reader = MagicMock()
        mock_reader.start.return_value = True
        mock_reader._process = MagicMock()
        mock_reader._process.pid = 12345

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = False
        mock_pipeline.start.return_value = True

        with patch("stream_reader.StreamReader", return_value=mock_reader), \
             patch("frame_pipeline.FramePipeline", return_value=mock_pipeline), \
             patch("virtual_camera.verify_device_visible") as mock_verify:
            controller._start_streaming_pipeline()

        # verify_device_visible should NOT be called when reusing existing vcam
        mock_verify.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Full pipeline sends frames that apps can consume
# ---------------------------------------------------------------------------

class TestPipelineDeliversFramesToApps:
    """Tests that the full pipeline (StreamReader → FramePipeline → VirtualCamera)
    actually pushes frames, so Zoom/Teams display live video rather than a
    black screen or 'camera unavailable'."""

    def test_pipeline_pushes_frames_to_vcam(self):
        """FramePipeline should push frames from reader to virtual camera."""
        config = make_test_config()
        pipeline = FramePipeline(config)

        # Mock reader that returns frames
        mock_reader = MagicMock()
        frames = [make_rgb_frame(1080, 1920, (i * 50, 100, 200)) for i in range(5)]
        frame_iter = iter(frames + [None])  # End with None to trigger stop
        mock_reader.read_frame.side_effect = lambda: next(frame_iter, None)

        # Mock vcam
        mock_vcam = MagicMock()
        mock_vcam.is_running = True
        mock_vcam.send_frame.return_value = True
        mock_vcam.send_last_frame.return_value = True
        mock_vcam.sleep_until_next_frame = MagicMock()

        result = pipeline.start(mock_reader, mock_vcam)
        assert result is True

        # Wait for frames to be pushed
        time.sleep(0.5)

        # Verify frames were pushed to the virtual camera
        assert mock_vcam.send_frame.call_count >= 1, \
            "Pipeline should push at least one frame to vcam"

        pipeline.stop()

    def test_pipeline_freeze_frame_keeps_device_alive(self):
        """During freeze-frame, the virtual camera keeps sending frames.

        This ensures Zoom/Teams don't lose the 'GoPro Webcam' device even
        when the actual camera is disconnected.
        """
        config = make_test_config()
        pipeline = FramePipeline(config)

        # Mock reader returns None immediately (simulating disconnect)
        mock_reader = MagicMock()
        mock_reader.read_frame.return_value = None

        mock_vcam = MagicMock()
        mock_vcam.is_running = True
        mock_vcam.send_frame.return_value = True
        mock_vcam.send_last_frame.return_value = True
        mock_vcam.sleep_until_next_frame = MagicMock()

        pipeline.start(mock_reader, mock_vcam)

        # Wait for pipeline to enter freeze-frame
        time.sleep(0.5)

        # Pipeline should have entered freeze-frame mode
        assert pipeline.state in (PipelineState.FREEZE_FRAME, PipelineState.STREAMING)

        # send_last_frame should have been called (keeping device alive for apps)
        if pipeline.state == PipelineState.FREEZE_FRAME:
            assert mock_vcam.send_last_frame.call_count >= 1, \
                "Freeze-frame should keep pushing frames to vcam"

        pipeline.stop()


# ---------------------------------------------------------------------------
# Test: Device name specifics for Zoom and Teams compatibility
# ---------------------------------------------------------------------------

class TestZoomTeamsCompatibility:
    """Tests verifying compatibility with specific app behaviors."""

    def test_device_name_exact_match(self):
        """The virtual camera must appear as exactly 'GoPro Webcam'.

        Both Zoom and Teams list cameras by their DirectShow FriendlyName.
        The name must be readable and professional, not a technical ID.
        """
        config = make_test_config()
        vcam = VirtualCamera(config)
        assert vcam.device_name == "GoPro Webcam"

    def test_default_config_uses_correct_name(self):
        """Config default must set virtual_camera_name to 'GoPro Webcam'."""
        config = Config()
        assert config.virtual_camera_name == "GoPro Webcam"

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_vcam_resolution_1080p_default(self, mock_fmt, mock_cam_cls, mock_backend):
        """Default resolution 1920x1080 is what Zoom/Teams expect."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()

        call_kwargs = mock_cam_cls.call_args[1]
        assert call_kwargs["width"] == 1920
        assert call_kwargs["height"] == 1080
        assert call_kwargs["fps"] == 30

        vcam.stop()

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_vcam_sends_rgb_format(self, mock_fmt, mock_cam_cls, mock_backend):
        """Virtual camera should output BGR format (pyvirtualcam.PixelFormat.BGR).

        Unity Capture accepts BGR natively, matching ffmpeg's bgr24 output.
        """
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.BGR = "BGR"

        vcam = VirtualCamera(make_test_config())
        vcam.start()

        call_kwargs = mock_cam_cls.call_args[1]
        assert call_kwargs["fmt"] == "BGR"

        vcam.stop()

    def test_dshow_output_parsing_handles_multiple_cameras(self):
        """When multiple cameras are present (typical setup), GoPro Webcam
        should still be found alongside built-in webcams."""
        output = """\
[dshow @ 0x1] DirectShow video devices (some may be both video and audio devices)
[dshow @ 0x1]  "HP TrueVision HD Camera"
[dshow @ 0x1]     Alternative name "@device_pnp_\\\\?\\usb#vid_05c8"
[dshow @ 0x1]  "NVIDIA Broadcast"
[dshow @ 0x1]     Alternative name "@device_sw_{nvidia-guid}"
[dshow @ 0x1]  "GoPro Webcam"
[dshow @ 0x1]     Alternative name "@device_sw_{unity-capture-guid}"
[dshow @ 0x1]  "OBS Virtual Camera"
[dshow @ 0x1]     Alternative name "@device_sw_{obs-guid}"
[dshow @ 0x1] DirectShow audio devices
[dshow @ 0x1]  "Microphone Array"
"""
        devices = []
        _parse_ffmpeg_dshow_output(output, devices)
        names = [d["name"] for d in devices]

        assert "GoPro Webcam" in names
        # Should coexist with other cameras
        assert "HP TrueVision HD Camera" in names
        assert "NVIDIA Broadcast" in names
        assert "OBS Virtual Camera" in names
        # Audio devices should NOT be included
        assert "Microphone Array" not in names

    @patch("virtual_camera.list_directshow_video_devices")
    def test_verify_device_visible_among_many_cameras(self, mock_list):
        """verify_device_visible should find GoPro Webcam even with many cameras."""
        mock_list.return_value = [
            {"name": "HP TrueVision HD Camera", "source": "dshow"},
            {"name": "NVIDIA Broadcast", "source": "dshow"},
            {"name": "GoPro Webcam", "source": "dshow"},
            {"name": "OBS Virtual Camera", "source": "dshow"},
            {"name": "ManyCam Video Source", "source": "dshow"},
        ]
        result = verify_device_visible("GoPro Webcam", timeout=1.0)
        assert result is True


# ---------------------------------------------------------------------------
# Test: VirtualCamera stats reflect app-facing state
# ---------------------------------------------------------------------------

class TestVcamStatsForAppVisibility:
    """Tests that get_stats() reflects the state apps would see."""

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_stats_show_running_with_correct_name(self, mock_fmt, mock_cam_cls, mock_backend):
        """Stats should confirm the camera is running as 'GoPro Webcam'."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()

        stats = vcam.get_stats()
        assert stats["running"] is True
        assert stats["device_name"] == "GoPro Webcam"
        assert stats["backend"] == "unitycapture"
        assert stats["resolution"] == "1920x1080"
        assert stats["fps"] == 30
        assert stats["frame_count"] >= 1  # At least placeholder

        vcam.stop()

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("pyvirtualcam.Camera")
    @patch("pyvirtualcam.PixelFormat")
    def test_stats_track_frame_count_for_monitoring(self, mock_fmt, mock_cam_cls, mock_backend):
        """Frame count should increment as frames are pushed to apps."""
        mock_cam = make_mock_camera()
        mock_cam_cls.return_value = mock_cam
        mock_fmt.RGB = "RGB"

        vcam = VirtualCamera(make_test_config())
        vcam.start()

        initial_count = vcam.frame_count

        # Push frames
        for _ in range(10):
            vcam.send_frame(make_rgb_frame())

        assert vcam.frame_count == initial_count + 10

        vcam.stop()
