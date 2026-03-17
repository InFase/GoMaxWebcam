"""
test_virtual_camera_naming.py — Tests for virtual camera naming & discovery

Verifies that:
  1. Unity Capture registry device name is configured correctly
  2. OBS VirtualCam fallback is detected with appropriate warnings
  3. DirectShow device enumeration parses ffmpeg output correctly
  4. verify_device_visible() correctly finds (or times out on) devices
  5. check_virtual_camera_ready() returns correct status for each scenario
  6. VirtualCamera.start() configures device name before opening
  7. Device name mismatch is detected and logged

All tests use mocks — no actual virtual camera driver or registry needed.
"""

import sys
import os
import time
from unittest.mock import patch, MagicMock, call

import pytest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from virtual_camera import (
    VirtualCamera,
    check_backend_available,
    select_best_backend,
    configure_unity_capture_device_name,
    get_unity_capture_device_name,
    list_directshow_video_devices,
    _parse_ffmpeg_dshow_output,
    verify_device_visible,
    check_virtual_camera_ready,
    _BACKEND_PREFERENCE,
    _PLACEHOLDER_COLOR,
)
from config import Config

pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_config(**overrides) -> Config:
    """Create a config with defaults for testing."""
    config = Config()
    config.stream_width = 1920
    config.stream_height = 1080
    config.stream_fps = 30
    config.virtual_camera_name = "GoPro Webcam"
    config._config_path = ""
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


SAMPLE_FFMPEG_DSHOW_OUTPUT = """\
ffmpeg version 6.1 Copyright (c) 2000-2024 the FFmpeg developers
[dshow @ 000001] DirectShow video devices (some may be both video and audio devices)
[dshow @ 000001]  "Integrated Webcam"
[dshow @ 000001]     Alternative name "@device_pnp_\\\\?\\usb#vid_0bda"
[dshow @ 000001]  "GoPro Webcam"
[dshow @ 000001]     Alternative name "@device_sw_{some-guid}"
[dshow @ 000001]  "OBS Virtual Camera"
[dshow @ 000001]     Alternative name "@device_sw_{obs-guid}"
[dshow @ 000001] DirectShow audio devices
[dshow @ 000001]  "Microphone (Realtek)"
"""

SAMPLE_FFMPEG_NO_GOPRO = """\
[dshow @ 000001] DirectShow video devices (some may be both video and audio devices)
[dshow @ 000001]  "Integrated Webcam"
[dshow @ 000001]     Alternative name "@device_pnp_\\\\?\\usb#vid_0bda"
[dshow @ 000001] DirectShow audio devices
[dshow @ 000001]  "Microphone"
"""

# Modern ffmpeg format (no section headers, inline type suffixes)
SAMPLE_FFMPEG_MODERN_OUTPUT = """\
ffmpeg version 2025-11-17-git-e94439e49b-essentials_build Copyright (c) 2000-2025
[dshow @ 000001be5ff66d40] "Camera (NVIDIA Broadcast)" (video)
[dshow @ 000001be5ff66d40]   Alternative name "@device_sw_{860BB310-5D01-11D0-BD3B-00A0C911CE86}\\{7BBFF097}"
[dshow @ 000001be5ff66d40] "GoPro Webcam" (video)
[dshow @ 000001be5ff66d40]   Alternative name "@device_sw_{860BB310-5D01-11D0-BD3B-00A0C911CE86}\\{FDB60968}"
[dshow @ 000001be5ff66d40] "Microphone (Realtek(R) Audio)" (audio)
[dshow @ 000001be5ff66d40]   Alternative name "@device_cm_{33D9A762}\\wave_{F0E8F177}"
"""

SAMPLE_FFMPEG_MODERN_NO_GOPRO = """\
[dshow @ 000001] "Integrated Webcam" (video)
[dshow @ 000001]   Alternative name "@device_pnp_\\\\?\\usb#vid_0bda"
[dshow @ 000001] "Microphone" (audio)
"""


# ---------------------------------------------------------------------------
# Tests: _parse_ffmpeg_dshow_output
# ---------------------------------------------------------------------------

class TestParseFFmpegDshowOutput:
    """Tests for parsing ffmpeg -list_devices dshow output."""

    def test_parse_finds_video_devices(self):
        devices = []
        _parse_ffmpeg_dshow_output(SAMPLE_FFMPEG_DSHOW_OUTPUT, devices)
        names = [d["name"] for d in devices]
        assert "Integrated Webcam" in names
        assert "GoPro Webcam" in names
        assert "OBS Virtual Camera" in names

    def test_parse_excludes_audio_devices(self):
        devices = []
        _parse_ffmpeg_dshow_output(SAMPLE_FFMPEG_DSHOW_OUTPUT, devices)
        names = [d["name"] for d in devices]
        assert "Microphone (Realtek)" not in names

    def test_parse_excludes_alternative_names(self):
        """Alternative name lines starting with @ should be skipped."""
        devices = []
        _parse_ffmpeg_dshow_output(SAMPLE_FFMPEG_DSHOW_OUTPUT, devices)
        for dev in devices:
            assert not dev["name"].startswith("@"), f"Got alternative name: {dev['name']}"

    def test_parse_all_have_dshow_source(self):
        devices = []
        _parse_ffmpeg_dshow_output(SAMPLE_FFMPEG_DSHOW_OUTPUT, devices)
        assert all(d["source"] == "dshow" for d in devices)

    def test_parse_empty_output(self):
        devices = []
        _parse_ffmpeg_dshow_output("", devices)
        assert devices == []

    def test_parse_no_video_section(self):
        devices = []
        _parse_ffmpeg_dshow_output("some random ffmpeg output\nno devices here", devices)
        assert devices == []

    def test_parse_video_section_with_no_devices(self):
        output = (
            "[dshow @ 0x1] DirectShow video devices\n"
            "[dshow @ 0x1] DirectShow audio devices\n"
        )
        devices = []
        _parse_ffmpeg_dshow_output(output, devices)
        assert devices == []

    # --- Modern ffmpeg format tests (no section headers) ---

    def test_parse_modern_finds_video_devices(self):
        """Modern ffmpeg format with (video)/(audio) suffixes is parsed correctly."""
        devices = []
        _parse_ffmpeg_dshow_output(SAMPLE_FFMPEG_MODERN_OUTPUT, devices)
        names = [d["name"] for d in devices]
        assert "Camera (NVIDIA Broadcast)" in names
        assert "GoPro Webcam" in names

    def test_parse_modern_excludes_audio_devices(self):
        """Modern format: audio devices with (audio) suffix are excluded."""
        devices = []
        _parse_ffmpeg_dshow_output(SAMPLE_FFMPEG_MODERN_OUTPUT, devices)
        names = [d["name"] for d in devices]
        assert "Microphone (Realtek(R) Audio)" not in names

    def test_parse_modern_excludes_alternative_names(self):
        """Modern format: alternative name lines (starting with @) are skipped."""
        devices = []
        _parse_ffmpeg_dshow_output(SAMPLE_FFMPEG_MODERN_OUTPUT, devices)
        for dev in devices:
            assert not dev["name"].startswith("@"), f"Got alternative name: {dev['name']}"

    def test_parse_modern_all_have_dshow_source(self):
        devices = []
        _parse_ffmpeg_dshow_output(SAMPLE_FFMPEG_MODERN_OUTPUT, devices)
        assert len(devices) > 0
        assert all(d["source"] == "dshow" for d in devices)

    def test_parse_modern_no_gopro(self):
        """Modern format without GoPro device."""
        devices = []
        _parse_ffmpeg_dshow_output(SAMPLE_FFMPEG_MODERN_NO_GOPRO, devices)
        names = [d["name"] for d in devices]
        assert "Integrated Webcam" in names
        assert "GoPro Webcam" not in names
        assert "Microphone" not in names


# ---------------------------------------------------------------------------
# Tests: configure_unity_capture_device_name
# ---------------------------------------------------------------------------

class TestConfigureUnityCaptureDeviceName:
    """Tests for Unity Capture registry device name configuration."""

    @patch("virtual_camera.platform.system", return_value="Windows")
    def test_sets_friendly_name_in_registry(self, mock_system):
        """Should write FriendlyName to HKCU\\SOFTWARE\\UnityCapture\\Device 0."""
        mock_key = MagicMock()
        with patch("winreg.CreateKeyEx", return_value=mock_key) as mock_create, \
             patch("winreg.SetValueEx") as mock_set, \
             patch("winreg.CloseKey") as mock_close:

            result = configure_unity_capture_device_name("GoPro Webcam", device_index=0)

            assert result is True
            mock_create.assert_called_once()
            # Verify the key path includes Device 0
            call_args = mock_create.call_args
            assert "Device 0" in call_args[0][1]
            # Verify FriendlyName was set
            mock_set.assert_called_once()
            set_args = mock_set.call_args[0]
            assert set_args[0] == mock_key
            assert set_args[1] == "FriendlyName"
            # SetValueEx(key, name, reserved, type, value)
            assert set_args[4] == "GoPro Webcam"
            mock_close.assert_called_once_with(mock_key)

    @patch("virtual_camera.platform.system", return_value="Windows")
    def test_custom_device_index(self, mock_system):
        """Should use the specified device index in the registry path."""
        mock_key = MagicMock()
        with patch("winreg.CreateKeyEx", return_value=mock_key) as mock_create, \
             patch("winreg.SetValueEx"), \
             patch("winreg.CloseKey"):

            configure_unity_capture_device_name("GoPro Webcam", device_index=2)

            call_args = mock_create.call_args
            assert "Device 2" in call_args[0][1]

    @patch("virtual_camera.platform.system", return_value="Windows")
    def test_registry_error_returns_false(self, mock_system):
        """Should return False if registry write fails."""
        with patch("winreg.CreateKeyEx", side_effect=OSError("Access denied")):
            result = configure_unity_capture_device_name("GoPro Webcam")
            assert result is False

    @patch("virtual_camera.platform.system", return_value="Linux")
    def test_non_windows_returns_false(self, mock_system):
        """Should return False on non-Windows platforms."""
        result = configure_unity_capture_device_name("GoPro Webcam")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: get_unity_capture_device_name
# ---------------------------------------------------------------------------

class TestGetUnityCaptureDeviceName:
    """Tests for reading Unity Capture device name from registry."""

    @patch("virtual_camera.platform.system", return_value="Windows")
    def test_reads_friendly_name(self, mock_system):
        mock_key = MagicMock()
        with patch("winreg.OpenKey", return_value=mock_key), \
             patch("winreg.QueryValueEx", return_value=("GoPro Webcam", 1)), \
             patch("winreg.CloseKey"):

            result = get_unity_capture_device_name()
            assert result == "GoPro Webcam"

    @patch("virtual_camera.platform.system", return_value="Windows")
    def test_returns_none_when_key_missing(self, mock_system):
        with patch("winreg.OpenKey", side_effect=FileNotFoundError):
            result = get_unity_capture_device_name()
            assert result is None

    @patch("virtual_camera.platform.system", return_value="Linux")
    def test_non_windows_returns_none(self, mock_system):
        result = get_unity_capture_device_name()
        assert result is None


# ---------------------------------------------------------------------------
# Tests: list_directshow_video_devices
# ---------------------------------------------------------------------------

class TestListDirectShowVideoDevices:
    """Tests for DirectShow device enumeration."""

    @patch("virtual_camera.platform.system", return_value="Windows")
    def test_uses_ffmpeg_dshow(self, mock_system):
        """Should parse ffmpeg dshow output when available."""
        mock_result = MagicMock()
        mock_result.stderr = SAMPLE_FFMPEG_DSHOW_OUTPUT
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            devices = list_directshow_video_devices()

        names = [d["name"] for d in devices]
        assert "GoPro Webcam" in names
        assert "Integrated Webcam" in names

    @patch("virtual_camera.platform.system", return_value="Windows")
    def test_falls_back_to_powershell(self, mock_system):
        """When ffmpeg fails, should fall back to PowerShell WMI query."""
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

        names = [d["name"] for d in devices]
        assert "GoPro Webcam" in names
        assert all(d["source"] == "wmi" for d in devices)

    @patch("virtual_camera.platform.system", return_value="Linux")
    def test_non_windows_returns_empty(self, mock_system):
        devices = list_directshow_video_devices()
        assert devices == []

    @patch("virtual_camera.platform.system", return_value="Windows")
    def test_both_fail_returns_empty(self, mock_system):
        """If both ffmpeg and PowerShell fail, returns empty list."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            devices = list_directshow_video_devices()
        assert devices == []


# ---------------------------------------------------------------------------
# Tests: verify_device_visible
# ---------------------------------------------------------------------------

class TestVerifyDeviceVisible:
    """Tests for verify_device_visible() polling."""

    @patch("virtual_camera.list_directshow_video_devices")
    def test_finds_device_immediately(self, mock_list):
        mock_list.return_value = [
            {"name": "Integrated Webcam", "source": "dshow"},
            {"name": "GoPro Webcam", "source": "dshow"},
        ]
        result = verify_device_visible("GoPro Webcam", timeout=2.0)
        assert result is True

    @patch("virtual_camera.list_directshow_video_devices")
    def test_case_insensitive_match(self, mock_list):
        mock_list.return_value = [
            {"name": "gopro webcam", "source": "dshow"},
        ]
        result = verify_device_visible("GoPro Webcam", timeout=1.0)
        assert result is True

    @patch("virtual_camera.list_directshow_video_devices")
    def test_substring_match(self, mock_list):
        """Should match if expected name is a substring."""
        mock_list.return_value = [
            {"name": "Unity Video Capture - GoPro Webcam", "source": "dshow"},
        ]
        result = verify_device_visible("GoPro Webcam", timeout=1.0)
        assert result is True

    @patch("virtual_camera.list_directshow_video_devices")
    @patch("virtual_camera.time.sleep")
    def test_timeout_when_device_not_found(self, mock_sleep, mock_list):
        mock_list.return_value = [
            {"name": "Integrated Webcam", "source": "dshow"},
        ]
        # Override time.monotonic to simulate passage of time
        times = iter([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        with patch("virtual_camera.time.monotonic", side_effect=lambda: next(times)):
            result = verify_device_visible("GoPro Webcam", timeout=2.0)
        assert result is False

    @patch("virtual_camera.list_directshow_video_devices")
    @patch("virtual_camera.time.sleep")
    def test_finds_device_on_retry(self, mock_sleep, mock_list):
        """Device appears after first poll fails."""
        call_count = [0]

        def devices_with_delay():
            call_count[0] += 1
            if call_count[0] <= 1:
                return [{"name": "Integrated Webcam", "source": "dshow"}]
            return [
                {"name": "Integrated Webcam", "source": "dshow"},
                {"name": "GoPro Webcam", "source": "dshow"},
            ]

        mock_list.side_effect = lambda: devices_with_delay()
        times = iter([0.0, 0.5, 1.0, 1.5])
        with patch("virtual_camera.time.monotonic", side_effect=lambda: next(times)):
            result = verify_device_visible("GoPro Webcam", timeout=3.0)
        assert result is True


# ---------------------------------------------------------------------------
# Tests: check_virtual_camera_ready
# ---------------------------------------------------------------------------

class TestCheckVirtualCameraReady:
    """Tests for the pre-start readiness check."""

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("virtual_camera.get_unity_capture_device_name", return_value="GoPro Webcam")
    def test_unitycapture_with_correct_name(self, mock_get_name, mock_backend):
        with patch.dict("sys.modules", {"pyvirtualcam": MagicMock()}):
            result = check_virtual_camera_ready("GoPro Webcam")
        assert result["ready"] is True
        assert result["backend"] == "unitycapture"
        assert result["device_name_ok"] is True

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("virtual_camera.get_unity_capture_device_name", return_value="Unity Video Capture")
    @patch("virtual_camera.configure_unity_capture_device_name", return_value=True)
    def test_unitycapture_name_auto_configured(self, mock_configure, mock_get_name, mock_backend):
        with patch.dict("sys.modules", {"pyvirtualcam": MagicMock()}):
            result = check_virtual_camera_ready("GoPro Webcam")
        assert result["ready"] is True
        assert result["device_name_ok"] is True
        mock_configure.assert_called_once_with("GoPro Webcam")

    @patch("virtual_camera.select_best_backend", return_value="unitycapture")
    @patch("virtual_camera.get_unity_capture_device_name", return_value="Wrong Name")
    @patch("virtual_camera.configure_unity_capture_device_name", return_value=False)
    def test_unitycapture_name_config_fails(self, mock_configure, mock_get_name, mock_backend):
        with patch.dict("sys.modules", {"pyvirtualcam": MagicMock()}):
            result = check_virtual_camera_ready("GoPro Webcam")
        # Still ready (usable) but name might be wrong
        assert result["ready"] is True
        assert result["device_name_ok"] is False
        assert "could not set device name" in result["message"].lower()

    @patch("virtual_camera.select_best_backend", return_value="obs")
    def test_obs_backend_warns_about_name(self, mock_backend):
        with patch.dict("sys.modules", {"pyvirtualcam": MagicMock()}):
            result = check_virtual_camera_ready("GoPro Webcam")
        assert result["ready"] is True
        assert result["backend"] == "obs"
        assert result["device_name_ok"] is False
        assert "OBS Virtual Camera" in result["message"]

    @patch("virtual_camera.select_best_backend", return_value=None)
    def test_no_backend_not_ready(self, mock_backend):
        with patch.dict("sys.modules", {"pyvirtualcam": MagicMock()}):
            result = check_virtual_camera_ready("GoPro Webcam")
        assert result["ready"] is False
        assert "No virtual camera driver" in result["message"]

    def test_pyvirtualcam_not_installed(self):
        """When pyvirtualcam is missing, should report not ready."""
        with patch.dict("sys.modules", {"pyvirtualcam": None}):
            with patch("builtins.__import__", side_effect=ImportError("no pyvirtualcam")):
                result = check_virtual_camera_ready("GoPro Webcam")
        assert result["ready"] is False
        assert "pyvirtualcam" in result["message"].lower()


# ---------------------------------------------------------------------------
# Tests: VirtualCamera.start() device naming
# ---------------------------------------------------------------------------

class TestVirtualCameraDeviceNaming:
    """Tests that VirtualCamera.start() correctly configures the device name."""

    def test_unitycapture_configures_registry_before_opening(self):
        """start() should call configure_unity_capture_device_name before Camera()."""
        config = make_test_config()
        vcam = VirtualCamera(config)

        call_order = []

        def mock_configure(name, device_index=0):
            call_order.append("configure_registry")
            return True

        mock_cam = MagicMock()
        mock_cam.device = "GoPro Webcam"

        def mock_camera(**kwargs):
            call_order.append("create_camera")
            return mock_cam

        mock_pyvirtualcam = MagicMock()
        mock_pyvirtualcam.Camera = mock_camera
        mock_pyvirtualcam.PixelFormat.RGB = 0

        with patch("virtual_camera.select_best_backend", return_value="unitycapture"), \
             patch("virtual_camera.configure_unity_capture_device_name", side_effect=mock_configure), \
             patch.dict("sys.modules", {"pyvirtualcam": mock_pyvirtualcam}):
            result = vcam.start()

        assert result is True
        assert call_order == ["configure_registry", "create_camera"]

    def test_unitycapture_passes_device_name_to_camera(self):
        """Camera() should receive device='GoPro Webcam' for Unity Capture."""
        config = make_test_config(virtual_camera_name="GoPro Webcam")
        vcam = VirtualCamera(config)

        captured_kwargs = {}
        mock_cam = MagicMock()
        mock_cam.device = "GoPro Webcam"

        def mock_camera(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_cam

        mock_pyvirtualcam = MagicMock()
        mock_pyvirtualcam.Camera = mock_camera
        mock_pyvirtualcam.PixelFormat.RGB = 0

        with patch("virtual_camera.select_best_backend", return_value="unitycapture"), \
             patch("virtual_camera.configure_unity_capture_device_name", return_value=True), \
             patch.dict("sys.modules", {"pyvirtualcam": mock_pyvirtualcam}):
            vcam.start()

        assert captured_kwargs.get("device") == "GoPro Webcam"
        assert captured_kwargs.get("backend") == "unitycapture"

    def test_obs_backend_does_not_pass_device_name(self):
        """OBS backend should NOT receive a 'device' kwarg (not supported)."""
        config = make_test_config()
        vcam = VirtualCamera(config)

        captured_kwargs = {}
        mock_cam = MagicMock()
        mock_cam.device = "OBS Virtual Camera"

        def mock_camera(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_cam

        mock_pyvirtualcam = MagicMock()
        mock_pyvirtualcam.Camera = mock_camera
        mock_pyvirtualcam.PixelFormat.RGB = 0

        with patch("virtual_camera.select_best_backend", return_value="obs"), \
             patch.dict("sys.modules", {"pyvirtualcam": mock_pyvirtualcam}):
            vcam.start()

        assert "device" not in captured_kwargs
        assert captured_kwargs.get("backend") == "obs"

    def test_device_name_mismatch_logs_warning(self):
        """If actual device name doesn't match desired, should log a warning."""
        config = make_test_config(virtual_camera_name="GoPro Webcam")
        vcam = VirtualCamera(config)

        mock_cam = MagicMock()
        mock_cam.device = "Unity Video Capture"  # Wrong name

        mock_pyvirtualcam = MagicMock()
        mock_pyvirtualcam.Camera.return_value = mock_cam
        mock_pyvirtualcam.PixelFormat.RGB = 0

        with patch("virtual_camera.select_best_backend", return_value="unitycapture"), \
             patch("virtual_camera.configure_unity_capture_device_name", return_value=True), \
             patch.dict("sys.modules", {"pyvirtualcam": mock_pyvirtualcam}), \
             patch("virtual_camera.log") as mock_log:
            vcam.start()

        # Should have logged a warning about name mismatch
        warning_calls = [c for c in mock_log.warning.call_args_list
                         if "mismatch" in str(c).lower()]
        assert len(warning_calls) > 0, "Expected a warning about device name mismatch"

    def test_device_name_match_no_warning(self):
        """If actual device name matches, no mismatch warning should be logged."""
        config = make_test_config(virtual_camera_name="GoPro Webcam")
        vcam = VirtualCamera(config)

        mock_cam = MagicMock()
        mock_cam.device = "GoPro Webcam"

        mock_pyvirtualcam = MagicMock()
        mock_pyvirtualcam.Camera.return_value = mock_cam
        mock_pyvirtualcam.PixelFormat.RGB = 0

        with patch("virtual_camera.select_best_backend", return_value="unitycapture"), \
             patch("virtual_camera.configure_unity_capture_device_name", return_value=True), \
             patch.dict("sys.modules", {"pyvirtualcam": mock_pyvirtualcam}), \
             patch("virtual_camera.log") as mock_log:
            vcam.start()

        warning_calls = [c for c in mock_log.warning.call_args_list
                         if "mismatch" in str(c).lower()]
        assert len(warning_calls) == 0, "Should not warn when names match"

    def test_uses_config_device_name(self):
        """VirtualCamera should read device_name from config."""
        config = make_test_config(virtual_camera_name="My Custom Webcam")
        vcam = VirtualCamera(config)
        assert vcam.device_name == "My Custom Webcam"

    def test_default_device_name_is_gopro_webcam(self):
        """Default device name should be 'GoPro Webcam'."""
        config = make_test_config()
        vcam = VirtualCamera(config)
        assert vcam.device_name == "GoPro Webcam"


# ---------------------------------------------------------------------------
# Tests: Backend selection
# ---------------------------------------------------------------------------

class TestBackendSelection:
    """Tests for backend detection and selection."""

    def test_prefers_unitycapture_over_obs(self):
        assert _BACKEND_PREFERENCE[0] == "unitycapture"
        assert _BACKEND_PREFERENCE[1] == "obs"

    @patch("virtual_camera.check_backend_available")
    def test_selects_unitycapture_when_available(self, mock_check):
        mock_check.side_effect = lambda b: b == "unitycapture"
        result = select_best_backend()
        assert result == "unitycapture"

    @patch("virtual_camera.check_backend_available")
    def test_falls_back_to_obs(self, mock_check):
        mock_check.side_effect = lambda b: b == "obs"
        result = select_best_backend()
        assert result == "obs"

    @patch("virtual_camera.check_backend_available", return_value=False)
    def test_returns_none_when_nothing_available(self, mock_check):
        result = select_best_backend()
        assert result is None


# ---------------------------------------------------------------------------
# Tests: VirtualCamera stats include device name
# ---------------------------------------------------------------------------

class TestVirtualCameraStats:
    """Tests that get_stats() includes correct device naming info."""

    def test_stats_device_name_before_start(self):
        config = make_test_config()
        vcam = VirtualCamera(config)
        stats = vcam.get_stats()
        assert stats["device_name"] == "GoPro Webcam"
        assert stats["running"] is False

    def test_stats_device_name_after_start(self):
        config = make_test_config()
        vcam = VirtualCamera(config)

        mock_cam = MagicMock()
        mock_cam.device = "GoPro Webcam"

        mock_pyvirtualcam = MagicMock()
        mock_pyvirtualcam.Camera.return_value = mock_cam
        mock_pyvirtualcam.PixelFormat.RGB = 0

        with patch("virtual_camera.select_best_backend", return_value="unitycapture"), \
             patch("virtual_camera.configure_unity_capture_device_name", return_value=True), \
             patch.dict("sys.modules", {"pyvirtualcam": mock_pyvirtualcam}):
            vcam.start()

        stats = vcam.get_stats()
        assert stats["device_name"] == "GoPro Webcam"
        assert stats["running"] is True
        assert stats["backend"] == "unitycapture"
