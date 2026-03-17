"""
virtual_camera.py — Virtual camera output for GoPro Bridge

Creates a virtual webcam device named 'GoPro Webcam' that appears in
Windows device lists (Zoom, Teams, OBS, NVIDIA Broadcast, etc.).

Backend selection (Windows):
  - Unity Capture (preferred): Lightweight standalone driver, supports
    custom device names, does NOT require OBS installed.
    Install from: https://github.com/schellingb/UnityCapture/releases
  - OBS VirtualCam (fallback): Requires OBS Studio installed. Device
    name is always "OBS Virtual Camera" — cannot be customized.

The virtual camera is the LAST component to shut down during any
failure scenario. While the GoPro is disconnected or ffmpeg crashes,
we keep feeding the last good frame (freeze-frame) so downstream
apps never see the camera disappear.

Device naming:
  - Unity Capture: Device name set via both pyvirtualcam's 'device' param
    AND the Unity Capture registry key (HKCU\\SOFTWARE\\UnityCapture\\Device 0).
    This ensures Zoom and Teams list the device as 'GoPro Webcam'.
  - OBS VirtualCam: Device name is hard-coded as 'OBS Virtual Camera' and
    cannot be changed. A warning is logged when this fallback is used.

Verification:
  - verify_device_visible() enumerates DirectShow devices via ffmpeg to
    confirm 'GoPro Webcam' appears in the same list apps query.
  - check_virtual_camera_ready() pre-checks backend + naming before start.

Usage:
    from virtual_camera import VirtualCamera, check_virtual_camera_ready

    status = check_virtual_camera_ready(device_name="GoPro Webcam")
    if status["ready"]:
        vcam = VirtualCamera(config)
        vcam.start()                     # Opens the virtual camera device
        vcam.send_frame(numpy_rgb_array) # Push a frame to downstream apps
        vcam.stop()                      # Closes the device (app exit only)
"""

import os
import platform
import subprocess
import threading
import time
from typing import Optional

import numpy as np

from logger import get_logger

log = get_logger("virtual_camera")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Backend preference order — try Unity Capture first (custom names), then OBS
_BACKEND_PREFERENCE = ["unitycapture", "obs"]

# OBS Virtual Camera DirectShow filter CLSID — pyvirtualcam's native code
# searches for a device named exactly "OBS Virtual Camera". If this registry
# value has been renamed (e.g. to "GoMax Webcam"), the backend fails to find it.
# The name lives in TWO places:
#   1. CLSID default value (COM class name)
#   2. DirectShow filter category instance FriendlyName (what apps enumerate)
_OBS_VIRTUALCAM_CLSID = r"SOFTWARE\Classes\CLSID\{5C2CD55C-92AD-4999-8666-912BD3E70010}"
_OBS_VIRTUALCAM_INSTANCE = (
    r"SOFTWARE\Classes\CLSID\{860BB310-5D01-11D0-BD3B-00A0C911CE86}"
    r"\Instance\{5C2CD55C-92AD-4999-8666-912BD3E70010}"
)
_OBS_DEFAULT_NAME = "OBS Virtual Camera"

# When no frame has been received yet, show a solid dark-gray placeholder
# so downstream apps see a valid image (not garbage pixels).
_PLACEHOLDER_COLOR = (40, 40, 40)  # Dark gray RGB


# ---------------------------------------------------------------------------
# Backend availability check
# ---------------------------------------------------------------------------

def _get_obs_registry_name() -> Optional[str]:
    """Read the current OBS Virtual Camera device name from the registry.

    Checks the DirectShow instance FriendlyName first (what apps see),
    then falls back to the CLSID default value.
    """
    if platform.system() != "Windows":
        return None
    try:
        import winreg
        # Primary: DirectShow instance FriendlyName
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _OBS_VIRTUALCAM_INSTANCE)
            value, _ = winreg.QueryValueEx(key, "FriendlyName")
            winreg.CloseKey(key)
            return value
        except (OSError, FileNotFoundError):
            pass
        # Fallback: CLSID default
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _OBS_VIRTUALCAM_CLSID)
        value, _ = winreg.QueryValueEx(key, "")
        winreg.CloseKey(key)
        return value
    except (OSError, FileNotFoundError):
        return None


def _set_obs_registry_name(name: str) -> bool:
    """Set the OBS Virtual Camera device name in both registry locations.

    Updates:
      1. CLSID default value (COM class name)
      2. DirectShow filter category instance FriendlyName (what apps enumerate)

    Requires admin rights to write to HKLM. First tries direct write,
    then falls back to elevated reg.exe commands (triggers one UAC prompt).
    """
    if platform.system() != "Windows":
        return False

    # Try direct write first (works if already admin)
    try:
        import winreg
        # Update CLSID default
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, _OBS_VIRTUALCAM_CLSID,
            0, winreg.KEY_SET_VALUE,
        )
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, name)
        winreg.CloseKey(key)
        # Update DirectShow instance FriendlyName
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, _OBS_VIRTUALCAM_INSTANCE,
            0, winreg.KEY_SET_VALUE,
        )
        winreg.SetValueEx(key, "FriendlyName", 0, winreg.REG_SZ, name)
        winreg.CloseKey(key)
        return True
    except PermissionError:
        pass
    except (OSError, FileNotFoundError):
        return False

    # Fallback: use elevated reg.exe (triggers one UAC prompt via a batch)
    try:
        import ctypes
        import tempfile
        clsid_path = f"HKLM\\{_OBS_VIRTUALCAM_CLSID}"
        instance_path = f"HKLM\\{_OBS_VIRTUALCAM_INSTANCE}"

        # Write a small batch file to update both keys in one elevation
        bat_content = (
            f'@echo off\n'
            f'reg add "{clsid_path}" /ve /t REG_SZ /d "{name}" /f >nul 2>&1\n'
            f'reg add "{instance_path}" /v FriendlyName /t REG_SZ /d "{name}" /f >nul 2>&1\n'
        )
        bat_path = os.path.join(tempfile.gettempdir(), "gopro_bridge_reg.bat")
        with open(bat_path, "w") as f:
            f.write(bat_content)

        result = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", bat_path,
            None, None, 0,  # SW_HIDE
        )
        if result > 32:
            log.info("[EVENT:vcam_start] OBS VirtualCam renamed to '%s' via elevated batch", name)
            time.sleep(1.0)  # Wait for the batch + registry to settle
            return True
        else:
            log.warning("[EVENT:vcam_error] Elevated rename failed (code %d)", result)
            return False
    except Exception as e:
        log.debug("Elevated registry rename failed: %s", e)
        return False


def check_backend_available(backend: str) -> bool:
    """Check if a pyvirtualcam backend is installed and working.

    Attempts to import the backend module without actually opening a camera.
    Returns True if the backend driver appears to be installed.
    """
    try:
        import pyvirtualcam
        # pyvirtualcam doesn't expose a clean "list backends" API,
        # so we check by attempting to import the backend module.
        if backend == "unitycapture":
            from pyvirtualcam import _native_windows_unity_capture  # noqa: F401
            return True
        elif backend == "obs":
            from pyvirtualcam import _native_windows_obs  # noqa: F401
            return True
    except (ImportError, OSError, Exception) as e:
        log.debug("Backend '%s' not available: %s", backend, e)
    return False


def select_best_backend() -> Optional[str]:
    """Select the best available pyvirtualcam backend.

    Tries Unity Capture first (supports custom device names), then
    OBS VirtualCam as fallback.

    Returns:
        Backend name string ('unitycapture' or 'obs'), or None if
        no backend is available.
    """
    for backend in _BACKEND_PREFERENCE:
        if check_backend_available(backend):
            log.info("[EVENT:vcam_start] Selected backend: %s", backend)
            return backend

    log.warning(
        "[EVENT:vcam_error] No virtual camera backend found. "
        "Install Unity Capture from https://github.com/schellingb/UnityCapture/releases "
        "or OBS Studio from https://obsproject.com/"
    )
    return None


def detect_backend() -> dict:
    """Detect the available virtual camera backend at startup.

    Performs a non-blocking check for installed backends and returns a
    result dict describing what was found. The caller should use this to
    show a non-blocking warning in the GUI — never block startup.

    Priority: Unity Capture (recommended) > OBS VirtualCam (fallback).

    Returns:
        dict with keys:
          - backend (str or None): Detected backend name ('unitycapture',
            'obs'), or None if nothing is installed.
          - recommended (str): The recommended backend ('unitycapture').
          - is_recommended (bool): True if the detected backend matches
            the recommended one.
          - warning (str or None): Human-readable warning message if the
            detected backend is not the recommended one, or if no backend
            is found. None when Unity Capture is detected.
          - level (str): Severity level for GUI display — 'success',
            'warning', or 'error'.
    """
    result = {
        "backend": None,
        "recommended": "unitycapture",
        "is_recommended": False,
        "warning": None,
        "level": "error",
    }

    # Check pyvirtualcam availability first
    try:
        import pyvirtualcam  # noqa: F401
    except ImportError:
        result["warning"] = (
            "pyvirtualcam is not installed. Virtual camera output will not work. "
            "Install it with: pip install pyvirtualcam"
        )
        result["level"] = "error"
        log.warning("[EVENT:vcam_error] pyvirtualcam not installed")
        return result

    # Check for Unity Capture (recommended)
    if check_backend_available("unitycapture"):
        result["backend"] = "unitycapture"
        result["is_recommended"] = True
        result["warning"] = None
        result["level"] = "success"
        log.info("[EVENT:vcam_start] detect_backend: Unity Capture found (recommended)")
        return result

    # Check for OBS VirtualCam (fallback)
    if check_backend_available("obs"):
        result["backend"] = "obs"
        result["is_recommended"] = False
        result["warning"] = (
            "OBS VirtualCam detected as fallback. Device will appear as "
            "'OBS Virtual Camera' (name cannot be customized). "
            "For best results, install Unity Capture from "
            "https://github.com/schellingb/UnityCapture/releases"
        )
        result["level"] = "warning"
        log.warning(
            "[EVENT:vcam_start] detect_backend: OBS VirtualCam found (fallback). "
            "Unity Capture recommended for custom device naming."
        )
        return result

    # No backend found
    result["warning"] = (
        "No virtual camera backend found. Install Unity Capture "
        "(recommended) from https://github.com/schellingb/UnityCapture/releases "
        "or OBS Studio from https://obsproject.com/ as a fallback."
    )
    result["level"] = "error"
    log.warning("[EVENT:vcam_error] detect_backend: No virtual camera backend found")
    return result


# ---------------------------------------------------------------------------
# Unity Capture registry device name configuration
# ---------------------------------------------------------------------------

# Unity Capture stores per-device config under HKCU\SOFTWARE\UnityCapture
_UNITY_CAPTURE_REG_PATH = r"SOFTWARE\UnityCapture"


def configure_unity_capture_device_name(
    device_name: str = "GoPro Webcam",
    device_index: int = 0,
) -> bool:
    """Set the Unity Capture device's FriendlyName in the Windows registry.

    Unity Capture reads its device name from:
      HKCU\\SOFTWARE\\UnityCapture\\Device <N>\\FriendlyName

    Setting this before creating the pyvirtualcam.Camera ensures apps
    see 'GoPro Webcam' in their camera dropdown.

    Args:
        device_name: Desired device name (default 'GoPro Webcam').
        device_index: Unity Capture device index (0 = first virtual cam).

    Returns:
        True if the registry was updated successfully.
    """
    if platform.system() != "Windows":
        log.debug("Skipping Unity Capture registry config (not Windows)")
        return False

    try:
        import winreg

        key_path = rf"{_UNITY_CAPTURE_REG_PATH}\Device {device_index}"
        key = winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            key_path,
            0,
            winreg.KEY_WRITE,
        )
        winreg.SetValueEx(key, "FriendlyName", 0, winreg.REG_SZ, device_name)
        winreg.CloseKey(key)

        log.info(
            "[EVENT:vcam_start] Unity Capture Device %d name set to '%s' in registry",
            device_index, device_name,
        )
        return True
    except OSError as e:
        log.warning(
            "[EVENT:vcam_error] Failed to set Unity Capture device name: %s", e
        )
        return False


def get_unity_capture_device_name(device_index: int = 0) -> Optional[str]:
    """Read the current Unity Capture FriendlyName from the registry.

    Returns:
        The device name string, or None if the key doesn't exist.
    """
    if platform.system() != "Windows":
        return None

    try:
        import winreg

        key_path = rf"{_UNITY_CAPTURE_REG_PATH}\Device {device_index}"
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path)
        value, _ = winreg.QueryValueEx(key, "FriendlyName")
        winreg.CloseKey(key)
        return value
    except (OSError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# DirectShow device enumeration & verification
# ---------------------------------------------------------------------------

def list_directshow_video_devices() -> list[dict]:
    """List all DirectShow video capture devices visible to applications.

    Uses ffmpeg's dshow device enumeration (the same mechanism that Zoom,
    Teams, and other apps use). Falls back to PowerShell WMI query if
    ffmpeg is not available.

    Returns:
        List of dicts, each with 'name' (str) and 'source' ('dshow' or 'wmi').
    """
    if platform.system() != "Windows":
        return []

    devices = []

    # Method 1: ffmpeg dshow — matches what video-conferencing apps actually see
    try:
        result = subprocess.run(
            ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        # ffmpeg outputs device list to stderr (not stdout)
        output = result.stderr or ""
        _parse_ffmpeg_dshow_output(output, devices)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        log.debug("ffmpeg dshow device listing failed: %s", e)

    # Method 2: PowerShell WMI fallback
    if not devices:
        try:
            ps_cmd = (
                'Get-CimInstance Win32_PnPEntity | '
                'Where-Object { $_.PNPClass -eq "Camera" -or '
                '$_.PNPClass -eq "Image" } | '
                'Select-Object -ExpandProperty Name'
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            for line in result.stdout.strip().splitlines():
                name = line.strip()
                if name:
                    devices.append({"name": name, "source": "wmi"})
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            log.debug("PowerShell device listing failed: %s", e)

    return devices


def _parse_ffmpeg_dshow_output(output: str, devices: list[dict]) -> None:
    """Parse ffmpeg -list_devices dshow output to extract video device names.

    Supports two ffmpeg output formats:

    Legacy format (with section headers)::

        [dshow @ 0x...] DirectShow video devices (some may be both ...)
        [dshow @ 0x...]  "Device Name"
        [dshow @ 0x...]   Alternative name "@device_..."
        [dshow @ 0x...] DirectShow audio devices
        ...

    Modern format (no section headers, inline type suffix)::

        [dshow @ 0x...] "Device Name" (video)
        [dshow @ 0x...]   Alternative name "@device_..."
        [dshow @ 0x...] "Microphone" (audio)
        ...
    """
    in_video_section = False
    has_section_headers = "DirectShow video devices" in output

    for line in output.splitlines():
        if has_section_headers:
            # Legacy format: use section headers to scope
            if "DirectShow video devices" in line:
                in_video_section = True
                continue
            if "DirectShow audio devices" in line:
                in_video_section = False
                continue
            if in_video_section and '"' in line:
                try:
                    start = line.index('"') + 1
                    end = line.index('"', start)
                    name = line[start:end]
                    if not name.startswith("@"):
                        devices.append({"name": name, "source": "dshow"})
                except ValueError:
                    continue
        else:
            # Modern format: look for lines with (video) suffix
            if "[dshow @" not in line or '"' not in line:
                continue
            # Skip lines with (audio) suffix
            if "(audio)" in line:
                continue
            try:
                start = line.index('"') + 1
                end = line.index('"', start)
                name = line[start:end]
                # Skip "Alternative name" lines (start with @)
                if not name.startswith("@"):
                    devices.append({"name": name, "source": "dshow"})
            except ValueError:
                continue


def verify_device_visible(
    expected_name: str = "GoPro Webcam",
    timeout: float = 5.0,
) -> bool:
    """Verify that a virtual camera with the expected name is listed.

    Polls the DirectShow device list until the device appears or timeout
    is reached. This confirms that Zoom, Teams, etc. will see the device.

    Args:
        expected_name: Name to look for (case-insensitive substring match).
        timeout: Max seconds to wait for the device to appear.

    Returns:
        True if the device was found in the device list.
    """
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        devices = list_directshow_video_devices()
        for dev in devices:
            if expected_name.lower() in dev["name"].lower():
                log.info(
                    "[EVENT:vcam_start] Verified '%s' visible in DirectShow list "
                    "(source=%s, attempt=%d)",
                    expected_name, dev["source"], attempt,
                )
                return True
        time.sleep(0.5)

    device_names = [d["name"] for d in list_directshow_video_devices()]
    log.warning(
        "[EVENT:vcam_error] '%s' not found in DirectShow devices after %.1fs. "
        "Visible devices: %s",
        expected_name, timeout, device_names,
    )
    return False


def check_virtual_camera_ready(device_name: str = "GoPro Webcam") -> dict:
    """Pre-check that the virtual camera system is ready.

    Checks pyvirtualcam installation, backend availability, and device
    name configuration. Call this during startup to give the user a
    clear message about what's missing.

    Returns:
        dict with keys:
          - ready (bool): True if we can create a virtual camera
          - backend (str): Detected backend name or 'none'
          - message (str): Human-readable status
          - device_name_ok (bool): True if the name is configured correctly
    """
    result = {
        "ready": False,
        "backend": "none",
        "message": "",
        "device_name_ok": False,
    }

    # Check pyvirtualcam
    try:
        import pyvirtualcam  # noqa: F401
    except ImportError:
        result["message"] = (
            "pyvirtualcam is not installed.\n"
            "Install it with: pip install pyvirtualcam"
        )
        return result

    # Detect backend
    backend = select_best_backend()
    if backend is None:
        result["backend"] = "none"
        result["message"] = (
            "No virtual camera driver found.\n\n"
            "Please install one of:\n"
            "  1. Unity Capture (recommended):\n"
            "     https://github.com/schellingb/UnityCapture/releases\n\n"
            "  2. OBS VirtualCam:\n"
            "     Install OBS Studio from https://obsproject.com/"
        )
        return result

    result["backend"] = backend

    # Handle device naming
    if backend == "unitycapture":
        # Unity Capture: ensure registry name matches desired name
        current_name = get_unity_capture_device_name()
        if current_name == device_name:
            result["device_name_ok"] = True
        else:
            # Auto-configure the name
            if configure_unity_capture_device_name(device_name):
                result["device_name_ok"] = True
            else:
                result["message"] = (
                    f"Unity Capture installed but could not set device name "
                    f"to '{device_name}'. The camera may appear with a "
                    f"different name."
                )
                result["ready"] = True
                return result

        result["ready"] = True
        result["message"] = (
            f"Ready. Device will appear as '{device_name}' "
            f"(Unity Capture backend)"
        )

    elif backend == "obs":
        result["ready"] = True
        result["device_name_ok"] = False
        result["message"] = (
            f"OBS VirtualCam detected. Note: device will appear as "
            f"'OBS Virtual Camera' instead of '{device_name}'. "
            f"For custom naming, install Unity Capture."
        )

    return result


# ---------------------------------------------------------------------------
# Main VirtualCamera class
# ---------------------------------------------------------------------------

class VirtualCamera:
    """Manages the virtual camera device that downstream apps see.

    Key design decisions:
      - The camera device stays open for the entire app session.
      - If no new frames arrive (GoPro disconnected, ffmpeg crashed),
        the last frame stays frozen — downstream apps never lose the feed.
      - Thread-safe: send_frame() can be called from any thread.

    Attributes:
        width:   Frame width in pixels (default 1920 from config)
        height:  Frame height in pixels (default 1080 from config)
        fps:     Target frame rate (default 30 from config)
        device_name: Name shown in Windows camera lists
        backend: Which virtual camera driver is in use
        is_running: True if the virtual camera device is open
    """

    def __init__(self, config=None):
        """Initialize the virtual camera (does NOT open the device yet).

        Args:
            config: Config dataclass instance. If None, uses sensible
                    defaults (1920x1080 @ 30fps, name='GoPro Webcam').
        """
        # Import config type for defaults without circular import
        self.width: int = getattr(config, 'stream_width', 1920)
        self.height: int = getattr(config, 'stream_height', 1080)
        self.fps: int = getattr(config, 'stream_fps', 30)
        self.device_name: str = getattr(config, 'virtual_camera_name', 'GoPro Webcam')

        # Internal state
        self._cam = None               # pyvirtualcam.Camera instance
        self._lock = threading.Lock()   # Protects _cam and _last_frame
        self._last_frame: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._backend: Optional[str] = None
        self._running: bool = False

        # Stats for monitoring
        self._start_time: Optional[float] = None
        self._last_frame_time: Optional[float] = None

    # -- Properties ----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True if the virtual camera device is open and accepting frames."""
        with self._lock:
            return self._running and self._cam is not None

    @property
    def backend(self) -> Optional[str]:
        """The active backend name, or None if not started."""
        return self._backend

    @property
    def frame_count(self) -> int:
        """Total number of frames sent since start."""
        return self._frame_count

    @property
    def last_frame(self) -> Optional[np.ndarray]:
        """The last frame sent (used for freeze-frame on disconnect)."""
        with self._lock:
            return self._last_frame

    # -- Lifecycle -----------------------------------------------------------

    def start(self, preferred_backend: Optional[str] = None) -> bool:
        """Open the virtual camera device so it appears in Windows camera lists.

        This is the key initialization step — after this returns True,
        apps like Zoom, Teams, and NVIDIA Broadcast will see 'GoPro Webcam'
        in their camera dropdown.

        Args:
            preferred_backend: Force a specific backend ('unitycapture' or 'obs').
                If None, auto-selects the best available backend.

        Returns:
            True if the virtual camera was opened successfully.
        """
        if self.is_running:
            log.warning("[EVENT:vcam_start] Virtual camera already running")
            return True

        try:
            import pyvirtualcam
        except ImportError:
            log.error(
                "[EVENT:vcam_error] pyvirtualcam not installed. "
                "Run: pip install pyvirtualcam"
            )
            return False

        # Select backend
        backend = preferred_backend or select_best_backend()
        if backend is None:
            log.error("[EVENT:vcam_error] No virtual camera backend available")
            return False

        log.info(
            "[EVENT:vcam_start] Opening virtual camera: name='%s', "
            "resolution=%dx%d, fps=%d, backend=%s",
            self.device_name, self.width, self.height, self.fps, backend,
        )

        # Pre-configure device name in registry BEFORE opening
        obs_original_name = None
        if backend == "unitycapture":
            configure_unity_capture_device_name(self.device_name)
        elif backend == "obs":
            # pyvirtualcam's OBS backend hardcodes looking for "OBS Virtual Camera".
            # If the registry name has been changed (e.g. to "GoMax Webcam"),
            # temporarily rename it back so pyvirtualcam can find the device,
            # then rename it to the desired name after opening.
            obs_original_name = _get_obs_registry_name()
            if obs_original_name and obs_original_name != _OBS_DEFAULT_NAME:
                log.info(
                    "[EVENT:vcam_start] OBS VirtualCam registry name is '%s', "
                    "temporarily renaming to '%s' for pyvirtualcam",
                    obs_original_name, _OBS_DEFAULT_NAME,
                )
                _set_obs_registry_name(_OBS_DEFAULT_NAME)

        try:
            # Build kwargs — device name is only supported by Unity Capture
            cam_kwargs = {
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "fmt": pyvirtualcam.PixelFormat.BGR,
                "backend": backend,
            }

            # Unity Capture supports custom device names via the 'device' param
            if backend == "unitycapture":
                cam_kwargs["device"] = self.device_name

            cam = pyvirtualcam.Camera(**cam_kwargs)

            # Rename OBS device to desired name now that we've opened it
            if backend == "obs" and self.device_name != _OBS_DEFAULT_NAME:
                if _set_obs_registry_name(self.device_name):
                    log.info(
                        "[EVENT:vcam_start] OBS VirtualCam renamed to '%s' in registry",
                        self.device_name,
                    )
                else:
                    log.info(
                        "[EVENT:vcam_start] OBS VirtualCam backend — device appears as "
                        "'%s' (rename to '%s' requires admin rights)",
                        _OBS_DEFAULT_NAME, self.device_name,
                    )

            with self._lock:
                self._cam = cam
                self._backend = backend
                self._running = True
                self._frame_count = 0
                self._start_time = time.monotonic()
                self._last_frame = None

            # Send an initial placeholder frame so the camera isn't blank
            self._send_placeholder()

            log.info(
                "[EVENT:vcam_start] Virtual camera opened successfully: "
                "device='%s', backend=%s, actual_device='%s'",
                self.device_name, backend, cam.device,
            )

            # Warn if the actual device name doesn't match the requested name
            if cam.device and self.device_name and \
               cam.device.lower() != self.device_name.lower():
                log.warning(
                    "[EVENT:vcam_start] Device name mismatch: "
                    "requested='%s', actual='%s'",
                    self.device_name, cam.device,
                )

            return True

        except RuntimeError as e:
            # pyvirtualcam raises RuntimeError if the backend driver isn't found
            log.error(
                "[EVENT:vcam_error] Failed to open virtual camera: %s. "
                "Is the %s driver installed?",
                e, backend,
            )
            # Restore the original OBS registry name if we changed it
            if obs_original_name and obs_original_name != _OBS_DEFAULT_NAME:
                _set_obs_registry_name(obs_original_name)
            return False
        except Exception as e:
            log.exception("[EVENT:vcam_error] Unexpected error opening virtual camera: %s", e)
            if obs_original_name and obs_original_name != _OBS_DEFAULT_NAME:
                _set_obs_registry_name(obs_original_name)
            return False

    def stop(self):
        """Close the virtual camera device.

        After this, the camera disappears from Windows device lists.
        Only call this on app exit — during normal operation, keep the
        camera open and use freeze-frame for resilience.
        """
        with self._lock:
            cam = self._cam
            self._cam = None
            self._running = False

        if cam is not None:
            try:
                cam.close()
                log.info(
                    "[EVENT:vcam_stop] Virtual camera closed after %d frames",
                    self._frame_count,
                )
            except Exception as e:
                log.warning("[EVENT:vcam_error] Error closing virtual camera: %s", e)

        self._backend = None
        self._start_time = None

    # -- Frame sending -------------------------------------------------------

    def send_frame(self, frame: np.ndarray) -> bool:
        """Send a video frame to the virtual camera.

        The frame is also stored as the 'last frame' for freeze-frame
        recovery. If the GoPro disconnects, call send_last_frame() to
        keep feeding the frozen image.

        Args:
            frame: numpy array of shape (height, width, 3) with dtype uint8,
                   in BGR format (matching ffmpeg bgr24 output).

        Returns:
            True if the frame was sent successfully.
        """
        try:
            with self._lock:
                if self._cam is None or not self._running:
                    return False
                h, w = frame.shape[:2]
                if h != self.height or w != self.width:
                    frame = self._resize_frame(frame)
                self._cam.send(frame)
                self._last_frame = frame
                self._frame_count += 1
                self._last_frame_time = time.monotonic()
                return True
        except Exception as e:
            log.error("[EVENT:vcam_error] Failed to send frame: %s", e)
            return False

    def send_last_frame(self) -> bool:
        """Re-send the last frame (freeze-frame during disconnect).

        This keeps the virtual camera feed alive with the last good
        image while the GoPro is reconnecting. Downstream apps see a
        frozen image instead of a black/missing feed.

        Returns:
            True if a freeze-frame was sent, False if no frame available.
        """
        with self._lock:
            frame = self._last_frame
            cam = self._cam

        if cam is None or not self._running:
            return False

        if frame is None:
            # No frame ever received — send placeholder
            self._send_placeholder()
            return True

        try:
            with self._lock:
                if self._cam is not None:
                    self._cam.send(frame)
                    self._frame_count += 1
                    return True
            return False
        except Exception as e:
            log.debug("[EVENT:freeze_frame] Failed to send freeze-frame: %s", e)
            return False

    def sleep_until_next_frame(self):
        """Sleep until the next frame should be sent (maintains target FPS).

        Call this after send_frame() to maintain consistent frame timing.
        """
        with self._lock:
            cam = self._cam
        if cam is not None:
            try:
                cam.sleep_until_next_frame()
            except Exception:
                # If the camera was closed between check and sleep, ignore
                pass

    # -- Status & diagnostics ------------------------------------------------

    def get_stats(self) -> dict:
        """Return diagnostic statistics for the GUI dashboard.

        Returns dict with:
          - running: bool
          - backend: str or None
          - device_name: str
          - resolution: str (e.g. '1920x1080')
          - fps: int
          - frame_count: int
          - uptime_seconds: float
          - seconds_since_last_frame: float or None
        """
        now = time.monotonic()
        with self._lock:
            running = self._running
            last_time = self._last_frame_time

        return {
            "running": running,
            "backend": self._backend,
            "device_name": self.device_name,
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps,
            "frame_count": self._frame_count,
            "uptime_seconds": (now - self._start_time) if self._start_time else 0.0,
            "seconds_since_last_frame": (now - last_time) if last_time else None,
        }

    # -- Internal helpers ----------------------------------------------------

    def _send_placeholder(self):
        """Send a dark gray placeholder frame.

        Used on first start and when no real frames are available.
        Ensures downstream apps see valid video (not garbage pixels).
        """
        placeholder = np.full(
            (self.height, self.width, 3),
            _PLACEHOLDER_COLOR,
            dtype=np.uint8,
        )
        try:
            with self._lock:
                if self._cam is not None:
                    self._cam.send(placeholder)
                    self._last_frame = placeholder
                    self._frame_count += 1
                    log.debug("[EVENT:vcam_start] Placeholder frame sent")
        except Exception as e:
            log.debug("Failed to send placeholder: %s", e)

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame to match the virtual camera dimensions.

        Uses simple nearest-neighbor interpolation via numpy slicing.
        For production quality, cv2.resize would be better, but we
        avoid the OpenCV dependency for now.
        """
        h, w = frame.shape[:2]
        target_h, target_w = self.height, self.width

        if h == target_h and w == target_w:
            return frame

        # Create output array
        out = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Nearest-neighbor scaling via index mapping
        row_indices = (np.arange(target_h) * h // target_h).astype(int)
        col_indices = (np.arange(target_w) * w // target_w).astype(int)

        out = frame[row_indices][:, col_indices]
        return np.ascontiguousarray(out)

    # -- Reconfiguration for resolution changes --------------------------------

    def reconfigure(self, width: int, height: int, fps: int = None) -> bool:
        """Stop the virtual camera and reopen at a new resolution.

        This is the key method for resolution changes. The virtual camera
        device briefly disappears from downstream apps during reconfiguration,
        but this is unavoidable when dimensions change (pyvirtualcam requires
        a new Camera instance for different dimensions).

        Args:
            width: New frame width in pixels.
            height: New frame height in pixels.
            fps: New target frame rate (default: keep current).

        Returns:
            True if the camera was reopened at the new resolution.
        """
        old_w, old_h, old_fps = self.width, self.height, self.fps
        new_fps = fps if fps is not None else self.fps

        if old_w == width and old_h == height and old_fps == new_fps:
            log.info(
                "[EVENT:vcam_start] Resolution unchanged (%dx%d@%d), skipping reconfigure",
                width, height, new_fps,
            )
            return True

        log.info(
            "[EVENT:vcam_start] Reconfiguring virtual camera: %dx%d@%d → %dx%d@%d",
            old_w, old_h, old_fps, width, height, new_fps,
        )

        # Remember the backend so we reuse it
        prev_backend = self._backend

        # Stop the current device
        self.stop()

        # Update dimensions
        self.width = width
        self.height = height
        self.fps = new_fps
        self._frame_interval = 1.0 / max(1, new_fps) if hasattr(self, '_frame_interval') else None

        # Reopen at new dimensions
        if not self.start(preferred_backend=prev_backend):
            log.error(
                "[EVENT:vcam_error] Failed to reopen virtual camera at %dx%d@%d",
                width, height, new_fps,
            )
            # Try to fall back to old dimensions
            self.width, self.height, self.fps = old_w, old_h, old_fps
            if self.start(preferred_backend=prev_backend):
                log.warning(
                    "[EVENT:vcam_start] Fell back to previous resolution %dx%d@%d",
                    old_w, old_h, old_fps,
                )
            return False

        log.info(
            "[EVENT:vcam_start] Virtual camera reconfigured to %dx%d@%d",
            width, height, new_fps,
        )
        return True

    # -- Context manager support ---------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        """Ensure the camera is closed if the object is garbage collected."""
        if self._running:
            try:
                self.stop()
            except Exception:
                pass
