"""
dependency_checker.py — Runtime dependency detection and installation for GoPro Bridge.

Checks for and installs three dependencies required at runtime:
  1. ffmpeg         — video decoder used to decode the GoPro MPEG-TS stream
  2. Unity Capture  — virtual DirectShow camera driver (COM-registered DLL)
  3. Windows Firewall rule — inbound UDP allow on port 8554

All public methods are safe to call from background threads.
"""

import ctypes
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import winreg
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("gopro_bridge.dependency_checker")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FFMPEG_DOWNLOAD_URL = (
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
)
FFMPEG_INSTALL_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "GoProBridge",
    "ffmpeg",
)

# COM CLSID registered by UnityCaptureFilter64.dll
UNITY_CAPTURE_CLSID = "{5C2CD55C-92AD-4999-8666-912BD3E700AF}"
UNITY_CAPTURE_DLL_64 = "UnityCaptureFilter64.dll"
UNITY_CAPTURE_DLL_32 = "UnityCaptureFilter32.dll"
UNITY_CAPTURE_SUBDIR = "UnityCapture"

# Registry path for the CLSID check
_UNITY_CLSID_REG_PATH = rf"CLSID\{UNITY_CAPTURE_CLSID}"

# How long to poll the registry after regsvr32 before giving up (seconds)
UNITY_INSTALL_POLL_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class DependencyStatus:
    """Describes the current state of a single runtime dependency."""

    name: str
    """Logical name: 'ffmpeg', 'unity_capture', or 'firewall'."""

    installed: bool
    """True if the dependency is present and ready to use."""

    path: Optional[str] = None
    """Resolved path (executable or DLL) when installed; None otherwise."""

    error: Optional[str] = None
    """Human-readable error message when installed is False; None otherwise."""


# ---------------------------------------------------------------------------
# Helper: resolve app_dir
# ---------------------------------------------------------------------------


def _default_app_dir() -> str:
    """
    Return the directory that contains the application's entry-point.

    When running as a PyInstaller bundle, sys.executable is the .exe, so we
    use its directory.  When running from source, we walk up one level from
    this file's location (src/ -> project root).
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DependencyChecker:
    """
    Checks for and installs GoPro Bridge's three runtime dependencies.

    Parameters
    ----------
    app_dir:
        Directory where the application executable lives.  Used to locate
        the bundled UnityCapture DLL files.  When None, the value is derived
        from sys.executable (frozen) or this file's parent directory (source).
    """

    def __init__(self, app_dir: Optional[str] = None) -> None:
        self._app_dir: str = app_dir if app_dir is not None else _default_app_dir()
        log.debug("DependencyChecker initialised with app_dir=%s", self._app_dir)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def check_all(self) -> list[DependencyStatus]:
        """
        Check all three dependencies and return their statuses.

        Returns a list with three DependencyStatus objects in the order:
        ffmpeg, unity_capture, firewall.
        """
        return [
            self.check_ffmpeg(),
            self.check_unity_capture(),
            self.check_firewall(),
        ]

    # ------------------------------------------------------------------
    # ffmpeg
    # ------------------------------------------------------------------

    def check_ffmpeg(self) -> DependencyStatus:
        """
        Determine whether a usable ffmpeg binary is available.

        Search order:
          1. System PATH (shutil.which).
          2. %LOCALAPPDATA%/GoProBridge/ffmpeg/bin/ffmpeg.exe (app-private install).

        Returns a DependencyStatus with installed=True and the resolved path
        if found, or installed=False with an error message if not.
        """
        # 1. System PATH
        system_path = shutil.which("ffmpeg")
        if system_path:
            log.debug("ffmpeg found on PATH: %s", system_path)
            return DependencyStatus(name="ffmpeg", installed=True, path=system_path)

        # 2. App-private install location
        local_bin = os.path.join(FFMPEG_INSTALL_DIR, "bin", "ffmpeg.exe")
        if os.path.isfile(local_bin):
            log.debug("ffmpeg found at local install: %s", local_bin)
            return DependencyStatus(name="ffmpeg", installed=True, path=local_bin)

        msg = (
            "ffmpeg not found on PATH or in "
            f"{FFMPEG_INSTALL_DIR}. Install ffmpeg or use install_ffmpeg()."
        )
        log.info("ffmpeg check: not installed. %s", msg)
        return DependencyStatus(name="ffmpeg", installed=False, error=msg)

    def install_ffmpeg(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> DependencyStatus:
        """
        Download and install ffmpeg to %LOCALAPPDATA%/GoProBridge/ffmpeg/.

        The official Gyan.dev essentials build is downloaded as a ZIP, extracted
        so that the final layout is::

            %LOCALAPPDATA%/GoProBridge/ffmpeg/bin/ffmpeg.exe
            %LOCALAPPDATA%/GoProBridge/ffmpeg/bin/ffprobe.exe
            ...

        The top-level directory inside the ZIP (e.g. ``ffmpeg-7.1-essentials_build``)
        is stripped during extraction so binaries land directly under
        ``ffmpeg/bin/``.

        Parameters
        ----------
        progress_callback:
            Optional callable invoked during download as
            ``progress_callback(downloaded_bytes, total_bytes)``.
            ``total_bytes`` may be -1 if the server does not report Content-Length.

        Returns
        -------
        DependencyStatus
            installed=True and path to ffmpeg.exe on success, or
            installed=False with error describing the failure.
        """
        log.info("Starting ffmpeg download from %s", FFMPEG_DOWNLOAD_URL)

        # --- Download -------------------------------------------------------
        tmp_zip = None
        try:
            tmp_fd, tmp_zip = tempfile.mkstemp(suffix=".zip", prefix="ffmpeg_dl_")
            os.close(tmp_fd)

            with urllib.request.urlopen(FFMPEG_DOWNLOAD_URL, timeout=60) as response:
                total = int(response.headers.get("Content-Length", -1))
                downloaded = 0
                chunk_size = 65536  # 64 KiB

                with open(tmp_zip, "wb") as out_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback is not None:
                            try:
                                progress_callback(downloaded, total)
                            except Exception:  # noqa: BLE001
                                pass  # never let a callback crash the install

            log.info(
                "ffmpeg ZIP downloaded (%d bytes) to %s", downloaded, tmp_zip
            )

        except Exception as exc:  # noqa: BLE001
            _try_remove(tmp_zip)
            msg = f"Failed to download ffmpeg: {exc}"
            log.error(msg)
            return DependencyStatus(name="ffmpeg", installed=False, error=msg)

        # --- Extract --------------------------------------------------------
        try:
            os.makedirs(FFMPEG_INSTALL_DIR, exist_ok=True)

            with zipfile.ZipFile(tmp_zip, "r") as zf:
                members = zf.namelist()
                if not members:
                    raise ValueError("Downloaded ZIP appears to be empty.")

                # Determine the single top-level folder name inside the ZIP.
                # All entries should start with something like
                # "ffmpeg-7.1-essentials_build/..."
                top_dir = members[0].split("/")[0]
                log.debug("ZIP top-level directory: %s", top_dir)

                for member in members:
                    # Strip the top-level folder so everything lands under
                    # FFMPEG_INSTALL_DIR directly (giving us bin/ffmpeg.exe etc.)
                    if member.startswith(top_dir + "/"):
                        rel_path = member[len(top_dir) + 1:]
                    else:
                        rel_path = member

                    if not rel_path:
                        continue  # Skip the root entry itself

                    dest = os.path.join(FFMPEG_INSTALL_DIR, rel_path)

                    if member.endswith("/"):
                        os.makedirs(dest, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        with zf.open(member) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)

            log.info("ffmpeg extracted to %s", FFMPEG_INSTALL_DIR)

        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to extract ffmpeg ZIP: {exc}"
            log.error(msg)
            return DependencyStatus(name="ffmpeg", installed=False, error=msg)
        finally:
            _try_remove(tmp_zip)

        # --- Verify ---------------------------------------------------------
        ffmpeg_exe = os.path.join(FFMPEG_INSTALL_DIR, "bin", "ffmpeg.exe")
        if not os.path.isfile(ffmpeg_exe):
            msg = (
                f"ffmpeg.exe not found at expected location after extraction: "
                f"{ffmpeg_exe}"
            )
            log.error(msg)
            return DependencyStatus(name="ffmpeg", installed=False, error=msg)

        try:
            result = subprocess.run(
                [ffmpeg_exe, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "non-zero exit")
            log.info(
                "ffmpeg verified. First line: %s",
                result.stdout.splitlines()[0] if result.stdout else "(empty)",
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"ffmpeg installed but failed verification: {exc}"
            log.error(msg)
            return DependencyStatus(name="ffmpeg", installed=False, error=msg)

        return DependencyStatus(name="ffmpeg", installed=True, path=ffmpeg_exe)

    # ------------------------------------------------------------------
    # Unity Capture
    # ------------------------------------------------------------------

    def check_unity_capture(self) -> DependencyStatus:
        """
        Determine whether the Unity Capture virtual camera driver is registered.

        Checks the Windows registry for the COM CLSID
        ``{5C2CD55C-92AD-4999-8666-912BD3E700AF}`` under HKEY_CLASSES_ROOT.

        Returns a DependencyStatus with installed=True if the key exists, or
        installed=False with an error message if not.
        """
        try:
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, _UNITY_CLSID_REG_PATH):
                log.debug("Unity Capture CLSID registry key found.")
                return DependencyStatus(
                    name="unity_capture",
                    installed=True,
                    path=_UNITY_CLSID_REG_PATH,
                )
        except FileNotFoundError:
            msg = (
                f"Unity Capture CLSID {UNITY_CAPTURE_CLSID} not found in registry. "
                "Run install_unity_capture() to register the driver."
            )
            log.info("Unity Capture check: not installed. %s", msg)
            return DependencyStatus(name="unity_capture", installed=False, error=msg)
        except OSError as exc:
            msg = f"Registry query for Unity Capture CLSID failed: {exc}"
            log.warning(msg)
            return DependencyStatus(name="unity_capture", installed=False, error=msg)

    def install_unity_capture(self) -> DependencyStatus:
        """
        Register the Unity Capture DirectShow filter via regsvr32.

        Locates the DLL under ``<app_dir>/UnityCapture/`` (64-bit preferred,
        32-bit fallback) and invokes ``regsvr32 /s <dll>`` with UAC elevation
        via ``ShellExecuteW("runas",...)``.

        After launching the elevated process, polls the registry for up to
        ``UNITY_INSTALL_POLL_TIMEOUT`` seconds to confirm the CLSID appears.

        Returns
        -------
        DependencyStatus
            installed=True on confirmed registration, or installed=False with
            an error message describing the failure (DLL not found, UAC denied,
            registration timed out, etc.).
        """
        dll_path = self._locate_unity_capture_dll()
        if dll_path is None:
            searched = [
                os.path.join(self._app_dir, UNITY_CAPTURE_SUBDIR, UNITY_CAPTURE_DLL_64),
                os.path.join(self._app_dir, UNITY_CAPTURE_SUBDIR, UNITY_CAPTURE_DLL_32),
            ]
            msg = (
                "Unity Capture DLL not found. Searched:\n  "
                + "\n  ".join(searched)
            )
            log.error(msg)
            return DependencyStatus(name="unity_capture", installed=False, error=msg)

        log.info(
            "Registering Unity Capture DLL with UAC elevation: %s", dll_path
        )

        try:
            ret = ctypes.windll.shell32.ShellExecuteW(
                None,             # parent HWND
                "runas",          # verb — triggers UAC prompt
                "regsvr32",       # executable
                f'/s "{dll_path}"',  # arguments (/s = silent, no dialogs)
                None,             # working directory
                0,                # SW_HIDE
            )
        except (AttributeError, OSError) as exc:
            msg = f"ShellExecuteW call failed: {exc}"
            log.error(msg)
            return DependencyStatus(name="unity_capture", installed=False, error=msg)

        # ShellExecuteW returns > 32 on success, <= 32 on error
        # Code 5 means SE_ERR_ACCESSDENIED (user clicked "No" on the UAC dialog)
        if ret <= 32:
            if ret == 5:
                msg = "UAC elevation was denied by the user."
            else:
                msg = f"ShellExecuteW returned error code {ret}."
            log.warning("Unity Capture install: %s", msg)
            return DependencyStatus(name="unity_capture", installed=False, error=msg)

        log.debug(
            "regsvr32 launched (ShellExecuteW returned %d). "
            "Polling registry for up to %d seconds...",
            ret,
            UNITY_INSTALL_POLL_TIMEOUT,
        )

        # Poll the registry until the CLSID appears or we time out
        deadline = time.monotonic() + UNITY_INSTALL_POLL_TIMEOUT
        while time.monotonic() < deadline:
            time.sleep(0.5)
            status = self.check_unity_capture()
            if status.installed:
                log.info(
                    "Unity Capture CLSID confirmed in registry after install."
                )
                return status

        msg = (
            f"regsvr32 was launched but Unity Capture CLSID "
            f"{UNITY_CAPTURE_CLSID} did not appear in the registry within "
            f"{UNITY_INSTALL_POLL_TIMEOUT} seconds. "
            "Registration may have failed silently."
        )
        log.error(msg)
        return DependencyStatus(name="unity_capture", installed=False, error=msg)

    def _locate_unity_capture_dll(self) -> Optional[str]:
        """
        Return the absolute path to the best available UnityCapture DLL.

        Searches in multiple locations to handle both source and frozen layouts:
          - <app_dir>/UnityCapture/          (source layout, release zip)
          - <app_dir>/_internal/UnityCapture/ (PyInstaller frozen layout)

        Prefers the 64-bit variant.  Returns None if neither file exists.
        """
        search_dirs = [
            os.path.join(self._app_dir, UNITY_CAPTURE_SUBDIR),
            os.path.join(self._app_dir, "_internal", UNITY_CAPTURE_SUBDIR),
        ]
        for search_dir in search_dirs:
            for dll_name in (UNITY_CAPTURE_DLL_64, UNITY_CAPTURE_DLL_32):
                candidate = os.path.join(search_dir, dll_name)
                if os.path.isfile(candidate):
                    log.debug("Located Unity Capture DLL: %s", candidate)
                    return candidate
        return None

    # ------------------------------------------------------------------
    # Firewall
    # ------------------------------------------------------------------

    def check_firewall(self) -> DependencyStatus:
        """
        Check whether the Windows Firewall rule for UDP port 8554 exists.

        Delegates to :func:`firewall.firewall_rule_exists` from the project's
        existing ``firewall`` module.

        Returns
        -------
        DependencyStatus
            installed=True if the rule is present, installed=False otherwise.
        """
        try:
            from firewall import firewall_rule_exists  # local import to avoid circular deps

            exists = firewall_rule_exists()
            if exists:
                return DependencyStatus(
                    name="firewall",
                    installed=True,
                    path="Windows Firewall — inbound UDP 8554",
                )
            msg = (
                "Windows Firewall inbound rule for UDP port 8554 is not present. "
                "Run install_firewall() to create it."
            )
            log.info("Firewall check: rule not found.")
            return DependencyStatus(name="firewall", installed=False, error=msg)

        except Exception as exc:  # noqa: BLE001
            msg = f"Could not check firewall rule: {exc}"
            log.warning(msg)
            return DependencyStatus(name="firewall", installed=False, error=msg)

    def install_firewall(self) -> DependencyStatus:
        """
        Create the Windows Firewall inbound UDP 8554 rule.

        Delegates to :func:`firewall.ensure_firewall_rule` from the project's
        existing ``firewall`` module.  That function handles admin detection and
        UAC elevation internally.

        Returns
        -------
        DependencyStatus
            installed=True if the rule now exists, installed=False with an
            error message if creation failed.
        """
        try:
            from firewall import ensure_firewall_rule  # local import

            success = ensure_firewall_rule()
            if success:
                return DependencyStatus(
                    name="firewall",
                    installed=True,
                    path="Windows Firewall — inbound UDP 8554",
                )
            msg = (
                "ensure_firewall_rule() returned False. The rule may not have "
                "been created — UAC may have been denied or netsh failed."
            )
            log.error(msg)
            return DependencyStatus(name="firewall", installed=False, error=msg)

        except Exception as exc:  # noqa: BLE001
            msg = f"Error during firewall rule installation: {exc}"
            log.error(msg)
            return DependencyStatus(name="firewall", installed=False, error=msg)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _try_remove(path: Optional[str]) -> None:
    """Silently attempt to delete a file; ignore all errors."""
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
