#!/usr/bin/env python3
"""
GoMaxWebcam Installer — First-run setup and update helper.

Pure stdlib. No admin/UAC required. Works on Windows, macOS, Linux.
Run:  python install.py
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

def _enable_ansi_win() -> None:
    """Enable VT100 escape sequences on Windows 10+."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass

_enable_ansi_win()

BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
RESET  = "\033[0m"

# Use ASCII-safe symbols to avoid cp1252 encoding errors on Windows consoles
CHECK  = f"{GREEN}OK{RESET}"
CROSS  = f"{RED}FAIL{RESET}"
WARN   = f"{YELLOW}!{RESET}"

TOTAL_STEPS = 7


def step(n: int, msg: str, status: str = "") -> None:
    tag = f"[{n}/{TOTAL_STEPS}]"
    print(f"  {BOLD}{tag}{RESET} {status} {msg}")


def info(msg: str) -> None:
    print(f"       {msg}")


def error(msg: str) -> None:
    print(f"       {RED}{msg}{RESET}")


def hint(msg: str) -> None:
    print(f"       {DIM}{msg}{RESET}")


def banner() -> None:
    print()
    print(f"  {BOLD}{CYAN}GoMaxWebcam v2 Installer{RESET}")
    print(f"  {DIM}Cross-platform GoPro virtual camera utility{RESET}")
    print(f"  {DIM}{'=' * 46}{RESET}")
    print()


# ---------------------------------------------------------------------------
# Step 1 — Python version
# ---------------------------------------------------------------------------

def check_python() -> bool:
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 12):
        step(1, f"Python {ver_str} detected", CHECK)
        return True
    else:
        step(1, f"Python {ver_str} detected — {RED}too old{RESET}", CROSS)
        error("GoMaxWebcam requires Python 3.12 or newer.")
        hint("Download from https://www.python.org/downloads/")
        return False


# ---------------------------------------------------------------------------
# Step 2 — pip
# ---------------------------------------------------------------------------

def check_pip() -> bool:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, check=True, timeout=30,
        )
        step(2, "pip is available", CHECK)
        return True
    except Exception:
        step(2, "pip not found", CROSS)
        error("pip is required to install GoMaxWebcam.")
        hint(f"Try:  {sys.executable} -m ensurepip --upgrade")
        return False


# ---------------------------------------------------------------------------
# Step 3 — Install / upgrade
# ---------------------------------------------------------------------------

def _in_repo() -> bool:
    return (Path(__file__).resolve().parent / "pyproject.toml").is_file()


def install_package() -> bool:
    if _in_repo():
        info("Source checkout detected — using editable install.")
        cmd = [sys.executable, "-m", "pip", "install", "-e", ".", "--user", "-q"]
        cwd = str(Path(__file__).resolve().parent)
    else:
        info("Installing from PyPI.")
        cmd = [sys.executable, "-m", "pip", "install", "gomaxwebcam", "--user", "-q"]
        cwd = None

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=cwd)
        if r.returncode == 0:
            step(3, "GoMaxWebcam installed", CHECK)
            return True
        else:
            step(3, "Installation failed", CROSS)
            error(r.stderr.strip() if r.stderr else "Unknown pip error")
            return False
    except subprocess.TimeoutExpired:
        step(3, "Installation timed out (5 min)", CROSS)
        return False
    except Exception as exc:
        step(3, f"Installation error: {exc}", CROSS)
        return False


# ---------------------------------------------------------------------------
# Step 4 — Verify import
# ---------------------------------------------------------------------------

def verify_import() -> str | None:
    """Return version string or None on failure."""
    try:
        # Invalidate caches so a fresh install is visible
        importlib.invalidate_caches()
        r = subprocess.run(
            [sys.executable, "-c",
             "from importlib.metadata import version; print(version('gomaxwebcam'))"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            ver = r.stdout.strip()
            step(4, f"gomaxwebcam {ver} importable", CHECK)
            return ver
        else:
            step(4, "Failed to import gomaxwebcam", CROSS)
            error(r.stderr.strip() if r.stderr else "import failed")
            return None
    except Exception as exc:
        step(4, f"Verification error: {exc}", CROSS)
        return None


# ---------------------------------------------------------------------------
# Step 5 — Virtual camera driver
# ---------------------------------------------------------------------------

def check_virtual_camera() -> bool:
    plat = sys.platform

    if plat == "win32":
        # Check Unity Capture
        import winreg
        found = False
        for name, key_path in [
            ("Unity Capture",
             r"SOFTWARE\Classes\CLSID\{5C2CD55C-92AD-4999-8666-912BD3E700F3}"),
            ("OBS Virtual Camera",
             r"SOFTWARE\Classes\CLSID\{A3FCE0F5-3493-419F-958A-ABA1250EC20B}"),
        ]:
            try:
                winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                step(5, f"Virtual camera driver found ({name})", CHECK)
                found = True
                break
            except FileNotFoundError:
                continue
            except Exception:
                continue
        if not found:
            step(5, "No virtual camera driver detected", WARN)
            info("GoMaxWebcam needs a virtual camera to output video.")
            info("")
            info(f"{BOLD}Recommended:{RESET} Unity Capture (included in this repo)")
            hint("  1. Open UnityCapture/Install/ folder")
            hint("  2. Right-click Install.bat -> Run as administrator")
            info("")
            info(f"{BOLD}Alternative:{RESET} OBS Virtual Camera")
            hint("  Install OBS Studio from https://obsproject.com")
        return found

    elif plat == "darwin":
        obs_path = Path("/Library/CoreMediaIO/Plug-Ins/DAL/obs-mac-virtualcam.plugin")
        if obs_path.exists():
            step(5, "OBS Virtual Camera found", CHECK)
            return True
        else:
            step(5, "No virtual camera driver detected", WARN)
            info("Install OBS Studio to get OBS Virtual Camera:")
            hint("  brew install --cask obs")
            return False

    else:  # Linux
        try:
            r = subprocess.run(
                ["modinfo", "v4l2loopback"],
                capture_output=True, timeout=10,
            )
            if r.returncode == 0:
                step(5, "v4l2loopback module found", CHECK)
                return True
        except Exception:
            pass
        step(5, "v4l2loopback not detected", WARN)
        info("Install the v4l2loopback kernel module:")
        hint("  sudo apt install v4l2loopback-dkms   # Debian/Ubuntu")
        hint("  sudo modprobe v4l2loopback")
        return False


# ---------------------------------------------------------------------------
# Step 6 — ffmpeg
# ---------------------------------------------------------------------------

def check_ffmpeg() -> bool:
    if shutil.which("ffmpeg"):
        step(6, "ffmpeg is on PATH", CHECK)
        return True
    else:
        step(6, "ffmpeg not found on PATH", WARN)
        if sys.platform == "win32":
            hint("  Download from https://www.gyan.dev/ffmpeg/builds/")
            hint("  Or:  winget install Gyan.FFmpeg")
        elif sys.platform == "darwin":
            hint("  brew install ffmpeg")
        else:
            hint("  sudo apt install ffmpeg   # Debian/Ubuntu")
        return False


# ---------------------------------------------------------------------------
# Step 7 — Shortcuts / launchers
# ---------------------------------------------------------------------------

def _desktop_path() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("USERPROFILE", Path.home())) / "Desktop"
    elif sys.platform == "darwin":
        return Path.home() / "Desktop"
    else:
        # XDG
        return Path(os.environ.get("XDG_DESKTOP_DIR", Path.home() / "Desktop"))


def create_shortcuts() -> str | None:
    """Create platform shortcuts. Return path description or None."""
    python_exe = sys.executable
    plat = sys.platform

    if plat == "win32":
        return _create_windows_shortcuts(python_exe)
    elif plat == "darwin":
        return _create_macos_shortcut(python_exe)
    else:
        return _create_linux_shortcut(python_exe)


def _create_windows_shortcuts(python_exe: str) -> str | None:
    desktop = _desktop_path()
    if not desktop.is_dir():
        step(7, "Desktop folder not found — skipping shortcut", WARN)
        return None

    # Prefer pythonw if available (no console window)
    python_dir = Path(python_exe).parent
    pythonw = python_dir / "pythonw.exe"
    if not pythonw.is_file():
        pythonw = Path(python_exe)  # fallback

    # .bat file (visible console — useful for debugging)
    bat_path = desktop / "GoMaxWebcam.bat"
    bat_content = textwrap.dedent(f"""\
        @echo off
        title GoMaxWebcam
        "{python_exe}" -m gomaxwebcam %*
        if errorlevel 1 pause
    """)
    bat_path.write_text(bat_content, encoding="utf-8")

    # .vbs wrapper (hides console window)
    vbs_path = desktop / "GoMaxWebcam.vbs"
    vbs_content = textwrap.dedent(f"""\
        Set WshShell = CreateObject("WScript.Shell")
        WshShell.Run Chr(34) & "{pythonw}" & Chr(34) & " -m gomaxwebcam", 0, False
    """)
    vbs_path.write_text(vbs_content, encoding="utf-8")

    # Start Menu shortcut (user-level, no admin needed)
    start_menu = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    start_menu_note = ""
    if start_menu.is_dir():
        sm_bat = start_menu / "GoMaxWebcam.bat"
        sm_bat.write_text(bat_content, encoding="utf-8")
        start_menu_note = f"\n       Start Menu: {sm_bat}"

    locations = f"{desktop / 'GoMaxWebcam.vbs'} (silent)\n       Desktop bat: {bat_path}{start_menu_note}"
    step(7, "Desktop shortcuts created", CHECK)
    return locations

def _create_macos_shortcut(python_exe: str) -> str | None:
    desktop = _desktop_path()
    if not desktop.is_dir():
        step(7, "Desktop folder not found — skipping shortcut", WARN)
        return None

    cmd_path = desktop / "GoMaxWebcam.command"
    cmd_content = textwrap.dedent(f"""\
        #!/bin/bash
        # GoMaxWebcam launcher
        "{python_exe}" -m gomaxwebcam "$@"
    """)
    cmd_path.write_text(cmd_content, encoding="utf-8")
    cmd_path.chmod(0o755)

    step(7, "Desktop shortcut created", CHECK)
    return str(cmd_path)


def _create_linux_shortcut(python_exe: str) -> str | None:
    apps_dir = Path.home() / ".local" / "share" / "applications"
    apps_dir.mkdir(parents=True, exist_ok=True)

    desktop_entry = apps_dir / "gomaxwebcam.desktop"
    content = textwrap.dedent(f"""\
        [Desktop Entry]
        Type=Application
        Name=GoMaxWebcam
        Comment=GoPro virtual camera utility
        Exec="{python_exe}" -m gomaxwebcam
        Terminal=false
        Categories=AudioVideo;Video;
        StartupNotify=true
    """)
    desktop_entry.write_text(content, encoding="utf-8")
    desktop_entry.chmod(0o755)

    step(7, "Application shortcut created", CHECK)
    return str(desktop_entry)


# ---------------------------------------------------------------------------
# Re-run: update check
# ---------------------------------------------------------------------------

def _check_source_updates() -> str | None:
    """If in a git repo, check for upstream commits. Return summary or None."""
    repo = Path(__file__).resolve().parent
    git = shutil.which("git")
    if not git or not (repo / ".git").is_dir():
        return None

    try:
        subprocess.run(
            [git, "fetch", "--quiet"],
            capture_output=True, timeout=30, cwd=str(repo),
        )
        r = subprocess.run(
            [git, "log", "HEAD..origin/v2-rewrite", "--oneline"],
            capture_output=True, text=True, timeout=10, cwd=str(repo),
        )
        if r.returncode == 0 and r.stdout.strip():
            lines = r.stdout.strip().splitlines()
            return f"{len(lines)} new commit(s) on origin/v2-rewrite"
    except Exception:
        pass
    return None


def _check_pypi_updates() -> str | None:
    """Check PyPI for a newer version. Return summary or None."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade",
             "gomaxwebcam", "--dry-run", "-q"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0 and "Would install" in r.stdout:
            return "Newer version available on PyPI"
    except Exception:
        pass
    return None


def _prompt_yn(question: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    try:
        answer = input(f"       {question} {suffix} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not answer:
        return default_yes
    return answer.startswith("y")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    banner()

    # --- Detect re-run ---
    already_installed = False
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import gomaxwebcam"],
            capture_output=True, timeout=10,
        )
        already_installed = r.returncode == 0
    except Exception:
        pass

    if already_installed:
        print(f"  {CHECK} GoMaxWebcam is already installed.\n")

        update_msg = None
        if _in_repo():
            update_msg = _check_source_updates()
        else:
            update_msg = _check_pypi_updates()

        if update_msg:
            info(f"{YELLOW}{update_msg}{RESET}")
            if _prompt_yn("Update now?"):
                print()
                install_package()
                verify_import()
                print()
        else:
            info("Already up to date.")
            print()

        if _prompt_yn("Recreate desktop shortcut?"):
            loc = create_shortcuts()
            if loc:
                info(f"Shortcut: {loc}")
            print()

        _print_summary(already_installed=True)
        return

    # --- Fresh install ---
    if not check_python():
        sys.exit(1)

    if not check_pip():
        sys.exit(1)

    install_package()

    version = verify_import()
    if not version:
        error("Installation verification failed. Check the errors above.")
        sys.exit(1)

    check_virtual_camera()
    check_ffmpeg()

    loc = create_shortcuts()
    print()
    _print_summary(shortcut_location=loc)


def _print_summary(
    already_installed: bool = False,
    shortcut_location: str | None = None,
) -> None:
    print(f"  {BOLD}{GREEN}{'Update' if already_installed else 'Setup'} complete!{RESET}")
    print()
    print(f"  {BOLD}How to run:{RESET}")
    info("Desktop shortcut  — double-click GoMaxWebcam on your desktop")
    info(f"Terminal           — {sys.executable} -m gomaxwebcam")
    info("If on PATH         — gomaxwebcam")
    print()
    print(f"  {BOLD}Dashboard:{RESET}")
    info("Opens automatically in your browser on launch.")
    info("Default: http://127.0.0.1:<port> (port is random each run)")
    print()
    print(f"  {BOLD}Update later:{RESET}")
    if _in_repo():
        info("git pull && python install.py")
    else:
        info("python install.py   (re-run this script)")
    print()
    print(f"  {BOLD}Uninstall:{RESET}")
    info("python install.py --uninstall")
    print()


def uninstall() -> None:
    """Remove GoMaxWebcam: pip package, shortcuts, config, and cache."""
    print()
    print(f"  {BOLD}{RED}GoMaxWebcam Uninstaller{RESET}")
    print(f"  {DIM}{'=' * 46}{RESET}")
    print()

    removed = []
    kept = []

    # 1. Remove desktop shortcuts
    print(f"  {BOLD}[1/4]{RESET} Removing shortcuts...")
    desktop = _desktop_path()
    shortcut_files = []
    if sys.platform == "win32":
        shortcut_files = [
            desktop / "GoMaxWebcam.bat",
            desktop / "GoMaxWebcam.vbs",
        ]
        start_menu = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "GoMaxWebcam.bat"
        shortcut_files.append(start_menu)
    elif sys.platform == "darwin":
        shortcut_files = [desktop / "GoMaxWebcam.command"]
    else:
        shortcut_files = [
            Path.home() / ".local" / "share" / "applications" / "gomaxwebcam.desktop",
        ]

    for f in shortcut_files:
        if f.is_file():
            try:
                f.unlink()
                removed.append(f"Shortcut: {f}")
                info(f"{GREEN}Removed{RESET} {f}")
            except Exception as e:
                kept.append(f"Shortcut: {f} ({e})")
                info(f"{RED}Failed{RESET} {f}: {e}")
        else:
            info(f"{DIM}Not found: {f}{RESET}")

    # 2. Remove config directory
    print(f"\n  {BOLD}[2/4]{RESET} Removing config...")
    from platformdirs import user_config_dir, user_runtime_dir
    config_dirs = []
    try:
        config_dirs.append(Path(user_config_dir("GoMaxWebcam-v2")))
    except Exception:
        pass
    try:
        config_dirs.append(Path(user_runtime_dir("GoMaxWebcam-v2")))
    except Exception:
        pass

    for d in config_dirs:
        if d.is_dir():
            try:
                shutil.rmtree(d)
                removed.append(f"Config: {d}")
                info(f"{GREEN}Removed{RESET} {d}")
            except Exception as e:
                kept.append(f"Config: {d} ({e})")
                info(f"{RED}Failed{RESET} {d}: {e}")
        else:
            info(f"{DIM}Not found: {d}{RESET}")

    # 3. Remove cohn_db.json (credential cache)
    cohn_db = Path.cwd() / "cohn_db.json"
    if cohn_db.is_file():
        try:
            cohn_db.unlink()
            removed.append(f"Credentials: {cohn_db}")
            info(f"{GREEN}Removed{RESET} {cohn_db}")
        except Exception as e:
            kept.append(f"Credentials: {cohn_db} ({e})")

    # 4. Pip uninstall
    print(f"\n  {BOLD}[3/4]{RESET} Uninstalling pip package...")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "gomaxwebcam", "-y"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            removed.append("pip package: gomaxwebcam")
            info(f"{GREEN}Uninstalled{RESET} gomaxwebcam pip package")
        else:
            if "not installed" in r.stderr.lower() or "not installed" in r.stdout.lower():
                info(f"{DIM}Package not installed via pip{RESET}")
            else:
                kept.append(f"pip package ({r.stderr.strip()[:60]})")
                info(f"{RED}Failed:{RESET} {r.stderr.strip()[:80]}")
    except Exception as e:
        kept.append(f"pip package ({e})")
        info(f"{RED}Failed:{RESET} {e}")

    # 5. Summary
    print(f"\n  {BOLD}[4/4]{RESET} Summary")
    print()
    if removed:
        print(f"  {GREEN}Removed:{RESET}")
        for item in removed:
            info(f"  {item}")
    if kept:
        print(f"\n  {YELLOW}Could not remove:{RESET}")
        for item in kept:
            info(f"  {item}")

    if not kept:
        print(f"\n  {BOLD}{GREEN}GoMaxWebcam fully uninstalled.{RESET}")
    else:
        print(f"\n  {BOLD}{YELLOW}Partially uninstalled.{RESET} Remove the items above manually.")

    if _in_repo():
        print(f"\n  {DIM}Note: Source code in {Path.cwd()} was NOT deleted.")
        print(f"  To remove: delete this folder manually.{RESET}")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--uninstall", "uninstall", "--remove", "remove"):
        if _prompt_yn(f"{RED}Uninstall GoMaxWebcam?{RESET} This removes the package, shortcuts, and config.", default_yes=False):
            uninstall()
        else:
            print("  Cancelled.")
    else:
        main()
