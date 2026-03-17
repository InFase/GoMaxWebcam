"""
Firewall rule management for GoPro Bridge.

On first run, creates a persistent Windows Firewall inbound rule allowing
UDP traffic on port 8554 (the port GoPro uses to push MPEG-TS video).
Requests UAC elevation only when needed — subsequent runs skip this step.

The rule is created via `netsh advfirewall` so it persists across reboots.
A marker file in %APPDATA%/GoProBridge/ tracks whether the rule was created,
but we also verify the rule actually exists in Windows Firewall as a safety check.
"""

import ctypes
import json
import os
import subprocess
import sys
import tempfile

from logger import get_logger

log = get_logger("firewall")

# Constants
RULE_NAME = "GoPro Bridge - UDP Stream (Port 8554)"
UDP_PORT = 8554
APPDATA_DIR = os.path.join(os.environ.get("APPDATA", ""), "GoProBridge")
FIREWALL_MARKER = os.path.join(APPDATA_DIR, ".firewall_rule_created")


def is_admin() -> bool:
    """Check if the current process has administrator privileges."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except (AttributeError, OSError):
        # Not on Windows or ctypes not available
        return False


def firewall_rule_exists() -> bool:
    """
    Check if our firewall rule already exists in Windows Firewall.

    Uses `netsh advfirewall firewall show rule` to look for our named rule.
    Returns True if the rule is found, False otherwise.
    """
    try:
        result = subprocess.run(
            [
                "netsh", "advfirewall", "firewall", "show", "rule",
                f"name={RULE_NAME}"
            ],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        # netsh returns 0 and shows rule details if found,
        # or returns 1 / "No rules match" if not found
        if result.returncode == 0 and RULE_NAME in result.stdout:
            log.info("[EVENT:firewall] Firewall rule '%s' already exists", RULE_NAME)
            return True
        else:
            log.info("[EVENT:firewall] Firewall rule '%s' not found", RULE_NAME)
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        log.warning("[EVENT:firewall] Could not check firewall rule: %s", e)
        return False


def _create_firewall_rule_direct() -> bool:
    """
    Create the firewall rule directly (must be running as admin).

    Creates an inbound allow rule for UDP port 8554 so the GoPro's
    MPEG-TS video stream can reach our app.
    """
    try:
        result = subprocess.run(
            [
                "netsh", "advfirewall", "firewall", "add", "rule",
                f"name={RULE_NAME}",
                "dir=in",
                "action=allow",
                "protocol=UDP",
                f"localport={UDP_PORT}",
                "profile=any",
                "enable=yes",
                f"description=Allow inbound UDP on port {UDP_PORT} for GoPro MPEG-TS video stream",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode == 0:
            log.info("[EVENT:firewall] Firewall rule '%s' created successfully for UDP port %d", RULE_NAME, UDP_PORT)
            return True
        else:
            log.error(
                "[EVENT:firewall] Failed to create firewall rule. netsh returned %d: %s",
                result.returncode,
                result.stderr.strip() or result.stdout.strip(),
            )
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        log.error("[EVENT:firewall] Error creating firewall rule: %s", e)
        return False


def _write_marker():
    """Write a marker file so we know the rule was created by us."""
    try:
        os.makedirs(APPDATA_DIR, exist_ok=True)
        with open(FIREWALL_MARKER, "w", encoding="utf-8") as f:
            json.dump({
                "rule_name": RULE_NAME,
                "port": UDP_PORT,
                "protocol": "UDP",
                "created_by": "GoPro Bridge first-run setup",
            }, f, indent=2)
        log.debug("Firewall marker written to %s", FIREWALL_MARKER)
    except OSError as e:
        log.warning("Could not write firewall marker file: %s", e)


def _marker_exists() -> bool:
    """Check if the marker file from a previous successful setup exists."""
    return os.path.isfile(FIREWALL_MARKER)


def _request_uac_elevation() -> bool:
    """
    Re-launch this script (or a helper) with UAC elevation to create
    the firewall rule. Shows the standard Windows UAC prompt.

    We create a small temporary script that:
      1. Creates the netsh rule
      2. Writes the marker file
      3. Exits

    This way the main app only needs elevation for this one-time setup,
    not for normal operation.
    """
    # Build a small helper script that creates the rule
    helper_code = f'''
import subprocess
import sys
import os
import json

RULE_NAME = {RULE_NAME!r}
UDP_PORT = {UDP_PORT!r}
APPDATA_DIR = {APPDATA_DIR!r}
FIREWALL_MARKER = {FIREWALL_MARKER!r}

def main():
    # Create firewall rule
    result = subprocess.run(
        [
            "netsh", "advfirewall", "firewall", "add", "rule",
            f"name={{RULE_NAME}}",
            "dir=in",
            "action=allow",
            "protocol=UDP",
            f"localport={{UDP_PORT}}",
            "profile=any",
            "enable=yes",
            f"description=Allow inbound UDP on port {{UDP_PORT}} for GoPro MPEG-TS video stream",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    if result.returncode == 0:
        # Write marker
        os.makedirs(APPDATA_DIR, exist_ok=True)
        with open(FIREWALL_MARKER, "w", encoding="utf-8") as f:
            json.dump({{
                "rule_name": RULE_NAME,
                "port": UDP_PORT,
                "protocol": "UDP",
                "created_by": "GoPro Bridge first-run setup",
            }}, f, indent=2)
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

    # Write helper script to a temp file
    try:
        helper_path = os.path.join(tempfile.gettempdir(), "gopro_bridge_firewall_setup.py")
        with open(helper_path, "w", encoding="utf-8") as f:
            f.write(helper_code)
    except OSError as e:
        log.error("Could not write firewall helper script: %s", e)
        return False

    # Find the Python executable
    python_exe = sys.executable
    if not python_exe:
        log.error("Cannot find Python executable for UAC elevation.")
        return False

    log.info("[EVENT:firewall] Requesting UAC elevation to create firewall rule...")

    try:
        # ShellExecuteW with "runas" verb triggers the UAC prompt
        # Parameters: hwnd, verb, file, params, directory, show_cmd
        # SW_SHOWNORMAL = 1, SW_HIDE = 0
        ret = ctypes.windll.shell32.ShellExecuteW(
            None,           # parent window handle
            "runas",        # verb — triggers UAC elevation
            python_exe,     # program to run
            f'"{helper_path}"',  # arguments
            None,           # working directory
            0,              # SW_HIDE — don't show a console window
        )
        # ShellExecuteW returns >32 on success, <=32 on failure
        # 5 = SE_ERR_ACCESSDENIED (user clicked "No" on UAC)
        if ret <= 32:
            if ret == 5:
                log.warning("[EVENT:firewall] User declined UAC elevation prompt")
            else:
                log.error("[EVENT:firewall] ShellExecuteW failed with code %d", ret)
            return False

        log.info("[EVENT:firewall] UAC elevation requested — waiting for firewall rule creation...")

        # Wait for the marker file to appear (the elevated process writes it)
        # Poll for up to 30 seconds
        import time
        for _ in range(60):
            time.sleep(0.5)
            if _marker_exists():
                log.info("[EVENT:firewall] Firewall rule created successfully via UAC elevation")
                return True

        # If we get here, the elevated process may have failed
        log.warning(
            "[EVENT:firewall] Timed out waiting for firewall rule creation — "
            "UAC prompt may have been dismissed or rule creation failed"
        )
        return False

    except (AttributeError, OSError) as e:
        log.error("UAC elevation failed: %s", e)
        return False


def ensure_firewall_rule() -> bool:
    """
    Main entry point: ensure the UDP 8554 firewall rule exists.

    Logic:
      1. If the rule already exists in Windows Firewall → done, return True
      2. If we're already admin → create it directly
      3. Otherwise → request UAC elevation via a helper script

    Returns True if the rule exists (or was just created), False if setup failed.
    This should be called early in app startup, before attempting to receive
    the GoPro's UDP stream.
    """
    # Quick check: does the rule already exist?
    if firewall_rule_exists():
        if not _marker_exists():
            _write_marker()  # Sync marker with actual state
        return True

    log.info(
        "Firewall rule for UDP port %d not found. Setting up first-run firewall rule...",
        UDP_PORT,
    )

    # If we already have admin rights, create directly
    if is_admin():
        success = _create_firewall_rule_direct()
        if success:
            _write_marker()
        return success

    # Not admin — need UAC elevation
    success = _request_uac_elevation()
    if not success:
        log.error(
            "[EVENT:firewall] Could not create firewall rule — GoPro video stream on UDP port %d "
            "may be blocked by Windows Firewall. Allow manually or re-run as administrator.",
            UDP_PORT,
        )
    return success


def remove_firewall_rule() -> bool:
    """
    Remove the firewall rule (for uninstall/cleanup).
    Requires admin privileges.
    """
    if not firewall_rule_exists():
        log.info("Firewall rule not found, nothing to remove.")
        return True

    if not is_admin():
        log.warning("Admin privileges required to remove firewall rule.")
        return False

    try:
        result = subprocess.run(
            [
                "netsh", "advfirewall", "firewall", "delete", "rule",
                f"name={RULE_NAME}",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode == 0:
            log.info("Firewall rule removed successfully.")
            # Remove marker file
            try:
                if os.path.isfile(FIREWALL_MARKER):
                    os.remove(FIREWALL_MARKER)
            except OSError:
                pass
            return True
        else:
            log.error("Failed to remove firewall rule: %s", result.stderr.strip())
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        log.error("Error removing firewall rule: %s", e)
        return False
