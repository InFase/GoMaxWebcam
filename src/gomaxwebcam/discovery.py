"""
discovery.py — Async USB device enumeration and GoPro detection for v2.

Detects connected GoPro cameras by scanning the USB bus for devices with
GoPro's vendor ID (0x2672). Extracts serial numbers from USB device IDs
and computes the camera's NCM IP address from the serial.

This module is designed for the v2 asyncio architecture:
  - All blocking OS calls (PowerShell/WMI, pnputil) are run via
    asyncio.to_thread() so they don't block the event loop.
  - Results are returned as GoProDeviceInfo dataclass instances.

Discovery chain:
  1. WMI via PowerShell — structured JSON, most reliable on Windows
  2. pnputil /enum-devices — fallback if PowerShell unavailable
  3. Serial → IP computation — Open GoPro formula: 172.2X.1YZ.51

Platform: Windows 11 primary. On non-Windows, enumeration returns
empty (camera detection deferred to open-gopro SDK mDNS).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

log = logging.getLogger("gomaxwebcam.discovery")

# ---------------------------------------------------------------------------
# GoPro USB identifiers
# ---------------------------------------------------------------------------

# All GoPro cameras use vendor ID 0x2672 (GoPro, Inc.)
GOPRO_VENDOR_ID = 0x2672
GOPRO_VENDOR_IDS = frozenset({0x2672})

# Known GoPro USB product IDs across models.
# Unknown PIDs under vendor 0x2672 are still treated as GoPro devices.
GOPRO_KNOWN_PIDS: frozenset[int] = frozenset({
    0x0004,  # Hero 3
    0x0006,  # Hero 3+ Silver
    0x0007,  # Hero 3+ Black
    0x0011,  # Hero 3+ Black (alternate)
    0x000E,  # Hero 4 Black
    0x0049,  # Hero 8 Black
    0x0052,  # Hero 9 Black
    0x0059,  # Webcam/USB mode (Hero 9+)
    0x000D,  # RNDIS Ethernet gadget
    0x0043,  # Alternate USB config
})

# GoPro HTTP API port
GOPRO_API_PORT = 8080


class DiscoveryMethod(Enum):
    """Identifies which method found the camera."""
    WMI = auto()
    PNPUTIL = auto()
    SERIAL_COMPUTED = auto()


@dataclass
class GoProDeviceInfo:
    """A GoPro camera discovered on the USB bus.

    Attributes:
        vendor_id: USB vendor ID (always 0x2672 for GoPro).
        product_id: USB product ID (model-dependent).
        description: Human-readable device description from OS.
        serial_number: Full serial extracted from USB device ID, or None.
        serial_suffix: Last 3 digits of serial (used for IP computation).
        camera_ip: Computed NCM IP address (172.2X.1YZ.51), or None.
        discovery_method: How this device was found.
    """
    vendor_id: int = GOPRO_VENDOR_ID
    product_id: int = 0
    description: str = "GoPro Device"
    serial_number: Optional[str] = None
    serial_suffix: Optional[str] = None
    camera_ip: Optional[str] = None
    discovery_method: Optional[DiscoveryMethod] = None

    @property
    def usb_id_str(self) -> str:
        """USB VID:PID as hex string."""
        return f"{self.vendor_id:04X}:{self.product_id:04X}"

    @property
    def is_known_model(self) -> bool:
        """True if the product ID is in the known GoPro PID table."""
        return self.product_id in GOPRO_KNOWN_PIDS

    def __str__(self) -> str:
        parts = [f"GoPro [{self.usb_id_str}] '{self.description}'"]
        if self.serial_number:
            parts.append(f"serial={self.serial_number}")
        if self.camera_ip:
            parts.append(f"ip={self.camera_ip}")
        if self.discovery_method:
            parts.append(f"via {self.discovery_method.name}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Serial → IP computation (Open GoPro spec)
# ---------------------------------------------------------------------------

def compute_ip_from_serial(serial: str) -> Optional[str]:
    """Compute the GoPro's USB NCM IP address from its serial number.

    Open GoPro spec: the USB IP is 172.2X.1YZ.51 where XYZ are the
    last three digits of the serial number.

    Example: serial "C3531350067212" → last 3 digits "212"
             → 172.22.112.51

    Args:
        serial: Full serial number string (must have ≥3 trailing digits).

    Returns:
        IP address string, or None if serial is invalid.
    """
    if not serial or len(serial) < 3:
        return None
    last3 = serial[-3:]
    if not last3.isdigit():
        return None
    x, y, z = last3[0], last3[1], last3[2]
    return f"172.2{x}.1{y}{z}.51"


def compute_serial_suffix(serial: str) -> Optional[str]:
    """Extract the last 3 digits of a GoPro serial number.

    These 3 digits determine the USB NCM IP address and are used
    by open-gopro SDK as the 'serial' parameter for WiredGoPro.

    Returns:
        3-character string of digits, or None if invalid.
    """
    if not serial or len(serial) < 3:
        return None
    last3 = serial[-3:]
    if not last3.isdigit():
        return None
    return last3


# ---------------------------------------------------------------------------
# USB device ID parsing
# ---------------------------------------------------------------------------

def parse_vid_pid(device_id: str) -> tuple[int, int]:
    """Extract vendor ID and product ID from a USB device ID string.

    Device IDs look like: USB\\VID_2672&PID_0059\\C3531350067212

    Returns:
        (vendor_id, product_id) as integers, or (0, 0) on failure.
    """
    vid_match = re.search(r"VID_([0-9A-Fa-f]{4})", device_id, re.IGNORECASE)
    pid_match = re.search(r"PID_([0-9A-Fa-f]{4})", device_id, re.IGNORECASE)
    vid = int(vid_match.group(1), 16) if vid_match else 0
    pid = int(pid_match.group(1), 16) if pid_match else 0
    return vid, pid


def extract_serial_from_device_id(device_id: str) -> Optional[str]:
    """Extract serial number from a USB DeviceID string.

    The USB composite device entry has the format:
        USB\\VID_2672&PID_0059\\C3531350067212

    The serial is the last segment, but only on entries that do NOT
    contain '&MI_' (which indicates a child interface, not the
    composite device).

    Returns:
        Serial string, or None if not found or not a composite entry.
    """
    if "&MI_" in device_id.upper():
        return None
    parts = device_id.replace("\\\\", "\\").split("\\")
    if len(parts) >= 3:
        candidate = parts[-1]
        # Serial numbers are alphanumeric, typically 10+ chars
        if len(candidate) >= 8 and candidate.isalnum():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Synchronous enumeration backends (run via to_thread)
# ---------------------------------------------------------------------------

def _enumerate_via_wmi_sync() -> list[GoProDeviceInfo]:
    """Use PowerShell/WMI to find GoPro USB devices (blocking).

    Queries Win32_PnPEntity for devices whose DeviceID contains
    GoPro's vendor ID string 'VID_2672'.
    """
    if sys.platform != "win32":
        return []

    vid_filters = " -or ".join(
        f"$_.DeviceID -like '*VID_{vid:04X}*'" for vid in GOPRO_VENDOR_IDS
    )
    ps_cmd = (
        f"Get-WmiObject Win32_PnPEntity | "
        f"Where-Object {{ {vid_filters} }} | "
        f"Select-Object DeviceID, Name, Description | "
        f"ConvertTo-Json -Compress"
    )

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        log.debug("WMI enumeration failed: %s", e)
        return []

    if result.returncode != 0 or not result.stdout.strip():
        return []

    raw = result.stdout.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Failed to parse WMI JSON: %.200s", raw)
        return []

    if isinstance(data, dict):
        data = [data]

    devices: list[GoProDeviceInfo] = []
    serial: Optional[str] = None

    for entry in data:
        device_id = entry.get("DeviceID", "")
        name = entry.get("Name", "") or entry.get("Description", "Unknown GoPro")

        vid, pid = parse_vid_pid(device_id)
        if vid not in GOPRO_VENDOR_IDS:
            continue

        entry_serial = extract_serial_from_device_id(device_id)
        if entry_serial:
            serial = entry_serial

        devices.append(GoProDeviceInfo(
            vendor_id=vid,
            product_id=pid,
            description=name,
            discovery_method=DiscoveryMethod.WMI,
        ))

    # Attach serial + computed IP to all devices (same physical camera)
    if serial:
        suffix = compute_serial_suffix(serial)
        ip = compute_ip_from_serial(serial)
        for d in devices:
            d.serial_number = serial
            d.serial_suffix = suffix
            d.camera_ip = ip

    return devices


def _enumerate_via_pnputil_sync() -> list[GoProDeviceInfo]:
    """Fallback: use pnputil to find GoPro USB devices (blocking).

    Runs 'pnputil /enum-devices /connected' and scans output for
    GoPro's vendor ID.
    """
    if sys.platform != "win32":
        return []

    vid_strs = {f"vid_{vid:04X}".lower() for vid in GOPRO_VENDOR_IDS}

    try:
        result = subprocess.run(
            ["pnputil", "/enum-devices", "/connected"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        log.debug("pnputil enumeration failed: %s", e)
        return []

    if result.returncode != 0:
        return []

    devices: list[GoProDeviceInfo] = []
    serial: Optional[str] = None
    current_instance_id = ""
    current_name = ""

    def _check_and_add() -> None:
        nonlocal current_instance_id, current_name, serial
        if current_instance_id and any(v in current_instance_id.lower() for v in vid_strs):
            vid, pid = parse_vid_pid(current_instance_id)
            if vid in GOPRO_VENDOR_IDS:
                entry_serial = extract_serial_from_device_id(current_instance_id)
                if entry_serial:
                    serial = entry_serial
                devices.append(GoProDeviceInfo(
                    vendor_id=vid,
                    product_id=pid,
                    description=current_name or "GoPro Device",
                    discovery_method=DiscoveryMethod.PNPUTIL,
                ))
        current_instance_id = ""
        current_name = ""

    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("Instance ID:"):
            _check_and_add()
            current_instance_id = line.split(":", 1)[1].strip()
        elif line.startswith("Device Description:") or line.startswith("Name:"):
            current_name = line.split(":", 1)[1].strip()

    _check_and_add()  # Handle last entry

    if serial:
        suffix = compute_serial_suffix(serial)
        ip = compute_ip_from_serial(serial)
        for d in devices:
            d.serial_number = serial
            d.serial_suffix = suffix
            d.camera_ip = ip

    return devices


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------

async def enumerate_usb_gopro_devices() -> list[GoProDeviceInfo]:
    """Enumerate USB-connected GoPro cameras asynchronously.

    Tries WMI first, then falls back to pnputil. Both are run via
    asyncio.to_thread() to avoid blocking the event loop.

    On non-Windows platforms, returns empty (open-gopro SDK handles
    mDNS discovery directly).

    Returns:
        List of GoProDeviceInfo for each GoPro found on the USB bus.
    """
    if sys.platform != "win32":
        log.debug("USB enumeration skipped (non-Windows platform)")
        return []

    # Try WMI first
    try:
        devices = await asyncio.to_thread(_enumerate_via_wmi_sync)
        if devices:
            log.info("Found %d GoPro device(s) via WMI", len(devices))
            for d in devices:
                log.debug("  %s", d)
            return devices
    except Exception as e:
        log.debug("WMI enumeration failed: %s", e)

    # Fallback to pnputil
    try:
        devices = await asyncio.to_thread(_enumerate_via_pnputil_sync)
        if devices:
            log.info("Found %d GoPro device(s) via pnputil", len(devices))
            for d in devices:
                log.debug("  %s", d)
            return devices
    except Exception as e:
        log.debug("pnputil enumeration failed: %s", e)

    log.debug("No GoPro devices found on USB bus")
    return []


async def find_gopro_device() -> Optional[GoProDeviceInfo]:
    """Find a single GoPro camera on the USB bus.

    Convenience wrapper that returns the first device with a serial
    number (i.e., the composite USB device entry). If no device has
    a serial, returns the first device found. Returns None if no
    GoPro is connected.

    Returns:
        GoProDeviceInfo or None.
    """
    devices = await enumerate_usb_gopro_devices()
    if not devices:
        return None

    # Prefer the entry with serial (composite device)
    for d in devices:
        if d.serial_number:
            return d

    # Fall back to first entry
    return devices[0]


def enumerate_usb_gopro_devices_sync() -> list[GoProDeviceInfo]:
    """Synchronous version of enumerate_usb_gopro_devices.

    For use in non-async contexts (e.g., tray icon status checks).
    """
    if sys.platform != "win32":
        return []

    try:
        devices = _enumerate_via_wmi_sync()
        if devices:
            return devices
    except Exception:
        pass

    try:
        devices = _enumerate_via_pnputil_sync()
        if devices:
            return devices
    except Exception:
        pass

    return []
