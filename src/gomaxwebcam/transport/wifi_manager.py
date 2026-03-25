"""
transport/wifi_manager.py — Cross-platform WiFi AP discovery and connection manager.

Detects GoPro hotspot SSIDs (pattern: "GP-XXXX" or "GP-<serial>") and
manages OS-level WiFi connections to join the camera's access point.

Supports three platforms via OS command-line tools:
  - Windows: netsh wlan
  - macOS: airport scan + networksetup
  - Linux: nmcli (NetworkManager)

This module is used by the WiFiAPTransport to handle the network layer
before HTTP API commands are sent to the camera at 10.5.5.9:8080.

GoPro WiFi AP details (Open GoPro spec):
  - SSID pattern: "GP-<last 4 of serial>" or custom name set by user
  - Default gateway: 10.5.5.9
  - Camera HTTP port: 8080
  - Password: stored via keyring or entered by user
  - Security: WPA2-PSK
"""

from __future__ import annotations

import asyncio
import logging
import platform
import re
import shutil
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

log = logging.getLogger("gomaxwebcam.transport.wifi_manager")

# GoPro WiFi AP constants
GOPRO_AP_IP = "10.5.5.9"
GOPRO_AP_PORT = 8080
GOPRO_SSID_PATTERN = re.compile(r"^GP[-_ ]?\w{4,}", re.IGNORECASE)

# Connection timeouts
WIFI_SCAN_TIMEOUT_S = 30.0
WIFI_CONNECT_TIMEOUT_S = 30.0
WIFI_DISCONNECT_TIMEOUT_S = 10.0


class WiFiManagerError(Exception):
    """Base exception for WiFi manager operations."""


class WiFiScanError(WiFiManagerError):
    """Error scanning for WiFi networks."""


class WiFiConnectError(WiFiManagerError):
    """Error connecting to a WiFi network."""


class WiFiPlatformError(WiFiManagerError):
    """Platform not supported or required tool missing."""


class WiFiConnectionState(Enum):
    """State of the WiFi connection manager."""
    IDLE = auto()           # Not connected to any GoPro AP
    SCANNING = auto()       # Scanning for GoPro SSIDs
    CONNECTING = auto()     # Joining a GoPro AP
    CONNECTED = auto()      # Connected to a GoPro AP
    DISCONNECTING = auto()  # Leaving the GoPro AP
    ERROR = auto()          # Error state


@dataclass(frozen=True)
class WiFiNetwork:
    """Represents a discovered WiFi network."""
    ssid: str
    signal_strength: int = 0       # dBm or percentage (platform-dependent)
    security: str = ""             # e.g. "WPA2-Personal"
    is_gopro: bool = False         # True if SSID matches GoPro pattern
    bssid: str = ""                # MAC address of AP (if available)
    channel: int = 0               # WiFi channel (if available)


@dataclass
class WiFiManagerStats:
    """Statistics for the WiFi manager."""
    scans_performed: int = 0
    connections_attempted: int = 0
    connections_succeeded: int = 0
    connections_failed: int = 0
    last_error: Optional[str] = None
    last_connected_ssid: Optional[str] = None
    previous_ssid: Optional[str] = None  # SSID before we connected to GoPro


def is_gopro_ssid(ssid: str) -> bool:
    """Check if an SSID looks like a GoPro WiFi AP.

    GoPro cameras create hotspots with names like:
      - "GP12345678" (serial-based)
      - "GP-1234" (short form)
      - Custom names set by the user (not detectable)

    Returns True for SSIDs matching the GoPro pattern.
    """
    return bool(GOPRO_SSID_PATTERN.match(ssid))


class WiFiManager:
    """Cross-platform WiFi AP discovery and connection manager.

    Provides async methods to scan for WiFi networks, detect GoPro APs,
    connect to a chosen SSID, and restore the previous connection.

    Platform backends:
      - Windows: netsh wlan
      - macOS: system_profiler + networksetup
      - Linux: nmcli

    Usage:
        manager = WiFiManager()
        networks = await manager.scan()
        gopro_nets = [n for n in networks if n.is_gopro]
        if gopro_nets:
            await manager.connect(gopro_nets[0].ssid, password="mypassword")
            # ... use GoPro at 10.5.5.9:8080 ...
            await manager.disconnect()  # restores previous WiFi
    """

    def __init__(self):
        self._state = WiFiConnectionState.IDLE
        self._stats = WiFiManagerStats()
        self._connected_ssid: Optional[str] = None
        self._previous_ssid: Optional[str] = None
        self._platform = platform.system().lower()
        self._state_listeners: list = []

        # Validate platform support
        self._backend = self._resolve_backend()

    # -- Properties --

    @property
    def state(self) -> WiFiConnectionState:
        return self._state

    @property
    def stats(self) -> WiFiManagerStats:
        return self._stats

    @property
    def connected_ssid(self) -> Optional[str]:
        return self._connected_ssid

    @property
    def previous_ssid(self) -> Optional[str]:
        return self._previous_ssid

    @property
    def is_connected(self) -> bool:
        return self._state == WiFiConnectionState.CONNECTED

    @property
    def platform_name(self) -> str:
        return self._platform

    # -- State management --

    def _set_state(self, new_state: WiFiConnectionState) -> None:
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        log.info("[WiFiManager] State: %s -> %s", old.name, new_state.name)
        for listener in self._state_listeners:
            try:
                listener(old, new_state)
            except Exception:
                log.exception("Error in WiFi manager state listener")

    def add_state_listener(self, callback) -> None:
        """Register a callback(old_state, new_state) for state transitions."""
        self._state_listeners.append(callback)

    # -- Public async API --

    async def scan(self, timeout: float = WIFI_SCAN_TIMEOUT_S) -> list[WiFiNetwork]:
        """Scan for available WiFi networks.

        Returns a list of WiFiNetwork objects, sorted by signal strength
        (strongest first). GoPro SSIDs are flagged with is_gopro=True.

        Raises WiFiScanError on failure.
        """
        self._set_state(WiFiConnectionState.SCANNING)
        self._stats.scans_performed += 1

        try:
            networks = await asyncio.wait_for(
                self._backend.scan(),
                timeout=timeout,
            )

            # Flag GoPro networks
            flagged = []
            for net in networks:
                if is_gopro_ssid(net.ssid):
                    flagged.append(WiFiNetwork(
                        ssid=net.ssid,
                        signal_strength=net.signal_strength,
                        security=net.security,
                        is_gopro=True,
                        bssid=net.bssid,
                        channel=net.channel,
                    ))
                else:
                    flagged.append(net)

            # Sort by signal strength (strongest first)
            flagged.sort(key=lambda n: n.signal_strength, reverse=True)

            gopro_count = sum(1 for n in flagged if n.is_gopro)
            log.info(
                "WiFi scan found %d networks (%d GoPro)",
                len(flagged), gopro_count,
            )

            self._set_state(WiFiConnectionState.IDLE)
            return flagged

        except asyncio.TimeoutError:
            log.error("WiFi scan timed out after %.1fs", timeout)
            self._stats.last_error = "Scan timeout"
            self._set_state(WiFiConnectionState.ERROR)
            raise WiFiScanError("WiFi scan timed out")
        except WiFiScanError:
            self._set_state(WiFiConnectionState.ERROR)
            raise
        except Exception as e:
            log.error("WiFi scan failed: %s", e)
            self._stats.last_error = str(e)
            self._set_state(WiFiConnectionState.ERROR)
            raise WiFiScanError(f"WiFi scan failed: {e}") from e

    async def scan_for_gopro(self, timeout: float = WIFI_SCAN_TIMEOUT_S) -> list[WiFiNetwork]:
        """Scan specifically for GoPro WiFi APs.

        Convenience method that filters scan results to only GoPro SSIDs.
        """
        all_networks = await self.scan(timeout=timeout)
        return [n for n in all_networks if n.is_gopro]

    async def get_current_ssid(self) -> Optional[str]:
        """Get the currently connected WiFi SSID.

        Returns None if not connected to any WiFi network.
        """
        try:
            return await self._backend.get_current_ssid()
        except Exception as e:
            log.debug("Could not get current SSID: %s", e)
            return None

    async def connect(
        self,
        ssid: str,
        password: str,
        timeout: float = WIFI_CONNECT_TIMEOUT_S,
    ) -> bool:
        """Connect to a WiFi network (typically a GoPro AP).

        Saves the current SSID so it can be restored on disconnect.
        Creates a WiFi profile if needed (Windows) and joins the network.

        Args:
            ssid: The SSID to connect to.
            password: The WiFi password (WPA2-PSK).
            timeout: Maximum time to wait for connection.

        Returns True on successful connection.
        Raises WiFiConnectError on failure.
        """
        if self._state == WiFiConnectionState.CONNECTED:
            if self._connected_ssid == ssid:
                log.info("Already connected to %s", ssid)
                return True
            # Disconnect from current GoPro AP first
            await self.disconnect()

        self._set_state(WiFiConnectionState.CONNECTING)
        self._stats.connections_attempted += 1

        # Remember current SSID for restoration
        try:
            self._previous_ssid = await self.get_current_ssid()
            self._stats.previous_ssid = self._previous_ssid
            log.info("Current WiFi: %s (will restore on disconnect)", self._previous_ssid)
        except Exception:
            self._previous_ssid = None

        try:
            success = await asyncio.wait_for(
                self._backend.connect(ssid, password),
                timeout=timeout,
            )

            if success:
                self._connected_ssid = ssid
                self._stats.connections_succeeded += 1
                self._stats.last_connected_ssid = ssid
                self._set_state(WiFiConnectionState.CONNECTED)
                log.info("Connected to WiFi AP: %s", ssid)
                return True
            else:
                self._stats.connections_failed += 1
                self._stats.last_error = f"Connection to {ssid} failed"
                self._set_state(WiFiConnectionState.ERROR)
                raise WiFiConnectError(f"Failed to connect to {ssid}")

        except asyncio.TimeoutError:
            log.error("WiFi connection timed out after %.1fs", timeout)
            self._stats.connections_failed += 1
            self._stats.last_error = "Connection timeout"
            self._set_state(WiFiConnectionState.ERROR)
            raise WiFiConnectError(f"Connection to {ssid} timed out")
        except WiFiConnectError:
            raise
        except Exception as e:
            log.error("WiFi connection failed: %s", e)
            self._stats.connections_failed += 1
            self._stats.last_error = str(e)
            self._set_state(WiFiConnectionState.ERROR)
            raise WiFiConnectError(f"Connection failed: {e}") from e

    async def disconnect(self, restore_previous: bool = True) -> None:
        """Disconnect from the current GoPro WiFi AP.

        If restore_previous is True and we have a saved SSID,
        attempts to reconnect to the previous WiFi network.
        """
        if self._state not in (WiFiConnectionState.CONNECTED, WiFiConnectionState.CONNECTING):
            self._connected_ssid = None
            self._set_state(WiFiConnectionState.IDLE)
            return

        self._set_state(WiFiConnectionState.DISCONNECTING)
        ssid_to_restore = self._previous_ssid if restore_previous else None

        try:
            await asyncio.wait_for(
                self._backend.disconnect(),
                timeout=WIFI_DISCONNECT_TIMEOUT_S,
            )
        except Exception as e:
            log.warning("Error disconnecting from WiFi: %s", e)

        self._connected_ssid = None

        # Try to restore previous WiFi
        if ssid_to_restore:
            log.info("Restoring previous WiFi: %s", ssid_to_restore)
            try:
                await asyncio.wait_for(
                    self._backend.reconnect_to(ssid_to_restore),
                    timeout=WIFI_CONNECT_TIMEOUT_S,
                )
                log.info("Restored WiFi connection to: %s", ssid_to_restore)
            except Exception as e:
                log.warning("Could not restore previous WiFi '%s': %s", ssid_to_restore, e)

        self._previous_ssid = None
        self._set_state(WiFiConnectionState.IDLE)
        log.info("WiFi manager disconnected")

    async def restore_previous(self) -> bool:
        """Restore the previous WiFi connection without full disconnect cycle.

        Useful when the GoPro AP connection was lost unexpectedly and we
        want to reconnect to the home network.

        Returns True if restoration succeeded.
        """
        ssid = self._previous_ssid
        if not ssid:
            log.info("No previous SSID to restore")
            return False

        log.info("Restoring previous WiFi: %s", ssid)
        try:
            success = await asyncio.wait_for(
                self._backend.reconnect_to(ssid),
                timeout=WIFI_CONNECT_TIMEOUT_S,
            )
            if success:
                self._connected_ssid = None
                self._previous_ssid = None
                self._set_state(WiFiConnectionState.IDLE)
                log.info("Restored WiFi connection to: %s", ssid)
                return True
            else:
                log.warning("Could not restore previous WiFi '%s'", ssid)
                return False
        except Exception as e:
            log.warning("Restore previous WiFi failed: %s", e)
            return False

    async def verify_gopro_reachable(self, timeout: float = 5.0) -> bool:
        """Verify that the GoPro is reachable at 10.5.5.9 after WiFi connection.

        Sends a simple HTTP request to the camera's info endpoint.
        """
        import aiohttp

        url = f"http://{GOPRO_AP_IP}:{GOPRO_AP_PORT}/gp/gpControl/info"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    reachable = resp.status == 200
                    log.info("GoPro reachability check: %s (status=%d)", reachable, resp.status)
                    return reachable
        except Exception as e:
            log.debug("GoPro not reachable at %s: %s", url, e)
            return False

    # -- Backend resolution --

    def _resolve_backend(self) -> _WiFiBackend:
        """Select the platform-appropriate WiFi backend."""
        if self._platform == "windows":
            return _WindowsWiFiBackend()
        elif self._platform == "darwin":
            return _MacOSWiFiBackend()
        elif self._platform == "linux":
            return _LinuxWiFiBackend()
        else:
            raise WiFiPlatformError(f"Unsupported platform: {self._platform}")


# ======================================================================
# Platform backends (private)
# ======================================================================

class _WiFiBackend:
    """Abstract backend for platform-specific WiFi operations."""

    async def scan(self) -> list[WiFiNetwork]:
        raise NotImplementedError

    async def get_current_ssid(self) -> Optional[str]:
        raise NotImplementedError

    async def connect(self, ssid: str, password: str) -> bool:
        raise NotImplementedError

    async def disconnect(self) -> bool:
        raise NotImplementedError

    async def reconnect_to(self, ssid: str) -> bool:
        raise NotImplementedError

    @staticmethod
    async def _run_cmd(
        *args: str,
        timeout: float = 30.0,
        check: bool = False,
    ) -> tuple[int, str, str]:
        """Run a subprocess command asynchronously.

        Returns (returncode, stdout, stderr).
        """
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if check and proc.returncode != 0:
            raise WiFiManagerError(
                f"Command {args[0]} failed (rc={proc.returncode}): {stderr.strip()}"
            )

        return proc.returncode, stdout, stderr


class _WindowsWiFiBackend(_WiFiBackend):
    """Windows WiFi backend using netsh wlan commands."""

    async def scan(self) -> list[WiFiNetwork]:
        """Scan for WiFi networks using 'netsh wlan show networks mode=bssid'."""
        rc, stdout, stderr = await self._run_cmd(
            "netsh", "wlan", "show", "networks", "mode=bssid",
        )
        if rc != 0:
            raise WiFiScanError(f"netsh scan failed: {stderr.strip()}")

        return self._parse_netsh_networks(stdout)

    async def get_current_ssid(self) -> Optional[str]:
        """Get current WiFi SSID using 'netsh wlan show interfaces'."""
        rc, stdout, _ = await self._run_cmd(
            "netsh", "wlan", "show", "interfaces",
        )
        if rc != 0:
            return None

        for line in stdout.splitlines():
            line = line.strip()
            # Match "SSID" but not "BSSID"
            if line.startswith("SSID") and not line.startswith("BSSID"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    ssid = parts[1].strip()
                    if ssid:
                        return ssid
        return None

    async def connect(self, ssid: str, password: str) -> bool:
        """Connect to WiFi using netsh. Creates a profile first."""
        # Create a temporary XML WiFi profile
        profile_xml = self._create_wifi_profile_xml(ssid, password)

        # Add the profile
        import tempfile
        import os

        profile_path = os.path.join(tempfile.gettempdir(), f"gomaxwebcam_wifi_{ssid}.xml")
        try:
            with open(profile_path, "w", encoding="utf-8") as f:
                f.write(profile_xml)

            rc, stdout, stderr = await self._run_cmd(
                "netsh", "wlan", "add", "profile",
                f"filename={profile_path}",
            )
            if rc != 0:
                log.warning("Profile add returned rc=%d: %s", rc, stderr.strip())
        finally:
            try:
                os.unlink(profile_path)
            except OSError:
                pass

        # Connect using the profile
        rc, stdout, stderr = await self._run_cmd(
            "netsh", "wlan", "connect",
            f"name={ssid}",
            f"ssid={ssid}",
        )

        if rc != 0:
            log.error("netsh connect failed: %s", stderr.strip())
            return False

        # Wait for connection to establish
        for _ in range(15):  # Up to 15 seconds
            await asyncio.sleep(1.0)
            current = await self.get_current_ssid()
            if current == ssid:
                return True

        log.error("WiFi connect to '%s' did not establish within timeout", ssid)
        return False

    async def disconnect(self) -> bool:
        """Disconnect from current WiFi."""
        rc, _, _ = await self._run_cmd(
            "netsh", "wlan", "disconnect",
        )
        return rc == 0

    async def reconnect_to(self, ssid: str) -> bool:
        """Reconnect to a previously known WiFi network."""
        rc, _, stderr = await self._run_cmd(
            "netsh", "wlan", "connect",
            f"name={ssid}",
        )

        if rc != 0:
            log.warning("Could not reconnect to '%s': %s", ssid, stderr.strip())
            return False

        # Wait briefly for reconnection
        for _ in range(10):
            await asyncio.sleep(1.0)
            current = await self.get_current_ssid()
            if current == ssid:
                return True

        return False

    @staticmethod
    def _parse_netsh_networks(output: str) -> list[WiFiNetwork]:
        """Parse 'netsh wlan show networks mode=bssid' output."""
        networks = []
        current_ssid = ""
        current_signal = 0
        current_security = ""
        current_bssid = ""
        current_channel = 0

        for line in output.splitlines():
            line = line.strip()

            if line.startswith("SSID") and not line.startswith("BSSID"):
                # Save previous network if any
                if current_ssid:
                    networks.append(WiFiNetwork(
                        ssid=current_ssid,
                        signal_strength=current_signal,
                        security=current_security,
                        bssid=current_bssid,
                        channel=current_channel,
                    ))
                # Parse new SSID
                parts = line.split(":", 1)
                current_ssid = parts[1].strip() if len(parts) == 2 else ""
                current_signal = 0
                current_security = ""
                current_bssid = ""
                current_channel = 0

            elif line.startswith("BSSID"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    current_bssid = parts[1].strip()

            elif line.startswith("Signal") or line.startswith("Sig"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    sig_str = parts[1].strip().rstrip("%")
                    try:
                        current_signal = int(sig_str)
                    except ValueError:
                        current_signal = 0

            elif "Authentication" in line or "Auth" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    current_security = parts[1].strip()

            elif line.startswith("Channel"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    try:
                        current_channel = int(parts[1].strip())
                    except ValueError:
                        current_channel = 0

        # Don't forget last network
        if current_ssid:
            networks.append(WiFiNetwork(
                ssid=current_ssid,
                signal_strength=current_signal,
                security=current_security,
                bssid=current_bssid,
                channel=current_channel,
            ))

        return networks

    @staticmethod
    def _create_wifi_profile_xml(ssid: str, password: str) -> str:
        """Create a Windows WiFi profile XML for WPA2-PSK."""
        # Escape XML special characters
        import xml.sax.saxutils as saxutils
        ssid_esc = saxutils.escape(ssid)
        pass_esc = saxutils.escape(password)

        return f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{ssid_esc}</name>
    <SSIDConfig>
        <SSID>
            <name>{ssid_esc}</name>
        </SSID>
    </SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>manual</connectionMode>
    <MSM>
        <security>
            <authEncryption>
                <authentication>WPA2PSK</authentication>
                <encryption>AES</encryption>
                <useOneX>false</useOneX>
            </authEncryption>
            <sharedKey>
                <keyType>passPhrase</keyType>
                <protected>false</protected>
                <keyMaterial>{pass_esc}</keyMaterial>
            </sharedKey>
        </security>
    </MSM>
</WLANProfile>"""


class _MacOSWiFiBackend(_WiFiBackend):
    """macOS WiFi backend using system_profiler and networksetup."""

    def __init__(self):
        self._wifi_interface: Optional[str] = None

    async def _get_wifi_interface(self) -> str:
        """Get the WiFi interface name (usually 'en0' on Mac)."""
        if self._wifi_interface:
            return self._wifi_interface

        rc, stdout, _ = await self._run_cmd(
            "networksetup", "-listallhardwareports",
        )
        if rc == 0:
            lines = stdout.splitlines()
            for i, line in enumerate(lines):
                if "Wi-Fi" in line or "AirPort" in line:
                    # Next line should have "Device: enX"
                    if i + 1 < len(lines):
                        match = re.search(r"Device:\s+(\w+)", lines[i + 1])
                        if match:
                            self._wifi_interface = match.group(1)
                            return self._wifi_interface

        # Fallback
        self._wifi_interface = "en0"
        return self._wifi_interface

    async def scan(self) -> list[WiFiNetwork]:
        """Scan for WiFi networks using CoreWLAN via system_profiler or airport."""
        iface = await self._get_wifi_interface()

        # Try airport scan (faster, more reliable output)
        airport_path = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"
        rc, stdout, stderr = await self._run_cmd(
            airport_path, "-s",
        )

        if rc == 0:
            return self._parse_airport_scan(stdout)

        # Fallback to system_profiler
        rc, stdout, stderr = await self._run_cmd(
            "system_profiler", "SPAirPortDataType",
        )
        if rc != 0:
            raise WiFiScanError(f"WiFi scan failed on macOS: {stderr.strip()}")

        return self._parse_system_profiler(stdout)

    async def get_current_ssid(self) -> Optional[str]:
        """Get current WiFi SSID on macOS."""
        iface = await self._get_wifi_interface()

        # Try networksetup first (works on modern macOS)
        rc, stdout, _ = await self._run_cmd(
            "networksetup", "-getairportnetwork", iface,
        )
        if rc == 0:
            # Output: "Current Wi-Fi Network: SSID_NAME"
            match = re.search(r"Current Wi-Fi Network:\s*(.+)", stdout)
            if match:
                return match.group(1).strip()

        return None

    async def connect(self, ssid: str, password: str) -> bool:
        """Connect to WiFi on macOS using networksetup."""
        iface = await self._get_wifi_interface()

        rc, stdout, stderr = await self._run_cmd(
            "networksetup", "-setairportnetwork", iface, ssid, password,
        )

        if rc != 0:
            log.error("macOS WiFi connect failed: %s", stderr.strip())
            return False

        # Verify connection
        for _ in range(15):
            await asyncio.sleep(1.0)
            current = await self.get_current_ssid()
            if current == ssid:
                return True

        return False

    async def disconnect(self) -> bool:
        """Disconnect from WiFi on macOS."""
        iface = await self._get_wifi_interface()

        rc, _, _ = await self._run_cmd(
            "networksetup", "-setairportpower", iface, "off",
        )
        await asyncio.sleep(0.5)
        rc2, _, _ = await self._run_cmd(
            "networksetup", "-setairportpower", iface, "on",
        )
        return rc == 0

    async def reconnect_to(self, ssid: str) -> bool:
        """Reconnect to a known WiFi network on macOS.

        networksetup can reconnect to known networks by name
        (password is stored in the Keychain).
        """
        iface = await self._get_wifi_interface()
        rc, _, stderr = await self._run_cmd(
            "networksetup", "-setairportnetwork", iface, ssid,
        )

        if rc != 0:
            return False

        for _ in range(10):
            await asyncio.sleep(1.0)
            current = await self.get_current_ssid()
            if current == ssid:
                return True

        return False

    @staticmethod
    def _parse_airport_scan(output: str) -> list[WiFiNetwork]:
        """Parse output of 'airport -s' command.

        Output format (fixed-width columns):
                                    SSID BSSID             RSSI CHANNEL HT CC SECURITY
                                MyWiFi 00:11:22:33:44:55 -65  6       Y  -- WPA2(PSK/AES/AES)
        """
        networks = []
        lines = output.strip().splitlines()

        if not lines:
            return networks

        # Skip header line
        for line in lines[1:]:
            # Airport output is right-aligned SSID, then fixed columns
            # Parse from the right side since SSID can contain spaces
            match = re.match(
                r"\s*(.+?)\s+([0-9a-fA-F:]{17})\s+(-?\d+)\s+(\d+(?:,[\+\-]\d+)?)\s+\S+\s+\S+\s+(.+)$",
                line,
            )
            if match:
                ssid = match.group(1).strip()
                bssid = match.group(2)
                rssi = int(match.group(3))
                channel_str = match.group(4).split(",")[0]
                security = match.group(5).strip()

                try:
                    channel = int(channel_str)
                except ValueError:
                    channel = 0

                networks.append(WiFiNetwork(
                    ssid=ssid,
                    signal_strength=rssi,
                    security=security,
                    bssid=bssid,
                    channel=channel,
                ))

        return networks

    @staticmethod
    def _parse_system_profiler(output: str) -> list[WiFiNetwork]:
        """Parse system_profiler SPAirPortDataType output (fallback)."""
        networks = []
        # This parser handles the basic case; system_profiler output is verbose
        current_ssid = ""

        for line in output.splitlines():
            line_stripped = line.strip()

            # Look for SSID entries under "Other Local Wi-Fi Networks:"
            if line_stripped.endswith(":") and not line_stripped.startswith("PHY Mode"):
                potential_ssid = line_stripped.rstrip(":")
                if potential_ssid and not any(kw in potential_ssid for kw in [
                    "Wi-Fi", "spairport", "Interfaces", "Type", "Status",
                    "Current Network", "Security", "Hardware",
                ]):
                    current_ssid = potential_ssid

            elif "RSSI" in line_stripped and current_ssid:
                parts = line_stripped.split(":")
                rssi = 0
                if len(parts) == 2:
                    try:
                        rssi = int(parts[1].strip())
                    except ValueError:
                        pass
                networks.append(WiFiNetwork(
                    ssid=current_ssid,
                    signal_strength=rssi,
                ))
                current_ssid = ""

        return networks


class _LinuxWiFiBackend(_WiFiBackend):
    """Linux WiFi backend using nmcli (NetworkManager)."""

    def __init__(self):
        if not shutil.which("nmcli"):
            log.warning("nmcli not found — WiFi AP transport may not work on this system")

    async def scan(self) -> list[WiFiNetwork]:
        """Scan for WiFi networks using nmcli."""
        # Trigger a fresh scan
        await self._run_cmd("nmcli", "device", "wifi", "rescan")
        await asyncio.sleep(2.0)  # Give scan time to complete

        # List networks
        rc, stdout, stderr = await self._run_cmd(
            "nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY,BSSID,CHAN",
            "device", "wifi", "list",
        )
        if rc != 0:
            raise WiFiScanError(f"nmcli scan failed: {stderr.strip()}")

        return self._parse_nmcli_list(stdout)

    async def get_current_ssid(self) -> Optional[str]:
        """Get current WiFi SSID on Linux."""
        rc, stdout, _ = await self._run_cmd(
            "nmcli", "-t", "-f", "ACTIVE,SSID",
            "device", "wifi", "list",
        )
        if rc != 0:
            return None

        for line in stdout.splitlines():
            # Format: "yes:MySSID" (terse mode, colon-separated)
            parts = line.split(":", 1)
            if len(parts) == 2 and parts[0].strip().lower() == "yes":
                ssid = parts[1].strip()
                if ssid:
                    return ssid
        return None

    async def connect(self, ssid: str, password: str) -> bool:
        """Connect to WiFi on Linux using nmcli."""
        rc, stdout, stderr = await self._run_cmd(
            "nmcli", "device", "wifi", "connect", ssid,
            "password", password,
        )

        if rc != 0:
            log.error("nmcli connect failed: %s", stderr.strip())
            return False

        # Verify
        for _ in range(10):
            await asyncio.sleep(1.0)
            current = await self.get_current_ssid()
            if current == ssid:
                return True

        return False

    async def disconnect(self) -> bool:
        """Disconnect from WiFi on Linux."""
        # Find the WiFi device
        rc, stdout, _ = await self._run_cmd(
            "nmcli", "-t", "-f", "DEVICE,TYPE",
            "device", "status",
        )

        wifi_device = None
        for line in stdout.splitlines():
            parts = line.split(":")
            if len(parts) >= 2 and parts[1].strip() == "wifi":
                wifi_device = parts[0].strip()
                break

        if wifi_device:
            rc, _, _ = await self._run_cmd(
                "nmcli", "device", "disconnect", wifi_device,
            )
            return rc == 0

        return False

    async def reconnect_to(self, ssid: str) -> bool:
        """Reconnect to a known WiFi network on Linux."""
        rc, _, stderr = await self._run_cmd(
            "nmcli", "connection", "up", ssid,
        )

        if rc != 0:
            # Try connecting without password (uses saved connection)
            rc, _, _ = await self._run_cmd(
                "nmcli", "device", "wifi", "connect", ssid,
            )

        if rc != 0:
            return False

        for _ in range(10):
            await asyncio.sleep(1.0)
            current = await self.get_current_ssid()
            if current == ssid:
                return True

        return False

    @staticmethod
    def _parse_nmcli_list(output: str) -> list[WiFiNetwork]:
        """Parse nmcli terse output (-t -f SSID,SIGNAL,SECURITY,BSSID,CHAN)."""
        networks = []
        seen_ssids: set[str] = set()

        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue

            # nmcli terse format uses ':' as separator
            # But BSSID contains colons, so we need to handle this carefully
            # Format: SSID:SIGNAL:SECURITY:BSSID:CHAN
            # Split from the right for CHAN, then from the right for BSSID (17 chars xx:xx:xx:xx:xx:xx)
            parts = line.split(":")

            if len(parts) < 5:
                continue

            # CHAN is last
            chan_str = parts[-1].strip()
            # BSSID is 6 colon-separated hex pairs = parts[-7:-1]
            # Actually, let's be smarter: BSSID is always 17 chars with ':'
            # Reconstruct and parse differently

            # Simpler approach: split on first colon for SSID, then parse rest
            # Actually nmcli with -t escapes colons in SSIDs with \:
            # Let's just do a best-effort parse
            try:
                channel = int(chan_str)
            except ValueError:
                channel = 0

            # Reconstruct: join all but try to identify BSSID (6 hex pairs)
            # For simplicity, take first field as SSID, second as signal
            ssid = parts[0].strip().replace("\\:", ":")
            if not ssid or ssid in seen_ssids:
                continue
            seen_ssids.add(ssid)

            try:
                signal = int(parts[1].strip())
            except (ValueError, IndexError):
                signal = 0

            security = parts[2].strip() if len(parts) > 2 else ""

            # BSSID: try to reconstruct from parts[3:-1]
            bssid_parts = parts[3:-1]
            bssid = ":".join(bssid_parts) if len(bssid_parts) == 6 else ""

            networks.append(WiFiNetwork(
                ssid=ssid,
                signal_strength=signal,
                security=security,
                bssid=bssid,
                channel=channel,
            ))

        return networks
