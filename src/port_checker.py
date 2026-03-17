"""
port_checker.py — Check that UDP port 8554 is available before starting the stream.

The GoPro pushes its MPEG-TS video stream to UDP port 8554 on the host machine.
If another app (like the official GoPro Webcam Utility or a previous zombie ffmpeg
process) is already bound to that port, the stream won't arrive and nothing works.

This module checks port availability on startup and provides clear, actionable
error messages so the user knows exactly what to do.
"""

import socket
import subprocess
from dataclasses import dataclass
from typing import Optional

from logger import get_logger

log = get_logger("port_checker")


@dataclass
class PortConflict:
    """Details about what's occupying the port."""
    port: int
    protocol: str  # "UDP" or "TCP"
    pid: Optional[int] = None
    process_name: Optional[str] = None

    @property
    def user_message(self) -> str:
        """Human-readable message about the conflict."""
        if self.process_name and self.pid:
            return (
                f"{self.protocol} port {self.port} is already in use by "
                f"'{self.process_name}' (PID {self.pid}).\n\n"
                f"To fix this:\n"
                f"  1. Close '{self.process_name}', or\n"
                f"  2. Open Task Manager → find '{self.process_name}' → End Task, or\n"
                f"  3. Run in terminal: taskkill /PID {self.pid} /F\n\n"
                f"Common culprits: GoPro Webcam Utility, a stuck ffmpeg process, "
                f"or another streaming app."
            )
        elif self.pid:
            return (
                f"{self.protocol} port {self.port} is already in use by "
                f"PID {self.pid} (unknown process).\n\n"
                f"To fix this:\n"
                f"  Run in terminal: taskkill /PID {self.pid} /F"
            )
        else:
            return (
                f"{self.protocol} port {self.port} is already in use by another application.\n\n"
                f"To fix this:\n"
                f"  1. Close the official GoPro Webcam Utility if running\n"
                f"  2. Close any streaming software that might use this port\n"
                f"  3. Check Task Manager for stuck ffmpeg.exe processes"
            )


def check_udp_port_available(port: int) -> Optional[PortConflict]:
    """Check if a UDP port is available for binding.

    Tries to bind a UDP socket to the port. If it fails, the port is in use.
    Then tries to identify which process holds it using netstat.

    Args:
        port: The UDP port number to check (e.g. 8554).

    Returns:
        None if the port is available, or a PortConflict with details if not.
    """
    log.debug("[EVENT:port_check] Checking if UDP port %d is available...", port)

    # Try binding a UDP socket to the port
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # SO_REUSEADDR is intentionally NOT set — we want to detect conflicts
        sock.bind(("0.0.0.0", port))
        log.info("[EVENT:port_check] UDP port %d is available", port)
        return None  # Port is free
    except OSError as e:
        log.warning("[EVENT:port_check] UDP port %d is NOT available: %s", port, e)
        # Port is in use — try to find who's using it
        conflict = PortConflict(port=port, protocol="UDP")
        _identify_port_owner(conflict)
        return conflict
    finally:
        sock.close()


def _identify_port_owner(conflict: PortConflict) -> None:
    """Use netstat to find which process owns the port.

    Fills in conflict.pid and conflict.process_name if possible.
    This is best-effort — it may not work without admin privileges.
    """
    try:
        # netstat -ano shows all connections with PIDs
        # We look for UDP entries on our port
        result = subprocess.run(
            ["netstat", "-ano", "-p", "UDP"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        if result.returncode != 0:
            log.debug("netstat failed: %s", result.stderr)
            return

        for line in result.stdout.splitlines():
            # Look for lines like:  UDP    0.0.0.0:8554    *:*    12345
            line = line.strip()
            if not line.startswith("UDP"):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            local_addr = parts[1]
            # Check if this line matches our port
            if local_addr.endswith(f":{conflict.port}"):
                try:
                    conflict.pid = int(parts[-1])
                    log.debug("Port %d held by PID %d", conflict.port, conflict.pid)
                except ValueError:
                    pass
                break

        # If we found a PID, try to get the process name
        if conflict.pid:
            _get_process_name(conflict)

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        log.debug("Could not run netstat to identify port owner: %s", e)


def _get_process_name(conflict: PortConflict) -> None:
    """Get the process name for a PID using tasklist."""
    if not conflict.pid:
        return

    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {conflict.pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        if result.returncode != 0:
            return

        # Output format: "process.exe","12345","Console","1","12,345 K"
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and line.startswith('"'):
                # Extract process name from first quoted field
                name = line.split('"')[1]
                if name and name.lower() != "info:":
                    conflict.process_name = name
                    log.debug("PID %d is '%s'", conflict.pid, name)
                    break

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        log.debug("Could not get process name for PID %d: %s", conflict.pid, e)


def find_available_port(preferred_port: int, max_attempts: int = 10) -> int:
    """Try preferred_port, then preferred_port+1, etc. until one is free.

    This provides ephemeral port auto-selection: if the configured port
    (e.g. 8554) is occupied, we silently try 8555, 8556, … without
    persisting the change back to config.

    Args:
        preferred_port: The port to try first (e.g. 8554).
        max_attempts: Maximum number of ports to try before giving up.

    Returns:
        The first available port number.

    Raises:
        PortInUseError: If no port in the range is available.
    """
    for offset in range(max_attempts):
        candidate = preferred_port + offset
        if candidate > 65535:
            break
        conflict = check_udp_port_available(candidate)
        if conflict is None:
            if offset > 0:
                log.info(
                    "[EVENT:port_check] Preferred port %d busy, using port %d instead",
                    preferred_port, candidate,
                )
            return candidate
        log.debug(
            "[EVENT:port_check] Port %d occupied, trying next...", candidate,
        )

    # All candidates occupied — raise with info about the preferred port
    conflict = check_udp_port_available(preferred_port)
    if conflict is None:
        # Race: port freed up between checks — use it
        return preferred_port
    raise PortInUseError(conflict)


def check_port_and_raise(port: int) -> None:
    """Check port availability and raise RuntimeError if in use.

    This is the simple one-call function for use in startup sequences.

    Args:
        port: The UDP port to check.

    Raises:
        PortInUseError: If the port is occupied, with a user-friendly message.
    """
    conflict = check_udp_port_available(port)
    if conflict is not None:
        raise PortInUseError(conflict)


class PortInUseError(RuntimeError):
    """Raised when the required UDP port is already in use."""

    def __init__(self, conflict: PortConflict):
        self.conflict = conflict
        super().__init__(conflict.user_message)
