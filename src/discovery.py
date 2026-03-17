"""
discovery.py — USB device enumeration and GoPro network discovery

Detects a connected GoPro Hero 12 by USB vendor/product ID, then finds
the corresponding NCM network interface and resolves the camera's IP.

Discovery fallback chain (discover_gopro_ip_chain):
  1. USB interface scan (ipconfig + psutil) — fastest, most reliable
  2. IP range scan (brute-force 172.2x.1xx.51) — works without adapter name
  3. mDNS (_gopro-web._tcp.local.) — works even with unusual IP ranges

GoPro USB identifiers:
  Vendor ID:  0x0A70 (2672 decimal) — GoPro, Inc.
  Known PIDs: 0x000D (Hero 12 Black USB Ethernet/RNDIS)

When the GoPro connects over USB, Windows creates a virtual network
adapter using NCM (Network Control Model). The camera's HTTP API is
available on port 8080 at the adapter's gateway IP (typically 172.2x.1xx.51).
"""

import re
import socket
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from logger import get_logger

log = get_logger(__name__)

# --- GoPro USB identifiers ---
# GoPro has used multiple vendor IDs across product generations:
#   0x2672 — Hero 12 and newer (decimal 9842)
#   0x0A70 — Older GoPro models (decimal 2672)
GOPRO_VENDOR_ID = 0x2672   # Current GoPro VID (Hero 12+)
GOPRO_VENDOR_IDS = {0x2672, 0x0A70}  # All known GoPro VIDs

# Known GoPro USB product IDs (Hero 12 uses NCM/RNDIS mode)
# Multiple PIDs are possible depending on firmware and USB mode
GOPRO_KNOWN_PIDS = {
    0x0059,  # Hero 12 webcam/USB mode
    0x000D,  # Hero 12 RNDIS Ethernet gadget
    0x0011,  # Alternate NCM mode
    0x0043,  # Alternate USB config
}

# GoPro camera IPs are typically in 172.2x.1xx.0/24 range, with .51 being the camera
GOPRO_IP_PATTERN = re.compile(r"172\.\d{1,3}\.\d{1,3}\.\d{1,3}")
GOPRO_CAMERA_SUFFIX = ".51"  # Camera is usually at x.x.x.51

# --- IP range scan constants ---
GOPRO_SECOND_OCTET_RANGE = range(20, 30)   # 172.20.x.x through 172.29.x.x
GOPRO_THIRD_OCTET_RANGE = range(100, 200)  # 172.2x.100.x through 172.2x.199.x
GOPRO_CAMERA_HOST = 51                      # Camera is at .51 on the subnet
GOPRO_CANDIDATE_HOSTS = [51, 1, 101]        # Candidate host octets to probe per /24 subnet
IP_RANGE_SCAN_TIMEOUT = 0.5                 # Aggressive timeout for parallel scan
IP_RANGE_SCAN_MAX_WORKERS = 50              # Parallel connection attempts
GOPRO_API_PORT = 8080                       # GoPro HTTP API port
GOPRO_STATUS_ENDPOINT = "/gopro/webcam/status"  # Endpoint to verify GoPro identity

# --- mDNS constants ---
GOPRO_MDNS_SERVICE_TYPE = "_gopro-web._tcp.local."
GOPRO_MDNS_ALT_SERVICE_TYPES = [
    "_gopro-web._tcp.local.",
    "_http._tcp.local.",         # Some GoPro firmware versions use generic HTTP
]
MDNS_DEFAULT_TIMEOUT = 3.0  # Seconds to listen for mDNS announcements
MDNS_PORT = 5353
MDNS_ADDR = "224.0.0.251"


class DiscoveryMethod(Enum):
    """Identifies which discovery method found the camera's IP."""
    USB_SCAN = "usb_scan"
    IP_SCAN = "ip_scan"
    MDNS = "mdns"


@dataclass
class GoProDevice:
    """Represents a discovered GoPro camera on USB."""
    vendor_id: int
    product_id: int
    description: str
    interface_name: Optional[str] = None
    camera_ip: Optional[str] = None
    host_ip: Optional[str] = None
    discovery_method: Optional[DiscoveryMethod] = None

    @property
    def usb_id_str(self) -> str:
        return f"{self.vendor_id:04X}:{self.product_id:04X}"

    def __str__(self) -> str:
        status = f"GoPro [{self.usb_id_str}] '{self.description}'"
        if self.camera_ip:
            status += f" @ {self.camera_ip}"
        if self.discovery_method:
            status += f" (via {self.discovery_method.value})"
        return status


# ============================================================================
# USB device enumeration
# ============================================================================

def enumerate_usb_gopro_devices() -> list[GoProDevice]:
    """Enumerate USB devices and return any with GoPro's vendor ID.

    Uses Windows Setup API (setupapi.dll) via ctypes to scan the USB bus
    for devices matching GoPro's vendor ID (0x0A70). This is the most
    reliable detection method -- it works even before network interfaces
    come up.

    Returns:
        List of GoProDevice objects found on the USB bus.
    """
    devices = []

    # Try WMI via PowerShell first (simpler and more reliable than raw SetupAPI)
    try:
        devices = _enumerate_via_wmi()
        if devices:
            return devices
    except Exception as e:
        log.debug("[EVENT:discovery_start] WMI enumeration failed, trying pnputil: %s", e)

    # Fallback: parse pnputil output
    try:
        devices = _enumerate_via_pnputil()
        if devices:
            return devices
    except Exception as e:
        log.debug("[EVENT:discovery_start] pnputil enumeration failed: %s", e)

    return devices


def _enumerate_via_wmi() -> list[GoProDevice]:
    """Use PowerShell/WMI to find GoPro USB devices.

    Queries Win32_PnPEntity for devices whose DeviceID contains
    GoPro's vendor ID string 'VID_0A70'. This catches the camera
    regardless of which USB mode or product ID it's using.
    """
    # Build a filter that matches ANY known GoPro VID
    vid_filters = " -or ".join(
        f"$_.DeviceID -like '*VID_{vid:04X}*'" for vid in GOPRO_VENDOR_IDS
    )
    ps_cmd = (
        f"Get-WmiObject Win32_PnPEntity | "
        f"Where-Object {{ {vid_filters} }} | "
        f"Select-Object DeviceID, Name, Description | "
        f"ConvertTo-Json -Compress"
    )

    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_cmd],
        capture_output=True, text=True, timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    if result.returncode != 0 or not result.stdout.strip():
        return []

    import json
    raw = result.stdout.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("[EVENT:discovery_failed] Failed to parse WMI JSON output: %s", raw[:200])
        return []

    if isinstance(data, dict):
        data = [data]

    devices = []
    for entry in data:
        device_id = entry.get("DeviceID", "")
        name = entry.get("Name", "") or entry.get("Description", "Unknown GoPro")

        vid, pid = _parse_vid_pid(device_id)
        if vid in GOPRO_VENDOR_IDS:
            devices.append(GoProDevice(
                vendor_id=vid,
                product_id=pid,
                description=name,
            ))

    return devices


def _enumerate_via_pnputil() -> list[GoProDevice]:
    """Fallback: use pnputil to find GoPro USB devices.

    Runs 'pnputil /enum-devices /connected' and scans output for
    GoPro's vendor ID. Less structured than WMI but works without
    PowerShell.
    """
    vid_strs = {f"VID_{vid:04X}".lower() for vid in GOPRO_VENDOR_IDS}

    result = subprocess.run(
        ["pnputil", "/enum-devices", "/connected"],
        capture_output=True, text=True, timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    if result.returncode != 0:
        return []

    devices = []
    current_instance_id = ""
    current_name = ""

    def _check_and_add():
        nonlocal current_instance_id, current_name
        if current_instance_id and any(v in current_instance_id.lower() for v in vid_strs):
            vid, pid = _parse_vid_pid(current_instance_id)
            if vid in GOPRO_VENDOR_IDS:
                devices.append(GoProDevice(
                    vendor_id=vid,
                    product_id=pid,
                    description=current_name or "GoPro Device",
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

    return devices


def _parse_vid_pid(device_id: str) -> tuple[int, int]:
    """Extract vendor ID and product ID from a USB device ID string.

    Device IDs look like: USB\\VID_0A70&PID_000D\\...
    Returns (vendor_id, product_id) as integers, or (0, 0) on failure.
    """
    vid_match = re.search(r"VID_([0-9A-Fa-f]{4})", device_id, re.IGNORECASE)
    pid_match = re.search(r"PID_([0-9A-Fa-f]{4})", device_id, re.IGNORECASE)

    vid = int(vid_match.group(1), 16) if vid_match else 0
    pid = int(pid_match.group(1), 16) if pid_match else 0
    return vid, pid


# ============================================================================
# IP discovery methods (individual strategies)
# ============================================================================

def discover_gopro_ip() -> Optional[str]:
    """Find the GoPro's IP address by scanning network interfaces.

    When connected via USB, the GoPro creates an NCM network adapter.
    The camera's IP is typically the gateway of that interface,
    ending in .51 on a 172.x.x.x subnet.

    This is the "USB scan" method -- it uses ipconfig and psutil to find
    the GoPro adapter by name/pattern. It's the fastest method but can
    fail if the adapter has a generic name.

    Returns:
        Camera IP string (e.g. '172.20.123.51') or None if not found.
    """
    # Strategy 1: Parse ipconfig for GoPro-related adapters
    ip = _discover_via_ipconfig()
    if ip:
        return ip

    # Strategy 2: Scan 172.x.x.x interfaces and probe port 8080
    ip = _discover_via_interface_scan()
    if ip:
        return ip

    return None


def _discover_via_ipconfig() -> Optional[str]:
    """Parse 'ipconfig /all' to find GoPro's network interface.

    Looks for network adapters that match GoPro patterns (name contains
    'GoPro', 'RNDIS', 'NCM', or 'Remote NDIS') and extracts the
    gateway IP.
    """
    try:
        result = subprocess.run(
            ["ipconfig", "/all"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception as e:
        log.debug("ipconfig failed: %s", e)
        return None

    if result.returncode != 0:
        return None

    gopro_adapter_patterns = [
        re.compile(r"gopro", re.IGNORECASE),
        re.compile(r"remote ndis", re.IGNORECASE),
        re.compile(r"rndis", re.IGNORECASE),
        re.compile(r"ncm", re.IGNORECASE),
    ]

    lines = result.stdout.splitlines()
    in_gopro_section = False
    section_ips = []
    all_172_ips = []  # Collect ALL 172.x.x.x IPs as fallback

    for i, line in enumerate(lines):
        if line and not line.startswith(" ") and ":" in line:
            if in_gopro_section and section_ips:
                return _pick_camera_ip(section_ips)

            in_gopro_section = any(p.search(line) for p in gopro_adapter_patterns)
            section_ips = []
            continue

        if in_gopro_section:
            ipv4_match = re.search(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", line)
            if ipv4_match:
                ip = ipv4_match.group(1)
                if GOPRO_IP_PATTERN.match(ip):
                    section_ips.append(ip)

            if "default gateway" in line.lower() or "gateway" in line.lower():
                gw_match = re.search(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", line)
                if gw_match:
                    gw_ip = gw_match.group(1)
                    if GOPRO_IP_PATTERN.match(gw_ip):
                        return gw_ip

        # Collect all 172.x.x.x IPs regardless of adapter name
        ipv4_match = re.search(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", line)
        if ipv4_match:
            ip = ipv4_match.group(1)
            if GOPRO_IP_PATTERN.match(ip) and ip not in all_172_ips:
                all_172_ips.append(ip)

    if in_gopro_section and section_ips:
        return _pick_camera_ip(section_ips)

    # Fallback: if no named GoPro adapter found, probe all 172.x.x.x IPs
    # The adapter may have a generic name like "Ethernet 3"
    if all_172_ips:
        log.debug("[EVENT:discovery_start] No named GoPro adapter, probing %d 172.x IPs", len(all_172_ips))
        for ip in all_172_ips:
            candidate = _pick_camera_ip([ip])
            if candidate and _probe_gopro_http(candidate):
                log.info("[EVENT:discovery_found] Found GoPro at %s via ipconfig fallback", candidate)
                return candidate

    return None


def _discover_via_interface_scan() -> Optional[str]:
    """Scan network interfaces for GoPro-like 172.x.x.x subnets.

    For each interface with a 172.x.x.x address, computes the .51
    address on the same subnet and probes port 8080 for the GoPro
    HTTP API.
    """
    try:
        import psutil
        addrs = psutil.net_if_addrs()
    except ImportError:
        log.debug("psutil not available, trying socket-based scan")
        return _discover_via_socket_scan()

    candidates = []
    for iface_name, iface_addrs in addrs.items():
        for addr in iface_addrs:
            if addr.family == socket.AF_INET and GOPRO_IP_PATTERN.match(addr.address):
                parts = addr.address.split(".")
                candidate = f"{parts[0]}.{parts[1]}.{parts[2]}.51"
                candidates.append((iface_name, addr.address, candidate))

    for iface_name, host_ip, camera_ip in candidates:
        if _probe_gopro_http(camera_ip):
            log.info(
                "[EVENT:discovery_found] Found GoPro at %s on interface '%s' (host %s)",
                camera_ip, iface_name, host_ip,
            )
            return camera_ip

    return None


def _discover_via_socket_scan() -> Optional[str]:
    """Fallback IP discovery without psutil, using ipconfig parsing.

    Parses ipconfig output to find all 172.x.x.x addresses, then
    probes the .51 IP on each subnet for GoPro's HTTP API.
    """
    try:
        result = subprocess.run(
            ["ipconfig"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception:
        return None

    candidates = []
    for match in GOPRO_IP_PATTERN.finditer(result.stdout):
        ip = match.group(0)
        parts = ip.split(".")
        candidate = f"{parts[0]}.{parts[1]}.{parts[2]}.51"
        if candidate not in candidates:
            candidates.append(candidate)

    for camera_ip in candidates:
        if _probe_gopro_http(camera_ip):
            log.info("[EVENT:discovery_found] Found GoPro at %s via ipconfig scan", camera_ip)
            return camera_ip

    return None


def _discover_via_ip_range_scan(
    timeout: float = IP_RANGE_SCAN_TIMEOUT,
    max_workers: int = IP_RANGE_SCAN_MAX_WORKERS,
    verify_endpoint: bool = True,
) -> Optional[str]:
    """Brute-force scan known GoPro USB-tethering IP ranges on port 8080.

    When ipconfig parsing and interface scanning both fail, this method
    systematically probes all candidate IPs in the 172.2x.1xx.0/24 subnets
    on port 8080 for GoPro HTTP endpoints. For each /24 subnet, it probes
    multiple candidate host addresses (.51, .1, .101) to handle different
    firmware or configuration variations.

    The scan is two-phase:
      1. **Fast TCP probe**: Concurrent port-open checks on all candidates
         using short timeouts. This quickly eliminates non-responsive IPs.
      2. **HTTP verification** (optional): Hits /gopro/webcam/status on any
         IP that passed the TCP probe to confirm it's actually a GoPro,
         not some other service on port 8080.

    Args:
        timeout: Per-probe TCP connection timeout in seconds.
        max_workers: Maximum concurrent probe threads.
        verify_endpoint: If True, confirm TCP-reachable IPs are GoPro cameras
            via HTTP endpoint check. Set to False for faster scanning when
            false positives are unlikely (e.g., dedicated USB interface).

    Returns:
        Camera IP string if found, or None.
    """
    candidates = _generate_gopro_candidate_ips()

    log.info(
        "[EVENT:discovery_start] IP range scan: probing %d candidate IPs across "
        "172.{20-29}.{100-199}.0/24 subnets on port %d "
        "(timeout=%.1fs, workers=%d, verify=%s)",
        len(candidates), GOPRO_API_PORT, timeout, max_workers, verify_endpoint,
    )

    found_ip = None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ip = {
            executor.submit(_probe_gopro_http, ip, GOPRO_API_PORT, timeout): ip
            for ip in candidates
        }

        for future in as_completed(future_to_ip):
            ip = future_to_ip[future]
            try:
                if future.result():
                    # Phase 2: Verify this is actually a GoPro endpoint
                    if verify_endpoint:
                        log.debug(
                            "IP range scan: TCP open at %s, verifying GoPro endpoint...",
                            ip,
                        )
                        if not _verify_gopro_endpoint(ip, GOPRO_API_PORT, timeout=2.0):
                            log.debug(
                                "IP range scan: %s has open port %d but is not a GoPro",
                                ip, GOPRO_API_PORT,
                            )
                            continue

                    found_ip = ip
                    log.info(
                        "[EVENT:discovery_found] IP range scan found GoPro at %s",
                        ip,
                    )
                    # Cancel remaining probes (best-effort)
                    for f in future_to_ip:
                        f.cancel()
                    break
            except Exception as e:
                log.debug("IP range scan probe error for %s: %s", ip, e)

    if not found_ip:
        log.debug("[EVENT:discovery_failed] IP range scan: no GoPro found in scanned range")

    return found_ip


def _generate_gopro_candidate_ips(
    host_octets: list[int] | None = None,
) -> list[str]:
    """Generate all candidate GoPro IPs in the known USB-tethering range.

    Scans 172.2x.1xx.0/24 subnets by generating IPs for each candidate
    host octet on every /24 subnet in the range. The primary candidate
    is .51 (standard GoPro camera address), with .1 and .101 as fallbacks
    for non-standard firmware configurations.

    Args:
        host_octets: List of fourth-octet values to probe per /24 subnet.
            Defaults to GOPRO_CANDIDATE_HOSTS [51, 1, 101].

    Returns:
        List of IP strings. Primary host (.51) IPs are listed first for
        each subnet to maximize chance of early discovery.
        E.g. ['172.20.100.51', '172.20.100.1', '172.20.100.101',
              '172.20.101.51', ...].
    """
    if host_octets is None:
        host_octets = GOPRO_CANDIDATE_HOSTS

    candidates = []
    for second in GOPRO_SECOND_OCTET_RANGE:
        for third in GOPRO_THIRD_OCTET_RANGE:
            for host in host_octets:
                candidates.append(f"172.{second}.{third}.{host}")
    return candidates


# ============================================================================
# mDNS Discovery Implementation
# ============================================================================

def _build_mdns_query(service_name: str) -> bytes:
    """Build a minimal DNS-SD PTR query packet for the given service.

    Constructs a raw DNS query (RFC 1035) asking for PTR records of
    the specified mDNS service type. This avoids needing the zeroconf
    library as a dependency.

    Args:
        service_name: Fully qualified mDNS service name
                      (e.g. '_gopro-web._tcp.local.')

    Returns:
        Raw DNS query bytes ready for UDP send.
    """
    # DNS header: ID=0, flags=0 (standard query), QDCOUNT=1
    header = struct.pack("!HHHHHH", 0, 0, 1, 0, 0, 0)

    # Encode the QNAME (domain name in DNS label format)
    qname = b""
    for part in service_name.rstrip(".").split("."):
        encoded = part.encode("utf-8")
        qname += bytes([len(encoded)]) + encoded
    qname += b"\x00"  # Root label terminator

    # QTYPE=PTR (12), QCLASS=IN (1) with unicast response bit
    question = qname + struct.pack("!HH", 12, 0x8001)

    return header + question


def _parse_mdns_response(data: bytes) -> list[str]:
    """Extract IP addresses from an mDNS response packet.

    Parses all record sections of an mDNS response looking for A records
    (type 1) that contain IPv4 addresses matching GoPro's 172.x.x.x range.

    Args:
        data: Raw mDNS response bytes.

    Returns:
        List of IPv4 address strings found in the response.
    """
    if len(data) < 12:
        return []

    ips = []

    try:
        _id, _flags, qdcount, ancount, nscount, arcount = struct.unpack(
            "!HHHHHH", data[:12]
        )
    except struct.error:
        return []

    offset = 12

    # Skip question section
    for _ in range(qdcount):
        offset = _skip_dns_name(data, offset)
        if offset is None or offset + 4 > len(data):
            return []
        offset += 4  # Skip QTYPE and QCLASS

    # Parse answer + authority + additional sections for A records
    total_records = ancount + nscount + arcount
    for _ in range(total_records):
        if offset is None or offset >= len(data):
            break

        name_end = _skip_dns_name(data, offset)
        if name_end is None or name_end + 10 > len(data):
            break

        rtype, _rclass, _ttl, rdlength = struct.unpack(
            "!HHIH", data[name_end:name_end + 10]
        )
        rdata_start = name_end + 10

        if rdata_start + rdlength > len(data):
            break

        # A record (type 1) with 4 bytes = IPv4 address
        if rtype == 1 and rdlength == 4:
            ip = socket.inet_ntoa(data[rdata_start:rdata_start + 4])
            if GOPRO_IP_PATTERN.match(ip):
                ips.append(ip)

        offset = rdata_start + rdlength

    return ips


def _skip_dns_name(data: bytes, offset: int) -> Optional[int]:
    """Skip over a DNS name (handling label and pointer formats).

    Returns the offset after the name, or None on error.
    """
    if offset >= len(data):
        return None

    max_offset = len(data)
    jumps = 0

    while offset < max_offset:
        length = data[offset]

        if length == 0:
            return offset + 1

        if (length & 0xC0) == 0xC0:
            # Pointer (2 bytes)
            return offset + 2

        offset += 1 + length
        jumps += 1

        if jumps > 128:
            return None

    return None


def _discover_via_mdns(timeout: float = MDNS_DEFAULT_TIMEOUT) -> Optional[str]:
    """Discover GoPro via mDNS service discovery (_gopro-web._tcp).

    Sends mDNS queries for known GoPro service types and listens for
    responses containing A records with GoPro IP addresses. Each
    candidate IP is verified with a TCP probe to port 8080.

    This is the last resort -- it works even when USB enumeration fails
    and the IP isn't in the common subnet range.

    Args:
        timeout: How long to listen for mDNS responses (seconds).

    Returns:
        Camera IP string if found via mDNS, None otherwise.
    """
    log.info(
        "[EVENT:discovery_start] Trying mDNS discovery for %s...",
        GOPRO_MDNS_SERVICE_TYPE,
    )

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # SO_REUSEPORT not available on Windows -- ignore gracefully
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass

        sock.settimeout(timeout)

        # Bind to any interface on mDNS port (or ephemeral if port in use)
        try:
            sock.bind(("", MDNS_PORT))
        except OSError:
            sock.bind(("", 0))

        # Join multicast group
        mreq = struct.pack(
            "4s4s",
            socket.inet_aton(MDNS_ADDR),
            socket.inet_aton("0.0.0.0"),
        )
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except OSError as e:
            log.debug("Could not join mDNS multicast group: %s", e)

        # Send queries for all known GoPro service types
        for svc in GOPRO_MDNS_ALT_SERVICE_TYPES:
            query = _build_mdns_query(svc)
            sock.sendto(query, (MDNS_ADDR, MDNS_PORT))
            log.debug("mDNS query sent for %s", svc)

        # Listen for responses
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sock.settimeout(remaining)
            try:
                data, addr = sock.recvfrom(4096)
                ips = _parse_mdns_response(data)
                for ip in ips:
                    if _probe_gopro_http(ip, 8080, 2.0):
                        log.info(
                            "[EVENT:discovery_found] mDNS found GoPro at %s (from %s)",
                            ip, addr,
                        )
                        sock.close()
                        return ip
                    else:
                        log.debug("mDNS candidate %s failed HTTP probe", ip)
            except socket.timeout:
                break
            except OSError as e:
                log.debug("mDNS recv error: %s", e)
                break

        sock.close()

    except OSError as e:
        log.debug("mDNS discovery socket error: %s", e)

    log.debug("mDNS discovery: no GoPro found")
    return None


# ============================================================================
# Utility functions
# ============================================================================

def _pick_camera_ip(ips: list[str]) -> Optional[str]:
    """From a list of IPs on a GoPro interface, determine the camera's IP.

    The camera is typically at .51 on the subnet. If we have the host IP,
    we compute the .51 address.
    """
    for ip in ips:
        if ip.endswith(".51"):
            return ip

    if ips:
        parts = ips[0].split(".")
        candidate = f"{parts[0]}.{parts[1]}.{parts[2]}.51"
        return candidate

    return None


def _probe_gopro_http(ip: str, port: int = 8080, timeout: float = 2.0) -> bool:
    """Check if a GoPro HTTP API is reachable at the given IP.

    Attempts a TCP connection to port 8080 to verify the camera is
    responding. This is faster than a full HTTP request.

    Args:
        ip: IP address to probe.
        port: HTTP API port (default 8080).
        timeout: Connection timeout in seconds.

    Returns:
        True if the port is open (camera is likely there).
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except (socket.error, OSError) as e:
        log.debug("Probe %s:%d failed: %s", ip, port, e)
        return False


def _verify_gopro_endpoint(ip: str, port: int = 8080, timeout: float = 2.0) -> bool:
    """Verify that an IP is actually a GoPro by hitting its HTTP API.

    Unlike _probe_gopro_http which only checks TCP connectivity, this
    function makes a real HTTP GET request to the GoPro webcam status
    endpoint and validates the response looks like a GoPro.

    This is used as a secondary validation step after the fast TCP probe
    confirms the port is open, to avoid false positives from other HTTP
    servers on port 8080.

    Args:
        ip: IP address to verify.
        port: HTTP API port (default 8080).
        timeout: HTTP request timeout in seconds.

    Returns:
        True if the endpoint responds with valid GoPro status JSON.
    """
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError

    url = f"http://{ip}:{port}{GOPRO_STATUS_ENDPOINT}"
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                log.debug("GoPro verify %s: HTTP %d", ip, resp.status)
                return False
            body = resp.read(4096).decode("utf-8", errors="replace")

            # GoPro webcam/status returns JSON with a "status" field
            # (integer webcam state). Any valid JSON with "status" key
            # is a strong indicator this is a real GoPro.
            import json
            try:
                data = json.loads(body)
                if "status" in data:
                    log.debug(
                        "GoPro verify %s: confirmed (status=%s)", ip, data["status"]
                    )
                    return True
                # Some firmware versions use "error" field for errors
                if "error" in data:
                    log.debug(
                        "GoPro verify %s: error response (likely GoPro): %s",
                        ip, data.get("error"),
                    )
                    return True
            except (json.JSONDecodeError, ValueError):
                # Non-JSON response — probably not a GoPro
                log.debug("GoPro verify %s: non-JSON response", ip)
                return False

    except HTTPError as e:
        # HTTP errors from the GoPro are still valid (it responded!)
        # GoPro returns 500 when not in webcam mode, etc.
        log.debug("GoPro verify %s: HTTP error %d (still a GoPro)", ip, e.code)
        return True
    except (URLError, OSError, socket.timeout) as e:
        log.debug("GoPro verify %s: connection failed: %s", ip, e)
        return False

    return False


# ============================================================================
# Fallback Chain Orchestrator
# ============================================================================

def discover_gopro_ip_chain(
    probe_timeout: float = 2.0,
    mdns_timeout: float = 3.0,
    ip_scan_max_workers: int = IP_RANGE_SCAN_MAX_WORKERS,
) -> tuple[Optional[str], Optional[DiscoveryMethod]]:
    """Fallback chain orchestrator: tries discovery methods in sequence.

    Attempts to find the GoPro's IP address using three methods, returning
    as soon as one succeeds:

      1. **USB scan** (fastest, most reliable): Parse network interfaces
         (ipconfig, psutil) looking for the GoPro NCM adapter by name.

      2. **IP scan** (medium speed): Probe all IPs in the known GoPro
         172.2x.1xx.51 range on port 8080 concurrently. Works even when
         the adapter has a generic name that ipconfig parsing misses.

      3. **mDNS** (slowest, last resort): Query for _gopro-web._tcp.local.
         mDNS service announcements. Works even when the IP is outside
         the common range.

    Each method is wrapped in a try/except so a failure in one method
    never prevents the next method from being attempted.

    Args:
        probe_timeout: TCP connect timeout per IP for IP range scan.
        mdns_timeout: How long to listen for mDNS responses.
        ip_scan_max_workers: Thread pool size for IP range scan.

    Returns:
        Tuple of (camera_ip, discovery_method) on success,
        or (None, None) if all methods fail.
    """
    log.info("[EVENT:discovery_start] Starting IP discovery fallback chain...")

    # --- Method 1: USB interface scan (ipconfig + psutil) ---
    log.info("[EVENT:discovery_start] Chain step 1/3: USB interface scan")
    try:
        ip = discover_gopro_ip()
        if ip:
            log.info("[EVENT:discovery_found] Chain: USB scan found GoPro at %s", ip)
            return ip, DiscoveryMethod.USB_SCAN
    except Exception as e:
        log.warning(
            "[EVENT:discovery_failed] Chain: USB scan raised exception: %s", e
        )

    log.info("[EVENT:discovery_failed] Chain step 1/3: USB scan found nothing")

    # --- Method 2: IP range scan ---
    log.info("[EVENT:discovery_start] Chain step 2/3: IP range scan")
    try:
        ip = _discover_via_ip_range_scan(
            timeout=probe_timeout,
            max_workers=ip_scan_max_workers,
        )
        if ip:
            log.info("[EVENT:discovery_found] Chain: IP scan found GoPro at %s", ip)
            return ip, DiscoveryMethod.IP_SCAN
    except Exception as e:
        log.warning(
            "[EVENT:discovery_failed] Chain: IP scan raised exception: %s", e
        )

    log.info("[EVENT:discovery_failed] Chain step 2/3: IP scan found nothing")

    # --- Method 3: mDNS ---
    log.info("[EVENT:discovery_start] Chain step 3/3: mDNS discovery")
    try:
        ip = _discover_via_mdns(timeout=mdns_timeout)
        if ip:
            log.info("[EVENT:discovery_found] Chain: mDNS found GoPro at %s", ip)
            return ip, DiscoveryMethod.MDNS
    except Exception as e:
        log.warning(
            "[EVENT:discovery_failed] Chain: mDNS raised exception: %s", e
        )

    log.info(
        "[EVENT:discovery_failed] Chain: all 3 methods exhausted -- no GoPro IP found"
    )
    return None, None


# ============================================================================
# Discovery timeout mechanism
# ============================================================================


class DiscoveryTimeout(Exception):
    """Raised when discovery does not complete within the configured timeout.

    This is specifically for the case where a GoPro USB device is detected
    (vendor ID match on the USB bus) but the full discovery process --
    including NCM network interface initialization and IP resolution --
    does not complete within the allowed time window.

    Attributes:
        device: The partially-discovered GoProDevice (USB info only, no IP).
        elapsed: How many seconds elapsed before the timeout triggered.
        timeout: The configured timeout value in seconds.
    """

    def __init__(self, device: GoProDevice, elapsed: float, timeout: float):
        self.device = device
        self.elapsed = elapsed
        self.timeout = timeout
        super().__init__(
            f"Discovery timed out after {elapsed:.1f}s (limit {timeout:.1f}s): "
            f"USB device {device.usb_id_str} found but IP not resolved"
        )


def timed_full_discovery(
    overall_timeout: float = 30.0,
    probe_timeout: float = 5.0,
    poll_interval: float = 2.0,
) -> Optional[GoProDevice]:
    """Full discovery with an overall timeout for the USB-detected-but-no-IP case.

    This wraps full_discovery() with a deadline-based polling loop. It handles
    the scenario where the GoPro is physically connected via USB but the NCM
    network adapter hasn't initialized yet (common after USB plug-in; the
    adapter can take 5-15 seconds to come up).

    Flow:
      1. Run full_discovery() once.
      2. If no USB device found at all -> return None immediately (no timeout).
      3. If USB device found WITH IP -> return the device immediately (success).
      4. If USB device found WITHOUT IP -> poll discover_gopro_ip() until
         the overall_timeout expires, then raise DiscoveryTimeout.

    Args:
        overall_timeout: Max seconds to wait for complete discovery (USB + IP)
            when a USB device is detected. Configurable via
            config.discovery_overall_timeout. Default 30s.
        probe_timeout: Timeout for individual HTTP probes during IP discovery.
            Passed through to full_discovery(). Default 5s.
        poll_interval: Seconds between IP resolution attempts during the
            waiting phase. Default 2s.

    Returns:
        A fully-populated GoProDevice (with camera_ip set), or None if no
        USB device was found at all.

    Raises:
        DiscoveryTimeout: If a USB device was detected but IP resolution
            did not complete within overall_timeout seconds. The exception
            carries the partial GoProDevice so callers can inform the user.
    """
    start_time = time.monotonic()
    deadline = start_time + overall_timeout

    log.info(
        "[EVENT:discovery_start] Starting timed discovery "
        "(overall_timeout=%.1fs, poll_interval=%.1fs)",
        overall_timeout, poll_interval,
    )

    # Step 1: Initial discovery attempt
    device = full_discovery(timeout=probe_timeout)

    if device is None:
        # No USB device at all -- nothing to wait for
        log.info("[EVENT:discovery_failed] No GoPro USB device found on bus")
        return None

    if device.camera_ip is not None:
        # Full success on first try
        elapsed = time.monotonic() - start_time
        log.info(
            "[EVENT:discovery_found] Timed discovery complete in %.1fs: %s",
            elapsed, device,
        )
        return device

    # Step 2: USB device found but no IP -- poll until timeout
    log.info(
        "[EVENT:discovery_start] USB device %s found but no IP. "
        "Polling for network interface (timeout %.1fs)...",
        device.usb_id_str, overall_timeout,
    )

    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        remaining = deadline - time.monotonic()

        if remaining <= 0:
            break

        # Wait before next poll (but don't overshoot the deadline)
        wait_time = min(poll_interval, remaining)
        time.sleep(wait_time)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        log.debug(
            "[EVENT:discovery_start] IP resolution attempt %d (%.1fs remaining)...",
            attempt, remaining,
        )

        camera_ip = discover_gopro_ip()
        if camera_ip:
            device.camera_ip = camera_ip
            elapsed = time.monotonic() - start_time
            log.info(
                "[EVENT:discovery_found] Timed discovery complete in %.1fs "
                "(after %d IP poll(s)): %s",
                elapsed, attempt, device,
            )
            return device

    # Step 3: Timeout -- USB found but IP never resolved
    elapsed = time.monotonic() - start_time
    log.warning(
        "[EVENT:discovery_timeout] Discovery timed out after %.1fs: "
        "USB device %s found but network interface never came up "
        "(%d IP resolution attempt(s))",
        elapsed, device.usb_id_str, attempt,
    )
    raise DiscoveryTimeout(device=device, elapsed=elapsed, timeout=overall_timeout)


# ============================================================================
# Main entry point
# ============================================================================

def full_discovery(timeout: float = 5.0) -> Optional[GoProDevice]:
    """Complete GoPro discovery: USB enumeration + fallback IP resolution chain.

    This is the main entry point for discovering a GoPro on app launch.
    It first checks if a GoPro is physically connected via USB (by vendor ID),
    then resolves the camera's network IP using the fallback chain:
      1. USB interface scan (ipconfig/psutil)
      2. IP range scan (probe common subnets)
      3. mDNS service discovery

    If USB enumeration itself fails, the chain is still attempted --
    the camera might be connected but USB enumeration may have failed
    (e.g., WMI/PowerShell not available).

    Args:
        timeout: Max time for HTTP probe during IP discovery.

    Returns:
        A fully-populated GoProDevice if found, or None.
    """
    log.info("[EVENT:discovery_start] Starting GoPro discovery...")

    # Step 1: Check USB bus for GoPro hardware
    usb_devices = enumerate_usb_gopro_devices()

    if not usb_devices:
        log.info("[EVENT:discovery_failed] No GoPro USB devices found on bus")

        # Even without USB confirmation, try the IP chain -- the camera
        # might be connected but USB enumeration may have failed
        camera_ip, method = discover_gopro_ip_chain(probe_timeout=timeout)
        if camera_ip:
            log.info(
                "[EVENT:discovery_found] GoPro found via %s at %s "
                "(USB enumeration had failed)",
                method.value if method else "unknown",
                camera_ip,
            )
            device = GoProDevice(
                vendor_id=GOPRO_VENDOR_ID,
                product_id=0,
                description="GoPro (discovered via network)",
                camera_ip=camera_ip,
                discovery_method=method,
            )
            return device
        return None

    log.info(
        "[EVENT:discovery_start] Found %d GoPro USB device(s): %s",
        len(usb_devices),
        [d.usb_id_str for d in usb_devices],
    )

    # Step 2: Find the camera's IP using the fallback chain
    camera_ip, method = discover_gopro_ip_chain(probe_timeout=timeout)

    if camera_ip:
        device = usb_devices[0]
        device.camera_ip = camera_ip
        device.discovery_method = method
        log.info(
            "[EVENT:discovery_found] GoPro discovery complete via %s: %s",
            method.value if method else "unknown",
            device,
        )
        return device
    else:
        # USB device found but network not ready yet -- still return the device
        # so the caller knows the camera is physically connected
        device = usb_devices[0]
        log.warning(
            "[EVENT:discovery_failed] GoPro USB device found (%s) but all IP "
            "discovery methods failed. Camera may still be initializing.",
            device.usb_id_str,
        )
        return device


# --- Module self-test ---
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.DEBUG, format="%(levelname)s: %(message)s")

    print("=== GoPro USB Device Enumeration ===\n")

    print("Scanning USB bus for GoPro devices (VID 0A70)...")
    devices = enumerate_usb_gopro_devices()

    if devices:
        for dev in devices:
            print(f"  Found: {dev}")
            if dev.product_id in GOPRO_KNOWN_PIDS:
                print(f"    -> Known GoPro product ID: 0x{dev.product_id:04X}")
            else:
                print(f"    -> Unknown product ID: 0x{dev.product_id:04X} (may be a new model/mode)")
    else:
        print("  No GoPro devices found on USB bus.")

    print("\n--- Fallback chain IP discovery ---")
    found_ip, found_method = discover_gopro_ip_chain()
    if found_ip:
        print(f"  Camera IP: {found_ip} (via {found_method.value})")
        print(f"  API endpoint: http://{found_ip}:8080/gopro/webcam/status")
    else:
        print("  Could not resolve GoPro network IP.")

    print("\n--- Full discovery ---")
    device = full_discovery()
    if device:
        print(f"  Result: {device}")
    else:
        print("  No GoPro discovered.")
