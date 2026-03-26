"""
provisioning.py — BLE provisioning service using open-gopro SDK.

Wraps open-gopro's WirelessGoPro to handle the complete COHN provisioning
lifecycle:

  1. BLE scan → discover GoPro cameras
  2. BLE connect → pair with selected camera
  3. WiFi connect → connect camera to home network via BLE command
  4. COHN provision → create TLS certificate, retrieve credentials
  5. Credential return → ip_address, username, password, certificate

open-gopro owns the full BLE stack (bleak adapter, GATT client, protobuf
commands) and COHN credential persistence (cohn_db.json via TinyDB).
This module never duplicates that functionality.

Architecture:
    BLEProvisioningService
      └── WirelessGoPro (open-gopro SDK)
            ├── BLE scan + connect (bleak)
            ├── BLE commands (scan_wifi, connect_wifi, cohn_*)
            ├── CohnFeature (provision, credentials, cohn_db)
            └── Credential persistence (TinyDB → cohn_db.json)

Thread model:
    All methods are async — run on the asyncio event loop.
    Status callbacks fire synchronously on the same loop.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from returns.pipeline import is_successful as _is_successful
except ImportError:  # pragma: no cover – returns not installed
    def _is_successful(result: Any) -> bool:  # type: ignore[misc]
        """Fallback when returns library is not installed."""
        return not isinstance(result, Exception)

log = logging.getLogger("gomaxwebcam.provisioning")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class COHNCredentials:
    """Credentials returned from COHN provisioning.

    Contains everything needed to connect to the camera via COHN HTTPS.

    Attributes:
        ip_address: Camera IP on the home network.
        username: COHN HTTP basic auth username.
        password: COHN HTTP basic auth password.
        certificate: PEM-encoded TLS certificate for the camera.
        camera_serial: Camera serial suffix (last 4 digits).
        ssid: WiFi network the camera is connected to.
        provisioned: Whether provisioning completed successfully.
    """
    ip_address: str = ""
    username: str = ""
    password: str = ""
    certificate: str = ""
    camera_serial: str = ""
    ssid: str = ""
    provisioned: bool = False

    @property
    def is_complete(self) -> bool:
        """All credential fields are populated."""
        return bool(
            self.ip_address
            and self.username
            and self.password
            and self.certificate
        )


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


class ProvisionPhase(Enum):
    """Phases of the BLE provisioning process."""
    IDLE = auto()
    BLE_SCANNING = auto()
    BLE_CONNECTING = auto()
    WIFI_CONNECTING = auto()
    COHN_PROVISIONING = auto()
    COMPLETE = auto()
    FAILED = auto()


@dataclass
class ProvisionProgress:
    """Progress update from the provisioning service.

    Published via callbacks for dashboard consumption.
    """
    phase: ProvisionPhase = ProvisionPhase.IDLE
    progress_pct: int = 0
    message: str = ""
    error: str = ""
    camera_serial: str = ""

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.name,
            "progress_pct": self.progress_pct,
            "message": self.message,
            "error": self.error,
            "camera_serial": self.camera_serial,
        }


ProgressCallback = Callable[[ProvisionProgress], None]


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_DEFAULT_COHN_DB = Path("cohn_db.json")


# ---------------------------------------------------------------------------
# BLEProvisioningService
# ---------------------------------------------------------------------------


class BLEProvisioningService:
    """BLE provisioning service using open-gopro SDK.

    Provides the complete BLE → COHN provisioning workflow:
      1. Scan for GoPro cameras
      2. Connect via BLE and join camera to home WiFi
      3. Provision COHN (TLS certificate + credentials)
      4. Return credentials for COHNTransport to use

    Credentials are persisted by open-gopro in cohn_db.json (TinyDB).
    No keyring duplication — open-gopro owns credential storage.

    Args:
        cohn_db_path: Path to the COHN credential database.
            Defaults to "cohn_db.json" in the working directory.

    Usage:
        service = BLEProvisioningService()
        creds = await service.provision(
            target="1234",
            wifi_ssid="HomeNetwork",
            wifi_password="secret",
            on_progress=lambda p: print(p.message),
        )
        if creds and creds.provisioned:
            print(f"Camera at {creds.ip_address}")

        await service.close()
    """

    def __init__(
        self,
        cohn_db_path: Path = _DEFAULT_COHN_DB,
    ) -> None:
        self._cohn_db_path = cohn_db_path
        self._gopro: Any = None  # WirelessGoPro instance during provisioning
        self._cancelled = False

        # For test injection
        self._gopro_factory: Any = None

    # ------------------------------------------------------------------
    # Public API: Provision
    # ------------------------------------------------------------------

    async def provision(
        self,
        target: str | None = None,
        wifi_ssid: str = "",
        wifi_password: str = "",
        on_progress: ProgressCallback | None = None,
        ble_timeout: int = 15,
        provision_timeout: int = 60,
        ble_address: str | None = None,
    ) -> COHNCredentials | None:
        """Run full BLE → COHN provisioning.

        Connects to the GoPro via BLE, connects camera to WiFi,
        provisions COHN, and returns credentials.

        Args:
            target: Camera serial suffix (last 4 digits). None = first found.
            wifi_ssid: Home WiFi SSID to connect the camera to.
            wifi_password: Home WiFi password.
            on_progress: Callback for progress updates.
            ble_timeout: BLE connection timeout in seconds.
            provision_timeout: COHN provisioning timeout in seconds.
            ble_address: BLE MAC address (e.g. "F7:23:0C:86:42:F5") for
                Windows scan fallback when device name is missing.

        Returns:
            COHNCredentials on success, None on failure.
        """
        self._cancelled = False
        self._ble_address = ble_address
        notify = on_progress or (lambda p: None)

        try:
            # Phase 1: Create WirelessGoPro and open BLE connection
            notify(ProvisionProgress(
                phase=ProvisionPhase.BLE_SCANNING,
                progress_pct=5,
                message="Scanning for GoPro cameras via Bluetooth...",
                camera_serial=target or "",
            ))

            gopro = await self._create_and_open_gopro(
                target=target,
                ble_timeout=ble_timeout,
                notify=notify,
            )
            if gopro is None:
                return None

            self._gopro = gopro
            camera_serial = self._get_identifier(gopro) or target or ""

            if self._cancelled:
                await self._close_gopro()
                notify(ProvisionProgress(
                    phase=ProvisionPhase.FAILED,
                    message="Provisioning cancelled",
                    camera_serial=camera_serial,
                ))
                return None

            # Phase 2: Connect camera to WiFi (if credentials provided)
            if wifi_ssid and wifi_password:
                notify(ProvisionProgress(
                    phase=ProvisionPhase.WIFI_CONNECTING,
                    progress_pct=30,
                    message=f"Connecting camera to WiFi '{wifi_ssid}'...",
                    camera_serial=camera_serial,
                ))

                wifi_ok = await self._connect_camera_to_wifi(
                    gopro, wifi_ssid, wifi_password,
                )
                if not wifi_ok:
                    notify(ProvisionProgress(
                        phase=ProvisionPhase.FAILED,
                        progress_pct=30,
                        message=f"Failed to connect camera to WiFi '{wifi_ssid}'",
                        error=f"WiFi connection failed for SSID '{wifi_ssid}'",
                        camera_serial=camera_serial,
                    ))
                    await self._close_gopro()
                    return None

            if self._cancelled:
                await self._close_gopro()
                return None

            # Phase 3: Provision COHN
            notify(ProvisionProgress(
                phase=ProvisionPhase.COHN_PROVISIONING,
                progress_pct=50,
                message="Provisioning COHN certificate...",
                camera_serial=camera_serial,
            ))

            credentials = await self._provision_cohn(
                gopro, provision_timeout, camera_serial, wifi_ssid,
            )

            if credentials is None or not credentials.provisioned:
                notify(ProvisionProgress(
                    phase=ProvisionPhase.FAILED,
                    progress_pct=50,
                    message="COHN provisioning failed",
                    error="Failed to provision COHN on camera",
                    camera_serial=camera_serial,
                ))
                await self._close_gopro()
                return None

            # Persist credentials to local cache so they survive app restarts.
            # open-gopro may auto-persist to its own cohn_db, but we explicitly
            # cache here as a safety net using the same DB path.
            self._persist_credentials(credentials)

            # Success
            notify(ProvisionProgress(
                phase=ProvisionPhase.COMPLETE,
                progress_pct=100,
                message=f"COHN provisioned! Camera at {credentials.ip_address}",
                camera_serial=camera_serial,
            ))

            log.info(
                "COHN provisioning complete: ip=%s, user=%s, serial=%s",
                credentials.ip_address, credentials.username, camera_serial,
            )

            return credentials

        except asyncio.CancelledError:
            notify(ProvisionProgress(
                phase=ProvisionPhase.FAILED,
                message="Provisioning cancelled",
            ))
            return None
        except Exception as e:
            log.error("Provisioning failed: %s", e, exc_info=True)
            notify(ProvisionProgress(
                phase=ProvisionPhase.FAILED,
                message=f"Provisioning failed: {e}",
                error=str(e),
            ))
            return None
        finally:
            await self._close_gopro()

    # ------------------------------------------------------------------
    # Public API: Stored credentials
    # ------------------------------------------------------------------

    def get_stored_credentials(self, camera_serial: str) -> COHNCredentials | None:
        """Retrieve stored COHN credentials from open-gopro's cohn_db.

        Reads from the TinyDB database that open-gopro manages.
        Does NOT connect to the camera — purely a local database lookup.

        Args:
            camera_serial: Camera serial suffix (last 4 digits).

        Returns:
            COHNCredentials if found, None otherwise.
        """
        try:
            from tinydb import TinyDB
            from open_gopro.database.cohn_db import CohnDb

            db = TinyDB(str(self._cohn_db_path), indent=4)
            cohn_db = CohnDb(db)
            info = cohn_db.search_credentials(camera_serial)
            db.close()

            if info and info.is_complete:
                return COHNCredentials(
                    ip_address=info.ip_address,
                    username=info.username,
                    password=info.password,
                    certificate=info.certificate,
                    camera_serial=camera_serial,
                    provisioned=True,
                )
            return None

        except Exception as e:
            log.warning("Failed to read stored credentials: %s", e)
            return None

    # ------------------------------------------------------------------
    # Public API: Cancel / Close
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request cancellation of any active provisioning."""
        self._cancelled = True
        log.info("BLE provisioning cancellation requested")

    async def close(self) -> None:
        """Clean up any open BLE connections."""
        await self._close_gopro()

    # ------------------------------------------------------------------
    # Internal: Create and open WirelessGoPro
    # ------------------------------------------------------------------

    async def _create_and_open_gopro(
        self,
        target: str | None,
        ble_timeout: int,
        notify: ProgressCallback,
    ) -> Any:
        """Create a WirelessGoPro and open BLE + COHN connection.

        Returns the gopro instance on success, None on failure.
        """
        try:
            gopro = self._make_gopro(target)

            notify(ProvisionProgress(
                phase=ProvisionPhase.BLE_CONNECTING,
                progress_pct=15,
                message=f"Connecting to GoPro via Bluetooth"
                        + (f" (target: {target})..." if target else "..."),
                camera_serial=target or "",
            ))

            await gopro.open(timeout=ble_timeout, retries=3)

            log.info("BLE connection established")
            notify(ProvisionProgress(
                phase=ProvisionPhase.BLE_CONNECTING,
                progress_pct=25,
                message="Bluetooth connected",
                camera_serial=self._get_identifier(gopro) or target or "",
            ))

            return gopro

        except ImportError as e:
            log.error("open-gopro or bleak not installed: %s", e)
            notify(ProvisionProgress(
                phase=ProvisionPhase.FAILED,
                message="open-gopro SDK not installed",
                error=str(e),
            ))
            return None
        except Exception as e:
            log.error("BLE connection failed: %s", e, exc_info=True)
            notify(ProvisionProgress(
                phase=ProvisionPhase.FAILED,
                progress_pct=15,
                message=f"Bluetooth connection failed: {e}",
                error=str(e),
                camera_serial=target or "",
            ))
            return None

    def _make_gopro(self, target: str | None) -> Any:
        """Create a WirelessGoPro instance.

        Uses the factory override if set (for testing), otherwise
        creates a real WirelessGoPro with BLE-only interface for provisioning.

        The target should be the serial suffix (last 4 digits from the
        device name, e.g. "7212" from "GoPro 7212"), NOT a BLE MAC address.
        The SDK uses target as a regex matched against device NAMES.

        We use BLE-only (not COHN) to prevent the SDK from terminating
        the PC's WiFi connection during provisioning.
        """
        if self._gopro_factory is not None:
            return self._gopro_factory(target)

        from open_gopro import WirelessGoPro

        # Convert BLE address to serial suffix if needed.
        # BLE addresses look like "F7:23:0C:86:42:F5" (contains colons).
        # Serial suffixes look like "7212" (4 digits, no colons).
        if target and ":" in target:
            log.warning(
                "Target '%s' looks like a BLE address, not a serial suffix. "
                "The SDK matches target against device names. "
                "Passing None to scan for any GoPro.",
                target,
            )
            target = None

        gopro = WirelessGoPro(
            target=target,
            cohn_db=self._cohn_db_path,
            interfaces={
                WirelessGoPro.Interface.BLE,  # BLE only — prevents WiFi disconnect
            },
        )

        # Workaround: On Windows, GoPro BLE advertisements often don't include
        # the device name (shows as "" or "."), causing the SDK's scan_callback
        # to never match on target regex. Monkey-patch the BLE controller's scan
        # to also check the device address against a known BLE address.
        if self._ble_address and ":" in self._ble_address:
            self._patch_scan_for_address(gopro, self._ble_address)

        return gopro

    def _patch_scan_for_address(self, gopro: Any, ble_address: str) -> None:
        """Monkey-patch the SDK's BLE scan to match by address when name is empty.

        On Windows, GoPro BLE advertisements frequently arrive without the
        device name. The SDK's scan_callback only checks token.match(name),
        which fails when name is empty. This patch adds a fallback that
        also matches by the known BLE address.
        """
        try:
            controller = gopro._ble._controller  # BleakWrapperController
            original_scan = controller.scan

            async def patched_scan(token, timeout=5, service_uuids=None):
                """Scan with address fallback for Windows name-missing issue."""
                import re
                from bleak import BleakScanner, BleakClient
                from bleak.backends.device import BLEDevice as BleakDevice
                from bleak.backends.scanner import AdvertisementData

                stop_event = asyncio.Event()
                devices = {}
                target_addr = ble_address.upper()

                log.info(
                    "Patched scan: looking for %s (name) or %s (address)",
                    token.pattern, target_addr,
                )

                def scan_callback(device: BleakDevice, adv_data: AdvertisementData):
                    name = adv_data.local_name or device.name or ""
                    addr = (device.address or "").upper()

                    if name and name not in devices:
                        devices[name] = device

                    # Match by name (SDK default) OR by address (our fallback)
                    if token.match(name) or addr == target_addr:
                        if not stop_event.is_set():
                            log.info("Matched device: %s (%s)", name or "(no name)", addr)
                            devices[addr] = device
                            stop_event.set()

                uuids = [] if service_uuids is None else [str(u) for u in service_uuids]
                async with BleakScanner(
                    timeout=timeout,
                    detection_callback=scan_callback,
                    service_uuids=uuids if uuids else None,
                ):
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout)
                    except asyncio.TimeoutError:
                        from open_gopro.domain.exceptions import FailedToFindDevice
                        raise FailedToFindDevice

                # Return the matched device
                matched = [d for d in devices.values()
                           if (d.address or "").upper() == target_addr]
                if not matched:
                    matched = [d for n, d in devices.items() if token.match(n)]
                if not matched:
                    from open_gopro.domain.exceptions import FailedToFindDevice
                    raise FailedToFindDevice

                log.info("Patched scan found %d device(s)", len(matched))
                return matched[0]

            controller.scan = patched_scan
            log.debug("BLE scan patched for address-based matching: %s", ble_address)
        except AttributeError:
            log.warning("Could not patch BLE scan — SDK internals may have changed")

    # ------------------------------------------------------------------
    # Internal: WiFi connection
    # ------------------------------------------------------------------

    async def _connect_camera_to_wifi(
        self,
        gopro: Any,
        ssid: str,
        password: str,
    ) -> bool:
        """Connect the camera to a WiFi network via BLE commands.

        Uses open-gopro's BLE commands to:
          1. Scan for available WiFi networks
          2. Connect camera to the specified network

        Returns True on success, False on failure.
        """
        try:
            # Scan for WiFi networks on the camera
            log.info("Requesting camera WiFi scan...")
            scan_result = await gopro.ble_command.scan_wifi_networks()
            if not scan_result.ok:
                log.warning("WiFi scan command failed, trying direct connect")

            # Wait a moment for scan to complete
            await asyncio.sleep(2)

            # Connect camera to the specified WiFi network
            log.info("Connecting camera to WiFi '%s'...", ssid)
            connect_result = await gopro.ble_command.request_wifi_connect_new(
                ssid=ssid, password=password,
            )

            if connect_result.ok:
                log.info("Camera connected to WiFi '%s'", ssid)
                return True
            else:
                log.warning(
                    "WiFi connect command returned non-OK: %s",
                    connect_result,
                )
                # Some cameras may still connect despite non-OK response
                # Wait and check COHN status
                await asyncio.sleep(3)
                return True

        except Exception as e:
            log.error("WiFi connection failed: %s", e, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Internal: COHN provisioning
    # ------------------------------------------------------------------

    async def _provision_cohn(
        self,
        gopro: Any,
        timeout: int,
        camera_serial: str,
        wifi_ssid: str,
    ) -> COHNCredentials | None:
        """Provision COHN on the camera using open-gopro's CohnFeature.

        Uses gopro.cohn.configure() which:
          1. Checks if COHN is already provisioned
          2. Clears existing certificate if needed
          3. Creates new TLS certificate
          4. Waits for camera to connect to network
          5. Stores credentials in cohn_db.json

        Returns COHNCredentials on success, None on failure.
        """
        try:
            result = await gopro.cohn.configure(
                force_reprovision=True,
                timeout=timeout,
            )

            if _is_successful(result):
                cohn_info = result.unwrap()
                log.info(
                    "COHN configured: ip=%s, user=%s",
                    cohn_info.ip_address, cohn_info.username,
                )
                return COHNCredentials(
                    ip_address=cohn_info.ip_address,
                    username=cohn_info.username,
                    password=cohn_info.password,
                    certificate=cohn_info.certificate,
                    camera_serial=camera_serial,
                    ssid=wifi_ssid,
                    provisioned=True,
                )
            else:
                error = result.failure()
                log.error("COHN provisioning failed: %s", error)
                return None

        except Exception as e:
            log.error("COHN provisioning error: %s", e, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Internal: Credential persistence
    # ------------------------------------------------------------------

    def _persist_credentials(self, credentials: COHNCredentials) -> bool:
        """Persist COHN credentials to the local cache file.

        Uses CohnCredentialStore to write to the same TinyDB file that
        open-gopro uses. This ensures credentials survive app restarts
        even if open-gopro's auto-persistence didn't trigger.

        Args:
            credentials: The provisioned COHN credentials to cache.

        Returns:
            True if credentials were persisted, False on error.
        """
        if not credentials.is_complete:
            log.warning(
                "Skipping credential persistence: incomplete credentials "
                "(ip=%s, user=%s, serial=%s)",
                credentials.ip_address, credentials.username,
                credentials.camera_serial,
            )
            return False

        try:
            from gomaxwebcam.transport.cohn_persistence import (
                CohnCredentialStore,
                StoredCOHNCredentials,
            )

            store = CohnCredentialStore(db_path=self._cohn_db_path)
            stored = StoredCOHNCredentials(
                ip_address=credentials.ip_address,
                username=credentials.username,
                password=credentials.password,
                certificate=credentials.certificate,
                camera_serial=credentials.camera_serial,
            )
            ok = store.store_credentials(stored)
            if ok:
                log.info(
                    "Credentials cached to %s for serial %s",
                    self._cohn_db_path, credentials.camera_serial,
                )
            else:
                log.warning(
                    "CohnCredentialStore.store_credentials returned False "
                    "for serial %s",
                    credentials.camera_serial,
                )
            return ok

        except Exception as e:
            log.warning(
                "Failed to persist credentials for serial %s: %s",
                credentials.camera_serial, e,
            )
            return False

    # ------------------------------------------------------------------
    # Internal: Cleanup
    # ------------------------------------------------------------------

    async def _close_gopro(self) -> None:
        """Close the WirelessGoPro instance if open."""
        if self._gopro is not None:
            try:
                await self._gopro.close()
            except Exception as e:
                log.debug("Error closing GoPro: %s", e)
            self._gopro = None

    @staticmethod
    def _get_identifier(gopro: Any) -> str:
        """Safely get the camera identifier."""
        try:
            return gopro.identifier
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def _inject_gopro_factory(self, factory: Callable) -> None:
        """For testing: inject a factory that creates mock WirelessGoPro."""
        self._gopro_factory = factory

    def _inject_gopro(self, mock_gopro: Any) -> None:
        """For testing: inject a mock WirelessGoPro instance directly."""
        self._gopro = mock_gopro
