"""
ble/scanner.py — BLE scanner for discovering GoPro cameras.

Uses bleak (cross-platform BLE library) to scan for GoPro cameras
advertising the GoPro BLE service UUID.

The scanner:
  1. Scans for BLE advertisements matching the GoPro service UUID
  2. Filters by name prefix ("GoPro " or "GP-")
  3. Returns DiscoveredGoPro dataclass instances with device info
  4. Supports both one-shot scan and continuous scanning with callback

All methods are async — runs on the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from gomaxwebcam.ble.uuids import (
    GOPRO_SERVICE_UUID,
    GOPRO_NAME_PREFIXES,
)

log = logging.getLogger("gomaxwebcam.ble.scanner")


@dataclass(frozen=True)
class DiscoveredGoPro:
    """A GoPro camera discovered via BLE scan.

    Attributes:
        address: BLE MAC address (or UUID on macOS).
        name: Advertised BLE name (e.g. "GoPro 1234").
        rssi: Signal strength in dBm.
        serial_suffix: Last 4 chars of camera name (used as serial hint).
        model_hint: Parsed model string from advertisement data, if available.
    """
    address: str
    name: str
    rssi: int = -100
    serial_suffix: str = ""
    model_hint: str = ""


class BLEScanner:
    """Scans for GoPro cameras via BLE advertisements.

    Args:
        scan_timeout: Default scan duration in seconds.
        adapter: BLE adapter identifier (None = system default).
    """

    def __init__(
        self,
        scan_timeout: float = 10.0,
        adapter: str | None = None,
    ):
        self._scan_timeout = scan_timeout
        self._adapter = adapter
        self._scanner = None  # BleakScanner instance, created lazily

    async def scan_once(
        self,
        timeout: float | None = None,
    ) -> list[DiscoveredGoPro]:
        """Perform a single BLE scan and return all discovered GoPros.

        Args:
            timeout: Scan duration in seconds (default: self._scan_timeout).

        Returns:
            List of DiscoveredGoPro instances found during the scan.
            Empty list if no GoPros found or BLE is unavailable.
        """
        timeout = timeout or self._scan_timeout
        log.info("Starting BLE scan for GoPro cameras (timeout=%.1fs)", timeout)

        try:
            from bleak import BleakScanner

            scanner = BleakScanner(
                service_uuids=[GOPRO_SERVICE_UUID],
                adapter=self._adapter,
            )

            devices = await scanner.discover(timeout=timeout)
            gopros = []

            for device in devices:
                if self._is_gopro(device):
                    gopro = self._make_discovered(device)
                    gopros.append(gopro)
                    log.info(
                        "Found GoPro: %s (%s, RSSI=%ddBm)",
                        gopro.name, gopro.address, gopro.rssi,
                    )

            log.info("BLE scan complete: %d GoPro(s) found", len(gopros))
            return gopros

        except ImportError:
            log.error("bleak not installed — BLE scanning unavailable")
            return []
        except Exception as e:
            log.error("BLE scan failed: %s", e)
            return []

    async def scan_continuous(
        self,
        callback: Callable[[DiscoveredGoPro], None],
        duration: float | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Continuously scan for GoPros, calling back on each discovery.

        Runs until duration expires or stop_event is set.

        Args:
            callback: Called with each DiscoveredGoPro found.
            duration: Max scan duration in seconds (None = until stop_event).
            stop_event: Set this event to stop scanning.
        """
        if stop_event is None:
            stop_event = asyncio.Event()

        duration = duration or self._scan_timeout * 3
        log.info("Starting continuous BLE scan (duration=%.1fs)", duration)

        seen: set[str] = set()

        try:
            from bleak import BleakScanner

            def _detection_callback(device, advertisement_data):
                if device.address in seen:
                    return
                if self._is_gopro(device):
                    seen.add(device.address)
                    gopro = self._make_discovered(device, advertisement_data)
                    log.info(
                        "Discovered GoPro: %s (%s)",
                        gopro.name, gopro.address,
                    )
                    callback(gopro)

            scanner = BleakScanner(
                detection_callback=_detection_callback,
                service_uuids=[GOPRO_SERVICE_UUID],
                adapter=self._adapter,
            )

            await scanner.start()
            self._scanner = scanner

            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=duration,
                )
            except asyncio.TimeoutError:
                pass  # Normal — scan duration elapsed
            finally:
                await scanner.stop()
                self._scanner = None

            log.info("Continuous scan stopped, found %d GoPro(s)", len(seen))

        except ImportError:
            log.error("bleak not installed — BLE scanning unavailable")
        except Exception as e:
            log.error("Continuous BLE scan failed: %s", e)

    async def stop(self) -> None:
        """Stop any ongoing continuous scan."""
        if self._scanner is not None:
            try:
                await self._scanner.stop()
            except Exception as e:
                log.warning("Error stopping scanner: %s", e)
            self._scanner = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_gopro(device) -> bool:
        """Check if a BLE device is a GoPro camera.

        Matches on:
          1. Service UUID matches GOPRO_SERVICE_UUID, OR
          2. Device name starts with known GoPro prefix
        """
        name = device.name or ""
        if any(name.startswith(prefix) for prefix in GOPRO_NAME_PREFIXES):
            return True

        # Check metadata for service UUIDs if available
        if hasattr(device, "metadata"):
            uuids = device.metadata.get("uuids", [])
            if GOPRO_SERVICE_UUID in uuids:
                return True

        return False

    @staticmethod
    def _make_discovered(device, adv_data=None) -> DiscoveredGoPro:
        """Create a DiscoveredGoPro from a bleak device."""
        name = device.name or "Unknown GoPro"
        rssi = -100

        if adv_data is not None:
            rssi = getattr(adv_data, "rssi", -100) or -100
        elif hasattr(device, "rssi"):
            rssi = device.rssi or -100

        # Extract serial suffix from name (last 4 chars after "GoPro " prefix)
        serial_suffix = ""
        if name.startswith("GoPro "):
            serial_suffix = name[6:].strip()
        elif name.startswith("GP-"):
            serial_suffix = name[3:].strip()

        return DiscoveredGoPro(
            address=str(device.address),
            name=name,
            rssi=rssi,
            serial_suffix=serial_suffix,
        )

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def _inject_scanner(self, mock_scanner) -> None:
        """For testing: inject a mock BleakScanner."""
        self._scanner = mock_scanner
