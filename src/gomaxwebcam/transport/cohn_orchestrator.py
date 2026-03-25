"""
transport/cohn_orchestrator.py — Integration orchestrator for BLE provisioning → COHN transport.

Sequences the full COHN setup using open-gopro SDK:
  1. BLE scan to discover the GoPro camera (via open-gopro / bleak)
  2. BLE connect to the camera (via open-gopro WirelessGoPro)
  3. WiFi connect — camera joins home network (via BLE command)
  4. COHN provisioning — TLS certificate, credentials (via open-gopro CohnFeature)
  5. Retrieve COHN credentials (IP, username, password, certificate)
  6. Hand off to COHNTransport for camera streaming

The hand-rolled BLE stack (ble/scanner.py, ble/gatt_client.py, ble/cohn.py)
is dead code — open-gopro owns the entire BLE + COHN lifecycle.

Provides status callbacks at every step so the dashboard can show
real-time provisioning progress via SSE.

Error handling:
  - Each step has a timeout and retry logic
  - Failures at any step produce a clear error message for the dashboard
  - Partial provisioning state is preserved for retry
  - Cancellation is supported at any point

Architecture:
    COHNOrchestrator
      ├── BLEProvisioningService (open-gopro wrapper)
      │     └── WirelessGoPro (BLE scan + connect + COHN provision)
      └── COHNTransport (HTTPS streaming) ← handoff target

Thread model:
    All methods run on the asyncio event loop.
    Status callbacks are fired synchronously on the same loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

log = logging.getLogger("gomaxwebcam.transport.cohn_orchestrator")


# ---------------------------------------------------------------------------
# Orchestration types
# ---------------------------------------------------------------------------


class OrchestratorPhase(Enum):
    """High-level phases of the COHN orchestration."""
    IDLE = auto()
    BLE_SCANNING = auto()
    BLE_CONNECTING = auto()
    PROVISIONING = auto()
    COHN_DISCOVERING = auto()
    COHN_CONNECTING = auto()
    COHN_STREAMING = auto()
    COMPLETE = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class OrchestratorStatus:
    """Full status of the COHN orchestration process.

    Published via callbacks for dashboard SSE consumption.
    """
    phase: OrchestratorPhase = OrchestratorPhase.IDLE
    progress_pct: int = 0           # 0-100 across all phases
    message: str = ""               # Human-readable status
    error: str = ""                 # Error detail if phase == FAILED
    camera_name: str = ""           # BLE name of the camera
    camera_address: str = ""        # BLE address
    wifi_ssid: str = ""             # Target WiFi SSID
    camera_ip: str = ""             # Camera IP after provisioning
    transport_state: str = ""       # COHNTransport state name
    elapsed_s: float = 0.0          # Total elapsed seconds

    def to_dict(self) -> dict:
        """Serialize to dict for JSON/SSE."""
        return {
            "phase": self.phase.name,
            "progress_pct": self.progress_pct,
            "message": self.message,
            "error": self.error,
            "camera_name": self.camera_name,
            "camera_address": self.camera_address,
            "wifi_ssid": self.wifi_ssid,
            "camera_ip": self.camera_ip,
            "transport_state": self.transport_state,
            "elapsed_s": round(self.elapsed_s, 1),
        }


# Callback type: called with OrchestratorStatus on each phase/progress change
StatusCallback = Callable[[OrchestratorStatus], None]


# ---------------------------------------------------------------------------
# Timeouts and retry config
# ---------------------------------------------------------------------------

BLE_SCAN_TIMEOUT_S = 15.0
BLE_CONNECT_TIMEOUT_S = 15.0
PROVISION_TIMEOUT_S = 90.0
COHN_DISCOVER_TIMEOUT_S = 15.0
COHN_CONNECT_TIMEOUT_S = 20.0
COHN_STREAM_START_TIMEOUT_S = 25.0

MAX_BLE_RETRIES = 2
MAX_PROVISION_RETRIES = 2
MAX_COHN_CONNECT_RETRIES = 3
RETRY_DELAY_S = 2.0


# ---------------------------------------------------------------------------
# COHNOrchestrator
# ---------------------------------------------------------------------------


class COHNOrchestrator:
    """Sequences BLE provisioning then hands off to COHN transport.

    Uses BLEProvisioningService (wrapping open-gopro SDK) for the BLE
    and COHN provisioning phases.  The hand-rolled BLE stack is dead code.

    Usage (from dashboard endpoint):

        orchestrator = COHNOrchestrator()

        # Option A: Full provisioning + connect
        transport = await orchestrator.provision_and_connect(
            wifi_ssid="MyNetwork",
            wifi_password="secret",
            on_status=lambda s: sse_publish(s.to_dict()),
        )

        # Option B: Connect with existing credentials
        transport = await orchestrator.connect_with_credentials(
            credentials=stored_creds,
            on_status=lambda s: sse_publish(s.to_dict()),
        )

    Args:
        ble_scan_timeout: BLE scan duration in seconds.
        ble_adapter: BLE adapter identifier (None = system default).
    """

    def __init__(
        self,
        ble_scan_timeout: float = BLE_SCAN_TIMEOUT_S,
        ble_adapter: str | None = None,
    ):
        self._ble_scan_timeout = ble_scan_timeout
        self._ble_adapter = ble_adapter
        self._status = OrchestratorStatus()
        self._callbacks: list[StatusCallback] = []
        self._cancelled = False
        self._start_time: float = 0.0

        # Components — created during orchestration
        self._provisioning_service: Any = None
        self._transport: Any = None  # COHNTransport

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def provision_and_connect(
        self,
        wifi_ssid: str,
        wifi_password: str,
        ble_address: str | None = None,
        on_status: StatusCallback | None = None,
    ) -> Any:
        """Full orchestration: BLE provision → COHN connect.

        Uses open-gopro SDK for the BLE + COHN provisioning, then
        creates a COHNTransport for streaming.

        Args:
            wifi_ssid: Home WiFi network SSID.
            wifi_password: Home WiFi network password.
            ble_address: Optional specific BLE address/serial suffix.
                If None, scans for the first available GoPro.
            on_status: Callback for status updates (for dashboard SSE).

        Returns:
            COHNTransport instance in CONNECTED state, or None on failure.
        """
        if on_status:
            self._callbacks.append(on_status)

        self._cancelled = False
        self._start_time = time.monotonic()

        try:
            # Phase 1-3: BLE scan + connect + COHN provision
            # (via BLEProvisioningService / open-gopro)
            credentials = await self._phase_provision_via_sdk(
                wifi_ssid=wifi_ssid,
                wifi_password=wifi_password,
                target=ble_address,
            )

            if credentials is None:
                if self._status.phase != OrchestratorPhase.FAILED:
                    self._fail("COHN provisioning failed")
                return None

            if self._cancelled:
                return self._fail_cancelled()

            # Persist credentials to local cache for restart survival.
            # BLEProvisioningService also persists, but this is a safety net
            # in case the orchestrator obtained credentials by a different path.
            self._cache_credentials(credentials)

            # Phase 4: COHN Transport Connect
            transport = await self._phase_cohn_connect(credentials)
            if transport is None:
                if self._status.phase != OrchestratorPhase.FAILED:
                    self._fail("Failed to connect via COHN")
                return None

            # Done!
            self._update_status(
                OrchestratorPhase.COMPLETE, 100,
                f"COHN connected! Camera streaming at {credentials.ip_address}",
            )
            self._transport = transport
            return transport

        except asyncio.CancelledError:
            return self._fail_cancelled()
        except Exception as e:
            log.error("COHN orchestration failed: %s", e, exc_info=True)
            self._fail(f"Unexpected error: {e}")
            return None
        finally:
            await self._cleanup_provisioning()
            if on_status and on_status in self._callbacks:
                self._callbacks.remove(on_status)

    async def connect_with_credentials(
        self,
        credentials: Any,
        on_status: StatusCallback | None = None,
    ) -> Any:
        """Connect to COHN using previously stored credentials.

        Skips BLE provisioning entirely — goes straight to COHN transport.
        Used when credentials are already saved in open-gopro's cohn_db.

        Args:
            credentials: COHNCredentials or CohnInfo from previous provisioning.
            on_status: Callback for status updates.

        Returns:
            COHNTransport instance in CONNECTED state, or None on failure.
        """
        if on_status:
            self._callbacks.append(on_status)

        self._cancelled = False
        self._start_time = time.monotonic()

        try:
            self._status.camera_ip = getattr(credentials, "ip_address", "")
            self._status.wifi_ssid = getattr(credentials, "ssid", "")

            transport = await self._phase_cohn_connect(credentials)
            if transport is None:
                if self._status.phase != OrchestratorPhase.FAILED:
                    self._fail("Failed to connect via COHN")
                return None

            self._update_status(
                OrchestratorPhase.COMPLETE, 100,
                f"COHN connected at {self._status.camera_ip}",
            )
            self._transport = transport
            return transport

        except Exception as e:
            log.error("COHN connect with credentials failed: %s", e, exc_info=True)
            self._fail(f"Connection failed: {e}")
            return None
        finally:
            if on_status and on_status in self._callbacks:
                self._callbacks.remove(on_status)

    def cancel(self) -> None:
        """Request cancellation of the orchestration."""
        self._cancelled = True
        log.info("COHN orchestration cancellation requested")

        # Cancel provisioning service if active
        if self._provisioning_service is not None:
            self._provisioning_service.cancel()

    @property
    def status(self) -> OrchestratorStatus:
        """Current orchestration status."""
        return self._status

    @property
    def transport(self) -> Any:
        """The COHN transport after successful orchestration (or None)."""
        return self._transport

    @property
    def is_active(self) -> bool:
        """True if orchestration is currently in progress."""
        return self._status.phase not in (
            OrchestratorPhase.IDLE,
            OrchestratorPhase.COMPLETE,
            OrchestratorPhase.FAILED,
            OrchestratorPhase.CANCELLED,
        )

    # ------------------------------------------------------------------
    # Phase 1-3: BLE Provision via open-gopro SDK
    # ------------------------------------------------------------------

    async def _phase_provision_via_sdk(
        self,
        wifi_ssid: str,
        wifi_password: str,
        target: str | None = None,
    ) -> Any:
        """Run BLE scan + connect + COHN provisioning via open-gopro.

        Delegates to BLEProvisioningService which wraps WirelessGoPro.
        Maps provisioning progress to orchestrator status updates.

        Returns COHNCredentials on success, None on failure.
        """
        from gomaxwebcam.provisioning import (
            BLEProvisioningService,
            ProvisionPhase,
            ProvisionProgress,
        )

        service = BLEProvisioningService(
            cohn_db_path=self._get_cohn_db_path(),
        )
        self._provisioning_service = service

        self._status.wifi_ssid = wifi_ssid

        # Map provisioning phases to orchestrator phases
        phase_map = {
            ProvisionPhase.BLE_SCANNING: OrchestratorPhase.BLE_SCANNING,
            ProvisionPhase.BLE_CONNECTING: OrchestratorPhase.BLE_CONNECTING,
            ProvisionPhase.WIFI_CONNECTING: OrchestratorPhase.PROVISIONING,
            ProvisionPhase.COHN_PROVISIONING: OrchestratorPhase.PROVISIONING,
            ProvisionPhase.COMPLETE: OrchestratorPhase.PROVISIONING,
            ProvisionPhase.FAILED: OrchestratorPhase.FAILED,
        }

        def _on_progress(progress: ProvisionProgress) -> None:
            orch_phase = phase_map.get(
                progress.phase, OrchestratorPhase.PROVISIONING,
            )
            # Scale provisioning progress to 0-60% of orchestrator progress
            scaled_pct = min(int(progress.progress_pct * 0.6), 60)

            if progress.phase == ProvisionPhase.FAILED:
                self._fail(progress.error or progress.message)
            else:
                self._update_status(orch_phase, scaled_pct, progress.message)

            if progress.camera_serial:
                self._status.camera_name = f"GoPro ({progress.camera_serial})"

        credentials = await service.provision(
            target=target,
            wifi_ssid=wifi_ssid,
            wifi_password=wifi_password,
            on_progress=_on_progress,
            ble_timeout=int(self._ble_scan_timeout),
            provision_timeout=int(PROVISION_TIMEOUT_S),
        )

        if credentials is not None and credentials.provisioned:
            self._status.camera_ip = credentials.ip_address
            self._update_status(
                OrchestratorPhase.PROVISIONING, 60,
                f"COHN provisioned! Camera IP: {credentials.ip_address}",
            )
            return credentials

        return None

    def _get_cohn_db_path(self):
        """Get the path to the COHN credential database."""
        from pathlib import Path
        try:
            import platformdirs
            config_dir = Path(platformdirs.user_config_dir("GoMaxWebcam-v2"))
            config_dir.mkdir(parents=True, exist_ok=True)
            return config_dir / "cohn_db.json"
        except ImportError:
            return Path("cohn_db.json")

    # ------------------------------------------------------------------
    # Phase 4: COHN Transport Connect
    # ------------------------------------------------------------------

    async def _phase_cohn_connect(self, credentials: Any) -> Any:
        """Create and connect the COHN transport.

        Returns COHNTransport in CONNECTED state, or None on failure.
        """
        ip = getattr(credentials, "ip_address", "")
        username = getattr(credentials, "username", "")
        password = getattr(credentials, "password", "")
        cert_path = getattr(credentials, "certificate", None)
        ssid = getattr(credentials, "ssid", "")

        self._status.camera_ip = ip
        self._status.wifi_ssid = ssid

        # Phase 4a: Discover (verify reachable)
        self._update_status(
            OrchestratorPhase.COHN_DISCOVERING, 65,
            f"Verifying camera at {ip}...",
        )

        for attempt in range(1, MAX_COHN_CONNECT_RETRIES + 1):
            try:
                from gomaxwebcam.transport.cohn import COHNTransport

                transport = COHNTransport(
                    ip_address=ip,
                    username=username,
                    password=password,
                    ssl_cert_path=cert_path if cert_path and not cert_path.startswith("-----") else None,
                )

                # If certificate is PEM text (not a path), we'll handle it
                # by disabling SSL verification (camera uses self-signed certs)

                # Inject credentials
                transport._inject_credentials(username, password)
                transport._inject_camera_ip(ip)

                # Discover (sets base_url and validates reachability)
                discovered = await asyncio.wait_for(
                    transport.discover(timeout=COHN_DISCOVER_TIMEOUT_S),
                    timeout=COHN_DISCOVER_TIMEOUT_S + 5,
                )

                if not discovered:
                    if attempt < MAX_COHN_CONNECT_RETRIES:
                        self._update_status(
                            OrchestratorPhase.COHN_DISCOVERING, 65,
                            f"Camera not reachable at {ip}, retrying ({attempt}/{MAX_COHN_CONNECT_RETRIES})...",
                        )
                        await asyncio.sleep(RETRY_DELAY_S * attempt)
                        continue
                    self._fail(f"Camera not reachable at {ip}")
                    return None

                # Phase 4b: Connect (HTTPS session)
                self._update_status(
                    OrchestratorPhase.COHN_CONNECTING, 80,
                    f"Connecting to camera at {ip} via HTTPS...",
                )

                connected = await asyncio.wait_for(
                    transport.connect(),
                    timeout=COHN_CONNECT_TIMEOUT_S + 5,
                )

                if connected:
                    self._status.transport_state = "CONNECTED"

                    # Phase 4c: Start stream
                    self._update_status(
                        OrchestratorPhase.COHN_STREAMING, 90,
                        f"Starting stream from {ip}...",
                    )

                    try:
                        stream_info = await asyncio.wait_for(
                            transport.start_stream(),
                            timeout=COHN_STREAM_START_TIMEOUT_S,
                        )
                        if stream_info is not None:
                            self._status.transport_state = "STREAMING"
                            self._update_status(
                                OrchestratorPhase.COHN_STREAMING, 95,
                                f"COHN streaming from {ip} (port {stream_info.port})",
                            )
                        else:
                            log.warning(
                                "Stream start returned None, transport is connected but not streaming"
                            )
                            self._update_status(
                                OrchestratorPhase.COHN_STREAMING, 95,
                                f"COHN transport connected to {ip} (stream start deferred)",
                            )
                    except (asyncio.TimeoutError, Exception) as e:
                        log.warning("Stream start failed (non-fatal): %s", e)
                        self._update_status(
                            OrchestratorPhase.COHN_STREAMING, 95,
                            f"COHN connected to {ip} (stream start deferred)",
                        )

                    return transport

                if attempt < MAX_COHN_CONNECT_RETRIES:
                    self._update_status(
                        OrchestratorPhase.COHN_CONNECTING, 80,
                        f"COHN connect failed, retrying ({attempt}/{MAX_COHN_CONNECT_RETRIES})...",
                    )
                    await transport.disconnect()
                    await asyncio.sleep(RETRY_DELAY_S * attempt)

            except asyncio.TimeoutError:
                log.warning("COHN connect attempt %d timed out", attempt)
                if attempt < MAX_COHN_CONNECT_RETRIES:
                    await asyncio.sleep(RETRY_DELAY_S * attempt)
            except Exception as e:
                log.warning("COHN connect attempt %d failed: %s", attempt, e)
                if attempt < MAX_COHN_CONNECT_RETRIES:
                    await asyncio.sleep(RETRY_DELAY_S * attempt)

        self._fail(f"Failed to connect to camera via COHN at {ip}")
        return None

    # ------------------------------------------------------------------
    # Status management
    # ------------------------------------------------------------------

    def _update_status(
        self,
        phase: OrchestratorPhase,
        progress_pct: int,
        message: str,
    ) -> None:
        """Update status and notify all callbacks."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0

        self._status = OrchestratorStatus(
            phase=phase,
            progress_pct=progress_pct,
            message=message,
            camera_name=self._status.camera_name,
            camera_address=self._status.camera_address,
            wifi_ssid=self._status.wifi_ssid,
            camera_ip=self._status.camera_ip,
            transport_state=self._status.transport_state,
            elapsed_s=elapsed,
        )

        log.info("[COHN Orchestrator] %d%% - %s (%.1fs)", progress_pct, message, elapsed)
        self._notify_callbacks()

    def _fail(self, error_msg: str) -> None:
        """Set orchestration to FAILED and notify callbacks."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0

        self._status = OrchestratorStatus(
            phase=OrchestratorPhase.FAILED,
            progress_pct=self._status.progress_pct,
            message=f"Failed: {error_msg}",
            error=error_msg,
            camera_name=self._status.camera_name,
            camera_address=self._status.camera_address,
            wifi_ssid=self._status.wifi_ssid,
            camera_ip=self._status.camera_ip,
            transport_state=self._status.transport_state,
            elapsed_s=elapsed,
        )

        log.error("[COHN Orchestrator] FAILED: %s", error_msg)
        self._notify_callbacks()

    def _fail_cancelled(self) -> None:
        """Set orchestration to CANCELLED and notify callbacks."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0

        self._status = OrchestratorStatus(
            phase=OrchestratorPhase.CANCELLED,
            progress_pct=self._status.progress_pct,
            message="Provisioning cancelled by user",
            camera_name=self._status.camera_name,
            camera_address=self._status.camera_address,
            wifi_ssid=self._status.wifi_ssid,
            camera_ip=self._status.camera_ip,
            elapsed_s=elapsed,
        )

        log.info("[COHN Orchestrator] Cancelled by user")
        self._notify_callbacks()
        return None

    def _notify_callbacks(self) -> None:
        """Fire all registered status callbacks."""
        for cb in self._callbacks:
            try:
                cb(self._status)
            except Exception:
                log.exception("Error in orchestrator status callback")

    # ------------------------------------------------------------------
    # Credential caching
    # ------------------------------------------------------------------

    def _cache_credentials(self, credentials: Any) -> bool:
        """Cache COHN credentials to local persistence after provisioning.

        Writes credentials to the CohnCredentialStore so they survive
        app restarts. Uses the same DB path as BLEProvisioningService.

        Args:
            credentials: COHNCredentials from provisioning (must have
                ip_address, username, password, certificate, camera_serial).

        Returns:
            True if cached successfully, False on error.
        """
        ip = getattr(credentials, "ip_address", "")
        username = getattr(credentials, "username", "")
        password = getattr(credentials, "password", "")
        certificate = getattr(credentials, "certificate", "")
        serial = getattr(credentials, "camera_serial", "")

        if not (ip and username and password):
            log.debug(
                "Skipping credential cache: incomplete credentials "
                "(ip=%s, user=%s, serial=%s)", ip, username, serial,
            )
            return False

        try:
            from gomaxwebcam.transport.cohn_persistence import (
                CohnCredentialStore,
                StoredCOHNCredentials,
            )

            store = CohnCredentialStore(db_path=self._get_cohn_db_path())
            stored = StoredCOHNCredentials(
                ip_address=ip,
                username=username,
                password=password,
                certificate=certificate,
                camera_serial=serial,
            )
            ok = store.store_credentials(stored)
            if ok:
                log.info(
                    "Credentials cached for serial %s (ip=%s)", serial, ip,
                )
            return ok

        except Exception as e:
            log.warning("Failed to cache credentials: %s", e)
            return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cleanup_provisioning(self) -> None:
        """Clean up provisioning service resources."""
        if self._provisioning_service is not None:
            try:
                await self._provisioning_service.close()
            except Exception as e:
                log.debug("Provisioning service cleanup error: %s", e)
            self._provisioning_service = None

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def _inject_provisioning_service(self, mock_service: Any) -> None:
        """For testing: inject a mock BLEProvisioningService."""
        self._provisioning_service = mock_service
